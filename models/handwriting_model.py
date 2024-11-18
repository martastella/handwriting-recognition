# Handwriting model for conditional and unconditional generation 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

is_cuda_available = torch.cuda.is_available()

class AttentionWindow(nn.Module):
    def __init__(self, padded_text_length, cell_size, heads):
        super(AttentionWindow, self).__init__()
        # Linear layer that outputs 3*heads components, corresponding to different parts of attention (alpha, beta, previous_attention)
        self.linear = nn.Linear(cell_size, 3*heads) # cell size as input, output size is 3*heads
        self.padded_text_length = padded_text_length

    def forward(
            self, 
            x,
            old_attention,
            one_hots,
            text_lengths):
        
        # Apply the linear transformation to get parameters alpha, beta and previous attention
        parameters = self.linear(x).exp()  # Exponentiate to ensure positive values
        alpha, beta, previous_attention = parameters.chunk(3, dim=-1)
        
        # Update the attention focus (previous_attention) with the new one
        attention = old_attention + previous_attention

        # Create indices for the padded sequence length (for computing distance influence)
        idxs = torch.from_numpy(np.array(range(self.padded_text_length + 1))).type(torch.FloatTensor)
        if is_cuda_available:
            idxs = idxs.cuda()  # Move idxs to the GPU if CUDA is available
        idxs = torch.autograd.Variable(idxs, requires_grad=False)

        # Compute the distance influence of each index from the current attention focus, scaled by beta
        distance_influence = -beta.unsqueeze(2) * (attention.unsqueeze(2).repeat(1, 1, self.padded_text_length + 1) - idxs) ** 2
        
        # Compute the final attention values after applying the attention mechanism
        scaled_attention_scores = (alpha.unsqueeze(2) * distance_influence.exp()).sum(dim=1) * (self.padded_text_length / text_lengths)
        
        # Compute the final weighted attention by applying the attention scores to the one-hot encoded inputs
        final_weighted_attention = (scaled_attention_scores.narrow(-1, 0, self.padded_text_length).unsqueeze(2) * one_hots).sum(dim=1) 
        
        return final_weighted_attention, attention, scaled_attention_scores
    
class MixtureOfGaussiansHandwritingModel(nn.Module):
    def __init__(self, cell_size, nclusters):
        super(MixtureOfGaussiansHandwritingModel, self).__init__()
        
        # Initialize two LSTM layers
        self.lstm_first = nn.LSTM(input_size=3, hidden_size=cell_size, num_layers=1, batch_first=True)
        self.lstm_second = nn.LSTM(input_size=cell_size + 3, hidden_size=cell_size, num_layers=1, batch_first=True)
        
        # Linear layer to combine the LSTM outputs and produce parameters for the mixture of Gaussians
        self.linear = nn.Linear(cell_size * 2, 1 + nclusters * 6)
        
        # Tanh activation for scaling rho parameter
        self.tanh = nn.Tanh()
        
    def forward(
            self, 
            x, 
            prev_state_first, 
            prev_state_second):
        
        # First LSTM forward pass
        hidden_state_1, (hidden_state_1_n, cell_state_1_n) = self.lstm_first(x, prev_state_first)
        
        # Create a skip connection by concatenating the output of the first LSTM with the input
        skip_connection_input = torch.cat([hidden_state_1, x], dim=-1)
        
        # Second LSTM forward pass with the skip connection
        hidden_state_2, (hidden_state_2_n, cell_state_2_n) = self.lstm_second(skip_connection_input, prev_state_second)
        
        # Combine both hidden states (from both LSTM layers) into a single tensor for the next layer
        combined_hidden_states = torch.cat([hidden_state_1, hidden_state_2], dim=-1)
        
        # Compute parameters for the mixture of Gaussians (MoG)
        params = self.linear(combined_hidden_states)
        mog_params = params.narrow(-1, 0, params.size()[-1]-1)
        
        # Split the MoG parameters into individual components
        mixing_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho_raw = mog_params.chunk(6, dim=-1)
        
        # Apply softmax to the weights to get the final mixture weights
        mixing_weights = F.softmax(mixing_weights, dim=-1)
        
        # Apply Tanh to rho for scaling
        rho = self.tanh(rho_raw)
        
        # The final output (sigmoid) determines the end of the sequence (binary decision)
        end_of_sequence = F.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
        
        return (
            end_of_sequence, 
            mixing_weights, 
            mu_1, 
            mu_2, 
            log_sigma_1, 
            log_sigma_2,
            rho, 
            (hidden_state_1_n, cell_state_1_n), 
            (hidden_state_2_n, cell_state_2_n)
        )

class LSTMWithAttention(nn.Module):
    def __init__(self, padded_text_length, len_vocabulary, cell_size, heads):
        super(LSTMWithAttention, self).__init__()
        # Initialize LSTMCell with input size: 3 (e.g., previous states, etc.) + vocab size
        self.lstm = nn.LSTMCell(3 + len_vocabulary, cell_size) 
        # Initialize attention mechanism
        self.window = AttentionWindow(padded_text_length, cell_size, heads) 
        
    def forward(
            self, 
            x, 
            onehots, 
            text_lengths, 
            previous_attention_weights, 
            previous_attention_state, 
            lstm_state):
        
        hidden_states = []  # Stores hidden states across time steps
        attention_weights = []  # Stores attention weights across time steps
        
         # Loop through the sequence (iterate over time steps)
        for _ in range(x.size()[1]): 
            # Concatenate input at current time step with the previous attention weights
            cell_input = torch.cat([x.narrow(1, _, 1).squeeze(1), previous_attention_weights], dim=-1)
            
            # Compute LSTM cell output
            lstm_state = self.lstm(cell_input, lstm_state)
            
            # Update attention weights and state using the attention window
            previous_attention_weights, previous_attention_state, attention_scores = self.window(
                lstm_state[0], previous_attention_state, onehots, text_lengths)
            
            # Store the hidden states and attention weights
            hidden_states.append(lstm_state[0])
            attention_weights.append(previous_attention_weights)

        # Stack and permute results to match expected dimensions (sequence_length, batch_size, dim)
        final_attention_weights = torch.stack(attention_weights, dim=0).permute(1, 0, 2)
        final_hidden_states = torch.stack(hidden_states, dim=0).permute(1, 0, 2)
        
        return (
            final_attention_weights,  # Attention weights (sequence_length, batch_size, attention_dim)
            final_hidden_states,  # Hidden states (sequence_length, batch_size, cell_size)
            lstm_state,  # Final LSTM state
            previous_attention_weights,  # Last attention weights
            previous_attention_state,  # Last attention state
            attention_scores  # Last attention scores
        )
    
class LSTMOutputLayer(nn.Module):
    def __init__(self, vocab_len, cell_size):
        super(LSTMOutputLayer, self).__init__()
        # LSTM layer with input size: 3 (e.g., previous states, etc.) + vocab_len + cell_size
        self.lstm = nn.LSTM(3 + vocab_len + cell_size,  # Input size: 3 + vocab_len + cell_size
                            cell_size,  # Hidden state size
                            num_layers=1,  # One layer LSTM
                            batch_first=True)  # Batch is the first dimension
        
    def forward(
            self, 
            x, 
            attention_weights, 
            first_lstm_hidden_states, 
            lstm_state):
        
        # Concatenate input data, attention weights, and hidden states from first LSTM layer
        lstm_input = torch.cat([x, attention_weights, first_lstm_hidden_states], dim=-1)
        
        # Pass the concatenated input through the LSTM layer
        lstm_output, lstm_state = self.lstm(lstm_input, lstm_state)
        
        return lstm_output, lstm_state


class HandwritingGenerationModel(nn.Module):
    def __init__(self, padded_text_len, len_vocabulary, cell_size, nclusters, heads):
        super(HandwritingGenerationModel, self).__init__()
        # Initialize the first LSTM with attention mechanism
        self.lstm1 = LSTMWithAttention(padded_text_len, len_vocabulary, cell_size, heads)
        # Initialize the second LSTM layer for output generation
        self.lstm2 = LSTMOutputLayer(len_vocabulary, cell_size)
        # Linear layer to combine outputs and generate final parameters
        self.linear = nn.Linear(cell_size*2, 1 + nclusters*6)
        # Tanh activation to scale rho
        self.tanh = nn.Tanh()
        
    def forward(
            self, 
            x, 
            onehots, 
            text_lens, 
            previous_attention_weights, 
            previous_attention_state, 
            lstm_state_1, 
            lstm_state_2, 
            bias=0.):
        
        # Initialize the LSTM states if not provided
        if lstm_state_1 is None:
            lstm_state_1 = (torch.zeros(1, x.size(0), lstm_state_1[0].size(-1)).to(x.device),
                             torch.zeros(1, x.size(0), lstm_state_1[0].size(-1)).to(x.device))
        
        if lstm_state_2 is None:
            lstm_state_2 = (torch.zeros(1, x.size(0), lstm_state_2[0].size(-1)).to(x.device),
                             torch.zeros(1, x.size(0), lstm_state_2[0].size(-1)).to(x.device))

        # First LSTM layer with attention mechanism
        attention_weights, hidden_states_1, lstm_state_1, previous_attention_weights, previous_attention_state, attention_scores = self.lstm1(x, onehots, text_lens, previous_attention_weights, previous_attention_state, lstm_state_1)
        
        # Second LSTM layer with context (hidden states and attention weights)
        hidden_states_2, lstm_state_2 = self.lstm2(x, attention_weights, hidden_states_1, lstm_state_2)
        
        # Concatenate the hidden states from both LSTM layers
        output_params = self.linear(torch.cat([hidden_states_1, hidden_states_2], dim=-1))
        
        # Extract mixture of Gaussian parameters from the output
        mog_parameters = output_params.narrow(-1, 0, output_params.size()[-1] - 1)
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho = mog_parameters.chunk(6, dim=-1)
        
        # Apply softmax to the weights, tanh to rho, and sigmoid to the end parameter
        weights = F.softmax(pre_weights * (1 + bias), dim=-1)
        rho = self.tanh(pre_rho)
        end = F.sigmoid(output_params.narrow(-1, output_params.size()[-1] - 1, 1))
        
        # Return the final generated parameters
        return (
            end, 
            weights, 
            mu_1, 
            mu_2, 
            log_sigma_1, 
            log_sigma_2, 
            rho, 
            previous_attention_weights, 
            previous_attention_state, 
            lstm_state_1, 
            lstm_state_2, 
            attention_scores
        )
