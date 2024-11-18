# Handwriting model for recognition
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        combined = torch.cat((hidden, encoder_outputs), dim=2)
        energy = self.attention(combined).squeeze(2)
        attention_weights = F.softmax(energy, dim=1).unsqueeze(1)

        context = torch.bmm(attention_weights, encoder_outputs)
        return context.squeeze(1)
        
class FocalCTLoss(nn.Module):
    def __init__(self, blank = 0, zero_infinity = False, gamma = 2):
        super(FocalCTLoss, self).__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')
        self.gamma = gamma
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        pt = torch.exp(-loss)
        focal_loss = ((1-pt)**self.gamma*loss).mean()
        return focal_loss
    
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x):
        res, _ = self.lstm(x)
        proj = self.proj(x)
        return res + proj

class DeepHandwritingRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=0.5):
        super(DeepHandwritingRecognitionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.residual_layers = nn.ModuleList([ResidualLSTM(hidden_size * 2, hidden_size) for _ in range(num_layers - 1)])
        self.attention = AttentionLayer(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 4, output_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 4)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.initial_lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        for layer in self.residual_layers:
            outputs = layer(outputs)

        hidden = outputs[:, -1, :]
        context = self.attention(hidden, outputs)
        
        combined = torch.cat((outputs, context.unsqueeze(1).repeat(1, outputs.size(1), 1)), dim=2)
        combined = self.dropout(combined)
        combined = self.batch_norm(combined.transpose(1, 2)).transpose(1, 2)
        
        output = self.fc(combined)
        log_probs = F.log_softmax(output, dim=-1)
        
        return log_probs