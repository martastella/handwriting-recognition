# Functions for the unconditional and conditional generation
import numpy as np
import torch
from torch.autograd import Variable
from utils.plots import plot_stroke, attention_plot

from models.handwriting_model import MixtureOfGaussiansHandwritingModel, HandwritingGenerationModel

is_cuda_available = torch.cuda.is_available()

def generate_unconditionally(
        cell_size=400, 
        num_clusters=20, 
        num_steps=800, 
        random_seed=42,
        model_checkpoint='save\\unconditional_handwriting_last_epoch_200.pt'):
    
    handwriting_model = MixtureOfGaussiansHandwritingModel(cell_size, num_clusters)

    model_checkpoint_data = torch.load(model_checkpoint, weights_only=True)
    handwriting_model.load_state_dict(model_checkpoint_data['model'])
    
    np.random.seed(random_seed)

    zero_tensor = torch.zeros((1,1,3)) # (batch_size, sequence_length, data_dimensions)

    initial_states  = [torch.zeros((1,1, cell_size))]*4  # [h1, c1, h2, c2] for both LSTM layers

    if is_cuda_available:
        handwriting_model.cuda()
        zero_tensor = zero_tensor.cuda()
        initial_states  = [state.cuda() for state in initial_states ]

    input_tensor = Variable(zero_tensor)

    h1_init, c1_init, h2_init, c2_init = initial_states
    prev_state = (h1_init, c1_init)  
    prev_state2 = (h2_init, c2_init) 
    
    stroke_record = [np.array([0, 0, 0])]  # (x, y, end_of_stroke)

    for _ in range(num_steps):
        end_of_stroke, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, p, prev_state, prev_state2 = handwriting_model(input_tensor, prev_state, prev_state2)

        end_probability = end_of_stroke.data[0][0][0]
        sample_end = np.random.binomial(1, end_probability.cpu().item())
        
        sample_index = np.random.choice(range(num_clusters), p=weights.data[0][0].cpu().numpy())
        
        mu = np.array([mu_1.data[0][0][sample_index].cpu(), mu_2.data[0][0][sample_index].cpu()])

        sigma_1 = log_sigma_1.exp().data[0][0][sample_index]  
        sigma_2 = log_sigma_2.exp().data[0][0][sample_index]  
        covariance = np.array([
            [sigma_1.cpu().item()**2, p.data[0][0][sample_index].cpu().item() * sigma_1.cpu().item() * sigma_2.cpu().item()],
            [p.data[0][0][sample_index].cpu().item() * sigma_1.cpu().item() * sigma_2.cpu().item(), sigma_2.cpu().item()**2]
        ])

        new_stroke_point = np.random.multivariate_normal(mu, covariance)
        
        stroke_point_with_end = np.insert(new_stroke_point, 0, sample_end)
        
        stroke_record.append(stroke_point_with_end)
        
        input_tensor = torch.from_numpy(stroke_point_with_end).type(torch.FloatTensor)
        
        if is_cuda_available:
            input_tensor = input_tensor.cuda()
        
        input_tensor = Variable(input_tensor, requires_grad=False)
        input_tensor = input_tensor.view((1, 1, 3))  # (batch_size, sequence_length, data_dimensions)
    
    plot_stroke(np.array(stroke_record))
    
    
def generate_conditionally(
        text, 
        cell_size=400, 
        num_clusters=20, 
        num_attention_params=10, 
        random_seed=42,
        bias=1., 
        bias2=1., 
        model_checkpoint='save\\conditional_handwriting_last_epoch_200.pt'):
    
    char_to_code = torch.load('char_to_code_generation.pt', weights_only=True)
    np.random.seed(random_seed)

    text += ' ' 
    
    handwriting_model = HandwritingGenerationModel(len(text), len(char_to_code)+1, cell_size, num_clusters, num_attention_params)
    handwriting_model.load_state_dict(torch.load(model_checkpoint, weights_only=True)['model'])
    
    one_hot_matrix = np.zeros((len(text), len(char_to_code)+1))
    for _ in range(len(text)):
        try:
            one_hot_matrix[_][char_to_code[text[_]]] = 1
        except:
            one_hot_matrix[_][-1] = 1
    
    input_tensor = torch.zeros((1, 1, 3)) 
    h1_init, c1_init = torch.zeros((1, cell_size)), torch.zeros((1, cell_size))  
    h2_init, c2_init = torch.zeros((1, 1, cell_size)), torch.zeros((1, 1, cell_size))  
    kappa_init = torch.zeros(1, num_attention_params)  
    
    one_hot_tensor = torch.from_numpy(one_hot_matrix).type(torch.FloatTensor)
    text_length_tensor = torch.from_numpy(np.array([[len(text)]])).type(torch.FloatTensor)
    
    if is_cuda_available:
        handwriting_model.cuda()
        input_tensor = input_tensor.cuda()
        h1_init, c1_init = h1_init.cuda(), c1_init.cuda()
        h2_init, c2_init = h2_init.cuda(), c2_init.cuda()
        kappa_init = kappa_init.cuda()
        one_hot_tensor = one_hot_tensor.cuda()
        text_length_tensor = text_length_tensor.cuda()
        
    current_input = Variable(input_tensor)
    h1_init, c1_init = Variable(h1_init), Variable(c1_init)
    h2_init, c2_init = Variable(h2_init), Variable(c2_init)
    prev_states = (h1_init, c1_init)  
    prev_states2 = (h2_init, c2_init) 
    kappa_old = Variable(kappa_init)
    one_hot_tensor = Variable(one_hot_tensor, requires_grad=False)
    attention_weight = one_hot_tensor.narrow(0, 0, 1)  
    text_length_tensor = Variable(text_length_tensor)
    
    stroke_record = [np.zeros(3)]  
    attention_weights_over_time = []  
    stop_generation = False  
    step_count = 0  

    while not stop_generation:    
        outputs = handwriting_model(current_input, one_hot_tensor, text_length_tensor, attention_weight, 
                                    kappa_old, prev_states, prev_states2, bias)
        
        (end_probability, weights, mu_x, mu_y, log_sigma_x, log_sigma_y, rho, attention_weight, kappa_old, 
         prev_states, prev_states2, phi) = outputs
        
        end_prob_value = end_probability.detach().cpu().item()  
        sample_end = np.random.binomial(1, end_prob_value)
        
        weights_np = weights.detach().cpu().numpy()
        sampled_cluster_index = np.random.choice(range(num_clusters), p=weights_np[0][0])
        
        mu = np.array([mu_x.detach().cpu()[0][0][sampled_cluster_index].item(), 
                       mu_y.detach().cpu()[0][0][sampled_cluster_index].item()])
        adjusted_log_sigma_x = log_sigma_x - bias2
        adjusted_log_sigma_y = log_sigma_y - bias2
        variance_x = adjusted_log_sigma_x.exp().detach().cpu()[0][0][sampled_cluster_index].item() ** 2
        variance_y = adjusted_log_sigma_y.exp().detach().cpu()[0][0][sampled_cluster_index].item() ** 2
        covariance = (rho.detach().cpu()[0][0][sampled_cluster_index] * 
                      adjusted_log_sigma_x.exp().detach().cpu()[0][0][sampled_cluster_index] * 
                      adjusted_log_sigma_y.exp().detach().cpu()[0][0][sampled_cluster_index]).item()
        covariance_matrix = np.array([[variance_x, covariance], [covariance, variance_y]])
        
        sampled_point = np.random.multivariate_normal(mu, covariance_matrix)
        
        stroke_point_with_end = np.insert(sampled_point, 0, sample_end)
        stroke_record.append(stroke_point_with_end)
        
        current_input = torch.from_numpy(stroke_point_with_end).type(torch.FloatTensor)
        if is_cuda_available:
            current_input = current_input.cuda()
        current_input = Variable(current_input, requires_grad=False)
        current_input = current_input.view(1, 1, 3)
        
        phi = phi.squeeze(0)
        attention_weights_over_time.append(phi.detach().cpu())  
        phi_values = phi.detach().cpu().numpy()  
        
        if step_count >= 20 and np.max(phi_values) == phi_values[-1]:
            stop_generation = True
        step_count += 1
    
    attention_weights_matrix = torch.stack(attention_weights_over_time).numpy().T
    plot_stroke(np.array(stroke_record))
    attention_plot(attention_weights_matrix)