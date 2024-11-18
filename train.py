# Training module for uncoditional and conditional generation
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os 
import re
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
plt.switch_backend('agg')

from models.handwriting_model import MixtureOfGaussiansHandwritingModel, HandwritingGenerationModel
from utils.train_utils import decay_learning_rate, save_checkpoint, load_data

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    print("🚀 CUDA is available! Training on GPU.")
else:
    print("⚠️ CUDA not available. Training on CPU.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='random_write', help='Task mode: "random_write" or "synthesis"')
    parser.add_argument('--cell_size', type=int, default=400, help='Size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=50, help='Mini-batch size for training')
    parser.add_argument('--timesteps', type=int, default=800, help='Sequence length for LSTM')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--model_save_dir', type=str, default='save', help='Directory to save the trained model')
    parser.add_argument('--learning_rate', type=float, default=8E-4, help='Initial learning rate for optimizer')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='Learning rate decay per epoch')
    parser.add_argument('--num_clusters', type=int, default=20, help='Number of Gaussian clusters for stroke prediction')
    parser.add_argument('--attention_clusters', type=int, default=10, help='Number of attention clusters on text input')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training ("cuda" or "cpu")')
    args = parser.parse_args()

    train_data = load_data('generation', 'train', args.device)
    validation_data = load_data('generation', 'validation', args.device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    if args.task == 'unconditional_handwriting':
        train_unconditional_handwriting(args, train_loader, validation_loader)
    elif args.task == 'conditional_handwriting':
        train_conditional_handwriting(args, train_loader, validation_loader)


def train_unconditional_handwriting(args, train_loader, validation_loader):
    model = MixtureOfGaussiansHandwritingModel(args.cell_size, args.num_clusters)
    if is_cuda_available:
        model = model.cuda()
    optimizer = optim.Adam([{'params': model.parameters()}], lr=args.learning_rate)

    checkpoint_files = [f for f in os.listdir(args.model_save_dir) if re.match(f"{args.task}_last_epoch_\\d+.pt", f)]
    if checkpoint_files != []:
        epochs = [int(re.search(r"(\d+).pt$", f).group(1)) for f in checkpoint_files]
        latest_epoch = max(epochs)

        checkpoint_name = f"{args.task}_last_epoch_{latest_epoch}.pt"
        checkpoint_path = os.path.join(args.model_save_dir, checkpoint_name)
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = latest_epoch  # Start from the next epoch
        lowest_val_loss = checkpoint['validation_loss']
        print(f"Resuming from epoch {start_epoch} with validation loss {lowest_val_loss:.4f}")
    else:
        print("No previous checkpoints found. Starting training from scratch.")
        start_epoch = 0
        lowest_val_loss = float('inf')

    init_states = [torch.zeros((1,args.batch_size,args.cell_size))]*4
    if is_cuda_available:
        init_states  = [state.cuda() for state in init_states]
    init_states  = [Variable(state, requires_grad = False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init = init_states

    training_loss, validation_loss_history = [], []
    
    start_time = time.time()

    print("\n" + "=" * 30)
    print("   🚀 Start training... 🚀")
    print("=" * 30 + "\n")
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        total_train_loss = 0

        for batch_index, (stroke_data, stroke_masks, text_onehot, text_lengths) in enumerate(train_loader):
            step_back = stroke_data.narrow(1,0,args.timesteps)
            x = Variable(step_back, requires_grad=False)
            stroke_masks = Variable(stroke_masks, requires_grad=False)
            stroke_masks = stroke_masks.narrow(1,0,args.timesteps)
            
            optimizer.zero_grad()

            outputs = model(x, (h1_init, c1_init), (h2_init, c2_init))
            end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho , prev, prev2 = outputs
            
            stroke_data = stroke_data.narrow(1,1,args.timesteps)
            y = Variable(stroke_data, requires_grad=False)

            loss = -calculate_log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, stroke_masks)/torch.sum(stroke_masks)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

            if batch_index % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Computed Loss: {:.6f}'.format(\
                    epoch+1, batch_index * len(stroke_data), len(train_loader.dataset),
                    100. * batch_index / len(train_loader),
                    loss.item()))

        average_train_loss = total_train_loss/(len(train_loader.dataset)//args.batch_size)
        training_loss.append(average_train_loss)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch+1, average_train_loss))

        loss = evaluate_model(args, model, validation_loader, h1_init, c1_init, h2_init, c2_init, None, None, unconditional=True)
        validation_loss = loss.item()
        validation_loss_history.append(validation_loss)
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(epoch+1, validation_loss))

        # Decay learning rate every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #    optimizer = decay_learning_rate(optimizer, args.decay_rate)

        # Save model checkpoint for the last epoch
        filename = f"{args.task}_last_epoch_{epoch + 1}.pt"

        old_checkpoint = f"{args.task}_last_epoch_{epoch}.pt"
        old_checkpoint_path = os.path.join(args.model_save_dir, old_checkpoint)
        if os.path.exists(old_checkpoint_path):
            os.remove(old_checkpoint_path)

        save_checkpoint(epoch, model, validation_loss, optimizer, args.model_save_dir, filename)

        print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds.")

def train_conditional_handwriting(args, train_loader, validation_loader):
    text_length, vocabulary_size = train_loader.dataset[0][2].size()
    
    model = HandwritingGenerationModel(text_length, vocabulary_size, args.cell_size, args.num_clusters, args.attention_clusters)
    if is_cuda_available:
        model = model.cuda()
    
    optimizer = optim.Adam([{'params': model.parameters()}], lr=args.learning_rate)

    checkpoint_files = [f for f in os.listdir(args.model_save_dir) if re.match(f"{args.task}_last_epoch_\\d+.pt", f)]
    if checkpoint_files != []:
        epochs = [int(re.search(r"(\d+).pt$", f).group(1)) for f in checkpoint_files]
        latest_epoch = max(epochs)
        
        checkpoint_name = f"{args.task}_last_epoch_{latest_epoch}.pt"
        checkpoint_path = os.path.join(args.model_save_dir, checkpoint_name)
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = latest_epoch  
        lowest_val_loss = checkpoint['validation_loss']
        print(f"Resuming from epoch {start_epoch} with validation loss {lowest_val_loss:.4f}")
    else:
        print("No previous checkpoints found. Starting training from scratch.")
        start_epoch = 0
        lowest_val_loss = float('inf')

    h1_init, c1_init = torch.zeros((args.batch_size, args.cell_size)), torch.zeros((args.batch_size, args.cell_size))
    h2_init, c2_init = torch.zeros((1, args.batch_size, args.cell_size)), torch.zeros((1, args.batch_size, args.cell_size))
    previous_kappa = torch.zeros(args.batch_size, args.attention_clusters)
    
    if is_cuda_available:
        h1_init, c1_init, h2_init, c2_init, previous_kappa = h1_init.cuda(), c1_init.cuda(), h2_init.cuda(), c2_init.cuda(), previous_kappa.cuda()
        
    h1_init, c1_init, h2_init, c2_init, previous_kappa = map(lambda t: Variable(t, requires_grad=False), 
                                                             [h1_init, c1_init, h2_init, c2_init, previous_kappa])
    
    training_loss, validation_loss_history = [], []

    start_time = time.time()
    
    print("\n" + "=" * 30)
    print("   🚀 Start training... 🚀")
    print("=" * 30 + "\n")
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        total_train_loss = 0
        
        for batch_idx, (input_data, masks, text_onehot, text_lengths) in enumerate(train_loader):
            step_back = input_data.narrow(1,0,args.timesteps)
            x = Variable(step_back, requires_grad=False)
            text_onehot = Variable(text_onehot, requires_grad = False)
            masks = Variable(masks, requires_grad=False)
            masks = masks.narrow(1,0,args.timesteps)
            text_lengths = Variable(text_lengths, requires_grad=False)
            
            w_old = text_onehot.narrow(1,0,1).squeeze()
            
            optimizer.zero_grad()
            
            outputs = model(x, text_onehot, text_lengths, w_old, previous_kappa, (h1_init, c1_init), (h2_init, c2_init))
            end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, prev, prev2, old_phi = outputs
            
            input_data = input_data.narrow(1,1,args.timesteps)
            y = Variable(input_data, requires_grad=False)
            loss = -calculate_log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)/torch.sum(masks)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(input_data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
        
        average_train_loss = total_train_loss / (len(train_loader.dataset) // args.batch_size)
        training_loss.append(average_train_loss)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch+1, average_train_loss))
        
        loss = evaluate_model(args, model, validation_loader, h1_init, c1_init, h2_init, c2_init, w_old, previous_kappa, unconditional=False)
        val_loss = loss.item()
        validation_loss_history.append(val_loss)
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(epoch+1, val_loss))
        
        # Decay learning rate every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     optimizer = decay_learning_rate(optimizer, args.decay_rate)
        
        filename = f"{args.task}_last_epoch_{epoch + 1}.pt"

        old_checkpoint = f"{args.task}_last_epoch_{epoch}.pt"
        old_checkpoint_path = os.path.join(args.model_save_dir, old_checkpoint)
        if os.path.exists(old_checkpoint_path):
            os.remove(old_checkpoint_path)

        save_checkpoint(epoch, model, val_loss, optimizer, args.model_save_dir, filename)

        print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds.")
        

def calculate_log_likelihood(stroke_end_probs, gaussian_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target, masks):
    y_0 = target.narrow(-1,0,1)
    y_1 = target.narrow(-1,1,1)
    y_2 = target.narrow(-1,2,1)
    
    end_loglik = (y_0*stroke_end_probs + (1-y_0)*(1-stroke_end_probs)).log().squeeze()
    
    const = 1E-20 # to prevent numerical error
    pi_term = torch.Tensor([2*np.pi])
    if is_cuda_available:
        pi_term = pi_term.cuda()
    pi_term = -Variable(pi_term, requires_grad = False).log()
    
    z = (y_1 - mu_1)**2/(log_sigma_1.exp()**2)\
        + ((y_2 - mu_2)**2/(log_sigma_2.exp()**2)) \
        - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 + log_sigma_2).exp())
    mog_lik1 =  pi_term -log_sigma_1 - log_sigma_2 - 0.5*((1-rho**2).log())
    mog_lik2 = z/(2*(1-rho**2))
    mog_loglik = ((gaussian_weights.log() + (mog_lik1 - mog_lik2)).exp().sum(dim=-1)+const).log()
    
    return (end_loglik*masks).sum() + ((mog_loglik)*masks).sum()


def evaluate_model(args, model, validation_loader, h1_init, c1_init, h2_init, c2_init, w_old, previous_kappa, unconditional):
    (validation_samples, masks, onehots, text_lens) = list(enumerate(validation_loader))[0][1]
    step_back2 = validation_samples.narrow(1,0,args.timesteps)
    masks = Variable(masks, requires_grad=False)
    masks = masks.narrow(1,0,args.timesteps)
        
    x = Variable(step_back2, requires_grad=False)
        
    validation_samples = validation_samples.narrow(1,1,args.timesteps)
    y = Variable(validation_samples, requires_grad = False)
    
    if unconditional:
        outputs = model(y, (h1_init, c1_init), (h2_init, c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho , prev, prev2 = outputs
    else:
        outputs = model(x, onehots, text_lens, w_old, previous_kappa, (h1_init, c1_init), (h2_init, c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, prev, prev2, old_phi = outputs

    loss = -calculate_log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)/torch.sum(masks)
    return loss

if __name__ == '__main__':
    main()
