# Training module for recognition task
import argparse
import os
import re
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CyclicLR

from models.recognition_model import DeepHandwritingRecognitionModel, FocalCTLoss
from utils.train_utils import load_data, decode_predictions

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Handwriting Recognition Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--model_save_dir', type=str, default='save', help='Directory to save the trained model')
    parser.add_argument('--char_to_code_path', type=str, required=True, help='Path to the char_to_code dictionary')
    args = parser.parse_args()

    device = torch.device(args.device)
    char_to_code = torch.load(args.char_to_code_path, weights_only=True)
    code_to_char = {v: k for k, v in char_to_code.items()}

    model = DeepHandwritingRecognitionModel(input_size=3, hidden_size=256, output_size=len(char_to_code), num_layers=4, dropout=0.5).to(device)

    train_data = load_data('recognition', 'train', args.device)
    validation_data = load_data('recognition', 'validation', args.device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

    train_handwriting_recognition(model, train_loader, validation_loader, char_to_code, code_to_char, args)

def train_handwriting_recognition(model, train_loader, validation_loader, char_to_code, code_to_char, args):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CyclicLR(optimizer, base_lr=args.learning_rate/10, max_lr=args.learning_rate, step_size_up=4*len(train_loader), mode='triangular2')

    blank_label = char_to_code['<PAD>']
    criterion = nn.CTCLoss(blank=blank_label, zero_infinity=True) # FocalCTLoss(blank=blank_label, zero_infinity=True)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    start_epoch = 0

    # Load checkpoint if exists
    checkpoint_files = [f for f in os.listdir(args.model_save_dir) if re.match(f"handwriting_recognition_last_epoch_\\d+.pt", f)]
    if checkpoint_files != []:
        epochs = [int(re.search(r"(\d+).pt$", f).group(1)) for f in checkpoint_files]
        latest_epoch = max(epochs)

        checkpoint_name = f"handwriting_recognition_last_epoch_{latest_epoch}.pt"
        checkpoint_path = os.path.join(args.model_save_dir, checkpoint_name)
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = latest_epoch 
        lowest_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch} with validation loss {lowest_val_loss:.4f}")
    else:
        print("No previous checkpoints found. Starting training from scratch.")
        start_epoch = 0
        lowest_val_loss = float('inf')

    print("\n" + "=" * 30)
    print("   🚀 Starting Handwriting Recognition Training... 🚀")
    print("=" * 30 + "\n")

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, (stroke_data, mask_data, onehot_data, text_len_data) in enumerate(train_loader):
            optimizer.zero_grad()

            if scaler is not None:
                with autocast(device_type='cuda'):
                    log_probs, targets, input_lengths, target_lengths = process_data(model, stroke_data, onehot_data, text_len_data, mask_data, args.device, char_to_code)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs, targets, input_lengths, target_lengths = process_data(model, stroke_data, onehot_data, text_len_data, mask_data, args.device, char_to_code)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            scheduler.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for stroke_data, mask_data, onehot_data, text_len_data in validation_loader:
                log_probs, targets, input_lengths, target_lengths = process_data(model, stroke_data, onehot_data, text_len_data, mask_data, args.device, char_to_code)

                val_loss = criterion(log_probs, targets, input_lengths, target_lengths)
                total_val_loss += val_loss.item()

                _, decoded_preds = log_probs.max(2)
                decoded_preds = decoded_preds.transpose(0, 1)
                predictions = decode_predictions(decoded_preds, code_to_char)

                decoded_targets = onehot_data.argmax(dim=-1)
                targets = decode_predictions(decoded_targets, code_to_char)

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        average_train_loss = total_train_loss / len(train_loader)
        average_val_loss = total_val_loss / len(validation_loader)
        
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")

        scheduler.step()
        
        # Print some sample predictions
        for i in range(min(5, len(all_predictions))):
            print(f"Prediction: {all_predictions[i]}")
            print(f"Target: {all_targets[i]}")
            print("---")

        # Save model checkpoint for the last epoch
        filename = f"handwriting_recognition_last_epoch_{epoch + 1}.pt"

        # Save model checkpoint for the current epoch
        old_checkpoint = f"handwriting_recognition_last_epoch_{epoch}.pt"
        old_checkpoint_path = os.path.join(args.model_save_dir, old_checkpoint)
        if os.path.exists(old_checkpoint_path):
            os.remove(old_checkpoint_path)

        # Save the current epoch's checkpoint
        save_checkpoint(epoch, model, average_val_loss, optimizer, args.model_save_dir, filename)

    print("\nTraining completed.")

def process_data(model, stroke_data, onehot_data, text_len_data, mask_data, device, char_to_code):
    stroke_data = stroke_data.to(device)
    onehot_data = onehot_data.to(device)
    text_len_data = text_len_data.squeeze().to(device)
    mask_data = mask_data.to(device)

    # Forward pass through the model
    log_probs = model(stroke_data, text_len_data)
    log_probs = log_probs.transpose(0,1)

    # Prepare target sequences for CTC loss
    targets = onehot_data.argmax(dim=-1)
    
    # Remove padding from targets
    targets_list = []
    for target, length in zip(targets, text_len_data):
        length = int(length.item())  # Convert to Python int
        targets_list.append(target[:length].tolist())
    targets = torch.LongTensor([item for sublist in targets_list for item in sublist]).to(device)

    # Calculate input lengths based on the output of the model
    batch_size = log_probs.size(1)
    seq_length = log_probs.size(0)
    input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(device)
    target_lengths = text_len_data.long()

    return log_probs, targets, input_lengths, target_lengths

def save_checkpoint(epoch, model, best_val_loss, optimizer, directory, filename):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        'optimizer_state_dict': optimizer.state_dict()
    }
    os.makedirs(directory, exist_ok=True)
    checkpoint_path = os.path.join(directory, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()