# Utils for training
import numpy as np
import os
import torch

import torch

def decode_predictions(predictions, code_to_char):
    decoded = []
    pad_token = 60

    for pred in predictions:
        text = ''
        for i in pred:
            if i.item() == pad_token:
                break
            text += code_to_char[i.item()]
        decoded.append(text.strip())
    
    return decoded

def decay_learning_rate(optimizer, decay_rate):
    state_dict = optimizer.state_dict()
    lr = state_dict['param_groups'][0]['lr']
    
    lr *= decay_rate
    
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
        
    optimizer.load_state_dict(state_dict)
    return optimizer

def save_checkpoint(epoch, model, validation_loss, optimizer, directory, filename='best.pt'):
    checkpoint = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'validation_loss': validation_loss,
        'optimizer': optimizer.state_dict()
    }

    try:
        os.makedirs(directory, exist_ok=True) 
        checkpoint_path = os.path.join(directory, filename)
        torch.save(checkpoint, checkpoint_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_data(task, data_type, device):
    if task == 'generation':
        strokes_path = f'preprocessed_generation/{data_type}_strokes_800.npy'
        masks_path = f'preprocessed_generation/{data_type}_masks_800.npy'
        onehot_path = f'preprocessed_generation/{data_type}_onehot_800.npy'
        text_lens_path = f'preprocessed_generation/{data_type}_text_lens.npy'
    elif task == 'recognition':
        strokes_path = f'preprocessed_recognition/{data_type}_strokes_800.npy'
        masks_path = f'preprocessed_recognition/{data_type}_masks_800.npy'
        onehot_path = f'preprocessed_recognition/{data_type}_onehot_800.npy'
        text_lens_path = f'preprocessed_recognition/{data_type}_text_lens.npy'
    else:
        raise ValueError("Unknown task type. Must be 'generation' or 'recognition'.")
    
    strokes = torch.from_numpy(np.load(strokes_path)).float().to(device) # the sequence of handwriting points
    masks = torch.from_numpy(np.load(masks_path)).float().to(device) # binary masks indicating valid stroke points
    onehot = torch.from_numpy(np.load(onehot_path)).float().to(device) # one-hot encoded representations of the text
    text_lens = torch.from_numpy(np.load(text_lens_path)).float().to(device) # the lengths of the input text sequences

    return torch.utils.data.TensorDataset(strokes, masks, onehot, text_lens)
