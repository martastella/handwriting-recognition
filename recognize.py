import numpy as np
import torch
from tqdm import tqdm

from models.recognition_model import DeepHandwritingRecognitionModel


def recognize_stroke(stroke_tensor, model, char_to_code):
    code_to_char = {v: k for k, v in char_to_code.items()}
    model.eval()
    with torch.no_grad():
        output = model(stroke_tensor.unsqueeze(0), torch.tensor([stroke_tensor.size(0)]))
        _, predicted = torch.max(output, dim=-1)
        predicted = predicted.squeeze().cpu().numpy()
    
    text = ''
    for i in predicted:
        if i != char_to_code['<PAD>']:
            text += code_to_char[i]
    return text.strip()


def recognize_handwriting(
    stroke_path: str,
    model_path: str,
    char_to_code_path: str,
    hidden_size: int = 256,
):
    char_to_code = torch.load(char_to_code_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepHandwritingRecognitionModel(
        input_size=3, hidden_size=hidden_size, output_size=len(char_to_code)
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    stroke = np.load(stroke_path, allow_pickle=True)
    if stroke.dtype == np.object_:
        strokes = [np.array(seq, dtype=np.float32) for seq in stroke]
    else:
        strokes = stroke.astype(np.float32)

    unique_texts = set()  

    for stroke_sequence in tqdm(strokes, desc="Processing strokes"):
        try:
            stroke_tensor = torch.FloatTensor(stroke_sequence).to(device)
            recognized_text = recognize_stroke(stroke_tensor, model, char_to_code)
            if recognized_text:  
                unique_texts.add(recognized_text)
        except ValueError as e:
            print(f"Error processing stroke sequence: {e}")

    return unique_texts
