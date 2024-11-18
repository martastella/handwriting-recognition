# Data preparation
import numpy as np
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='generation', help='Task mode: "generation" or "recognition"')
    args = parser.parse_args()

    # Begin training based on selected task
    if args.task == 'generation':
        preprocessing(args.task)
    elif args.task == 'recognition':
        preprocessing(args.task)

def preprocessing(task):
    strokes = np.load('data/strokes-py3.npy', encoding='latin1', allow_pickle=True)
    with open('data/sentences.txt') as f:
        texts = f.readlines()
        
    train_strokes = []
    train_texts = []
    validation_strokes = []
    validation_texts = []

    for _ in range(len(strokes)):
        if len(strokes[_]) <= 801:
            train_strokes.append(strokes[_])
            train_texts.append(texts[_])
        else:
            validation_strokes.append(strokes[_])
            validation_texts.append(texts[_])

    train_masks = np.zeros((len(train_strokes),800))
    for i in range(len(train_strokes)):
        train_masks[i][0:len(train_strokes[i])-1] = 1
        train_strokes[i] = np.vstack([train_strokes[i], np.zeros((801-len(train_strokes[i]), 3))])
        
    validation_masks = np.zeros((len(validation_strokes),1200))
    for i in range(len(validation_strokes)):
        validation_masks[i][0:len(validation_strokes[i])-1] = 1
        validation_strokes[i] = np.vstack([validation_strokes[i], np.zeros((1201-len(validation_strokes[i]), 3))])

    if task == 'generation':
        np.save('preprocessed_generation/train_strokes_800', np.stack(train_strokes))
        np.save('preprocessed_generation/train_masks_800', train_masks)
        np.save('preprocessed_generation/validation_strokes_800', np.stack(validation_strokes))
        np.save('preprocessed_generation/validation_masks_800', validation_masks)

        # convert each text sentence to an array of onehots
        char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-'

        char_to_code = {}
        code_to_char = {}
        c = 0
        for _ in char_list:
            char_to_code[_] = c
            code_to_char[c] = _
            c += 1
        torch.save(char_to_code, 'char_to_code_generation.pt')

        max_text_len = np.max(np.array([len(a) for a in validation_texts]))

        train_onehot_800 = []
        train_text_masks = []
        for t in train_texts:
            onehots = np.zeros((max_text_len, len(char_to_code)+1))
            mask = np.ones(max_text_len)
            for _ in range(len(t)):
                try:
                    onehots[_][char_to_code[t[_]]] = 1
                except:
                    onehots[_][-1] = 1
            mask[len(t):] = 0
            train_onehot_800.append(onehots)
            train_text_masks.append(mask)
        train_onehot_800 = np.stack(train_onehot_800)
        train_text_masks = np.stack(train_text_masks)
        train_text_lens = np.array([[len(a)] for a in train_texts])

        validation_onehot_800 = []
        validation_text_masks = []
        for t in validation_texts:
            onehots = np.zeros((max_text_len, len(char_to_code)+1))
            mask = np.ones(max_text_len)
            for _ in range(len(t)):
                try:
                    onehots[_][char_to_code[t[_]]] = 1
                except:
                    onehots[_][-1] = 1
            mask[len(t):] = 0
            validation_onehot_800.append(onehots)
            validation_text_masks.append(mask)
        validation_onehot_800 = np.stack(validation_onehot_800)
        validation_text_masks = np.stack(validation_text_masks)
        validation_text_lens = np.array([[len(a)] for a in validation_texts])

        np.save('preprocessed_generation/train_onehot_800', train_onehot_800)
        np.save('preprocessed_generation/validation_onehot_800', validation_onehot_800)
        np.save('preprocessed_generation/train_text_masks', train_text_masks)
        np.save('preprocessed_generation/validation_text_masks', validation_text_masks)
        np.save('preprocessed_generation/train_text_lens', train_text_lens)
        np.save('preprocessed_generation/validation_text_lens', validation_text_lens)

    elif task == 'recognition':
        np.save('preprocessed_recognition/train_strokes_800', np.stack(train_strokes))
        np.save('preprocessed_recognition/train_masks_800', train_masks)
        np.save('preprocessed_recognition/validation_strokes_800', np.stack(validation_strokes))
        np.save('preprocessed_recognition/validation_masks_800', validation_masks)

        char_list = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                    ',', '.', '"', "'", '?', '-', '>', '<PAD>']

        char_to_code = {}
        code_to_char = {}

        for idx, char in enumerate(char_list):
            char_to_code[char] = idx
            code_to_char[idx] = char

        torch.save(char_to_code, 'char_to_code_recognition.pt')

        max_text_len = np.max(np.array([len(a) for a in validation_texts]))

        train_onehot_800 = []
        train_text_masks = []
        for t in train_texts:
            onehots = np.zeros((max_text_len, len(char_to_code))) 
            mask = np.ones(max_text_len)
            for _ in range(len(t)):
                if t[_] in char_to_code:
                    onehots[_][char_to_code[t[_]]] = 1
                else:
                    onehots[_][char_to_code['<PAD>']] = 1 
            mask[len(t):] = 0
            train_onehot_800.append(onehots)
            train_text_masks.append(mask)
        train_onehot_800 = np.stack(train_onehot_800)
        train_text_masks = np.stack(train_text_masks)
        train_text_lens = np.array([[len(a)] for a in train_texts])

        validation_onehot_800 = []
        validation_text_masks = []
        for t in validation_texts:
            onehots = np.zeros((max_text_len, len(char_to_code)))  
            mask = np.ones(max_text_len)
            for _ in range(len(t)):
                if t[_] in char_to_code:
                    onehots[_][char_to_code[t[_]]] = 1
                else:
                    onehots[_][char_to_code['<PAD>']] = 1  
            mask[len(t):] = 0
            validation_onehot_800.append(onehots)
            validation_text_masks.append(mask)
        validation_onehot_800 = np.stack(validation_onehot_800)
        validation_text_masks = np.stack(validation_text_masks)
        validation_text_lens = np.array([[len(a)] for a in validation_texts])

        np.save('preprocessed_recognition/train_onehot_800', train_onehot_800)
        np.save('preprocessed_recognition/validation_onehot_800', validation_onehot_800)
        np.save('preprocessed_recognition/train_text_masks', train_text_masks)
        np.save('preprocessed_recognition/validation_text_masks', validation_text_masks)
        np.save('preprocessed_recognition/train_text_lens', train_text_lens)
        np.save('preprocessed_recognition/validation_text_lens', validation_text_lens)

if __name__ == '__main__':
    main()