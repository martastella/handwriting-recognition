# :writing_hand: Handwriting Synthesis and Recognition Model :writing_hand:

This project implements a handwriting synthesis model based on the paper "Generating Sequences With Recurrent Neural Networks" by Alex Graves. The model can generate handwriting in both unconditional and conditional (text-to-handwriting) modes. Additionally, it includes a handwriting recognition feature to convert handwritten strokes back into text.

The project supports:
- Handwriting Synthesis:
    - Unconditional Generation: Generates handwriting without specifying the content.
    - Conditional Generation: Generates handwriting for a given text.

- Handwriting Recognition:
    - Converts sequences of pen strokes into text using an enhanced recognition model with residual LSTMs and attention mechanisms.

## Project Features :dart:
1. Synthesis
    - Generates realistic handwriting with temporal stroke sequences.
    - Supports both unconditional and conditional text-based handwriting synthesis.
2. Recognition (STILL IN PROGRESS)
    - Recognizes handwritten strokes and converts them to text.
    - Enhanced features:
        - Residual connections in LSTM layers for gradient efficiency.
        - Attention mechanism for focused sequence decoding.
        - Support for Focal CTC Loss for imbalanced sequence training.

## Project Structure :card_index_dividers:

<pre>
handwriting_synthesis/
│ 
├── data/                        # Data directory
│ ├── sentences.txt              # Text data for synthesis
│ └── strokes-py3.npy            # Handwriting stroke data
│ 
├── models/                      # Model definitions
│ ├── handwriting_model.py       # Handwriting synthesis model
│ └── recognition_model.py       # Handwriting recognition model
│ 
├── utils/                       # Utility functions
│ ├── train_utils.py             # Helper functions for training and evaluation
│ └── plots.py                   # Functions for plotting handwriting
│ 
├── weights/                     # Directory for trained model weights
│ └── (Saved model weights)      
│ 
├── train.py                     # Training script for synthesis
├── train_recognition.py         # Training script for recognition
├── generate.py                  # Script for handwriting generation
├── recognize.py                 # Script for handwriting recognition
├── data_preprocessing.py        # Data preparation scripts
└── README.md                    # This file
</pre>

## Setup :gear:

1. Clone this repository:

    ```
    git clone https://github.com/yourusername/handwriting-synthesis-recognition.git cd handwriting-synthesis-recognition
    ```

2. Prepare your data:

    - Place your data files in the `data/` directory as shown in the project structure.
        - `strokes-py3.npy`: Preprocessed handwriting stroke sequences.
        - `sentences.txt`: Sentences for conditional handwriting generation.

3. Run the data preprocessing script to create the character-to-code dictionary:

    ```
    python data_preprocessing.py --task generation
    ```
    ```
    python data_preprocessing.py --task recognition
    ```

## Training Synthesis :runner:

Run training script specifying the task (for unconditional or conditional handwriting generation):

```
python train.py --task 'unconditional_handwriting'
```
```
python train.py --task 'conditional_handwriting'
```

## Training Recognition (To be improved...) :runner:

To train the handwriting recognition model, use:
```
python train_recognition.py --batch_size 32 --num_epochs 100 --learning_rate 0.001 --char_to_code_path char_to_code_recognition.pt 
```

You can adjust the hyperparameters as needed. The trained model will be saved in the `save/` directory.

## Generation :pencil2:

Visualize the results on `generation.ipynb`.

## Recognition (To be improved...) :mag:

Visualize the results on `recognition.ipynb`.


## References :books:

- Graves, A. (2013). Generating Sequences With Recurrent Neural Networks. [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)