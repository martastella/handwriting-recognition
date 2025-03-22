# :writing_hand: Handwriting Synthesis and Recognition Model :writing_hand:

![](generation1.png)
![](generation2.png)
![](generation3.png)

This project implements a handwriting synthesis model based on the paper "Generating Sequences With Recurrent Neural Networks" by Alex Graves. The model can generate handwriting in both unconditional and conditional (text-to-handwriting) modes. Additionally, it includes a handwriting recognition feature to convert handwritten text images into digital text.

The project supports:
- Handwriting Synthesis:
    - Unconditional Generation: Generates handwriting without specifying the content.
    - Conditional Generation: Generates handwriting for a given text.

- Handwriting Recognition:
    - Converts images of handwritten text into digital text using a Convolutional Neural Network (CNN) combined with a Bidirectional Long Short-Term Memory (BiLSTM) network and Connectionist Temporal Classification (CTC) loss.

## Project Features :dart:
1. Synthesis
    - Generates realistic handwriting with temporal stroke sequences.
    - Supports both unconditional and conditional text-based handwriting synthesis.
2. Recognition (STILL IN PROGRESS)
    - Recognizes handwritten text from images and converts them to digital text.
    - Enhanced features:
        - CNN for feature extraction from input images.
        - BiLSTM layers to capture contextual information in both forward and backward directions.
        - CTC loss function to handle unsegmented data and align predicted sequences with input data.

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
├── handwriting_recognition/     # Directory for handwriting recognition implementation
│ 
├── train.py                     # Training script for synthesis
├── generate.py                  # Script for handwriting generation
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

## Training Synthesis :runner:

Run training script specifying the task (for unconditional or conditional handwriting generation):

```
python train.py --task 'unconditional_handwriting'
```
```
python train.py --task 'conditional_handwriting'
```

You can adjust the hyperparameters as needed. The trained model will be saved in the `save/` directory.

## Training Recognition (To be improved...) :runner:

The handwriting recognition model is implemented in a Jupyter notebook located in the handwriting_recognition/ directory. To train the recognition model: 

1. Navigate to the recognition directory:
```
cd handwriting_recognition/
```

2. Open the jupiter notebook
```
jupyter notebook recognition_notebook.ipynb
```

3. Follow the instructions within the notebook to preprocess data, define the model architecture (CNN-BiLSTM with CTC loss), train the model, and evaluate its performance.


## Generation :pencil2:

Visualize the results on `generation.ipynb`.

## Recognition :mag:

The handwriting recognition implementation is provided in the `handwriting_recognition/recognition_notebook.ipynb` notebook. This notebook guides you through the process of training and evaluating the recognition model using a CNN-BiLSTM architecture with CTC loss.


## References :books:

- Graves, A. (2013). Generating Sequences With Recurrent Neural Networks. [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)
- CNN-BiLSTM model for English Handwriting Recognition. [ResearchGate](https://www.researchgate.net/publication/365494707_CNN-BiLSTM_model_for_English_Handwriting_Recognition_Comprehensive_Evaluation_on_the_IAM_Dataset)