import keras
import numpy as np


def create_model(tokenizer: keras.preprocessing.text.Tokenizer, embedding_matrix: np.ndarray,
                 max_input_length: int) -> keras.Sequential:
    """
    Needs tokenizer object used to create tokens and pretrained embedding matrix and length of the input sequences.
    Creates the keras model object, compiles it and returns it for fitting.
    """
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, weights=[embedding_matrix], output_dim=300,
                               trainable=False, input_length=max_input_length),
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.MaxPool1D(),
        keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu'),
        keras.layers.MaxPool1D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile('adam', keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    return model
