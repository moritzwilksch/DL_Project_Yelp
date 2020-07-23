import keras
import numpy as np
from app.architecture import create_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
import joblib

path = "../data/"  # needs to end with /

# Load data
print("Loading data...")
data = joblib.load(path + '3c_subsampled_data.pickle')
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']

# Load objects: tokenizer and embedding matrix
print("Loading prepared objects...")
tok = joblib.load(path + 'tokenizer.pickle')
embedding_matrix = joblib.load(path + 'embedding_matrix.pickle')

# Tokenize text
print("Tokenizing text...")
x_train = tok.texts_to_sequences(x_train['text'])
x_val = tok.texts_to_sequences(x_val['text'])

# Parameters
BATCH_SIZE = 256
MAX_INPUT_LENGTH = int(np.round(np.percentile([len(x) for x in x_train], 90)))  # 90th percentile crop
N_EPOCHS = 10

# Zero-pad sequences to max input length
print("Padding sequences...")
x_train = pad_sequences(x_train, MAX_INPUT_LENGTH)
x_val = pad_sequences(x_val, MAX_INPUT_LENGTH)

# One-Hot Encode class labels
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.values.reshape(-1, 1))
y_val = ohe.fit_transform(y_val.values.reshape(-1, 1))

# Instantiate Model
model: keras.Sequential = create_model(tokenizer=tok, embedding_matrix=embedding_matrix, max_input_length=MAX_INPUT_LENGTH)

# Fit model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_val, y_val))

# Save model
model.save(path + 'model.hdf5')
print("Model saved to disk!")
