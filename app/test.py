import keras
from sklearn.metrics import classification_report
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences

path = "/Users/moritz/Downloads/yelp_dataset/"

# Load data
data = joblib.load(path + '3c_subsampled_data.pickle')
x_test = data['x_test']
y_test = data['y_test']

# Load tokenizer for preprocessing
tok = joblib.load(path + 'tokenizer.pickle')

# Tokenize text
x_test = tok.texts_to_sequences(x_test['text'])

# Pad sequences
x_test = pad_sequences(x_test, 183)

# One-hot encode labels
ohe = OneHotEncoder()
y_test = ohe.fit_transform(y_test.values.reshape(-1, 1))

# Load model
model: keras.Sequential = keras.models.load_model(path + 'model.hdf5')

# Test set performance
print("=== Performance on test set for negative/neural/positive classes ===")
preds = model.predict_classes(x_test)
print("\n" + classification_report(np.argmax(y_test.toarray(), axis=1), preds))


