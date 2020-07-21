import keras
from sklearn.metrics import classification_report
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load data
data = joblib.load('3c_subsampled_data.pickle')
x_test = data['x_test']
y_test = data['y_test']

# Load tokenizer for preprocessing
tok = joblib.load('tokenizer.pickle')

# Tokenize text
x_test = tok.texts_to_sequences(x_test['text'])

# One-hot encode labels
ohe = OneHotEncoder()
y_test = ohe.fit_transform(y_test.values.reshape(-1, 1))

# Load model
model: keras.Sequential = keras.models.load_model('data/model.hdf5')

# Test set performance
print("=== Performance on test set for negative/neural/positive classes ===")
preds = model.predict_classes(x_test)
print("\n" + classification_report(np.argmax(y_test.toarray(), axis=1), preds))


