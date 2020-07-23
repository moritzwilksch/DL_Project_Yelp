from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

path = "../data/"  # needs to end with /

# Load data
print("Loading data...")
data = joblib.load(path + '3c_subsampled_data.pickle')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Convert corpus to Bag-of-words representation
print("Converting to BOW...")
cv = CountVectorizer()
x_train = cv.fit_transform(x_train['text'])
x_test = cv.transform(x_test['text'])

# Fitting Multinomial Naive Bayes
print("Fitting NB...")
nb = MultinomialNB()
nb.fit(x_train, y_train)

# Evaluation on TEST set
print(classification_report(y_test, nb.predict(x_test)))
