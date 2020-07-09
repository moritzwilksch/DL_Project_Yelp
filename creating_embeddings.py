# %%
import numpy as np
import joblib

# %%
data = joblib.load('DL-Yelp/subsampled_data.pickle')
x_train = data['x_train']
# y_train = data['y_train']
# x_val = data['x_val']
# y_val = data['y_val']
# x_test = data['x_test']
# y_test = data['y_test']
del data

# %%
import fasttext.util

# For first run: download model (6 GB)
# fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('DL-Yelp/cc.en.300.bin')

# %%
from gensim.utils import simple_preprocess

# Tokenize the text column to get the new column 'tokenized_text'
tokens_xtrain = [simple_preprocess(line, deacc=True) for line in x_train['text']]

# %%
from gensim.parsing.porter import PorterStemmer

ps = PorterStemmer()
import multiprocessing as mp

# Stem each word using all CPU cores
def stem(list_of_words):
    return [ps.stem(word.encode("ascii", errors="ignore").decode().strip()) for word in list_of_words]

with mp.Pool(mp.cpu_count()) as p:
    result = p.map(stem, tokens_xtrain)

# %%
from keras.preprocessing.text import Tokenizer

# Fit tokenizer to texts. This object is important, it saves the word -> index mapping
tok = Tokenizer()
tok.fit_on_texts(result)
tokens_xtrain = tok.texts_to_sequences(result)

# %%
embedding_matrix = np.zeros((len(tok.word_index) + 1, 300))

# Fill embedding matrix with vectors
for word, index in tok.word_index.items():
    embedding_matrix[index] = ft.get_word_vector(word)

# %%
# Save matrix and tokenizer object for later use
joblib.dump(embedding_matrix, 'embedding_matrix.pickle')
joblib.dump(tok, 'tokenizer.pickle')
