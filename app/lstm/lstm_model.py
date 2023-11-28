## Vectorizing/Embedding - Word2Vec

from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences

def w2v_preprocessing(sentences, vector_size, window, dtype='float32', padding='post'):

  """Returns a list of embedded, padded sentences (each sentence is a matrix).
  Takes vectorizing arguments "vector_size" and "window
  Takes padding arguments dtype & padding
  """
  word2vec = Word2Vec(sentences=sentences, vector_size=vector_size, window=window)
  embedded = [embed_sentence(word2vec, s) for s in sentences]

  return pad_sequences(embedded, dtype=dtype, padding=padding)

## Create LSTM Model

## 06-Deep-Learning/04-RNN-and-NLP/data-your-first-embedding/Your-first-embedding.ipynb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Flatten, LSTM
