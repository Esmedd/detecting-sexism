## Vectorizing/Embedding - Word2Vec

from gensim.models import Word2Vec
from tensorflow import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras import layers

import numpy as np


def tokenize(df_column, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
    """ tokenize a column
    df_column (pandas.Series): The DataFrame column containing text.
    filters: The set of characters to filter out. defaults to remove punctuation.
    lower: Whether to convert the text to lowercase. defaults to true.
    split: The split to use for splitting the text. Defaults to ' ' (space).

    Returns:
    list of lists: list where each element is a list of tokens from a row in the input column.
    """
    return df_column.astype(str).apply(lambda x: text_to_word_sequence(x, filters=filters, lower=lower, split=split)).tolist()

def embed_sentence(word2vec, sentence):
    """Embed a sentence using Word2Vec."""
    return [word2vec.wv[word] for word in sentence if word in word2vec.wv]

def w2v_preprocessing(sentences, vector_size, window, dtype='float32', padding='post'):
    """Returns a list of embedded, padded sentences (each sentence is a matrix).
    Takes vectorizing arguments "vector_size" and "window
    Takes padding arguments dtype & padding."""

    word2vec = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1)
    embedded = [embed_sentence(word2vec, s) for s in sentences]
    print("✅ Embedding complete.")

    # Determine the maximum length of the sequences
    max_length = max(len(seq) for seq in embedded)

    # Pad sequences
    padded = pad_sequences(embedded, maxlen=200, dtype=dtype, padding=padding)
    print("✅ Padding complete.")
    # returns both padded sequence and shape
    return np.array(padded), np.array(padded).shape

    ## Create LSTM Model

def create_lstm_model(input_shape):
    '''Create LSTM model.'''
    model = Sequential()

    model.add(layers.LSTM(50, input_shape = input_shape, return_sequences = False))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    print("✅ LSTM model creation complete.")
    return model


################### EXAMPLE ###############
# train_sentences = tokenize(data_clean['text'])
# padded_sentences, sequence_shape = w2v_preprocessing(train_sentences, vector_size = 100, window = 5)
# input_shape = sequence_shape[1:]
# model = create_lstm_model(input_shape=input_shape)



## 06-Deep-Learning/04-RNN-and-NLP/data-your-first-embedding/Your-first-embedding.ipynb
