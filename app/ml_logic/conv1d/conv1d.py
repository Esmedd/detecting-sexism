from app.packages.preprocessing.cleaning import *
import numpy as np
importpandas as pd

from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential, models, metrics

def c1d_preproc(X=df["text"], maxlen=100):
    """ Preprocess X data for a Conv1D model
    Takes a single column, X, as input returns the preprocessed X,
    the maxlen and the vocab size as output for use in initialize model function
    """
    X_text = X.to_numpy()
    X_word = [text_to_word_sequence(x) for x in X_text]

    tk = Tokenizer()
    tk.fit_on_texts(X_word)
    X_token = tk.texts_to_sequences(X_word)
    vocab_size = len(tk.word_index)

    X_token_pad = pad_sequences(X_token, dtype=float, padding='post', maxlen=maxlen)
    return X_token_pad, vocab_size, maxlen

def intialize_c1d(vocab_size, maxlen, embedding_size=100, loss='binary_crossentropy', optimizer='adam'):
    """Initialize and compile a Conv1D model
    Takes a vocab_size and maxlen ouputted by preproc, as well as embedding size
    returns a Conv1D model
    """
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
    model.add(models.Conv1D(20, kernel_size=3,padding='same', activation='relu'))
    model.add(models.Conv1D(20, kernel_size=4,padding='same', activation='relu'))
    model.add(models.Dropout(0.5))
    model.add(layers.GlobalMaxPooling1D())
    model.add(models.Dense(100, activation='relu'))
    model.add(models.Dense(1, activation='softmax'))

    precision = metrics.Precision()
    recall = metrics.Recall()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision, recall])

    print("✅ Model initialized and compiled")

    return model
