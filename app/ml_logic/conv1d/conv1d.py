from app.packages.preprocessing.cleaning import *
import numpy as np
import pandas as pd
from app.packages.utils import *
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential, models, metrics
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

@simple_time_and_memory_tracker
def preprocessing_cld(X, maxlen=100):
    """ Preprocess X data for a Conv1D model
    Takes a single column df, X (as a list), as input. Returns the preprocessed X,
    the maxlen and the vocab size as output for use in initialize model function
    """
    X_word = [text_to_word_sequence(x) for x in X]

    tk = Tokenizer()
    tk.fit_on_texts(X_word)
    X_token = tk.texts_to_sequences(X_word)
    vocab_size = len(tk.word_index)

    X_token_pad = pad_sequences(X_token, dtype=float, padding='post', maxlen=maxlen)
    return X_token_pad, vocab_size, maxlen

@simple_time_and_memory_tracker
def intialize_c1d(vocab_size, maxlen, embedding_size=100, loss='binary_crossentropy', optimizer='adam'):
    """Initialize and compile a Conv1D model
    Takes a vocab_size and maxlen ouputted by preproc, as well as embedding size
    returns a Conv1D model
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
    model.add(Conv1D(64, kernel_size=3,padding='same', activation='relu'))
    model.add(Conv1D(128, kernel_size=4,padding='same', activation='relu'))

    model.add(MaxPooling1D())
    #model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    precision = metrics.Precision()
    recall = metrics.Recall()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision, recall])

    print("✅ Model initialized and compiled")

    return model
