## Vectorizing/Embedding - Word2Vec

from gensim.models import Word2Vec
import gensim.downloader
from typing import Tuple
from tensorflow import keras
from keras.utils import pad_sequences
from app.packages.utils import *
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras import Model, Sequential, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras import metrics

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


def w2v_train_and_embed(X_train, vector_size, window, dtype='float32', padding='post'):

    """Returns a list of embedded, padded sentences (each sentence is a matrix).
    Takes vectorizing arguments "vector_size" and "window"
    Takes padding arguments dtype & padding
    """

    word2vec = Word2Vec(sentences=X_train, vector_size=vector_size, window=window)

    def embed_sentence(wv, sentence):
        return np.array([wv[i] for i in sentence if i in wv])

    wv = word2vec.wv

    embedded = [embed_sentence(wv, s) for s in X_train]

    return pad_sequences(embedded, dtype=dtype, padding=padding), word2vec

def w2v_embed(X_test, word2vec_model, max_length, dtype='float32', padding='post'):
    """
    Embed sentences using a trained Word2Vec model.
    """
    def embed_sentence(wv, sentence):
        return np.array([wv[i] for i in sentence if i in wv])

    # Embedding the sentences
    wv = word2vec_model.wv
    embedded_X = [embed_sentence(wv, s) for s in X_test]

    return pad_sequences(embedded_X, maxlen=max_length, dtype=dtype, padding=padding)

#########EXAMPLES#############
#X_train_padded, trained_word2vec_model = train_and_embed_word2vec(X_train, vector_size=50, window=5)
#X_test_padded = embed_word2vec(X_test, trained_word2vec_model, max_length=len(X_train_padded[0]))


def embed_preprocessing():
    pass

## Create LSTM Model

## 06-Deep-Learning/04-RNN-and-NLP/data-your-first-embedding/Your-first-embedding.ipynb



def initialize_lstm(lstm_units=50, lstm_activation='tanh', embedding:bool=False):

    if embedding == True:
        tk = Tokenizer()
        model_wiki = gensim.downloader.load('glove-twitter-200') # loads dataset (Glove Twitter, 100dimensions)
        embedding_dim = 200  # GloVe vectors dimension
        word_index = tk.word_index  # using fitted tokenizer
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim)) # initalize embedding matrix
        for word, i in word_index.items():
            try:
                embedding_vector = model_wiki[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                continue

    model = Sequential()
    if embedding == False:
        model.add(Masking())
    elif embedding == True:
        model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(LSTM(units=lstm_units, activation=lstm_activation, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation=lstm_activation))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))

    return model

def compile_lstm_model(model: Model, loss='binary_crossentropy', optimizer='rmsprop') -> Model:
    """
    Compile the Neural Network
    """
    precision = metrics.Precision()
    recall = metrics.Recall()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision, recall])

    print("✅ Model compiled")

    return model

@simple_time_and_memory_tracker
def train_lstm_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print("\nTraining model...")

    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )


    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=50,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )




    #print(f"✅ Model trained on {len(X)} rows with max val accuracy: {round(np.max(history.history['val_accuracy']), 2)}, max val recall: {round(np.max(history.history['val_recall']), 2)}, max val precision: {round(np.max(history.history['val_precision']), 2)}")

    return model, history

def evaluate_lstm_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(f"\nEvaluating model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    recall = metrics["recall"]

    print(f"✅ Model evaluated, recall: {round(recall, 2)}, accuracy: {round(accuracy, 2)}")

    return metrics


##### PRELIMINARY RESULTS:
#{'loss': 0.5114620327949524,
# 'accuracy': 0.7444544434547424,
# 'precision': 0.7088000178337097,
# 'recall': 0.6040909290313721}
