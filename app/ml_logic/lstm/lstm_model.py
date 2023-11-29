## Vectorizing/Embedding - Word2Vec

from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def w2v_preprocessing(X, vector_size, window, dtype='float32', padding='post'):

    """Returns a list of embedded, padded sentences (each sentence is a matrix).
    Takes vectorizing arguments "vector_size" and "window"
    Takes padding arguments dtype & padding
    """

    word2vec = Word2Vec(sentences=X, vector_size=vector_size, window=window)

    def embed_sentence(word2vec, sentence):
        wv = word2vec.wv
        return np.array([wv[i] for i in sentence if i in wv])

    embedded = [embed_sentence(word2vec, s) for s in X]

    return pad_sequences(embedded, dtype=dtype, padding=padding)

def embed_preprocessing():
    pass

## Create LSTM Model

## 06-Deep-Learning/04-RNN-and-NLP/data-your-first-embedding/Your-first-embedding.ipynb

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, LSTM, Masking
from keras.callbacks import EarlyStopping
from keras import metrics

def initialize_lstm(lstm_units=20, lstm_activation='tanh'):

    model = Sequential()
    model.add(Masking())
    model.add(LSTM(units=lstm_units, activation=lstm_activation))
    model.add(Dense(10, activation="tanh"))
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


def train_lstm_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64,
        patience=4,
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
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with max val accuracy: {round(np.min(history.history['val_accuracy']), 2)}, max val recall: {round(np.min(history.history['val_recall']), 2)}")

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
