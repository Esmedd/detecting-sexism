from typing import Tuple
from gensim.models import Word2Vec
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras import metrics
from tensorflow import keras
from keras import Model, Sequential, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras import metrics


## !! We are using the same preprocessing method for this model as for the LSTM model so we import/use the functions
## for instance ==> w2v_train_and_embed() and w2v_embed() functions

## Create GRU Model


def initialize_gru(gru_units=50, gru_activation='tanh'):

    model = Sequential()
    model.add(Masking())
    model.add(GRU(units=gru_units, activation=gru_activation, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(units=50, activation=gru_activation))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))

    return model

def compile_gru_model(model, loss='binary_crossentropy', optimizer='rmsprop'):
    precision = metrics.Precision()
    recall = metrics.Recall()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision, recall])

    print("✅ Model compiled")

    return model

def compile_gru_model_focal(model, loss='binary_focal_crossentropy', optimizer='rmsprop'):
    precision = metrics.Precision()
    recall = metrics.Recall()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision, recall])

    print("✅ Model compiled")

    return model


def train_gru_model(
        model,
        X,
        y,
        batch_size=64,
        patience=4,
        validation_data=None,  # overrides validation_split
        validation_split=0.2
) -> Tuple:
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

    print(f"✅ Model trained on {len(X)} rows with max val accuracy: {round(np.max(history.history['val_accuracy']), 2)}, max val recall: {round(np.max(history.history['val_recall']), 2)}, max val precision: {round(np.max(history.history['val_precision']), 2)}")

    return model, history

def train_gru_model_focal(
        model,
        X,
        y,
        batch_size=64,
        patience=4,
        validation_data=None,  # overrides validation_split
        validation_split=0.2
) -> Tuple:
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

    print(f"✅ Model trained on {len(X)} rows with max val accuracy: {round(np.max(history.history['val_accuracy']), 2)}, max val recall: {round(np.max(history.history['val_recall_1']), 2)}, max val precision: {round(np.max(history.history['val_precision_1']), 2)}")

    return model, history



def evaluate_gru_model(model, X, y, batch_size=64) -> Tuple:
    print(f"\nEvaluating model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    recall = metrics["recall"]

    print(f"✅ Model evaluated with loss:{loss}, recall: {round(recall, 2)}, accuracy: {round(accuracy, 2)}")

    return metrics
def evaluate_gru_model_focal(model, X, y, batch_size=64) -> Tuple:
    print(f"\nEvaluating model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    recall = metrics["recall_1"]

    print(f"✅ Model evaluated with loss:{loss}, recall: {round(recall, 2)}, accuracy: {round(accuracy, 2)}")

    return metrics

##### PRELIMINARY RESULTS:
# {'loss': 0.5017110705375671,
#  'accuracy': 0.7493237257003784,
#  'precision': 0.7381215691566467,
#  'recall': 0.5932504534721375}
