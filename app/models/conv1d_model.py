from app.packages.preprocessing.cleaning import *
import numpy as np
import pandas as pd
from app.packages.utils import *
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential, models, metrics, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

@simple_time_and_memory_tracker
def preprocessing_cld(X, maxlen=100):
    """ Preprocess X data for a Conv1D model
    Takes a single column df, X (as a list), as input. Returns the preprocessed X,
    the maxlen and the vocab size as output for use in initialize model function
    """
    X_word = [text_to_word_sequence(x) for x in X.tolist()]

    tk = Tokenizer()
    tk.fit_on_texts(X_word)
    X_token = tk.texts_to_sequences(X_word)
    vocab_size = len(tk.word_index)

    X_token_pad = pad_sequences(X_token, dtype=float, padding='post', maxlen=maxlen)
    return X_token_pad, vocab_size, maxlen


@simple_time_and_memory_tracker
def intialize_c1d(vocab_size, maxlen, embedding_size=100, loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), globalmax=True):
    """Initialize and compile a Conv1D model
    Takes: a vocab_size and maxlen ouputted by preproc, as well as embedding size,
    loss, optimizer and whether to use a global max layer or a max + flatten
    returns a Conv1D model
    ATT: vocab and maxlen MUST be the same as those outputted by preproc
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
    model.add(Conv1D(64, kernel_size=3,padding='same', activation='relu'))
    model.add(Conv1D(128, kernel_size=4,padding='same', activation='relu'))
    model.add(Conv1D(128, kernel_size=5,padding='same', activation='relu'))

    if globalmax:
        model.add(GlobalMaxPooling1D())
    else:
        model.add(MaxPooling1D())
        model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    precision = metrics.Precision()
    recall = metrics.Recall()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision, recall])

    print("✅ Model initialized and compiled")

    return model

@simple_time_and_memory_tracker
def train_c1d_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64,
        patience=4,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ):
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

    print(f"✅ Model trained on {len(X)} rows with max val accuracy: {round(np.min(history.history['val_accuracy']), 2)}, max val recall: {round(np.min(history.history['val_recall']), 2)}")

    return model, history

@simple_time_and_memory_tracker
def evaluate_c1d_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ):
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
