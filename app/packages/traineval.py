from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, metrics, Model
import numpy as np
from app.packages.utils import *

@simple_time_and_memory_tracker
def train_model(
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
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with max val accuracy: {round(np.min(history.history['val_accuracy']), 2)}, max val recall: {round(np.min(history.history['val_recall']), 2)}")

    return model, history

@simple_time_and_memory_tracker
def evaluate_model(
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
