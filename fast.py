import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os

from app.packages.data_storage.registry import load_model
from app.packages.preprocessing.preprocessing_ML import *
from app.interface.main import *

app = FastAPI()



# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# app.state.model = load_model("LSTM")
app.state.model = load_model_local("20231205-131210_LSTM_GLOVE_BIDIR")
assert app.state.model is not None
print("🏁 Model has been loaded")

# http://127.0.0.1:8000/predict?text=
@app.get("/predict")
def predict(text):
    """
    Make a single prediction.
    """
    print(text)
    l = []
    l.append(text)

    X_pred = pd.DataFrame({"text":l})
    print(3)
    sentence = X_pred["text"][0]
    print(4)
    print(X_pred.shape)

    y_pred = fast_pred("LSTM", app.state.model, X_pred, clean_param, preproc_params_LSTM_bidir)
    print(y_pred[0][0])
    print(5)

    return {sentence: float(y_pred[0][0])}



    # model = load_model("LSTM")
    # assert model is not None
    # print("🏁 Model has been loaded")
    # y_pred = pred("LSTM", X_pred, clean_param, preproc_params_LSTM_bidir)
    # return {X_pred:y_pred}
    #X_processed = preprocess_features(X_pred)
    #y_pred = model.predict(X_processed)
    #return {"prediction": float(y_pred[0][0])}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
