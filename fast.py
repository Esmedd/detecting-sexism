import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os

from app.packages.data_storage.registry import load_model, load_model_local
from app.packages.preprocessing.preprocessing_ML import *
from app.packages.preprocessing.translate import *
from app.interface.main import fast_pred
from app.models.elizaBERT import *
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
app.state.model_BERT = load_model_local("BERT_EN_UNCASED_BPUN_W")
app.state.model_BIDIR = load_model_local("20231205-131210_LSTM_GLOVE_BIDIR")
app.state.model_GLOVE = load_model_local("20231204-135910_LSTM_embed")
app.state.model_GRU = load_model_local("20231207-115704_GRU")


assert app.state.model_BERT is not None
print("🏁 Model has been loaded")

# http://127.0.0.1:8000/predict?text=
@app.get("/predict")
def predict(text, model_name:str="BERT"):
    """
    Make a single prediction.
    """
    model_name = model_name.upper()
    if model_name == "BERT":
        app.state.model = app.state.model_BERT
    if model_name == "Bidir":
        model_name = "LSTM"
        app.state.model = app.state.model_BIDIR
    if model_name == "Glove":
        model_name = "LSTM"
        app.state.model = app.state.model_GLOVE
    if model_name == "GRU":
        app.state.model = app.state.model_GRU

    try:
        if predict_language(text) != "en":
            text = translation(text)
    except:
        pass

    l = []
    l.append(text)
    X_pred = pd.DataFrame({"text":l})
    sentence = X_pred["text"][0]

    print("Starting: pred")
    y_pred = fast_pred(model_name, app.state.model, X_pred, clean_param, preproc_params_LSTM_bidir)
    print("pred is done")
    print(y_pred)


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

@app.get("/ping")
def pong():
    return {"response":"pong"}
