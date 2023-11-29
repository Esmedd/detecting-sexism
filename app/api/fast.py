import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.packages.data_storage.registry import load_model
from app.packages.preprocessing.preprocessing_ML import *
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?text=
@app.get("/predict")
def predict(text):
    """
    Make a single prediction.
    """

    X_pred = pd.DataFrame(dict(text=text))

    model = load_model()
    assert model is not None

    #X_processed = preprocess_features(X_pred)
    #y_pred = model.predict(X_processed)
    #return {"prediction": float(y_pred[0][0])}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
