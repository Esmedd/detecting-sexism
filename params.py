import os

##################  VARIABLES  #####################
preproc_params_LSTM = {
    "max_length":100,
    "vector_size":50,
    "window":5,
    "embed":False,
    "lower":True,
    "split":" ",
    "dtype":"float32",
    "padding":"post",
}

model_params_LSTM = {
    "lstm_units":50,
    "lstm_activation":"tanh",
    "loss":"binary_crossentropy",
    "optimizer":"rmsprop",
    "batch_size":64,
    "patience":2,
    "validation_split":0.2
}


preproc_params_GRU = {
    "max_length":100,
    "vector_size":50,
    "window":5,
    "lower":True,
    "split":" ",
    "dtype":"float32",
    "padding":"post",
}

model_params_GRU = {
    "gru_units":50,
    "gru_activation":"tanh",
    "loss":"binary_crossentropy",
    "optimizer":"rmsprop",
    "batch_size":64,
    "patience":2,
    "validation_split":0.2
}


##################   MLFLOW    #####################
MODEL_TARGET="mlflow"
MLFLOW_TRACKING_URI="https://mlflow.lewagon.ai"
MLFLOW_EXPERIMENT="You_are_not_sexist-experiment"
MLFLOW_MODEL_NAME="You_are_not_sexist-model"

##################     GCP     #####################
GCP_PROJECT = "youre-not-sexist" #os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = "" #os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = "europe-west9" #os.environ.get("GCP_REGION")

BQ_DATASET = "merged_df_en" #os.environ.get("BQ_DATASET")
BQ_REGION = "europe-west9" #os.environ.get("BQ_REGION")
BUCKET_NAME = "youre-not-sexist" #os.environ.get("BUCKET_NAME")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GCP_key")

##################  CONSTANTS  #####################
LOCAL_RAW_DATA_PATH = os.path.join(os.path.expanduser('~'),"code","Esmedd", "detecting-sexism", "data", "raw_data", "merged_df_en.csv")
LOCAL_CLEANED_DATA_PATH = os.path.join(os.path.expanduser('~'),"code","Esmedd", "detecting-sexism", "data", "cleaned_data")
LOCAL_PROCESSED_DATA_PATH = os.path.join(os.path.expanduser('~'),"code","Esmedd", "detecting-sexism", "data", "processed_data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code","Esmedd", "detecting-sexism", "training_outputs")
