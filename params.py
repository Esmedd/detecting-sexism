import os
from tensorflow.keras.optimizers import Adam

##################  VARIABLES  #####################
clean_param = {
    "concatenate":False,
    "url_label":"[URL]",
    "usr_label":"[USERNAME]",
    "functions":[True,True,True,True,True,True,True,True,True,True,True]
}

preproc_params_LSTM = {
    "max_length":100,
    "vector_size":50,
    "window":5,
    "embed":False,
    "lower":True,
    "split":" ",
    "dtype":"float32",
    "padding":"post",
    "bidirectional":False
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

preproc_params_LSTM_bidir = {
    "max_length":100,
    "vector_size":50,
    "window":5,
    "embed":True,
    "lower":True,
    "split":" ",
    "dtype":"float32",
    "padding":"post",
    "bidirectional":True
}

model_params_LSTM_bidir = {
    "lstm_units":50,
    "lstm_activation":"tanh",
    "loss":"binary_focal_crossentropy",
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

preproc_params_c1d = {
    "max_length":100,
    "vector_size":50,
    "window":5,
    "lower":False,
    "split":" ",
    "dtype":"float32",
    "padding":"post",
}

model_params_c1d = {
    "embedding_size":100,
    "loss":"binary_crossentropy",
    "optimizer":Adam(learning_rate=0.001),
    "globalmax":True,
    "complex":True,
    "batch_size":64,
    "patience":1,
    "validation_split":0.2
}

preproc_params_LSTM_embed = {
    "max_length":100,
    "vector_size":50,
    "window":5,
    "embed":True,
    "lower":True,
    "split":" ",
    "dtype":"float32",
    "padding":"post",
    "bidirectional":False
}

model_params_LSTM_embed = {
    "lstm_units":50,
    "lstm_activation":"tanh",
    "loss":"binary_crossentropy",
    "optimizer":"rmsprop",
    "batch_size":64,
    "patience":1,
    "validation_split":0.2
}

##################   MLFLOW    #####################
MODEL_TARGET="mlflow"
MLFLOW_TRACKING_URI="https://mlflow.lewagon.ai"
MLFLOW_EXPERIMENT="You_are_not_sexist-experiment"
MLFLOW_MODEL_NAME="You_are_not_sexist-model"

##################   Prefect    ####################
PREFECT_FLOW_NAME="you-are-not-sexist"
PREFECT_LOG_LEVEL="WARNING"

##################     GCP     #####################
GCP_PROJECT = "youre-not-sexist" #os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = "" #os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = "europe-west9" #os.environ.get("GCP_REGION")

BQ_DATASET = "merged_df_en" #os.environ.get("BQ_DATASET")
BQ_REGION = "europe-west9" #os.environ.get("BQ_REGION")
BUCKET_NAME = "youre-not-sexist" #os.environ.get("BUCKET_NAME")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GCP_key")

##################  CONSTANTS  #####################
LOCAL_RAW_DATA_PATH = os.path.join(os.path.expanduser('~'),"code","Esmedd", "detecting-sexism", "data", "raw_data", "merged_df_en.csv")
LOCAL_CLEANED_DATA_PATH = os.path.join(os.path.expanduser('~'),"code","Esmedd", "detecting-sexism", "data", "cleaned_data")
LOCAL_PROCESSED_DATA_PATH = os.path.join(os.path.expanduser('~'),"code","Esmedd", "detecting-sexism", "data", "processed_data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code","Esmedd", "detecting-sexism", "training_outputs")
