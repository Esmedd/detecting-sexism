from app.packages.preprocessing.cleaning import *
from app.packages.preprocessing.preprocessing_ML import *
# from app.packages.preprocessing.translate import *
import numpy as np
##################   Import Models ##################
from app.models.multinomial_model import *
from app.models.conv1d_model import *
from app.models.GRU_model import *
from app.models.lstm_model import *

from app.packages.data_storage.data_storage import *
from app.packages.data_storage.registry import *
from app.models.lstm_model import *

from sklearn.model_selection import train_test_split, cross_validate


DB_URL = "data/raw_data/merged_df_en.csv"
text_col = "text"
selected_col = ["text", "sexist_binary"]
concatenate = False
url_label = "[URL]"
usr_label = "[USERNAME]"
target = "sexist_binary"
preproc_name="test01"
split_ratio = 0.2
model_names = ["conv1d", "GRU", "LSTM", "multinomial", "BERT"]
model_name = "LSTM"

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


def clean_from_path():
    # Call Cleaning function and return a df
    data_from_csv = init_data(DB_URL)
    df = all_in_one(data_from_csv,text_col,selected_col,concatenate,url_label, usr_label)
    return df
    # Save the cleaned DataFrame locally and in Big Query

def clean_new(data:pd.DataFrame):
    df = all_in_one(data,text_col,selected_col,concatenate,url_label, usr_label)
    return df


def preprocess(model_name:str, cleaned_df:pd.DataFrame, preproc_params):
    """
    >>> Initialize preprocessing depending on 'model_name'

    model_name accepts these models :
    "conv1d", "GRU", "LSTM", "multinomial", "BERT"

    >>> 'params' argument takes a dictionnary to mention all parameters on your preprocessing
    example preprocessing parameters for LSTM :
    'params_LSTM = {
    >>> "max_length":100,
    >>> "vector_size":50,
    >>> "window":5,
    >>> "embed":False,
    >>> "lower":True,
    >>> "split":" ",
    >>> "dtype":"float32",
    >>> "padding":"post"}'

    You can refer to your model in preprocessing_ML.py to see all the variables you can mention
    """

    print("\n⭐️ Pre-processing : Starting")

    # Split X and y and preprocess X
    X = cleaned_df.drop(target, axis=1)
    y = cleaned_df[[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=split_ratio)

    if model_name == "conv1d":
        X_train_preproc, X_test_preproc = preproc_test(X_train, X_test, model_name, preproc_params)
    else:
        X_train_preproc, X_test_preproc = preproc_test(X_train, X_test, model_name, preproc_params)

    print("✅ preprocess() done \n")
    return X_train_preproc, X_test_preproc, y_train, y_test




@mlflow_run
def train(model_name:str,X_train_preproc, y_train, preproc_params: dict, model_params:dict):
    """
    >>> Initialize preprocessing and training depending on 'model_name' then load it on Ml-flow

    model_name accepts these models :
    "conv1d", "GRU", "LSTM", "multinomial", "BERT"

    >>> 'preproc_params' argument takes a dictionnary to mention all parameters on your preprocessing
    example preprocessing parameters for LSTM :
    'preproc_params_LSTM = {
    >>> "max_length":100,
    >>> "vector_size":50,
    >>> "window":5,
    >>> "embed":False,
    >>> "lower":True,
    >>> "split":" ",
    >>> "dtype":"float32",
    >>> "padding":"post"}'

    You can refer to your model in preprocessing_ML.py to see all the variables you can mention

    >>> 'model_params' argument takes a dictionnary to mention all parameters on your training
    example model parameters for LSTM :
    model_params_LSTM = {
    >>> "lstm_units":50,
    >>> "lstm_activation":"tanh",
    >>> "loss":"binary_crossentropy",
    >>> "optimizer":"rmsprop",
    >>> "batch_size":64,
    >>> "patience":2,
    >>> "validation_split":0.2}

    You can refer to your model in models."model_name".py to see all the variables you can mention.

    We could go further and add to dictionnary the number of neruons per layer, etc.
    """

    print("\n⭐️ Training : Starting")


    if model_name == "conv1d":
        X_train_preproc_conv = X_train_preproc[0][0]
        train_vocab_size = X_train_preproc[0][1]
        train_max_length = X_train_preproc[0][2]

    if preproc_params["embed"] == True:
        X_train_preproc_emb = X_train_preproc[0][0]
        train_word_index = X_train_preproc[0][1]
        train_vocab_size = X_train_preproc[0][2]



    # Train model using `model.py`
    # model = load_model(model_name=model_name)

    # if model is None:
    #     pass
    if model_name == "LSTM":
        if preproc_params["embed"] == True:
            model = initialize_lstm(lstm_units=model_params["lstm_units"],lstm_activation=model_params["lstm_activation"],max_length=preproc_params["max_length"], embedding=preproc_params["embed"], word_index=train_word_index)
            model = compile_lstm_model(model=model, loss=model_params["loss"], optimizer=model_params['optimizer'])
            model, history = train_lstm_model(model=model, X=X_train_preproc_emb, y=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])

        else:
            model = initialize_lstm(lstm_units=model_params["lstm_units"],lstm_activation=model_params["lstm_activation"], embedding=preproc_params["embed"])
            model = compile_lstm_model(model=model, loss=model_params["loss"], optimizer=model_params['optimizer'])
            model, history = train_lstm_model(model=model, X=X_train_preproc, y=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])

    if model_name == "multinomial":
        pass
    if model_name == "GRU":
        model = initialize_gru(gru_units=model_params["gru_units"], gru_activation=model_params["gru_activation"])
        model = compile_gru_model(model=model, loss=model_params["loss"], optimizer=model_params['optimizer'])
        model, history = train_gru_model(model=model, X=X_train_preproc, y=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])

    if model_name == "conv1d":
        model = intialize_c1d(vocab_size=train_vocab_size, maxlen=train_max_length, embedding_size=model_params["embedding_size"], loss=model_params["loss"], optimizer=model_params["optimizer"], globalmax=model_params["globalmax"], complex=model_params["complex"])
        model, history = train_c1d_model(model=model, X_train=X_train_preproc_conv, y_train=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])
    # if model_name == "BERT":
    #     BERT_preprocess()

    val_loss = np.min(history.history["loss"])

    params = dict(
        context="you_are_not_sexist",
        model_used=model_name,
        loss_used=model_params["loss"],
        row_count=len(X_train_preproc),
    )

    # Save results on the hard drive using taxifare.models.registry
    save_results(params=params, metrics=dict(val_loss=val_loss))

    # Save model weight on the hard drive (and optionally on GCS too!)
    try:
        if preproc_params["embed"] == True:
            model_name = f"{model_name}_embed"
    except:
        pass

    save_model(model_name=model_name,model=model)

    # The latest model should be moved to staging
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(model_name=model_name,current_stage="None", new_stage="Staging")

    print("✅ train() done \n")

@mlflow_run
def evaluate(model_name:str,X_test_preproc, y_test, preproc_params:dict,stage:str="Staging",batch_size:int=32) -> float:

    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print("\n⭐️ Evaluate : Starting")

    if model_name == "conv1d":
        X_test_preproc_conv = X_test_preproc[0][0]
        test_vocab_size = X_test_preproc[0][1]
        test_max_length = X_test_preproc[0][2]

    if preproc_params["embed"] == True:
        model_name = f"{model_name}_embed"
    model = load_model(model_name=model_name,stage=stage)
    assert model is not None


    if model_name == "LSTM" and preproc_params["embed"] == False:
        metrics = evaluate_lstm_model(model, X=X_test_preproc, y=y_test, batch_size=batch_size )
    if preproc_params["embed"] == True:
        print(X_test_preproc)
        print(X_test_preproc.shape)
        metrics = evaluate_lstm_model(model, X=X_test_preproc, y=y_test, batch_size=batch_size )
    if model_name == "multinomial":
        pass
    if model_name == "GRU":
        metrics = evaluate_gru_model(model,X_test_preproc, y_test, batch_size=batch_size)
    if model_name == "conv1d":
        metrics = evaluate_c1d_model(model, X_test=X_test_preproc_conv, y_test=y_test, batch_size=batch_size)
    # if model_name == "BERT":


    params = dict(
        context="evaluate", # Package behavior,
        row_count=len(X_test_preproc),
        model_used=model_name,
        loss_used=metrics["loss"],
    )

    save_results(params=params, metrics=metrics)

    print("✅ evaluate() done \n")


    return metrics

@simple_time_and_memory_tracker
def pred(model_name:str,X_pred: pd.DataFrame, preproc_params:dict,stage:str="Production") -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")
    X_clean = all_in_one(X_pred,text_col,selected_cols=["text"])

    try:
        if preproc_params["embed"] == True:
            model_name = f"{model_name}_embed"
    except:
        pass

    model = load_model(model_name=model_name, stage=stage)
    assert model is not None
    print(model_name)
    X_proc = preproc_pred(X_clean, model_name, preproc_params)

    print("\n🏁 Predict: Model has been load")

    if model_name == "conv1d":
        X_proc = X_proc[0][0]

    y_pred = model.predict(X_proc)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred

#@mlflow_run
# def evaluate(
#      stage: str = "Production") -> float


def test_main(model_name:str, preproc_params:dict, model_params:dict, data:pd.DataFrame=None) :
    if data == None:
        data = init_data(DB_URL)
    cleaned_df = clean_new(data)
    X_train_preproc, X_test_preproc, y_train, y_test = preprocess(model_name=model_name,cleaned_df=cleaned_df ,preproc_params=preproc_params)
    train(model_name, X_train_preproc, y_train, preproc_params,model_params)
    metrics = evaluate(model_name, X_test_preproc, y_test,preproc_params)


if __name__ == '__main__':
    try:
        pass
        #preprocess_and_train()
        # preprocess()
        # train()
        #pred()
    except:
        pass
        #import sys
        #import traceback

        #import ipdb
        #extype, value, tb = sys.exc_info()
        #traceback.print_exc()
        #ipdb.post_mortem(tb)
