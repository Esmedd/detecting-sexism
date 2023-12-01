from app.packages.preprocessing.cleaning import *
from app.packages.preprocessing.preprocessing_ML import *
# from app.packages.preprocessing.translate import *

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


def clean():
    # Call Cleaning function and return a df
    clean = cleaning(DB_URL)
    df = clean.all_in_one(clean.data,text_col,selected_col,concatenate,url_label, usr_label)

    # Save the cleaned DataFrame locally and in Big Query
    load_data_to_bq(
    df,
    gcp_project=GCP_PROJECT,
    bq_dataset=BQ_DATASET,
    table=f'df_cleaned',
    truncate=True
    )

def preprocess(model_name:str, params):
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
    # Call Cleaning function and return a df
    clean = cleaning(DB_URL)
    df = clean.all_in_one(clean.data,text_col,selected_col,concatenate,url_label, usr_label)

    # Split X and y and preprocess X
    X = df.drop(target, axis=1)
    y = df[[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=split_ratio)

    if model_name == "conv1d":
        X_train_preproc, X_test_preproc = preproc_test(X_train, X_test, model_name, params)
    else:
        X_train_preproc, X_test_preproc = preproc_test(X_train, X_test, model_name, params)

    print("✅ preprocess() done \n")
    return X_train_preproc, X_test_preproc, y_train, y_test




@mlflow_run
def train(model_name:str, preproc_params, model_params):
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

    X_train_preproc, X_test_preproc, y_train, y_test = preprocess(model_name, preproc_params)

    if model_name == "conv1d":
        X_train_preproc = X_train_preproc[0]
        X_test_preproc = X_test_preproc[0]

        train_vocab_size = X_train_preproc[1]
        test_vocab_size = X_test_preproc[1]

        train_max_length = X_train_preproc[2]
        test_max_length = X_test_preproc[2]

    # Train model using `model.py`
    # model = load_model(model_name=model_name)

    # if model is None:
    #     pass
    if model_name == "LSTM":
        if params["embed"] == True:
            pass
            # return Embed_LSTM_preproc(X_train, X_test, params)
        else:
            model = initialize_lstm(lstm_units=model_params["lstm_units"],lstm_activation=model_params["lstm_activation"])
            model = compile_lstm_model(model=model, loss=model_params["loss"], optimizer=model_params['optimizer'])
            model, history = train_lstm_model(model=model, X=X_train_preproc, y=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])


    if model_name == "multinomial":
        pass
    if model_name == "GRU":
        model = initialize_gru(gru_units=model_params["gru_units"], gru_activation=model_params["gru_activation"])
        model = compile_gru_model(model=model, loss=model_params["loss"], optimizer=model_params['optimizer'])
        model, history = train_gru_model(model=model, X=X_train_preproc, y=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])

    if model_name == "conv1d":
        model = intialize_c1d(vocab_size=train_vocab_size, maxlen=train_max_length, embedding_size=model_params["embedding_size"], loss=model_params["loss"], optimizer=model_params["optimizer"], globalmax=model_params["globalmax"])
        model, history = train_c1d_model(model=model, X=X_train_preproc, y=y_train, batch_size=model_params["batch_size"], patience=model_params["patience"],validation_data=None,validation_split=model_params["validation_split"])
    # if model_name == "BERT":
    #     BERT_preprocess()

    val_loss = np.min(history.history[model_params["loss"]])

    params = dict(
        context="you_are_not_sexist",
        model_used=model_name,
        loss_used=model_params["loss"],
        row_count=len(X_train_preproc),
    )

    # Save results on the hard drive using taxifare.models.registry
    save_results(params=params, metrics=dict(val_loss=val_loss))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")


####### TEST LSTM ########

@mlflow_run
def train_LSTM(model_name:str, loss="val_loss"):
    # query = f"""
    #     SELECT * EXCEPT(_0)
    #     FROM {GCP_PROJECT}.{BQ_DATASET}.df_processed_by_{model_name}
    #     ORDER BY _0 ASC
    # """
    # data_processed_cache_path = Path(LOCAL_PROCESSED_DATA_PATH).joinpath(f"df_processed_by_{model_name}.csv")


    # data_processed = get_data_with_cache(
    #     gcp_project=GCP_PROJECT,
    #     query=query,
    #     cache_path=data_processed_cache_path,
    #     data_has_header=False
    # )
    clean = cleaning(DB_URL)
    data = clean.all_in_one(clean.data, text_col, selected_col, False)

    X = data.drop(target, axis=1)
    y = data[[target]]

    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)


    max_length = 100

    X_train_token = tokenize(X_train.text)
    X_test_token = tokenize(X_test.text)
    X_train_truncated = [sentence[:max_length] for sentence in X_train_token]
    X_train_padded, trained_word2vec_model = w2v_train_and_embed(X_train_truncated, vector_size=50, window=5)
    print(X_train_padded.shape)
    # X_test_padded = w2v_embed(X_test_token, trained_word2vec_model, max_length=len(X_train_padded[0]))
    # model = initialize_lstm()
    # model = compile_lstm_model(model)
    # fitted_model, history = train_lstm_model(model, X_train_padded, y_train, validation_data = None, validation_split = 0.2)
    # Train model using `model.py`
    # model = load_model(model_name=model_name)

    # if model is None:
    # pass
        #model = initialize_model(input_shape=X_train.shape[1:])

    # model = compile_model(model, learning_rate=learning_rate)
    # model, history = train_model(
    #     model, X_train_processed, y_train,
    #     batch_size=batch_size,
    #     patience=patience,
    #     validation_data=(X_val_processed, y_val)
    # )

    # val_loss = np.min(history.history[loss])

    # params = dict(
    #     context="you_are_not_sexist",
    #     model_used=model_name,
    #     loss_used=loss,
    #     row_count=len(X_train),
    # )

    # # Save results on the hard drive using taxifare.models.registry
    # save_results(params=params, metrics=dict(val_loss=val_loss))

    # # Save model weight on the hard drive (and optionally on GCS too!)
    # save_model(model=fitted_model, model_name=model_name)

    # # The latest model should be moved to staging
    # if MODEL_TARGET == 'mlflow':
    #     mlflow_transition_model(model_name=model_name ,current_stage="None", new_stage="Staging")

    # print("✅ train() done \n")






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
