import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from app.interface.main import *
from app.packages.preprocessing.preprocessing_ML import *
from app.packages.data_storage.registry import *
from params import *

@task
def clean_data(clean_param:dict, data:pd.DataFrame=None):
    if type(data) != pd.DataFrame :
        data = init_data(DB_URL)
    cleaned_df = clean_new(data,clean_param)
    return cleaned_df

# @task
# def preprocess_new_data(new_data: pd.DataFrame, model_name:str, preproc_params:dict):
#     X_preproc, to_ignore = preproc_test(new_data,new_data, model_name, preproc_params)
#     return X_preproc


@task
def preprocess_new_data(cleaned_df: pd.DataFrame, model_name:str, preproc_params:dict):
    X_train_preproc, X_test_preproc, y_train, y_test = preprocess(model_name=model_name,cleaned_df=cleaned_df ,preproc_params=preproc_params)
    print("X_train",X_train_preproc)
    print("X_test",X_test_preproc)
    print("y_train",y_train)
    print("y_test",y_test)
    return X_train_preproc, X_test_preproc, y_train, y_test

@task
def preprocess_new_data_splitted(Train: pd.DataFrame,Test: pd.DataFrame, model_name:str, preproc_params:dict):
    X_train_preproc, X_test_preproc, y_train, y_test = preprocess_splitted(model_name=model_name,Train=Train, Test=Test,preproc_params=preproc_params)
    return X_train_preproc, X_test_preproc, y_train, y_test

@task
def evaluate_production_model(model_name:str,X_test_preproc, y_test, preproc_params:dict,stage:str="Production",batch_size:int=32):
    return evaluate(model_name, X_test_preproc, y_test, preproc_params, stage, batch_size)

@task
def new_train(model_name, X_train_preproc, y_train, preproc_params,model_params):
    return train(model_name, X_train_preproc, y_train, preproc_params,model_params)


@task
def transition_model(current_stage: str, new_stage: str):
    return mlflow_transition_model(current_stage,new_stage)


@flow(name=PREFECT_FLOW_NAME)
def train_flow(model_name:str,clean_params:dict , preproc_params:dict, model_params:dict, data:pd.DataFrame=None):
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """



    cleaned_df = clean_data.submit(clean_params, data)
    X_train_preproc, X_test_preproc, y_train, y_test = preprocess_new_data(model_name=model_name,cleaned_df=cleaned_df ,preproc_params=preproc_params)

    metrics = evaluate_production_model.submit(model_name, X_test_preproc, y_test, preproc_params)

    history = new_train.submit(model_name, X_train_preproc, y_train, preproc_params,model_params)

    new_acc = round(np.min((history.result()).history['val_recall']), 2)
    old_acc = round((metrics.result())["recall"], 2)

    print(f"🏁 new_acc: {new_acc} // old_acc: {old_acc}")
    if new_acc > old_acc:
        transition_model("Staging", "Production")
        print("⭐️ New model has been set on Production")

    return

@flow(name=PREFECT_FLOW_NAME)
def train_flow_splitted(model_name:str,clean_params:dict , preproc_params:dict, model_params:dict, Train:pd.DataFrame=None,Test:pd.DataFrame=None ):
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """


    Train = clean_data.submit(clean_params, Train)
    Test = clean_data.submit(clean_params, Test)
    preproc= preprocess_new_data_splitted.submit(model_name=model_name,Train=Train, Test=Test ,preproc_params=preproc_params, wait_for=[Train,Test])
    X_train_preproc, X_test_preproc, y_train, y_test=preproc.result()
    metrics = evaluate_production_model.submit(model_name, X_test_preproc, y_test, preproc_params)

    history = new_train.submit(model_name, X_train_preproc, y_train, preproc_params,model_params)

    history = history.result()
    metrics = metrics.result()

    print(f"Metrics Keys: {metrics.keys()} // History Keys : {history.keys()}")
    new_recall = round(np.min(history.history['val_recall']), 2)
    old_recall = round((metrics)["recall"], 2)

    new_acc = round(np.min(history.history['val_accuracy']), 2)
    old_acc = round((metrics)["accuracy"], 2)

    new_precision = round(np.min(history.history['val_precision']), 2)
    old_precision = round((metrics)["precision"], 2)


    print(f"🏁 new_acc: {new_acc} // old_acc: {old_acc}")
    print(f"🏁 new_recall: {new_recall} // old_recall: {old_recall}")
    print(f"🏁 new_precision: {new_precision} // old_precision: {old_precision}")

    if new_acc > old_acc:
        transition_model("Staging", "Production")
        print("⭐️ New model has been set on Production")

    return

    # batch_size = [128,256,512]#[64,128,256]
    # learning_rate = [0.1, 0.09, 0.05] #[0.1, 0.01, 0.005, 0.001]
    # patience = [2,4]#[2,4,6]


    # l = []
    # for b in batch_size:
    #     for lr in learning_rate:
    #         for pat in patience:
    #             dico = {}
    #             new_mae = re_train.submit(min_date, max_date, 0.2,lr,b,pat, wait_for=[data])
    #             dico["new_mae"] = new_mae
    #             dico["params"] = f"{lr},{b},{pat}"
    #             l.append(dico)

    # Compute your results as actual python object
    # l2 = []
    # old_mae_result = old_mae.result()

    # for i in range(0,len(l)):
    #     l2.append(l[i]["new_mae"].result())


    # ind = l2.index(min(l2))
    # lr , b , p = l[ind]["params"]

    # print(f"Best Params -  learning_rate: {lr} / batch_size: {b} / patience: {p}")
    # new_mae = re_train.submit(min_date, max_date, 0.2,lr,b,p)
    # new_mae_result = new_mae.result()

    # new_mae_result

    # Do something with the results (e.g. compare them)

    # Actually launch your workflow





if __name__ == "__main__":
    train_flow()
