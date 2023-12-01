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
def clean_data(data:pd.DataFrame=None):
    if data == None:
        data = init_data(DB_URL)
    cleaned_df = clean_new(data)
    return cleaned_df

@task
def preprocess_new_data(cleaned_df: pd.DataFrame, model_name:str, preproc_params:dict):
    X_train_preproc, X_test_preproc, y_train, y_test = preprocess(model_name=model_name,cleaned_df=cleaned_df ,preproc_params=preproc_params)
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
def train_flow(model_name:str, preproc_params:dict, model_params:dict):
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """


    cleaned_df = clean_data()
    X_train_preproc, X_test_preproc, y_train, y_test = preprocess_new_data(cleaned_df, model_name, preproc_params)
    evaluate_production_model(model_name, X_test_preproc, y_test, preproc_params)

    new_train(model_name, X_train_preproc, y_train, preproc_params,model_params)


    # Define your tasks
    # data = preprocess_new_data.submit(min_date, max_date)
    # old_mae = evaluate_production_model.submit(min_date, max_date,wait_for=[data])

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
    if new_mae_result < old_mae_result:
            transition_model("Staging", "Production")
    # Actually launch your workflow





if __name__ == "__main__":
    train_flow()
