from app.packages.preprocessing.cleaning import *
from app.packages.preprocessing.preprocessing_ML import *
# from app.packages.preprocessing.translate import *
from app.ml_logic.multinomial.multinomial_model import *
from app.packages.data_storage.data_storage import *
#from app.packages.data_storage.registry import *

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

def clean():
    pass
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    # X_train_preproc = preprocessing(X_train,text_col,False,False, True)
    # X_test_preproc = preprocessing(X_test,text_col,False,False, True)

    # # Temporary Baseline Model
    # results = cv_multinomial_baseline_model(X_train_preproc[selected_col], y_train)

def preprocess():
    """"""
    # Call Cleaning function and return a df
    clean = cleaning(DB_URL)
    df = clean.all_in_one(clean.data,text_col,selected_col,concatenate,url_label, usr_label)

    # Split X and y and preprocess X
    X = df.drop(target, axis=1)
    y = df[[target]]
    X_preproc = preprocessing(X,text_col,False,False, True)
    # Create a df from X preprocessed and y in order to save it
    data_processed = pd.DataFrame(np.concatenate((
        X_preproc,
        y
    ),axis=1))

    load_data_to_bq(
    data_processed,
    gcp_project=GCP_PROJECT,
    bq_dataset=BQ_DATASET,
    table=f'df_processed_by_{preproc_name}',
    truncate=True
    )

    print("✅ preprocess() done \n")

preprocess()

#@mlflow_run
def train():
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.df_processed_by_{preproc_name}
        ORDER BY _0 ASC
    """
    data_processed_cache_path = Path(LOCAL_PROCESSED_DATA_PATH).joinpath(f"df_processed_by_{preproc_name}.csv")


    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    X_processed = data_processed[:, :-1]
    y = data_processed[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y , split_ratio)



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
