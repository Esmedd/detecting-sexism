from app.packages.preprocessing.cleaning import *
from app.packages.preprocessing.preprocessing_ML import *
from app.packages.preprocessing.translate import *
from sklearn.model_selection import train_test_split, cross_validate

DB_URL = "data/raw_data/merged_df_en.csv"
text_col = "text"
selected_col = ["text", "sexist_binary"]
concatenate = False
url_label = "[URL]"
usr_label = "[USERNAME]"
target = "sexist_binary"

def main():
    clean = cleaning(DB_URL)
    df = clean.all_in_one(clean.data,text_col,selected_col,concatenate,url_label, usr_label)

    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    X_train_preproc = preprocessing(X_train,text_col,False,False, True)
    X_test_preproc = preprocessing(X_test,text_col,False,False, True)





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
