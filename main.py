from app.packages.preprocessing.cleaning import *
from app.packages.preprocessing.preprocessing_ML import *
from app.packages.preprocessing.translate import *

DB_URL = "data/raw_data/merged_df_en.csv"
text_col = "text"

def main():
    clean = cleaning(DB_URL)
    df = clean.all_in_one(clean.data,text_col)




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
