from googletrans import Translator, LANGUAGES, LANGCODES
from app.packages.preprocessing.cleaning import *
import numpy as np

def translation(path, text_col:str, selected_cols:list = None,method:str="concat", dest:str="en" ):
    """Function to translate automatically a text column.
    >>> 'path' -> csv to translate

    Will call Cleaning Class beforehand.

    -------- CLEANING PACKAGE --------
        >>> 'selected_cols' columns to keep in the Dataframe
        >>> 'text_col' text column to clean

        >>> 'method=splitted' will split the hashtags at each uppercase letter
        >>> 'method=concat' will keep the hashtag intact

        1st : drop_duplicates_from_one_col()
        2nd : drop_na()
        3rd : urls_remover()
        4th : username_remover()
        5th : emoji_replacer()
        6th : hashtag_adapter()
        7th : strip()

        example : clean.all_in_one(data=clean.data, text_col="text", selected_cols=["text", "sexist_binary"], method="splitted")
    """
    clean = cleaning(path)
    data = clean.all_in_one(data=clean.data, text_col=text_col, selected_cols=selected_cols, method=method)
    translator = Translator()
    data[text_col] = data[text_col].apply(lambda x: translator.translate(x, dest=dest).text)
    return data

def add_to_csv(dest_path:str, data: pd.DataFrame):
    """ Add the specified data to the csv mentionned in dest_path"""

    data.index = np.arange(len(pd.read_csv(dest_path)), len(pd.read_csv(dest_path))+len(data), 1)
    data.to_csv(dest_path,mode= "a", index=True, header=False )
    print("Data appended successfully.")
