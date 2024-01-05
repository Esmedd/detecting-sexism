import pandas as pd
import emoji
import re
import string
import unidecode
from app.packages.utils import *

@simple_time_and_memory_tracker
def init_data( path):
    """Create data from a path to a csv"""
    data = pd.read_csv(path)
    return data

# Drop duplicates from one selected column
@simple_time_and_memory_tracker
def drop_duplicates_from_one_col( data, duplicate_cols: str):
    """Drop duplicates based on One column
    example """
    print(f"Amount of duplicates on : {data[duplicate_cols].duplicated().value_counts()}")
    df = data[data[duplicate_cols].duplicated() == False]
    return df


# Construct Dataframe from selected columns and drop nan values
@simple_time_and_memory_tracker
def drop_na(data, selected_columns: list = None):
    """Returns Df with selected columns and drop na"""
    if selected_columns == None:
        data = data
    else:
        data = data[selected_columns]
    print(f"Amount of NaN on : {data.isna().sum()}")
    data = data.dropna()
    return data

# Remove Urls in text and replace them by "[URL]""
@simple_time_and_memory_tracker
def urls_remover( data, target_column:str, label:str="[URL]"):
    """Removes URLs from target column in a given data frame
    replaces them with a chosen label

    Args:
        data (pd.DataFrame): dataframe
        target_column (str): name of column to use
        label (str, optional): String label. Defaults to "[URL]".

    Returns:
        data (pd.DataFrame): returns updated dataframe
    """
    data[target_column] = data[target_column].str.replace(r'\s*https?://\S+(\s+|$)',label, regex=True)
    data[target_column] = data[target_column].str.replace(r'\s*http?://\S+(\s+|$)',label,regex=True)
    return data

    # Remove Usernames in text and replace them by "[USERNAME]"
@simple_time_and_memory_tracker
def username_remover( data, target_column:str, label:str="[USERNAME]"):
    """Removes usernames from target column in a given data frame
    replaces them with a chosen label

    Args:
        data (pd.DataFrame): dataframe
        target_column (str): name of column to use
        label (str, optional): String label. Defaults to "[USERNAME]".

    Returns:
        data (pd.DataFrame): returns updated dataframe
    """
    data[target_column] = data[target_column].str.replace(r'\s*@\S+(\s+|$)',label, regex=True)
    return data

# Replace Emojis in text and replace them by their descriptions wrapped in squared brackets
@simple_time_and_memory_tracker
def emoji_replacer( data, target_column:str):
    """Replaces emojis in a given column of a given dataframe
    with their text decription

    Args:
        data (pd.DataFrame): dataframe
        target_column (str): name of column to use

    Returns:
        data (pd.DataFrame): returns updated dataframe
    """
    data[target_column] = data[target_column].apply(lambda x:emoji.demojize(x, delimiters=("[", "]")) )
    return data

# Replace Hashtags by the full word or by separating words at each uppercase letter
@simple_time_and_memory_tracker
def hashtag_adapter(data:pd.DataFrame, target_column:str, concatenate:bool=True):
    """Will replace hashtags found in text by a method.
    >>> 'concatenate=False' will split the hashtags at each uppercase letter
    >>> 'concatenate=True' will keep the hashtag intact
    both methods wrap the hashtags in squared brackets"""

    if concatenate == False:
        data[target_column] = data[target_column].apply(lambda x: re.sub(r'#(\w+)', lambda x: '[' + re.sub(r'(?!^)(?=[A-Z])|(?<=\D)(?=\d)|(?<=\d)(?=\D)', " ", x.group(1)+"]"), x) )
        return data
    elif concatenate == True:
        data[target_column] = data[target_column].apply(lambda x: re.sub(r'#(\w+)', lambda x: '[' + re.sub(r'(?!^)(?=[A-Z])|(?<=\D)(?=\d)|(?<=\d)(?=\D)', "", x.group(1)+"]"), x) )
        return data
    else:
        raise ValueError("Wrong method")

@simple_time_and_memory_tracker
def remove_punctuation(data:pd.DataFrame, text_col:str): #remove punctuation
    """Removes punctuation in a given column of a given dataframe

    Args:
        data (pd.DataFrame): dataframe
        text_col (str): name of column to use
    """
    def remove_punct(text):
        for punctuation in string.punctuation:
            if punctuation != "[" and punctuation != "]":
                text = text.replace(punctuation,'')
        return text
    data[text_col] = data[text_col].apply(lambda x: remove_punct(x))
    return data

@simple_time_and_memory_tracker
def lower_case( data:pd.DataFrame, text_col:str):   #turn into lower case
    data[text_col] = data[text_col].apply(lambda x: x.lower())
    return data

@simple_time_and_memory_tracker
def remove_accents(data:pd.DataFrame, text_col:str):
    data[text_col] = data[text_col].apply(lambda x: unidecode.unidecode(x))
    return data

@simple_time_and_memory_tracker
def remove_numbers(data:pd.DataFrame, text_col:str):
    data[text_col] = data[text_col].apply(lambda x: "".join([word for word in x if not word.isdigit()]))
    return data

@simple_time_and_memory_tracker
def strip(data, dropna_cols: str):
    """Manage text columns with strip and splitting the text"""
    serie = data[dropna_cols].apply(lambda x: x.strip() if isinstance(x, str) else x)
    data[dropna_cols] = serie
    return data

def all_in_one( data : pd.DataFrame(),
                text_col: str, selected_cols:list = None,
                concatenate:bool=True, url_label:str="[URL]",
                usr_label:str="[USERNAME]", func_to_exec:list=[True]*11):
    """ Does a sequence of the above functions :
    >>> 'selected_cols' -> columns to keep in the Dataframe
    >>> 'text_col' -> text column to clean

    >>> 'concatenate=False' -> will split the hashtags at each uppercase letter
    >>> 'concatenate=True' -> will keep the hashtag intact

    >>> 'url_label' -> label to replace urls in text. Default : [URL]
    >>> 'usr_label' -> label to replace usernames in text. Default : [USERNAME]

    >>> 'func_to_exec' -> list of functions to use or not to use, accept boolean for each function
    >>> example : '[True, True, True, False, False, True, True, True, True, False, False]
    1st : drop_duplicates_from_one_col()
    2nd : drop_na()
    3rd : urls_remover()
    4th : username_remover()
    5th : emoji_replacer()
    6th : hashtag_adapter()
    7th : remove_punctuation
    8th : lower_case
    9th : remove_accents
    10th : remove_punctuation
    11th : strip()

    example : clean.all_in_one(data=clean.data, text_col="text", selected_cols=["text", "sexist_binary"], method="splitted")"""
    if func_to_exec[0] == True:
        data = drop_duplicates_from_one_col(data, text_col)
    if func_to_exec[1] == True:
        data = drop_na(data,selected_cols)
    if func_to_exec[2] == True:
        data = urls_remover(data,text_col, label=url_label)
    if func_to_exec[3] == True:
        data= username_remover(data,text_col, label=usr_label)
    if func_to_exec[4] == True:
        data = emoji_replacer(data,text_col)
    if func_to_exec[5] == True:
        data = hashtag_adapter(data,text_col, concatenate)
    if func_to_exec[6] == True:
        data = remove_punctuation(data,text_col)
    if func_to_exec[7] == True:
        data = lower_case(data,text_col)
    if func_to_exec[8] == True:
        data = remove_accents(data,text_col)
    if func_to_exec[9] == True:
        data = remove_numbers(data,text_col)
    if func_to_exec[10] == True:
        data = strip(data,text_col)

    print("✅ All in One is done")
    return data
