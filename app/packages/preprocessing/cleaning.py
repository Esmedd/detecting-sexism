import pandas as pd
import emoji
import re
import string
import unidecode
from app.packages.utils import *

@simple_time_and_memory_tracker
class cleaning:
    def __init__(self, path):
        """Create self.data from a path to a csv"""
        self.data = pd.read_csv(path)

    # Drop duplicates from one selected column
    @simple_time_and_memory_tracker
    def drop_duplicates_from_one_col(self, data, duplicate_cols: str):
        """Drop duplicates based on One column
        example """
        print(f"Amount of duplicates on : {data[duplicate_cols].duplicated().value_counts()}")
        df = data[data[duplicate_cols].duplicated() == False]
        return df


    # Construct Dataframe from selected columns and drop nan values
    @simple_time_and_memory_tracker
    def drop_na(self,data, selected_columns: list = None):
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
    def urls_remover(self, data, target_column:str, label:str="[URL]"):
        data[target_column] = data[target_column].str.replace(r'\s*https?://\S+(\s+|$)',label, regex=True)
        data[target_column] = data[target_column].str.replace(r'\s*http?://\S+(\s+|$)',label,regex=True)
        return data

    # Remove Usernames in text and replace them by "[USERNAME]"
    @simple_time_and_memory_tracker
    def username_remover(self, data, target_column:str, label:str="[USERNAME]"):
        data[target_column] = data[target_column].str.replace(r'\s*@\S+(\s+|$)',label, regex=True)
        return data

    # Replace Emojis in text and replace them by their descriptions wrapped in squared brackets
    @simple_time_and_memory_tracker
    def emoji_replacer(self, data, target_column:str):
        data[target_column] = data[target_column].apply(lambda x:emoji.demojize(x, delimiters=("[", "]")) )
        return data

    # Replace Hashtags by the full word or by separating words at each uppercase letter
    @simple_time_and_memory_tracker
    def hashtag_adapter(self,data:pd.DataFrame, target_column:str, concatenate:bool=True):
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
    def remove_punctuation(self,data:pd.DataFrame, text_col:str): #remove punctuation
        def remove_punct(text):
            for punctuation in string.punctuation:
                if punctuation != "[" and punctuation != "]":
                    text = text.replace(punctuation,'')
            return text
        data[text_col] = data[text_col].apply(lambda x: remove_punct(x))
        return data

    @simple_time_and_memory_tracker
    def lower_case(self, data:pd.DataFrame, text_col:str):   #turn into lower case
        data[text_col] = data[text_col].apply(lambda x: x.lower())
        return data

    @simple_time_and_memory_tracker
    def remove_accents(self,data:pd.DataFrame, text_col:str):
        data[text_col] = data[text_col].apply(lambda x: unidecode.unidecode(x))
        return data

    @simple_time_and_memory_tracker
    def remove_numbers(self,data:pd.DataFrame, text_col:str):
        data[text_col] = data[text_col].apply(lambda x: "".join([word for word in x if word.isalpha()]))
        return data

    @simple_time_and_memory_tracker
    def strip(self,data, dropna_cols: str):
        """Manage text columns with strip and splitting the text"""
        serie = data[dropna_cols].apply(lambda x: x.strip() if isinstance(x, str) else x)
        data[dropna_cols] = serie
        return data

    def all_in_one(self, data : pd.DataFrame(), text_col: str, selected_cols:list = None, concatenate:bool=True, url_label:str="[URL]", usr_label:str="[USERNAME]"):
        """ Does a sequence of the above functions :
        >>> 'selected_cols' -> columns to keep in the Dataframe
        >>> 'text_col' -> text column to clean

        >>> 'concatenate=False' -> will split the hashtags at each uppercase letter
        >>> 'concatenate=True' -> will keep the hashtag intact

        >>> 'url_label' -> label to replace urls in text. Default : [URL]
        >>> 'usr_label' -> label to replace usernames in text. Default : [USERNAME]

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
        data_dups = self.drop_duplicates_from_one_col(data, text_col)
        data_dropna = self.drop_na(data_dups,selected_cols)
        data_url = self.urls_remover(data_dropna,text_col, label=url_label)
        data_usr = self.username_remover(data_url,text_col, label=usr_label)
        data_emoji = self.emoji_replacer(data_usr,text_col)
        data_hashtag = self.hashtag_adapter(data_emoji,text_col, concatenate)
        data_punctuation = self.remove_punctuation(data_hashtag,text_col)
        data_lower = self.lower_case(data_punctuation,text_col)
        data_accent = self.remove_accents(data_lower,text_col)
        data_number = self.remove_punctuation(data_accent,text_col)
        data_strip = self.strip(data_number,text_col)

        print("✅ All in One is done")
        return data_strip
