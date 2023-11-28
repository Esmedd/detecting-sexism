import pandas as pd
import emoji
import re

class cleaning:
    def __init__(self, path):
        """Create self.data from a path to a csv"""
        self.data = pd.read_csv(path)

    # Drop duplicates from one selected column
    def drop_duplicates_from_one_col(self, data, duplicate_cols: str):
        """Drop duplicates based on One column
        example """
        print(f"Amount of duplicates on : {data[duplicate_cols].duplicated().value_counts()}")
        df = data[data[duplicate_cols].duplicated() == False]
        return df


    # Construct Dataframe from selected columns and drop nan values
    def drop_na(self,data, selected_columns: list = None):
        """Returns Df with selected columns and drop na"""
        if selected_columns == None:
            data = data
        else:
            data = data[selected_columns]
        print(f"Amount of NaN on : {data.isna().sum()}")
        data = data.dropna()
        return data

    # Remove Usernames in text and replace them by "[URL]""
    def urls_remover(self, data, target_column:str):
        data[target_column] = data[target_column].str.replace(r'\s*https?://\S+(\s+|$)','[URL]', regex=True)
        data[target_column] = data[target_column].str.replace(r'\s*http?://\S+(\s+|$)','[URL]',regex=True)
        return data

    # Remove Usernames in text and replace them by "[USERNAME]"
    def username_remover(self, data, target_column:str):
        data[target_column] = data[target_column].str.replace(r'\s*@\S+(\s+|$)','[USERNAME]', regex=True)
        return data

    # Replace Emojis in text and replace them by their descriptions wrapped in squared brackets
    def emoji_replacer(self, data, target_column:str):
        data[target_column] = data[target_column].apply(lambda x:emoji.demojize(x, delimiters=("[", "]")) )
        return data

    # Replace Hashtags by the full word or by separating words at each uppercase letter
    def hashtag_adapter(self,data, target_column:str, method: str):
        """Will replace hashtags found in text by a method.
        >>> 'method=splitted' will split the hashtags at each uppercase letter
        >>> 'method=concat' will keep the hashtag intact
        both methods wrap the hashtags in squared brackets"""

        if method == "splitted":
            data[target_column] = data[target_column].apply(lambda x: re.sub(r'#(\w+)', lambda x: '[' + re.sub(r'(?!^)(?=[A-Z])|(?<=\D)(?=\d)|(?<=\d)(?=\D)', " ", x.group(1)+"]"), x) )
            return data
        elif method == "concat":
            data[target_column] = data[target_column].apply(lambda x: re.sub(r'#(\w+)', lambda x: '[' + re.sub(r'(?!^)(?=[A-Z])|(?<=\D)(?=\d)|(?<=\d)(?=\D)', "", x.group(1)+"]"), x) )
            return data
        else:
            raise ValueError("Wrong method")

    def strip(self,data, dropna_cols: str):
        """Manage text columns with strip and splitting the text"""
        serie = data[dropna_cols].apply(lambda x: x.strip() if isinstance(x, str) else x)
        data[dropna_cols] = serie
        return data
    # START : MOVE TO PREPROC ---------------------------------------------------------

    # data[dropna_cols].apply(lambda x: x.split(split_sep) if isinstance(x, str) else x)
    # data_strip_n_split = serie.apply(lambda x: [i.replace(",","") for i in x] if isinstance(x, list) else x)

    # END : MOVE TO PREPROC -----------------------------------------------------------

    def all_in_one(self, data : pd.DataFrame(), text_col: str, selected_cols:list = None, method:str="concat"):
        """ Does a sequence of the above functions :
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

        example : clean.all_in_one(df.data,dropna_cols="text",split_cols="text",selected_cols=["text", "sexist_binary"])"""
        data_dups = self.drop_duplicates_from_one_col(data, text_col)
        data_dropna = self.drop_na(data_dups,selected_cols)
        data_url = self.urls_remover(data_dropna,text_col)
        data_usr = self.username_remover(data_url,text_col)
        data_emoji = self.emoji_replacer(data_usr,text_col)
        data_hashtag = self.hashtag_adapter(data_emoji,text_col, method)
        data_strip = self.strip(data_hashtag,text_col)
        print("✅ All in One is done")
        return data_strip
