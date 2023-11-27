import pandas as pd

class cleaning:
    def __init__(self, path):
        """Create self.data from a path to a csv"""
        self.data = pd.read_csv(path)

    def drop_duplicates_from_one_col(self, data, duplicate_cols: str):
        """Drop duplicates based on One column
        example """
        print(f"Amount of duplicates on : {data[duplicate_cols].duplicated().value_counts()}")
        df = data[data[duplicate_cols].duplicated() == False]
        print("✅ Drop duplicates is done")
        return df

    def drop_na(self,data, selected_columns: list = None):
        """Returns Df with selected columns and drop na"""
        if selected_columns == None:
            data = data
        else:
            data = data[selected_columns]
        print(f"Amount of NaN on : {data.isna().sum()}")
        data = data.dropna()
        print("✅ Drop NA is done")
        return data

    def strip_n_split(self,data, dropna_cols: str, split_sep:str=" "):
        """Manage text columns with strip and splitting the text"""
        serie = data[dropna_cols].apply(lambda x: x.strip().split(split_sep) if isinstance(x, str) else x)
        data_strip_n_split = serie.apply(lambda x: [i.replace(",","") for i in x] if isinstance(x, list) else x)
        data[dropna_cols] = data_strip_n_split
        print("✅ Strip & Split is done")
        return data

    def all_in_one(self, data : pd.DataFrame(), duplicate_cols: str, split_cols: str, selected_cols:list = None, split_sep:str=" "):
        """ Does a sequence of the above functions :
        1st : drop_duplicates_from_one_col()
        2nd : drop_na()
        3rd : strip_n_split()

        'split_sep' argument is used to define the separator to split the text. By default it used spaces, but can use a "." or ","

        example : clean.all_in_one(df.data,dropna_cols="text",split_cols="text",selected_cols=["text", "sexist_binary"])"""
        data_dups = self.drop_duplicates_from_one_col(data, duplicate_cols)
        data_dropna = self.drop_na(data_dups,selected_cols)
        data_split = self.strip_n_split(data_dropna, split_cols, split_sep)
        print("✅ All in One is done")
        print(data_dropna.head())
        return data_split
