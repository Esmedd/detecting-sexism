import pandas as pd
from app.packages.utils import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize

@simple_time_and_memory_tracker
def tokenize_words(data:pd.DataFrame, text_col:str): #tokenizes text to words
    df = data.copy()
    df[text_col] = df[text_col].apply(lambda x: word_tokenize(x))
    return df

@simple_time_and_memory_tracker
def tokenize_sentences(data:pd.DataFrame, text_col:str): #tokenizes text to sentences
    df = data.copy()
    df[text_col] = df[text_col].apply(lambda x: sent_tokenize(x))
    return df

@simple_time_and_memory_tracker
def remove_stop_words(data:pd.DataFrame, text_col:str): # remove stop words (english)
    df = data.copy()
    stop_words = set(stopwords.words('english')) # Make stopword list
    df[text_col] = df[text_col].apply(lambda x: [word for word in x if not word in stop_words])
    return df

@simple_time_and_memory_tracker
def lemmatizer(data:pd.DataFrame, text_col:str): # lemmatize text
    df = data.copy()
    lem = WordNetLemmatizer()
    # verb_lemmatized = [
    #     WordNetLemmatizer().lemmatize(word, pos='v')
    #     for word in tokens]
    # noun_lemmatized = [WordNetLemmatizer().lemmatize(word, pos='n')
    #                    for word in verb_lemmatized]
    # adj_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = 'a')
    #                   for word in noun_lemmatized]

    df[text_col] = df[text_col].apply(lambda x: [lem.lemmatize(word, pos="v")for word in x])
    return df


##############################################################################################################
###################### MAIN PREPROCESSING FUNCTION ##########################################################
###############################################################################################################

##### Preprocessing function. Optional parameters:
# punctuation = True (removes punctuation),
# stop_words = True (removes stop_words),
# sent_tokenize = False (words are tokenized, not sentences)
# lemmatize = False (sentences are not lemmatized)
# EXMPAMPLE # data['text_clean'] = data['text'].apply(lambda x: preprocessing_ML.preprocessing(x, stop_words=False))

def preprocessing(data:pd.DataFrame, text_col:str, stop_words=True, sent_tokenize=False, lemmatize=False):

    # if not isinstance(sentence, str):   ### Check if the input is not a string
    #     return ""

    # sentence = lower_case(sentence)     ### lower case
    # sentence = remove_accents(sentence) ### remove accents

    # if punctuation==True:               ### remove punctuation (if punctuation == True)
    #     for punctuation in string.punctuation:
    #         sentence = sentence.replace(punctuation, ' ')

    if sent_tokenize==True:             ### tokenize rows to words or sentences
        tokenized = tokenize_sentences(data,text_col)
    else:
        tokenized = tokenize_words(data,text_col)
        # tokenized = [word for word in tokenized if word.isalpha()] # Remove numbers

    if stop_words==True:                ### remove stop words (if stop_words==True)
        tokenized = remove_stop_words(tokenized, text_col)

    if lemmatize==True:                 ### lemmatize sentences
        tokenized = lemmatizer(tokenized, text_col)

    return tokenized
