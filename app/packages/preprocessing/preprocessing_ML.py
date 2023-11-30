import pandas as pd
from app.packages.utils import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import pad_sequences
import numpy as np
from gensim.models import Word2Vec

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

    df[text_col] = df[text_col].apply(lambda x: " ".join([lem.lemmatize(word, pos="v")for word in x]))
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

    print("✅ Preprocessing is done")
    return tokenized



##############################   UPDATING PREPROCESSING  ################################################



model_names = ["conv1d", "GRU", "LSTM", "multinomial", "BERT"]


def test_test(model_name:str):

    if model_name == "LSTM":
        LSTM_preprocess()
    if model_name == "multinomial":
        Multinomial_preprocess()
    if model_name == "GRU":
        GRU_preprocess()
    if model_name == "conv1d":
        Conv1d_preprocess()
    if model_name == "BERT":
        BERT_preprocess()

    def LSTM_preprocess():

        def tokenize(df_column, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
            """ tokenize a column
            df_column (pandas.Series): The DataFrame column containing text.
            filters: The set of characters to filter out. defaults to remove punctuation.
            lower: Whether to convert the text to lowercase. defaults to true.
            split: The split to use for splitting the text. Defaults to ' ' (space).

            Returns:
            list of lists: list where each element is a list of tokens from a row in the input column.
            """
            return df_column.astype(str).apply(lambda x: text_to_word_sequence(x, filters=filters, lower=lower, split=split)).tolist()

        def w2v_train_and_embed(X_train, vector_size, window, dtype='float32', padding='post'):

            """Returns a list of embedded, padded sentences (each sentence is a matrix).
            Takes vectorizing arguments "vector_size" and "window"
            Takes padding arguments dtype & padding
            """
            word2vec = Word2Vec(sentences=X_train, vector_size=vector_size, window=window)

            def embed_sentence(wv, sentence):
                return np.array([wv[i] for i in sentence if i in wv])
            wv = word2vec.wv
            embedded = [embed_sentence(wv, s) for s in X_train]
            return pad_sequences(embedded, dtype=dtype, padding=padding), word2vec

        def w2v_embed(X_test, word2vec_model, max_length, dtype='float32', padding='post'):
            """
            Embed sentences using a trained Word2Vec model.
            """
            def embed_sentence(wv, sentence):
                return np.array([wv[i] for i in sentence if i in wv])
            # Embedding the sentences
            wv = word2vec_model.wv
            embedded_X = [embed_sentence(wv, s) for s in X_test]
            return pad_sequences(embedded_X, maxlen=max_length, dtype=dtype, padding=padding)



    def Multinomial_preprocess():
        pass

    def GRU_preprocess():
        pass

    def Conv1d_preprocess():
        pass

    def BERT_preprocess():
        pass
