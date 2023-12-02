import pandas as pd
from app.packages.utils import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.utils import pad_sequences
import numpy as np
from gensim.models import Word2Vec, KeyedVectors


model_names = ["conv1d", "GRU", "LSTM", "multinomial", "BERT"]


def preproc_test(X_train:pd.DataFrame,X_test:pd.DataFrame, model_name:str, params : dict=None):
    def LSTM_preprocess(X_train, X_test):
        max_length = params["max_length"]
        vector_size= params["vector_size"]
        window= params["window"]

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
            word2vec.save(f"./training_outputs/W2V/{params['max_length']}_{vector_size}_{window}_{padding}_W2V.wordvectors")
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

        X_train_token = tokenize(X_train.text)
        X_test_token = tokenize(X_test.text)
        X_train_truncated = [sentence[:max_length] for sentence in X_train_token]
        X_train_padded, trained_word2vec_model = w2v_train_and_embed(X_train_truncated, vector_size=vector_size, window=window)
        X_test_padded = w2v_embed(X_test_token, trained_word2vec_model, max_length=len(X_train_padded[0]))
        print(f"Shape of X_train_padded in preproc: {X_train_padded.shape}")
        return X_train_padded, X_test_padded

    def Embed_LSTM_preproc(X_train:pd.DataFrame, X_test:pd.DataFrame, params:dict):
        tk = Tokenizer()
        tk.fit_on_texts(X_train.text)
        vocab_size = len(tk.word_index)
        print(f'There are {vocab_size} different words in your corpus')
        X_train_token = tk.texts_to_sequences(X_train.text)
        X_test_token = tk.texts_to_sequences(X_test.text)
        X_train_truncated = [sentence[:params["max_length"]] for sentence in X_train_token]
        X_test_truncated = [sentence[:params["max_length"]] for sentence in X_test_token]
        X_train_pad = pad_sequences(X_train_truncated, maxlen=params["max_length"])
        X_test_pad = pad_sequences(X_test_truncated, maxlen=params["max_length"])
        return X_train_pad, X_test_pad
    def Multinomial_preprocess(X_train, X_test):

        @simple_time_and_memory_tracker
        def tokenize_words(X_train, X_test): #tokenizes text to words
            X_train = X_train.apply(lambda x: word_tokenize(x))
            X_test = X_test.apply(lambda x: word_tokenize(x))
            return X_train, X_test

        @simple_time_and_memory_tracker
        def tokenize_sentences(X_train, X_test): #tokenizes text to sentences
            X_train = X_train.apply(lambda x: sent_tokenize(x))
            X_test = X_test.apply(lambda x: sent_tokenize(x))
            return X_train, X_test

        @simple_time_and_memory_tracker
        def remove_stop_words(X_train, X_test): # remove stop words (english)
            stop_words = set(stopwords.words('english')) # Make stopword list
            X_train = X_train.apply(lambda x: [word for word in x if not word in stop_words])
            X_test = X_test.apply(lambda x: [word for word in x if not word in stop_words])
            return X_train, X_test

        @simple_time_and_memory_tracker
        def lemmatizer(X_train, X_test): # lemmatize text
            lem = WordNetLemmatizer()
            X_train = X_train.apply(lambda x: " ".join([lem.lemmatize(word, pos="v")for word in x]))
            X_test = X_test.apply(lambda x: " ".join([lem.lemmatize(word, pos="v")for word in x]))
            return X_train, X_test

        def preprocessing(X_train, X_test, stop_words=True, sent_tokenize=False, lemmatize=False):
            if sent_tokenize==True:             ### tokenize rows to words or sentences
                X_train_token, X_test_token = tokenize_sentences(X_train,X_test)
            else:
                X_train_token, X_test_token = tokenize_words(X_train,X_test)
            if stop_words==True:                ### remove stop words (if stop_words==True)
                X_train_token, X_test_token = remove_stop_words(X_train_token, X_test_token)
            if lemmatize==True:                 ### lemmatize sentences
                X_train_token, X_test_token = lemmatizer(X_train_token, X_test_token)
            print("✅ Preprocessing is done")
            return X_train_token, X_test_token
        return preprocessing(X_train, X_test, stop_words=False, sent_tokenize=False, lemmatize=True)
    def GRU_preprocess(X_train, X_test):
        return LSTM_preprocess(X_train, X_test)

    def Conv1d_preprocess(X_train, X_test):
        max_length = params["max_length"]
        @simple_time_and_memory_tracker
        def preprocessing_cld(X : pd.DataFrame,maxlen=max_length):
            """ Preprocess X data for a Conv1D model
            Takes a single column df, X (as a list), as input. Returns the preprocessed X,
            the maxlen and the vocab size as output for use in initialize model function
            """
            X_l = X.values.tolist()
            X_word = [text_to_word_sequence("".join(x)) for x in X_l]

            tk = Tokenizer()
            tk.fit_on_texts(X_word)
            X_token = tk.texts_to_sequences(X_word)
            vocab_size = len(tk.word_index)

            X_token_pad = pad_sequences(X_token, dtype=params["dtype"], padding=params["padding"], maxlen=params["max_length"])
            return X_token_pad, vocab_size, max_length
        X_train_pad, train_vocab_size, train_max_length = preprocessing_cld(X_train)
        X_test_pad, test_vocab_size, test_max_length = preprocessing_cld(X_test)
        train = []
        test = []
        train.append([X_train_pad, train_vocab_size, train_max_length])
        test.append([X_test_pad, test_vocab_size, test_max_length])
        return  train, test

    def BERT_preprocess():
        pass

    if model_name == "LSTM":
        if params["embed"] == True:
            return Embed_LSTM_preproc(X_train, X_test, params)
        else:
            return LSTM_preprocess(X_train, X_test)
    if model_name == "multinomial":
        return Multinomial_preprocess(X_train, X_test)
    if model_name == "GRU":
        return GRU_preprocess(X_train, X_test)
    if model_name == "conv1d":
        return Conv1d_preprocess(X_train, X_test)

    if model_name == "BERT":
        BERT_preprocess()

def preproc_pred(X_pred:pd.DataFrame,model_name:str, params : dict=None):
    def LSTM_preprocess(X_pred, ):
        max_length = params["max_length"]
        vector_size= params["vector_size"]
        window= params["window"]

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


        def w2v_embed(X_pred, max_length, dtype='float32', padding='post'):
            """
            Embed sentences using a trained Word2Vec model.
            """
            word2vec_model = KeyedVectors.load(f"./training_outputs/W2V/{params['max_length']}_{params['vector_size']}_{params['window']}_{params['padding']}_W2V.wordvectors", mmap='r')
            def embed_sentence(wv, sentence):
                return np.array([wv[i] for i in sentence if i in wv])
            # Embedding the sentences
            wv = word2vec_model.wv
            embedded_X = [embed_sentence(wv, s) for s in X_pred]
            return pad_sequences(embedded_X, maxlen=max_length, dtype=dtype, padding=params['padding'])

        X_pred_token = tokenize(X_pred.text)
        X_pred_truncated = [sentence[:max_length] for sentence in X_pred_token]
        X_pred_padded = w2v_embed(X_pred_truncated, max_length=params["max_length"], dtype=params["dtype"],padding=params['padding'])
        print(f"Shape of X_pred_padded in preproc: {X_pred_padded.shape}")
        return X_pred_padded

    def Embed_LSTM_preproc(X_pred:pd.DataFrame, params:dict):
        tk = Tokenizer()
        tk.fit_on_texts(X_pred.text)
        vocab_size = len(tk.word_index)
        print(f'There are {vocab_size} different words in your corpus')
        X_pred_token = tk.texts_to_sequences(X_pred.text)
        X_pred_truncated = [sentence[:params["max_length"]] for sentence in X_pred_token]
        X_pred_pad = pad_sequences(X_pred_truncated, maxlen=params["max_length"])
        return X_pred_pad
    def Multinomial_preprocess(X_pred, ):

        @simple_time_and_memory_tracker
        def tokenize_words(X_pred): #tokenizes text to words
            X_pred = X_pred.apply(lambda x: word_tokenize(x))
            return X_pred

        @simple_time_and_memory_tracker
        def tokenize_sentences(X_pred): #tokenizes text to sentences
            X_pred = X_pred.apply(lambda x: sent_tokenize(x))
            return X_pred,

        @simple_time_and_memory_tracker
        def remove_stop_words(X_pred): # remove stop words (english)
            stop_words = set(stopwords.words('english')) # Make stopword list
            X_pred = X_pred.apply(lambda x: [word for word in x if not word in stop_words])
            return X_pred,

        @simple_time_and_memory_tracker
        def lemmatizer(X_pred): # lemmatize text
            lem = WordNetLemmatizer()
            X_pred = X_pred.apply(lambda x: " ".join([lem.lemmatize(word, pos="v")for word in x]))
            return X_pred

        def preprocessing(X_pred, stop_words=True, sent_tokenize=False, lemmatize=False):
            if sent_tokenize==True:             ### tokenize rows to words or sentences
                X_pred_token = tokenize_sentences(X_pred)
            else:
                X_pred_token = tokenize_words(X_pred)
            if stop_words==True:                ### remove stop words (if stop_words==True)
                X_pred_token = remove_stop_words(X_pred_token)
            if lemmatize==True:                 ### lemmatize sentences
                X_pred_token = lemmatizer(X_pred_token)
            print("✅ Preprocessing is done")
            return X_pred_token
        return preprocessing(X_pred,stop_words=False, sent_tokenize=False, lemmatize=True)
    def GRU_preprocess(X_pred, ):
        return LSTM_preprocess(X_pred)

    def Conv1d_preprocess(X_pred):
        max_length = params["max_length"]
        @simple_time_and_memory_tracker
        def preprocessing_cld(X : pd.DataFrame,maxlen=max_length):
            """ Preprocess X data for a Conv1D model
            Takes a single column df, X (as a list), as input. Returns the preprocessed X,
            the maxlen and the vocab size as output for use in initialize model function
            """
            X_l = X.values.tolist()
            X_word = [text_to_word_sequence("".join(x)) for x in X_l]

            tk = Tokenizer()
            tk.fit_on_texts(X_word)
            X_token = tk.texts_to_sequences(X_word)
            vocab_size = len(tk.word_index)

            X_token_pad = pad_sequences(X_token, dtype=params["dtype"], padding=params["padding"], maxlen=params["max_length"])
            return X_token_pad, vocab_size, max_length
        X_pred_pad, train_vocab_size, train_max_length = preprocessing_cld(X_pred)
        train = []
        train.append([X_pred_pad, train_vocab_size, train_max_length])
        return train

    def BERT_preprocess():
        pass

    if model_name == "LSTM":
        if params["embed"] == True:
            return Embed_LSTM_preproc(X_pred, params)
        else:
            return LSTM_preprocess(X_pred)
    if model_name == "multinomial":
        return Multinomial_preprocess(X_pred)
    if model_name == "GRU":
        return GRU_preprocess(X_pred)
    if model_name == "conv1d":
        return Conv1d_preprocess(X_pred)

    if model_name == "BERT":
        BERT_preprocess()
