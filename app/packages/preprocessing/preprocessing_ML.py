import pandas as pd
from app.packages.utils import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize


@simple_time_and_memory_tracker
def tokenize_words(text): #tokenizes text to words
    return word_tokenize(text)

@simple_time_and_memory_tracker
def tokenize_sentences(text): #tokenizes text to sentences
    return sent_tokenize(text)

@simple_time_and_memory_tracker
def remove_stop_words(tokens): # remove stop words (english)
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in tokens if not word in stop_words] # Remove Stop Words
    return without_stopwords

@simple_time_and_memory_tracker
def lemmatize(tokens): # lemmatize text
    verb_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos='v')
        for word in tokens]
    # noun_lemmatized = [WordNetLemmatizer().lemmatize(word, pos='n')
    #                    for word in verb_lemmatized]
    # adj_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = 'a')
    #                   for word in noun_lemmatized]
    return verb_lemmatized


##############################################################################################################
###################### MAIN PREPROCESSING FUNCTION ##########################################################
###############################################################################################################

##### Preprocessing function. Optional parameters:
# punctuation = True (removes punctuation),
# stop_words = True (removes stop_words),
# sent_tokenize = False (words are tokenized, not sentences)
# lemmatize = False (sentences are not lemmatized)
# EXMPAMPLE # data['text_clean'] = data['text'].apply(lambda x: preprocessing_ML.preprocessing(x, stop_words=False))

def preprocessing(sentence, punctuation=True, stop_words=True, sent_tokenize=False, lemmatize=False):

    # if not isinstance(sentence, str):   ### Check if the input is not a string
    #     return ""

    # sentence = lower_case(sentence)     ### lower case
    # sentence = remove_accents(sentence) ### remove accents

    # if punctuation==True:               ### remove punctuation (if punctuation == True)
    #     for punctuation in string.punctuation:
    #         sentence = sentence.replace(punctuation, ' ')

    if sent_tokenize==True:             ### tokenize rows to words or sentences
        tokenized = tokenize_sentences(sentence)
    else:
        tokenized = tokenize_words(sentence)
        tokenized = [word for word in tokenized if word.isalpha()] # Remove numbers

    if stop_words==True:                ### remove stop words (if stop_words==True)
        tokenized = remove_stop_words(tokenized)

    if lemmatize==True:                 ### lemmatize sentences
        tokenized = lemmatize(tokenized)

    return " ".join(tokenized)
