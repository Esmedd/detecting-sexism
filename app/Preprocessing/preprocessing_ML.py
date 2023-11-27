import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize

import string
import unidecode

def tokenize_words(text): #tokenizes text to words
    return word_tokenize(text)

def tokenize_sentences(text): #tokenizes text to sentences
    return sent_tokenize(text)

def remove_punctuation(text): #remove punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation,'')
    return text

def lower_case(text):   #turn into lower case
    return text.lower()

def remove_accents(text):
    return unidecode.unidecode(text) # remove accents

def remove_numbers(text):
        words_only = [word for word in text if word.isalpha()] # Remove numbers

def remove_stop_words(tokens): # remove stop words (english)
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    return without_stopwords

def lemmatize(tokens): # lemmatize text
    verb_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos='v')
        for word in tokens]
    # noun_lemmatized = [WordNetLemmatizer().lemmatize(word, pos='n')
    #                    for word in verb_lemmatized]
    # adj_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = 'a')
    #                   for word in noun_lemmatized]
    return verb_lemmatized

def preprocessing(sentence, punctuation=True, stop_words=True, sent_tokenize=False, lemmatize=False):

    ### lower case
    sentence = lower_case(sentence)

    ### remove accents
    sentence = remove_accents(sentence)

    #### remove punctuation (if punctuation == True)
    if punctuation==True:
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, ' ')

    ##### tokenize rows to words or sentences
    if sent_tokenize==True:
        tokenized = tokenize_sentences(sentence)
    else:
        tokenized = tokenize_words(sentence)
        tokenized = [word for word in tokenized if word.isalpha()] # Remove numbers

    ##### remove stop words (if stop_words==True)
    if stop_words==True:
        tokenized = remove_stop_words(tokenized)

    #### lemmatize sentences
    if lemmatize==True:
        tokenized = lemmatize(tokenized)

    return " ".join(tokenized)




#df['clean_text'] = df['title_comment'].apply(clean)
