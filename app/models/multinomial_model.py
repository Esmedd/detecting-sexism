import numpy as np
import pandas as pd
import sys
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import set_config; set_config("diagram")
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from app.packages.preprocessing.cleaning import *
from app.packages.preprocessing.preprocessing_ML import *
from params import *


@simple_time_and_memory_tracker
# Function that Grid search on pipeline
def grid_searching_NB(pipeNaiveBayes,parametersNB):
    return GridSearchCV(
        pipeNaiveBayes, param_grid=parametersNB, n_jobs=-1,
        verbose=1, scoring="accuracy", cv=5
    )


# grid of parameters by default
parametersNB = {
    #'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
    #'tfidfvectorizer__max_df': [0.6, 0.7, 0.8],
    #'tfidfvectorizer__min_df': [1, 2, 3],
    #'tfidfvectorizer__stop_words': [None, 'english'],
    'tfidfvectorizer__max_features': [500, 1000, 2000],
    'multinomialnb__alpha': [0.1, 0.5, 1.0],
}
@simple_time_and_memory_tracker
# function performing the preprocessing on cleaned data, defining a pipeline and fitting on a grid search
def multinomial_baseline_grid(X_cleaned, y_encoded, model=MultinomialNB(), stop_words=None, ngram_range=(1,2)):

    #X_preproc = X_cleaned.apply(lambda x: preprocessing(x, punctuation=True, stop_words=True, sent_tokenize=False, lemmatize=False))

    pipeNaiveBayes = make_pipeline(
        TfidfVectorizer(min_df=1, stop_words=stop_words, ngram_range=ngram_range),
        MultinomialNB()
    )
    search=grid_searching_NB(pipeNaiveBayes,parametersNB)
    search.fit(X_cleaned,y_encoded)
    best_model = search.best_estimator_

    return best_model


def cv_multinomial_baseline_model(X_cleaned, y_encoded, model=MultinomialNB(), ngram_range=(2,2)):

    vectorizer = CountVectorizer(ngram_range = ngram_range)
    model = model

    print(X_cleaned.head())

    X_bow = vectorizer.fit_transform(X_cleaned)

    cv_nb = cross_validate(
        model,
        X_bow,
        y_encoded,
        scoring = ["accuracy", "recall"]
    )
    print(f"recall is {round(cv_nb['test_recall'].mean(),2)}")
    return cv_nb
