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
from sklearn.metrics import accuracy_score, recall_score

@simple_time_and_memory_tracker

# Function that Grid search on pipeline
def grid_searching_NB(pipeNaiveBayes,parametersNB):
    return GridSearchCV(
        pipeNaiveBayes, parametersNB, n_jobs=-1,
        verbose=1, scoring="accuracy", cv=5
    )

# function defining a pipeline and fitting on a grid search on pre-processed data

def multinomial_baseline_grid_ctVec(X_train_preproc, y_train,X_test_preproc,y_test):
    """     Function that searches for the best parameters on the baseline multinomial model.
    t defines a pipe line using a CountVectorizer()
    Args:
    - X_train_preproc: Preprocessed training data.
    - y_train: Training labels.
    - X_test_preproc: Preprocessed test data.
    - y_test: Test labels.

    Returns:
    - best_params: Best parameters found during grid search.
    - best_model: Best model from the grid search.
    - accuracy: Accuracy on the test set.
    - recall: Recall on the test set.
    """

    # grid search parameters
    parametersNB = {'countvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
                   'multinomialnb__alpha': [0.1, 0.5, 1.0]}

    # Define the pipeline
    pipeNaiveBayes = make_pipeline(
        CountVectorizer(),
        MultinomialNB()
    )

    # grid search function
    search = grid_searching_NB(pipeNaiveBayes, parametersNB)
    search.fit(X_train_preproc, y_train)

    # Get the best model from the grid search
    best_model = search.best_estimator_

     # Get the best parameters
    best_params = search.best_params_
    print("Best Parameters:", best_params)

    # Make predictions
    y_pred = best_model.predict(X_test_preproc)

    # Calculate accuracy and recall
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Best Model Metrics - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

    return best_params, best_model, accuracy, recall



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



def multinomial_baseline_grid_tfid(X_train_preproc, y_train, X_test_preproc, y_test):
    """Function that searches for the best parameters on the baseline multinomial model.
    it defines a pipe line using a TfidfVectorizer()

    Args:
    - X_train_preproc: Preprocessed training data.
    - y_train: Training labels.
    - X_test_preproc: Preprocessed test data.
    - y_test: Test labels.

    Returns:
    - best_params: Best parameters found during grid search.
    - best_model: Best model from the grid search.
    - accuracy: Accuracy on the test set.
    - recall: Recall on the test set.
    """
    # grid search parameters
    parametersNB = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
                    'multinomialnb__alpha': [0.1, 0.5, 1.0]}

    # Define the pipeline with TfidfVectorizer
    pipeNaiveBayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    # grid search function
    search = grid_searching_NB(pipeNaiveBayes, parametersNB)
    search.fit(X_train_preproc, y_train)

    # Get the best model from the grid search
    best_model = search.best_estimator_

    # Get the best parameters
    best_params = search.best_params_
    print("Best Parameters:", best_params)

    # Make predictions
    y_pred = best_model.predict(X_test_preproc)

    # Calculate accuracy and recall
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Best Model Metrics - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

    return best_params, best_model, accuracy, recall



##### PRELIMINARY RESULTS:
# Best Parameters: {'countvectorizer__ngram_range': (1, 1), 'multinomialnb__alpha': 1.0}
# Best Model Metrics - Accuracy: 0.7417, Recall: 0.7458
