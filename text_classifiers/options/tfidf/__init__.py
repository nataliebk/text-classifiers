"""Module to train the tf-idf model."""

import logging

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score


def train_with_tfidf(train_data, text_col, label_col, max_features=2500, test_size=0.10):
    """Trains a simple text classifyer."""
    # pylint: disable=invalid-name

    # splitting data into train and validation sets
    # with a specified random state for reproducibility
    X_train, X_val, y_train, y_val = train_test_split(
        train_data[text_col], train_data[label_col], test_size=test_size, random_state=2020
    )

    # training a classifier using each word's tf-idf values as features
    classifier = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                        max_features=max_features,
                        stop_words=stopwords.words("dutch")
                )
            ),
            ("cls", LogisticRegression(
                solver="lbfgs", multi_class="auto", max_iter=500
                )
            )
        ]
    )

    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, predicted)
    prec = precision_score(y_val, predicted, average="weighted")

    logging.info("Accuracy on validation set: %s", accuracy)
    logging.info("Precision on validation set: %s", prec)

    return classifier
