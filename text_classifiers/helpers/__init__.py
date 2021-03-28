import logging

import nltk
from sklearn.model_selection import train_test_split


def check_stopwords():
    """Check whether nltk stopwords are available."""
    try:
        nltk.corpus.stopwords.words("dutch")
    except LookupError:
        nltk.download("stopwords")


def setup_logging(level):
    """Logging setup."""
    assert level >=0

    if level > 2:
        level = 2

    names = [
        "WARNING",
        "INFO",
        "DEBUG"
    ]

    logging.basicConfig(
        level=names[level], format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
    )
