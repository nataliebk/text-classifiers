import logging
import re
import unicodedata
from collections import Counter

# from html2text import html2text
from nltk.corpus import stopwords
from nltk import sent_tokenize
import nltk

# how to make it better????
nltk.download("stopwords")
nltk.download("punkt")


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


def normalize_text(text, replace_ascii=True, render_html=False):
    """
    Substitute non-ASCII letters/ special characters in text (string)
          with respective alternatives, and remove the following:

        - html tags/ code (if render_html=True)
        - urls (i.e. any pattern starting with http and www)
        - non-printable characters (e.g. escape sequences \n \t )
        - round brackets (to avoid splitting e.g. medewerk(st)er)
        - long strings (to remove e.g. image rendered as text)

    Note that rendering html can significantly slow down preprocessing,
        thus the default is False

    Returns a string with normalised text
    """
    # remove leading/ trailing spaces
    text = text.strip()

    # render html to plain text
    # if render_html:
    #     text = html2text(text)

    # replace non-ASCII characters
    if replace_ascii:
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode()

    # substitute patterns in text with specified values
    subs = {
        r"\n|\t|\r|\v|\f" : r" ",
        r"http\S+|www\S+" : r" ",
        r"\(|\)" : r""
    }
    for pattern in subs:
        text = re.sub(pattern, subs[pattern], text)

    # remove non-printable
    text = "".join(filter(str.isprintable, text))

    # remove long strings
    tokens = [token for token in text.split() if len(token.replace("_", "")) <= 45]
    text = " ".join(tokens)

    return text


def remove_punctuation(text):
    """
    Removes punctuation from the provided string by replacing it with space
    """
    text = re.sub("[^\w\s]|_", " ", text)

    return text


def remove_stopwords(tokens_list, language="english"):
    """
    Remove stopwords for the respective language from a list with tokens,
        using stopwords from nltk module
    """

    stop_words = set(stopwords.words(language))
    result_list = [token for token in tokens_list if token not in stop_words]

    return result_list


def tokenize_text(text, rm_nonalpha=True, rm_stopwords=True, language="english"):
    """
    Remove punctuation and split text into a list of tokens.

    Args:
        - text: single string
        - rm_nonalpha (True/ Flase, default=True): whether to remove non alphabetic tokens
        - rm_stopwords (True/ False, default=True): whether to remove stopwords for the respective language
        - language (default="dutch", supports all languages from nltk)

    Returns:
        List with tokens
    """
    # replace punctuation with space
    text = remove_punctuation(text)

    # get a list with tokens
    tokens = text.lower().split()

    # remove any tokens that are too short/long
    tokens = [token for token in tokens if len(token) > 1 and len(token) <= 35]

    # remove non alpha strings
    if rm_nonalpha:
        tokens = [token for token in tokens if token.isalpha()]

    # remove stopwords
    if rm_stopwords:
        tokens = remove_stopwords(tokens, language)

    return tokens


def preprocess_text(text,
                    language="english",
                    render_html=False,
                    rm_nonalpha=True,
                    rm_stopwords=True):
    """
    Normalise and tokenize an individual text document.
    """
    normalized = normalize_text(text, render_html=render_html)
    tokenized = tokenize_text(normalized,
                                language=language,
                                rm_nonalpha=rm_nonalpha,
                                rm_stopwords=rm_stopwords)

    return tokenized


class PreprocessedCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for document in self.corpus:
            for sentence in sent_tokenize(document):
                yield preprocess_text(sentence)