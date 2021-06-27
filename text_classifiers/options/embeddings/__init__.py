import logging

import numpy as np 
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score

from text_classifiers.helpers import preprocess_text


def mean_vector(tokens, wv_model):
    """Calculates a mean vector for a given list of tokens.

    Note that all tokens that are not in the provided word embeddings model
    vocabulary are ignored. Thus if none of the passed tokens
    are in vocabulary, the returned vector will contain only 0's.

    Returns a 1-d array, which is the mean vector.
    """
    # initialise mean vector
    mean_vect = np.zeros(wv_model.vector_size)
    # check that input is a list
    if isinstance(tokens, list):
        lowered_tokens = [token.lower() for token in tokens]
        vocab_tokens = [token for token in lowered_tokens if \
                        token in wv_model.key_to_index and token.isalpha()]
        # if not empty
        if vocab_tokens:
            mtrx = wv_model[vocab_tokens]

            if len(mtrx) == 1:
                mean_vect = mtrx[0]

            elif len(mtrx) > 1:
                mean_vect = np.mean(mtrx, axis = 0)
        # if none of the tokens are in vocabulary, return zero vector
        else:
            mean_vect = mean_vect

    else:
        raise ValueError('Input should be a list of tokens, no punctuation')

    return mean_vect


def create_embedding_features(corpus, wv):
    """
    Takes a text corpus and returns a numpy array of mean word vectors.

    Each document in the corpus is split into tokens, and word vectors
    associated with each of those tokens are averaged together to produce
    a mean vector for that document. Word vectors come from the provided 
    pre-trained word embeddings model. 
    """
    def generate_mean_vectors(corpus, wv):
        for text in corpus:
            tokenized = preprocess_text(text)
            yield mean_vector(tokenized, wv)

    return np.stack([vect for vect in generate_mean_vectors(corpus, wv)])


class EmbedFeatures():
    """Class that creates mean vector features using a word embeddings model."""
    def __init__(self, wv_model):
        self.wv_model = wv_model

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return create_embedding_features(X, self.wv_model)


def train_with_embeddings(train_data, text_col, label_col, embed_path=None, test_size=0.10):
    """Trains a text classifier using mean word embeddings as features."""
    if embed_path is not None:
        logging.info("Reading in the embeddings from %s...", embed_path)
        wv_model = KeyedVectors.load(embed_path, mmap='r')
    else:
        logging.info("Reading in word2vec-google-news-300 embeddings model...")
        wv_model = downloader.load("word2vec-google-news-300")

    # splitting data into train and validation sets
    # with a specified random state for reproducibility
    logging.info("Creating train / validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_data[text_col], train_data[label_col], 
        test_size=test_size, random_state=2020
    )

    logging.info("Training the model...")
    classifier = Pipeline(
        [
            ("meanvec", EmbedFeatures(wv_model)),
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