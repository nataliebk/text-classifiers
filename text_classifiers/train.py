import os
import pickle
import logging
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score

from text_classifiers.options.tfidf import train_with_tfidf
from text_classifiers.helpers import setup_logging


def read_data(file_path):
    if file_path.endswith(".json"):
        return pd.read_json(file_path, lines=True)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        logging.error("File format should be .json or .csv")
        return None


def get_train_test(dtf, directory, filename="data", test_size=0.15):
    """
    Saves train/test datasets to specified directory.
    Returns the datasets.
    """
    train, test = train_test_split(dtf, test_size=test_size, random_state=2020)
    train.to_csv(f"{directory}/{filename}_train.csv", index=False)
    test.to_csv(f"{directory}/{filename}_test.csv", index=False)
    logging.info("Train and test sets saved to %s", os.path.abspath(directory))
    return train, test


def evaluate_model(test_data, text_col, label_col, classifier):
    """Evaluates the model performance."""
    predicted = classifier.predict(test_data[text_col])
    accuracy = accuracy_score(test_data[label_col], predicted)
    precision = precision_score(test_data[label_col], predicted, average="weighted")

    logging.info("Accuracy on test set: %s", accuracy)
    logging.info("Precision on test set: %s", precision)


def select_classes(dtf, text_col="headline", label_col="category", top_n=6):
    """Select top N classes for training"""
    selected_categories = list(dtf[label_col].value_counts().keys()[:top_n])
    logging.info("Selected classes for training: %s", selected_categories)
    selected_df = dtf[dtf[label_col].isin(selected_categories)][[text_col, label_col]]
    return selected_df


def run():
    """Executes model training."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--verbose", "-v", action="count", default=1)
    arg_parser.add_argument("--model-option", default="tfidf",
                            help="Feature / model type to train."
                            "Available options are tfidf, embeddings, lstm or all.")
    arg_parser.add_argument("--raw-data-path", type=str,
                            default="data/raw/News_Category_Dataset_v2.json",
                            help="Path to the raw data file. Should be .json or .csv file.")
    arg_parser.add_argument("--train-test-directory", type=str,
                            default="data/processed",
                            help="Path to the raw data file.")
    arg_parser.add_argument("--model-directory", type=str,
                            default="data/trained_models",
                            help="Path to directory where the trained model should be saved to.")
    args = arg_parser.parse_args()

    setup_logging(args.verbose)

    logging.info("Reading in data...")
    raw_dtf = read_data(args.raw_data_path)

    if raw_dtf is None:
        return 

    logging.info("Splitting into train/test...")
    selected_dtf = select_classes(raw_dtf)
    train, test = get_train_test(selected_dtf, args.train_test_directory)

    logging.info("Training %s model...", args.model_option)
    classifier = train_with_tfidf(train,
                                  text_col="headline",
                                  label_col="category")

    # Evaluating the model on the test set
    evaluate_model(test,
                   text_col="headline",
                   label_col="category",
                   classifier=classifier)

    model_path = args.model_directory + "/" + args.model_option + ".pkl"
    logging.info("Saving the model to %s...", model_path)
    with open(model_path, "wb") as model_file:
        pickle.dump(classifier, model_file)
  