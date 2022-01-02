# Text classifiers

## Overview
A project to quickly train a baseline model to classify text, using any dataset provided a in JSON or CSV format.
The idea behind it is as follows. Sometimes we just want to quickly check whether we have a dataset with reliable
labels, i.e. whether it makes sense at all to spend time on training a model using the data at hand. It happened way too often that a lot of time was spent on text cleaning, only to realise in the end that there is too much noise in the labels to train a model.

This project can create a baseline text classifier with a single command, which can surely save some time.


## Data
Main dataset used in this project as an example is a 
news categories dataset (taken from kaggle)[https://www.kaggle.com/rmisra/news-category-dataset]. It consists of headlines and short descriptions of news articles.

Another example dataset (or data files really) was also taken from (kaggle)[https://www.kaggle.com/snapcrack/all-the-news]

## Quickstart
To train a classifier using example data, first download data from the above links into `data/raw` folder. Then:
```bash
poetry run train-classifier
```
This will train a simple classifier with features created using tf-idf technique.