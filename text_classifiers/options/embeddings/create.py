import time 

import pandas as pd
import gensim

from text_classifiers.helpers import PreprocessedCorpus 


def run():
    print("Reading in data...")
    articles1 = pd.read_csv("data/raw/articles1.csv")
    articles2 = pd.read_csv("data/raw/articles2.csv")
    articles3 = pd.read_csv("data/raw/articles3.csv")
    all_articles = pd.concat([articles1, articles2, articles3])

    all_content = all_articles["content"]

    print("Preprocessing corpus...")
    start_time = time.time()
    sentences = PreprocessedCorpus(all_content)
    print(f"--- Took {round(time.time() - start_time, 2)} seconds ---")

    print("Training the embeddings model...")
    start_time = time.time()
    model = gensim.models.Word2Vec(sentences=sentences, vector_size=300)
    print(f"--- Took {round(time.time() - start_time, 2)} seconds ---")

    embed_path = "data/trained_models/embeddings.wordvec"
    print("Saving the model to ", embed_path)
    model.wv.save(embed_path)
