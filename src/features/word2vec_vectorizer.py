import pandas as pd
import numpy as np
 
from gensim.models import Word2Vec
 
 
def document_vector(model, text):
 
    words = str(text).split()
 
    # keep only words the model actually learned
    vectors = [
        model.wv[w]
        for w in words
        if w in model.wv
    ]
 
    # empty or all-unknown review -> zero vector
    if not vectors:
        return np.zeros(model.vector_size)
 
    return np.mean(vectors, axis=0)
 
 
def train_word2vec():
 
    df = pd.read_csv(
        "data/processed/cleaned_reviews.csv"
    )
 
    df["clean_review"] = df["clean_review"].fillna("")
 
    # Word2Vec learns from lists of tokens,
    # so we split each cleaned review into words.
    sentences = [
        review.split()
        for review in df["clean_review"]
    ]
 
    # vector_size = length of each word vector
    # window      = neighbouring words used as context
    # min_count   = ignore words seen fewer than twice
    # workers     = CPU cores used to train
    w2v = Word2Vec(
        sentences,
        vector_size=200,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )
 
    w2v.save("models/word2vec.model")
 
    print(
        "Word2Vec Trained:",
        len(w2v.wv.index_to_key),
        "words"
    )
 
 
if __name__ == "__main__":
    train_word2vec()
