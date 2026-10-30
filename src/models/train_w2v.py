import pandas as pd
import numpy as np
import joblib
 
from gensim.models import Word2Vec
 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from src.features.word2vec_vectorizer import document_vector
 
df = pd.read_csv( "data/processed/cleaned_reviews.csv")
 
df["clean_review"] = df["clean_review"].fillna("")
 
w2v = Word2Vec.load("models/word2vec.model")
 
# Turn every review into one averaged 200-d vector
X = np.vstack([
    document_vector(w2v, text)
    for text in df["clean_review"]
])
 
y = df["sentiment"]
 
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)
 
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
 
model.fit(X_train, y_train)
 
joblib.dump(
    model,
    "models/sentiment_model_w2v.pkl"
)
 
print("Word2Vec Model Trained")
