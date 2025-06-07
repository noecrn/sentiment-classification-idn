import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def vectorize_texts(texts, max_features=3000):
    cleaned = texts.apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(cleaned)
    return X, vectorizer