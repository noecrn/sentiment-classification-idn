import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def advanced_clean_text(text):
    """Nettoyage avancé du texte"""
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Supprimer les mentions et hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Garder seulement les lettres et espaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début/fin
    text = text.strip()
    
    return text



def vectorize_texts_advanced(texts, max_features=10000):
    """Vectorisation avancée avec TF-IDF optimisé"""
    cleaned = texts.apply(advanced_clean_text)
    
    # TF-IDF avec paramètres optimisés
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),  # Unigrams, bigrams et trigrams
        min_df=3,           # Ignorer les mots qui apparaissent moins de 3 fois
        max_df=0.7,         # Ignorer les mots qui apparaissent dans plus de 70% des documents
        stop_words='english',
        sublinear_tf=True,  # Utiliser la transformation logarithmique
        use_idf=True,
        smooth_idf=True
    )
    
    X = vectorizer.fit_transform(cleaned)
    return X, vectorizer