# src/advanced_preprocess.py

import pandas as pd
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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

def get_text_statistics(text):
    """Extraire des statistiques textuelles simples et rapides"""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Compter les mots positifs/négatifs communs (approche naïve mais rapide)
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect', 
                      'happy', 'enjoyed', 'favorite', 'fantastic', 'recommend', 'top', 'beautiful'}
    negative_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'poor', 'waste', 'boring', 
                      'hate', 'disappointing', 'avoid', 'difficult', 'unfortunately', 'mediocre', 'fails'}
    
    # Compter les occurrences
    pos_count = sum(1 for word in words if word.lower() in positive_words)
    neg_count = sum(1 for word in words if word.lower() in negative_words)
    
    # Calculer un score naïf
    simple_polarity = (pos_count - neg_count) / max(1, word_count)
    
    return {
        'simple_polarity': simple_polarity,
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'positive_ratio': pos_count / max(1, word_count),
        'negative_ratio': neg_count / max(1, word_count)
    }

def vectorize_texts_advanced(texts, max_features=8000):
    """Vectorisation avancée avec TF-IDF optimisé"""
    cleaned = texts.apply(advanced_clean_text)
    
    # TF-IDF avec paramètres optimisés pour meilleure performance
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),  # Unigrams à trigrams (meilleur équilibre performance/complexité)
        min_df=3,           # Ignorer les mots trop rares pour éviter l'overfitting
        max_df=0.7,         # Ignorer les mots trop fréquents
        stop_words='english',
        sublinear_tf=True,  # Utiliser la transformation logarithmique
        use_idf=True,
        smooth_idf=True,
        norm='l2'           # Normalisation pour uniformiser les documents
    )
    
    X = vectorizer.fit_transform(cleaned)
    return X, vectorizer

def vectorize_with_features(texts, max_features=8000):
    """Vectorisation TF-IDF + Features statistiques (rapide, sans TextBlob)"""
    from scipy.sparse import hstack
    
    # 1. TF-IDF classique
    X_tfidf, vectorizer = vectorize_texts_advanced(texts, max_features)
    
    # 2. Features statistiques rapides
    print("🔄 Extraction des statistiques textuelles (approche rapide)...")
    text_stats = []
    for text in texts:
        stats = get_text_statistics(text)
        text_stats.append([
            stats['simple_polarity'],    # Notre propre calcul de polarité
            stats['positive_ratio'],     # % de mots positifs
            stats['negative_ratio'],     # % de mots négatifs
            stats['avg_word_length'],    # Longueur moyenne des mots
            stats['word_count'],         # Nombre de mots
            stats['char_count']          # Nombre de caractères
        ])
    
    X_features = np.array(text_stats)
    
    # 3. Combiner TF-IDF + Features
    X_combined = hstack([X_tfidf, X_features])
    
    return X_combined, vectorizer


def transform_with_features(texts, vectorizer):
    """Transforme des textes en utilisant un vectorizer déjà entraîné."""
    from scipy.sparse import hstack

    cleaned = texts.apply(advanced_clean_text)
    X_tfidf = vectorizer.transform(cleaned)

    text_stats = []
    for text in texts:
        stats = get_text_statistics(text)
        text_stats.append([
            stats["simple_polarity"],
            stats["positive_ratio"],
            stats["negative_ratio"],
            stats["avg_word_length"],
            stats["word_count"],
            stats["char_count"]
        ])

    X_features = np.array(text_stats)
    X_combined = hstack([X_tfidf, X_features])

    return X_combined

