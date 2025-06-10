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
<<<<<<< HEAD
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

=======
    return X, vectorizer
>>>>>>> parent of baace9a ([ADD] Implement advanced text preprocessing and feature extraction for sentiment analysis)
