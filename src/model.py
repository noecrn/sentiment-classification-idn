# src/ensemble_model.py

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_ensemble_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modèles individuels optimisés pour la vitesse
    lr = LogisticRegression(
        max_iter=1000,  # Réduit de 2000 à 1000
        C=1.0,
        random_state=42,
        n_jobs=-1  # Utilise tous les CPU disponibles
    )
    rf = RandomForestClassifier(
        n_estimators=50,  # Réduit de 100 à 50 arbres
        max_depth=8,      # Réduit de 10 à 8
        min_samples_split=5,
        random_state=42,
        n_jobs=-1  # Utilise tous les CPU disponibles
    )
    svm = SVC(
        C=1.0,
        kernel='linear',
        probability=True,
        random_state=42,
        cache_size=1000  # Augmente le cache pour plus de vitesse
    )
    
    # Ensemble voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('svm', svm)
        ],
        voting='soft',  # Utilise les probabilités
        n_jobs=-1  # Utilise tous les CPU disponibles
    )
    
    # Entraînement
    ensemble.fit(X_train, y_train)
    
    # Évaluation
    y_pred = ensemble.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"Ensemble Validation Accuracy: {acc:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_val, y_pred))
    
    # Validation croisée
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return ensemble
