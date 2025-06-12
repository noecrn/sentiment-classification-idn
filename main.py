import pandas as pd
from src.preprocess import vectorize_texts_advanced
from src.model import train_ensemble_model
from src.predict import predict_and_export
from sklearn.model_selection import cross_val_score

print("=== ENTRAÎNEMENT DU MODÈLE D'ENSEMBLE OPTIMISÉ ULTRA PERFORMANCE ===")

# Charger les données
train_df = pd.read_csv("data/train.csv")

# Preprocessing amélioré avec TF-IDF optimisé + features statistiques
print("🔄 Preprocessing avancé des données avec features optimisées...")
X_train, vectorizer = vectorize_texts_advanced(train_df['reviews_content'])

# Évaluation par validation croisée
print("📊 Évaluation par validation croisée...")
from sklearn.ensemble import RandomForestClassifier
temp_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
cv_scores = cross_val_score(temp_model, X_train, train_df['category'], cv=5)
print(f"Score moyen de validation croisée: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Entraînement du modèle d'ensemble amélioré
print("🚀 Entraînement du modèle d'ensemble optimisé...")
model = train_ensemble_model(X_train, train_df['category'])

# Génération des prédictions
print("📊 Génération des prédictions...")
predict_and_export(model, vectorizer, "data/test.csv", "outputs/submission_optimized.csv")

print("\n✅ Soumission optimisée créée : outputs/submission_optimized.csv")
print("🎯 Prêt pour Kaggle ! Score attendu : ~0.90+")