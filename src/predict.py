import pandas as pd
<<<<<<< HEAD
from .preprocess import transform_with_features

def predict_and_export(model, vectorizer, test_path, output_path):
    test_df = pd.read_csv(test_path)
    
    # Appliquer le même vectorizer que lors de l'entraînement
    X_test = transform_with_features(test_df["reviews_content"], vectorizer)
    
=======

def predict_and_export(model, vectorizer, test_path, output_path):
    test_df = pd.read_csv(test_path)
    texts = test_df['reviews_content'].apply(lambda x: x.lower())
    X_test = vectorizer.transform(texts)
>>>>>>> parent of baace9a ([ADD] Implement advanced text preprocessing and feature extraction for sentiment analysis)
    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "id": range(len(preds)),
        "category": preds
    })

    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
