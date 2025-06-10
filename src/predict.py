import pandas as pd
from .preprocess import vectorize_texts_with_features

def predict_and_export(model, vectorizer, test_path, output_path):
    test_df = pd.read_csv(test_path)
    
    # Utiliser la même fonction avec TextBlob features pour la prédiction
    X_test, _ = vectorize_texts_with_features(test_df['reviews_content'])
    
    preds = model.predict(X_test)
    if 'id' in test_df.columns:
        ids = test_df['id']
    else:
        ids = range(1, len(preds) + 1)
    
    submission = pd.DataFrame({
        "Row": ids,
        "Category": preds
    })

    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")