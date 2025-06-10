import pandas as pd
from .preprocess import advanced_clean_text

def predict_and_export(model, vectorizer, test_path, output_path):
    test_df = pd.read_csv(test_path)
    
    # Utiliser le même vectorizer déjà entraîné, PAS en créer un nouveau
    cleaned_texts = test_df['reviews_content'].apply(advanced_clean_text)
    X_test = vectorizer.transform(cleaned_texts)
    
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