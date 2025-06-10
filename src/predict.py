import pandas as pd
from .preprocess import transform_with_features

def predict_and_export(model, vectorizer, test_path, output_path):
    test_df = pd.read_csv(test_path)
    
    # Appliquer le même vectorizer que lors de l'entraînement
    X_test = transform_with_features(test_df["reviews_content"], vectorizer)
    
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
