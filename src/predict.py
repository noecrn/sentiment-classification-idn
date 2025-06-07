import pandas as pd

def predict_and_export(model, vectorizer, test_path, output_path):
    test_df = pd.read_csv(test_path)
    texts = test_df['reviews_content'].apply(lambda x: x.lower())
    X_test = vectorizer.transform(texts)
    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "id": range(len(preds)),
        "category": preds
    })

    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")