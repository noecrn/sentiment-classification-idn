import pandas as pd
from src.preprocess import vectorize_texts
from src.model import train_model
from src.predict import predict_and_export

# Load data
train_df = pd.read_csv("data/train.csv")

# Vectorize
X_train, vectorizer = vectorize_texts(train_df['reviews_content'])

# Train model
model = train_model(X_train, train_df['category'])

# Predict and export
predict_and_export(model, vectorizer, "data/test.csv", "outputs/submission.csv")