# src/train.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump

try:
    from src.clean import clean_email
except ImportError:
    try:
        from .clean import clean_email
    except ImportError:
        from clean import clean_email

LABELS = ["SPAM", "FRAUD", "NORMAL"]

def train_model(data_path: str = "data/final_dataset.csv", model_output_path: str = "models/classifier.joblib"):
    """
    Load data, clean email text, train multi-class classifier pipeline,
    print classification report, and save trained model to disk.
    """
    if not os.path.exists(data_path):
        # Check relative to parent directory if script is run from src/
        parent_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_path)
        if os.path.exists(parent_data_path):
            data_path = parent_data_path
        else:
            print(f"Error: '{data_path}' not found. Make sure the file is in the correct directory.")
            return None

    # 1) Load data with latin-1 encoding
    df = pd.read_csv(data_path, encoding='latin-1')

    # Rename columns to standard 'text' and 'label'
    df = df.rename(columns={"Email": "text", "Label": "label"})

    # Drop rows with missing text or labels
    df.dropna(subset=['text', 'label'], inplace=True)

    # Clean the email text
    df["text"] = df["text"].astype(str).map(clean_email)

    # 2) Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3) Create the pipeline: TF-IDF + Logistic Regression
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    # 4) Train the model
    print("Training model...")
    pipe.fit(X_train, y_train)

    # 5) Evaluate the model
    y_pred = pipe.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

    # 6) Save the trained model and labels
    output_dir = os.path.dirname(model_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dump({"model": pipe, "labels": LABELS}, model_output_path)
    print(f"✅ Model saved to -> {model_output_path}")
    return pipe

if __name__ == "__main__":
    train_model()
