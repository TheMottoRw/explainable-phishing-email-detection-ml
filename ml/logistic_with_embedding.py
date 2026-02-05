# ==================================================
# Explainable Phishing Detection
# Sentence Embeddings + Logistic Regression
# ==================================================

import os
import re
import joblib
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from lime.lime_text import LimeTextExplainer


# ==================================================
# Text Preprocessing
# ==================================================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================================================
# Generate Sentence Embeddings in Batches
# ==================================================
def generate_embeddings(texts, embedder, batch_size=256):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = embedder.encode(
            batch,
            show_progress_bar=False
        )
        embeddings.append(emb)
    return np.vstack(embeddings)


# ==================================================
# Train Model
# ==================================================
def train_model(csv_path, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(csv_path).dropna(subset=["text_combined", "label"])
    data["text"] = data["text_combined"].apply(preprocess_text)
    data["label"] = data["label"].replace(
        {"legitimate": 0, "phishing": 1}
    )

    X_text = data["text"].tolist()
    y = data["label"].values

    # Load sentence embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings (batch-wise)
    X_embeddings = generate_embeddings(X_text, embedder)

    # Train Logistic Regression
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight={0: 2, 1: 1},
        solver="liblinear"
    )

    classifier.fit(X_embeddings, y)

    # Save models
    joblib.dump(embedder, f"{model_dir}/sentence_embedder.pkl")
    joblib.dump(classifier, f"{model_dir}/phishing_classifier_embedding.pkl")

    print("Model training completed and saved.")


# ==================================================
# Evaluate Model
# ==================================================
def evaluate_model(csv_path, model_dir):
    embedder = joblib.load(f"{model_dir}/sentence_embedder.pkl")
    classifier = joblib.load(f"{model_dir}/phishing_classifier_embedding.pkl")

    data = pd.read_csv(csv_path).dropna(subset=["text_combined", "label"])
    data["text"] = data["text_combined"].apply(preprocess_text)
    data["label"] = data["label"].replace(
        {"legitimate": 0, "phishing": 1}
    )

    X_embeddings = generate_embeddings(
        data["text"].tolist(), embedder
    )
    y_true = data["label"].values
    y_pred = classifier.predict(X_embeddings)

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}\n")

    print(classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Phishing"]
    ))


# ==================================================
# Predict + Explain
# ==================================================
def predict_email(model_dir, email_text):
    embedder = joblib.load(f"{model_dir}/sentence_embedder.pkl")
    classifier = joblib.load(f"{model_dir}/phishing_classifier_embedding.pkl")

    cleaned = preprocess_text(email_text)
    embedding = embedder.encode([cleaned])

    prediction = classifier.predict(embedding)[0]
    probability = classifier.predict_proba(embedding)[0][prediction]

    label = "Phishing" if prediction == 1 else "Legitimate"

    print(f"\nPrediction: {label}")
    print(f"Confidence: {probability:.4f}")

    explainer = LimeTextExplainer(
        class_names=["Legitimate", "Phishing"]
    )

    explanation = explainer.explain_instance(
        cleaned,
        lambda texts: classifier.predict_proba(
            embedder.encode(texts, show_progress_bar=False)
        ),
        num_features=10
    )

    print("\nLIME Explanation:")
    for word, weight in explanation.as_list():
        print(f"{word}: {weight:.4f}")


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    DATASET_PATH = f"datasets/phishing-email-dataset/phishing_email.csv"
    MODEL_DIR = "models"

    train_model(DATASET_PATH, MODEL_DIR)
    evaluate_model(DATASET_PATH, MODEL_DIR)

    test_email = (
        "Please confirm your account information to avoid service disruption."
    )

    predict_email(MODEL_DIR, test_email)
