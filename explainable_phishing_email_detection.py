#!/usr/bin/env python3
"""
Explainable Phishing Email Detection using Machine Learning

- Loads datasets/phishing-email-dataset/phishing_email.csv
- Trains a TF-IDF + Logistic Regression pipeline
- Evaluates on a test split and prints metrics
- Saves/loads a model pipeline (vectorizer + classifier)
- Provides per-text prediction with a simple, human-readable explanation

Usage examples:
  Train and evaluate, then save model:
    python explainable_phishing_email_detection.py --train --evaluate --save-model

  Predict and explain a custom text (will train a temp model if no model is saved yet):
    python explainable_phishing_email_detection.py --predict "Your account is suspended, click here" --explain

  Evaluate an existing saved model on the dataset test split:
    python explainable_phishing_email_detection.py --evaluate --load-model

Notes:
- Explanations do NOT use LIME or SHAP. They are computed directly from the linear
  model coefficients and the TF‑IDF features for the given input text (coef × TF‑IDF).
  We list top tokens contributing to the predicted class.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

try:
    from joblib import dump, load
except Exception:  # pragma: no cover
    dump = load = None  # type: ignore


DEFAULT_DATASET = os.path.join(
    "datasets", "phishing-email-dataset", "phishing_email.csv"
)
DEFAULT_MODEL_DIR = os.path.join("models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "explainable_phishing_model.joblib")


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 50000
    ngram_range: Tuple[int, int] = (1, 2)
    C: float = 2.0
    penalty: str = "l2"
    class_weight: Optional[str] = "balanced"
    max_iter: int = 200


def load_dataset(path: str) -> Tuple[List[str], List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    # Expect columns: text_combined, label (0/1)
    expected_cols = {"text_combined", "label"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}. Found columns: {list(df.columns)}"
        )
    texts = df["text_combined"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def build_pipeline(cfg: TrainConfig) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        strip_accents="unicode",
    )
    clf = LogisticRegression(
        C=cfg.C,
        penalty=cfg.penalty,
        class_weight=cfg.class_weight,
        max_iter=cfg.max_iter,
        solver="liblinear" if cfg.penalty == "l1" else "lbfgs",
        n_jobs=None,
    )
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return pipe


def train_pipeline(
    texts: List[str], labels: List[int], cfg: TrainConfig
) -> Tuple[Pipeline, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=cfg.test_size, random_state=cfg.random_state, stratify=labels
    )
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)
    return pipe, (np.array(X_train, dtype=object), np.array(X_test, dtype=object), np.array(y_train), np.array(y_test))


def evaluate_pipeline(pipe: Pipeline, X_test: List[str], y_test: List[int]) -> Dict[str, float]:
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = float("nan")
    print("\nEvaluation Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "phishing"]))
    print(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nROC-AUC: {roc:.4f}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc}


def save_model(pipe: Pipeline, path: str = DEFAULT_MODEL_PATH) -> None:
    if dump is None:
        raise RuntimeError("joblib is required to save the model. Please `pip install joblib`."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(pipe, path)
    print(f"Saved model to {path}")


def load_model(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    if load is None:
        raise RuntimeError("joblib is required to load the model. Please `pip install joblib`.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train and save a model first.")
    pipe = load(path)
    return pipe


def predict_with_explanation(
    pipe: Pipeline, text: str, top_k: int = 10
) -> Tuple[int, float, List[Tuple[str, float]]]:
    """
    Returns:
      - predicted label (0=ham, 1=phishing)
      - probability for class 1 (phishing)
      - top_k feature contributions as (token, contribution)
    """
    # Predict
    prob = float(pipe.predict_proba([text])[0, 1])
    pred = int(prob >= 0.5)

    # Explanation from linear model weights
    vectorizer: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    X_vec = vectorizer.transform([text])  # sparse row
    # For class 1, contribution per feature is value * coef_1
    coefs = clf.coef_[0]  # shape (n_features,)

    # Extract non-zero indices for the sample to be efficient
    row = X_vec.tocoo()
    contribs = {}
    for idx, val in zip(row.col, row.data):
        contribs[idx] = val * coefs[idx]

    if not contribs:
        return pred, prob, []

    # Sort by absolute contribution favoring predicted class direction
    # If pred==1, sort descending by contribution; else ascending (more negative)
    items = list(contribs.items())
    if pred == 1:
        items.sort(key=lambda kv: kv[1], reverse=True)
    else:
        items.sort(key=lambda kv: kv[1])  # most negative first

    # Map feature indices to tokens
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(vectorizer.get_feature_names())

    top = []
    for idx, score in items[:top_k]:
        token = feature_names[idx]
        top.append((token, float(score)))

    return pred, prob, top


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Explainable Phishing Email Detection (TF-IDF + Logistic Regression)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET, help="Path to phishing_email.csv")
    parser.add_argument("--train", action="store_true", help="Train a model on the dataset")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model on a held-out test set")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model to disk")
    parser.add_argument("--load-model", action="store_true", help="Load an existing saved model instead of training")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to save/load the model joblib")

    parser.add_argument("--predict", type=str, default=None, help="A single input text to predict")
    parser.add_argument("--explain", action="store_true", help="Show an explanation for the --predict text")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens to show in explanation")

    parser.add_argument("--max-features", type=int, default=50000, help="Max TF-IDF features")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size (1 to n)")
    parser.add_argument("--C", type=float, default=2.0, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--l1", action="store_true", help="Use L1 penalty (otherwise L2)")
    parser.add_argument("--no-balanced", action="store_true", help="Disable class_weight=balanced")

    args = parser.parse_args(argv)

    cfg = TrainConfig(
        ngram_range=(1, max(1, int(args.ngram_max))),
        max_features=int(args.max_features),
        C=float(args.C),
        penalty="l1" if args.l1 else "l2",
        class_weight=None if args.no_balanced else "balanced",
    )

    texts: List[str] = []
    labels: List[int] = []
    X_train = X_test = y_train = y_test = None

    pipe: Optional[Pipeline] = None

    if args.load_model and os.path.exists(args.model_path):
        pipe = load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        # If we intend to train or evaluate, we need data
        if args.train or args.evaluate or args.predict is None:
            try:
                texts, labels = load_dataset(args.data)
            except Exception as e:
                print(f"Error loading dataset: {e}", file=sys.stderr)
                return 2

        if args.train or args.evaluate or (args.predict is not None and pipe is None):
            pipe, (X_train, X_test, y_train, y_test) = train_pipeline(texts, labels, cfg)
            print("Training completed.")
            if args.save_model:
                save_model(pipe, args.model_path)

    # Evaluate if requested (requires X_test/y_test; if loaded model, create a fresh split)
    if args.evaluate:
        if (X_test is None) or (y_test is None):
            # Need to split since we loaded a model only
            texts, labels = load_dataset(args.data)
            _, X_test_arr, _, y_test_arr = train_test_split(
                texts, labels, test_size=cfg.test_size, random_state=cfg.random_state, stratify=labels
            )
            X_test, y_test = np.array(X_test_arr, dtype=object), np.array(y_test_arr)
        evaluate_pipeline(pipe, list(X_test), list(y_test))  # type: ignore

    # Predict single input and optionally explain
    if args.predict is not None:
        if pipe is None:
            # If we get here, it means --load-model was given but not found, so we already trained above.
            texts, labels = load_dataset(args.data)
            pipe, _ = train_pipeline(texts, labels, cfg)
        pred, prob, top = predict_with_explanation(pipe, args.predict, top_k=args.top_k)
        label_name = "phishing" if pred == 1 else "ham"
        print("\nPrediction:")
        print(f"Text: {args.predict}")
        print(f"Predicted: {label_name} (prob={prob:.4f})")
        if args.explain and top:
            print("Top contributing tokens (token -> contribution):")
            for token, score in top:
                direction = "+" if score >= 0 else "-"
                print(f"  {token:20s} {direction}{abs(score):.5f}")
        elif args.explain:
            print("No informative tokens found to explain (likely due to preprocessing).")

    # If user didn't request anything, show help
    if not (args.train or args.evaluate or args.predict is not None):
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
