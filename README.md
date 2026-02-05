# Explainable Phishing Email Detection (ML)

This repository provides a minimal, end‑to‑end, explainable phishing email detector using classic machine learning:

- Input: datasets/phishing-email-dataset/phishing_email.csv (columns: `text_combined`, `label`)
- Model: TF‑IDF features + Logistic Regression
- Explainability: token‑level contributions computed from linear model weights (coef × TF‑IDF value) for a given input text

The solution is intentionally lightweight and dependency‑friendly, avoiding heavy frameworks. It includes a single Python script with CLI flags for training, evaluation, prediction, and explanations.

## File overview
- explainable_phishing_email_detection.py — main script (train/evaluate/predict/explain)
- datasets/phishing-email-dataset/phishing_email.csv — dataset used for training/evaluation
- models/ — folder created automatically to store the trained pipeline (joblib)

## Setup

1) Python version
- Python 3.9+ recommended (3.8+ should also work)

2) Create and activate a virtual environment (recommended)
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3) Install dependencies
```
pip install -U pip
pip install scikit-learn pandas numpy joblib
```
