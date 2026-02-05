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

## Usage

Ensure the dataset path exists (it is included in this repo):
- datasets/phishing-email-dataset/phishing_email.csv

Run `python explainable_phishing_email_detection.py -h` to see all options.

### Train and evaluate
```
python explainable_phishing_email_detection.py --train --evaluate --save-model
```
This will:
- Split the dataset into train/test (default 80/20, stratified)
- Train TF‑IDF + Logistic Regression
- Print evaluation metrics (accuracy, precision, recall, F1, ROC‑AUC)
- Save the trained pipeline to `models/explainable_phishing_model.joblib`

### Evaluate a saved model
```
python explainable_phishing_email_detection.py --evaluate --load-model
```
If a saved model is present at `models/explainable_phishing_model.joblib`, the script loads it and evaluates on a fresh test split from the dataset.

### Predict and get an explanation for a custom text
```
python explainable_phishing_email_detection.py \
  --predict "Your account is suspended. Verify immediately to avoid termination." \
  --explain --top-k 10
```
Output example:
```
Prediction:
Text: Your account is suspended. Verify immediately to avoid termination.
Predicted: phishing (prob=0.9732)
Top contributing tokens (token -> contribution):
  verify               +0.12345
  suspended            +0.09876
  immediately          +0.07654
  account              +0.05555
  avoid termination    +0.04021
```

### Useful options
- --data: path to CSV (default: datasets/phishing-email-dataset/phishing_email.csv)
- --max-features: limit vocabulary size for TF‑IDF (default: 50000)
- --ngram-max: include up to n‑grams of this length (default: 2)
- --C: inverse regularization strength for Logistic Regression (default: 2.0)
- --l1: use L1 penalty (sparser model)
- --no-balanced: disable class_weight=balanced (enabled by default)

## Notes on explainability
- The model is linear; a token’s sign and magnitude indicate its push toward the phishing (positive) or ham (negative) class for the specific input. The printed contributions are the product of the TF‑IDF value and the learned coefficient for class 1.
- Because TF‑IDF normalizes by document frequency and document length, contribution magnitudes are comparable within a prediction, but do not represent probabilities by themselves.

### Does this use LIME or SHAP?
- No. This repository does not use LIME or SHAP. Explanations are derived directly from the linear model’s coefficients (coef × TF‑IDF value) for the specific input text.
- Rationale: keeping the example lightweight with minimal dependencies and fast runtime while still providing faithful, token-level attributions for a linear model.
- If you want LIME or SHAP:
  - LIME: `pip install lime` and integrate `lime.lime_text.LimeTextExplainer` over the pipeline’s predict_proba.
  - SHAP: `pip install shap` and use `shap.LinearExplainer` with the trained classifier and TF‑IDF features; then map feature attributions back to tokens via the vectorizer vocabulary.
  - These additions are not included by default to keep dependencies minimal, but the code structure should make them straightforward to add.

## Reproducibility
- We set a fixed random_state for the train/test split to improve reproducibility.
- If you change feature settings (e.g., n‑grams, max features), saved models will not be compatible with prior ones.

## Troubleshooting
- ModuleNotFoundError (numpy/pandas/sklearn): ensure you ran the pip install commands above inside your virtual environment.
- Unicode issues when reading CSV: the script uses pandas’ default engine/encoding; you can add `encoding="utf-8"` in `pd.read_csv` if needed.

## License
This is a minimal example for educational purposes. Adapt as needed for production.
