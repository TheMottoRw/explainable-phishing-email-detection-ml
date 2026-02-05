# ==========================================
# Explainable Phishing Email Detection System
# With Data Preprocessing, ML, and LIME
# ==========================================

import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

from lime.lime_text import LimeTextExplainer

print(os.getcwd())
# ==========================================
# Function 1: Text Preprocessing
# ==========================================
def preprocess_text(text):
    """
    Cleans and normalizes email text.
    """
    if pd.isna(text):
        return ""

    text = text.lower()                         # lowercase
    text = re.sub(r"http\\S+|www\\S+", "", text) # remove URLs
    text = re.sub(r"[^a-z\\s]", " ", text)       # remove punctuation & numbers
    text = re.sub(r"\\s+", " ", text).strip()   # remove extra spaces

    return text


# ==========================================
# Function 2: Train and Save Model
# ==========================================
def train_model(csv_path, model_path):
    """
    Trains phishing detection model with preprocessing and saves it.
    """

    # Load dataset
    data = pd.read_csv(csv_path)
    # Drop missing rows
    data = data.dropna(subset=['text_combined', 'label'])

    # Apply preprocessing
    data['text'] = data['text_combined'].apply(preprocess_text)

    # Encode labels
    # data['label'] = data['label'].replace({'legitimate': 0, 'phishing': 1})

    X = data['text']
    y = data['label']
    

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # NLP + ML pipeline
    pipeline = model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(2,3),
            min_df=5, # ignore terms that appear in less than 2 documents
            max_df=0.8,
            use_idf=True,
            smooth_idf=True
        )),
        ('classifier', LogisticRegression(max_iter=1000,
                                          C=0.1,
                                          # class_weight='balanced',
                                          class_weight={0: 2, 1: 1}
                                          ))
    ])
    #start of improvement
    # Accuracy Improvement: Hyperparameter Tuning
    # param_grid = {
    #     'classifier__C': [0.1, 1.0, 10.0],  # Testing different regularization strengths
    #     'classifier__solver': ['lbfgs', 'liblinear']
    # }
    #
    # model = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    #
    # # Train
    # model.fit(X_train, y_train)
    # # Use the best found model for evaluation
    # best_model = model.best_estimator_
    # y_pred = best_model.predict(X_test)

 #end of improvement
    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("\nModel Evaluation Results")
    print("------------------------")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}\n")

    print("Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Phishing"]
    ))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"\nModel saved successfully at: {model_path}")


# ==========================================
# Function 3: Predict and Explain Email
# ==========================================
def predict_email(model_path, email_text):
    """
    Predicts and explains phishing decision using LIME.
    """

    # Load model
    model = joblib.load(model_path)
    # print(f"Email {type(email_text)}")
    # quit()

    # Preprocess input email
    cleaned_email = preprocess_text(email_text)

    # Prediction
    prediction = model.predict([cleaned_email])[0]
    probability = model.predict_proba([cleaned_email])[0][prediction]

    label = "Phishing" if prediction == 1 else "Legitimate"

    print(f"\nPrediction: {label}")
    print(f"Confidence: {probability:.4f}")

    # Explain using LIME
    explainer = LimeTextExplainer(
        class_names=["Legitimate", "Phishing"]
    )

    explanation = explainer.explain_instance(
        cleaned_email,
        model.predict_proba,
        num_features=10
    )

    print("\nLIME Explanation (Top Influential Words):")
    for word, weight in explanation.as_list():
        print(f"{word}: {weight:.4f}")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    DATASET_PATH = f"datasets/phishing-email-dataset/phishing_email.csv"
    MODEL_PATH = f"{os.getcwd()}/models/phishing_detection_model.pkl"

    # train_model(DATASET_PATH, MODEL_PATH)

    sample_email = (
        # "Urgent! Your bank account has been suspended. "
        # "Click the link to verify your details immediately."
        "Hello Asua, can you share with me your bank account number to support you"
        """
        Dear,
Transaction was declined. [OR-PFGVEM-18]

If you don't update your payment method, you'll lose all of the benefits provided by Premium Individual (Spotify: Music and Podcasts). These benefits include the following:

Ad-free music listening
Download to listen offline
Play songs in any order
High audio quality
To keep these benefits, you'll need to keep your subscription active. Update your payment method or use a different one.

"""
    )

    predict_email(MODEL_PATH, sample_email)
