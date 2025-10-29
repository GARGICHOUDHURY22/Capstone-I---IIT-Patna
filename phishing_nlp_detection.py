# ============================================================
# Phishing Email Detection using NLP and Logistic Regression
# Author: Gargi Choudhury
# Institution: Southern New Hampshire University
# License: Apache 2.0
# ============================================================

"""
Phishing Email Detection using NLP (Python)

This single-file program is written to be presentation-ready for a mathematics or data-science talk.
It includes:
 - brief mathematical notes (TF-IDF, logistic regression, evaluation metrics)
 - data preprocessing (HTML/url/email removal, tokenization, lemmatization)
 - feature extraction (TF-IDF)
 - model training with cross-validation (Logistic Regression + GridSearch)
 - evaluation (confusion matrix, precision/recall/F1, ROC AUC)
 - saving trained model

USAGE:
 1. Put a CSV file named phishing_emails.csv in the same folder with two columns: text and label.
    - text: the raw email text (string)
    - label: integer 1 for phishing, 0 for legitimate
 2. Install dependencies:
    pip install pandas scikit-learn nltk matplotlib seaborn joblib
 3. Run:
    python phishing_nlp_detection.py

Notes for a maths presentation:
 - Use the Markdown docstrings near the top for TF-IDF and logistic-regression formulas.
 - The code also prints metrics you can show on slides.
"""

# ----- Mathematical background (for slides) -----
# TF-IDF formula (for a term t in document d):
#   tf(t,d) = count(t in d)
#   idf(t) = log( (1 + N) / (1 + df(t)) ) + 1
#   tfidf(t,d) = tf(t,d) * idf(t)
# Logistic regression hypothesis (probability):
#   h_w(x) = sigmoid(w^T x + b) = 1 / (1 + exp(-w^T x - b))
# Loss (binary cross-entropy):
#   L = -[y log h_w(x) + (1-y) log(1 - h_w(x))]

# --------------------------------------
# Code
import re
import string
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# NLP tools
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Check and download NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print('Downloading NLTK data (stopwords, wordnet)...')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove URLs, emails, HTML tags, punctuation, numbers."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # remove HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'\d+', ' ', text)  # remove numbers
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  # punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
    return text


def tokenize_and_lemmatize(text: str):
    """Tokenize text, remove stopwords, and lemmatize."""
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMA.lemmatize(t) for t in tokens]
    return tokens


def load_data(csv_path: str = 'phishing_emails.csv') -> pd.DataFrame:
    """Load and preprocess the dataset."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Include 'text' and 'label' columns.")
    df = pd.read_csv(p)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df[['text', 'label']].dropna()
    df['text_clean'] = df['text'].apply(clean_text)
    return df


def build_pipeline_and_train(X_train, y_train):
    """Build TF-IDF + Logistic Regression pipeline with GridSearch."""
    vect = TfidfVectorizer(
        tokenizer=tokenize_and_lemmatize,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=3,
        strip_accents='unicode'
    )

    X_train_tfidf = vect.fit_transform(X_train)
    print('TF-IDF shape:', X_train_tfidf.shape)

    lr = LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced')
    param_grid = {'C': [0.01, 0.1, 1.0, 5.0]}
    gs = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    gs.fit(X_train_tfidf, y_train)

    print('Best params:', gs.best_params_)
    return vect, gs.best_estimator_


def evaluate_model(vect, model, X_test, y_test, plot=True):
    """Evaluate trained model and visualize performance."""
    X_test_tfidf = vect.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print('\n=== Evaluation ===')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1 score : {f1:.4f}')
    print(f'ROC AUC  : {roc:.4f}')
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    if plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'confusion_matrix': cm
    }


def top_informative_features(vect: TfidfVectorizer, model: LogisticRegression, n: int = 20):
    """Display most informative features (phishing vs. legitimate)."""
    if not hasattr(model, 'coef_'):
        print('Model has no coefficients to display.')
        return
    feature_names = np.array(vect.get_feature_names_out())
    coefs = model.coef_.ravel()
    top_pos = np.argsort(coefs)[-n:][::-1]
    top_neg = np.argsort(coefs)[:n]

    print('\nTop positive features (phishing indicators):')
    for i in top_pos:
        print(f'{feature_names[i]:20s} -> {coefs[i]:.4f}')

    print('\nTop negative features (legitimate indicators):')
    for i in top_neg:
        print(f'{feature_names[i]:20s} -> {coefs[i]:.4f}')


def main():
    """Main execution function."""
    print('Loading data...')
    df = load_data('phishing_emails.csv')

    X = df['text_clean'].values
    y = df['label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print('Train size:', X_train.shape[0], 'Test size:', X_test.shape[0])

    vect, model = build_pipeline_and_train(X_train, y_train)
    metrics = evaluate_model(vect, model, X_test, y_test, plot=True)
    top_informative_features(vect, model, n=15)

    joblib.dump({'vectorizer': vect, 'model': model}, 'phishing_detector.joblib')
    print('\nSaved trained pipeline to phishing_detector.joblib')


if __name__ == "__main__":
    main()
