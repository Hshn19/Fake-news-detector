import re
import string

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

import joblib

# Load datasets
true_df = pd.read_csv("Data/True.csv")
fake_df = pd.read_csv("Data/Fake.csv")

# Add labels
true_df['label'] = 1  # Real news
fake_df['label'] = 0  # Fake news

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Preprocess Text Data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# Build and Train Models
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model_lr = LogisticRegression()
model_lr.fit(X_train_vec, y_train)

model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train)

# Evaluate and compare
lr_preds = model_lr.predict(X_test_vec)
nb_preds = model_nb.predict(X_test_vec)

print("Logistic Regression Report:\n", classification_report(y_test, lr_preds))
print("Naive Bayes Report:\n", classification_report(y_test, nb_preds))

# Save models
joblib.dump(model_lr, "Models/model_lr.pkl")
joblib.dump(model_nb, "Models/model_nb.pkl")
joblib.dump(vectorizer, "Models/vectorizer.pkl")





