import os
import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# Initialize empty DataFrame
combined_df = pd.DataFrame(columns=["title", "text", "label"])

# Autoload all CSVs in Data folder
data_folder = "Data"
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        path = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(path)

            # Guess label if not explicitly labeled
            if 'label' in df.columns:
                # Normalize labels
                df['label'] = df['label'].map({
                    'FAKE': 0, 'fake': 0, 'Fake': 0,
                    'REAL': 1, 'real': 1, 'Real': 1,
                    0: 0, 1: 1
                })
                df = df.dropna(subset=['label'])
                df['label'] = df['label'].astype(int)
            else:
                # Infer from filename
                if "fake" in filename.lower():
                    df['label'] = 0
                elif "true" in filename.lower() or "real" in filename.lower():
                    df['label'] = 1
                else:
                    print(f"Skipping {filename}: Cannot determine label.")
                    continue

            # Ensure 'title' and 'text' columns exist
            if not all(col in df.columns for col in ['title', 'text']):
                print(f"Skipping {filename}: Missing 'title' or 'text'.")
                continue

            combined_df = pd.concat([combined_df, df[['title', 'text', 'label']]], ignore_index=True)
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Combine and clean text
# Clean and standardize label column
combined_df = combined_df[combined_df['label'].isin([0, 1])]  # keep only 0s and 1s
combined_df['label'] = combined_df['label'].astype(int)       # force label to integer type
print("Unique labels in dataset:", combined_df['label'].unique())

# Train/Test Split
X = combined_df['text']
y = combined_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Linear SVM": LinearSVC()
}

# Evaluate Models
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    print(f"\n{name} Report:\n", classification_report(y_test, preds))
    joblib.dump(model, f"Models/model_{name.replace(' ', '_').lower()}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "Models/vectorizer.pkl")





