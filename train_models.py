import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

import joblib

# ----------------------------- #
#        Data Preparation      #
# ----------------------------- #

data_dir = "Data"
dfs = []
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        if 'label' not in df.columns:
            if 'title' in df.columns and 'text' in df.columns:
                if "fake" in file.lower():
                    df['label'] = 0
                elif "true" in file.lower():
                    df['label'] = 1
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Combine title and text
df['text'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['text'] = df['text'].apply(clean_text)
df = df[df['label'].isin([0, 1])]

X = df['text']
y = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------- #
#        Model Training         #
# ----------------------------- #

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Linear SVM": LinearSVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1": f1,
        "cv_mean": cv_mean,
        "cv_std": cv_std
    }

    print(f"\n{name} Report:")
    print(classification_report(y_test, preds))
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Train CV Mean : {cv_mean:.4f} ± {cv_std:.4f}")

# ----------------------------- #
#         Visualizations        #
# ----------------------------- #

# Accuracy and F1 Score Bar Chart
scores_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [res["accuracy"] for res in results.values()],
    "F1 Score": [res["f1"] for res in results.values()]
})

scores_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Model Accuracy and F1 Score Comparison")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.savefig("model_scores.png")
plt.show()

# Cross-Validation vs Test Accuracy Plot
cv_vs_test = pd.DataFrame({
    "Model": list(results.keys()),
    "CV Mean": [r["cv_mean"] for r in results.values()],
    "Test Accuracy": [r["accuracy"] for r in results.values()]
})

cv_vs_test.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Cross-Validation vs Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("cv_vs_test_accuracy.png")
plt.show()

# Confusion Matrices
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()
for i, (name, res) in enumerate(results.items()):
    preds = res["model"].predict(X_test_vec)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=axs[i], cmap='Blues')
    axs[i].set_title(f"{name} Confusion Matrix")
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

# ROC Curves
plt.figure(figsize=(10, 6))
for name, res in results.items():
    model = res["model"]
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_vec)[:, 1]
        else:
            probs = model.decision_function(X_test_vec)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    except:
        continue

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()

# Precision-Recall Curves
plt.figure(figsize=(10, 6))
for name, res in results.items():
    model = res["model"]
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_vec)[:, 1]
        else:
            probs = model.decision_function(X_test_vec)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.plot(recall, precision, label=name)
    except:
        continue

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curves.png")
plt.show()

# ----------------------------- #
#          Save Models          #
# ----------------------------- #

os.makedirs("Models", exist_ok=True)
joblib.dump(vectorizer, "Models/vectorizer.pkl")
for name, res in results.items():
    filename = f"Models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(res["model"], filename)






