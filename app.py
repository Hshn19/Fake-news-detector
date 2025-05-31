import streamlit as st
import joblib

# Load the vectorizer
vectorizer = joblib.load("Models/vectorizer.pkl")

st.title("📰 Fake News Detector")

# Dropdown to choose model
model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Naive Bayes"])

# Load the selected model
if model_choice == "Logistic Regression":
    model = joblib.load("Models/model_lr.pkl")
else:
    model = joblib.load("Models/model_nb.pkl")

# Option 1: Text input
user_input = st.text_area("Paste a news article here:")

# Option 2: File uploader
uploaded_file = st.file_uploader("...or upload a .txt file", type=["txt"])
if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    st.write("📄 Uploaded content:")
    st.write(file_content)
    user_input = file_content  # override with file content

if st.button("Predict"):
    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        result = "REAL 🟢" if prediction == 1 else "FAKE 🔴"
        st.subheader(f"Prediction using {model_choice}: {result}")
    else:
        st.warning("Please enter some text or upload a file.")

