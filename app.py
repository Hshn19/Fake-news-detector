import numpy as np
import streamlit as st
import joblib
import re
import string


# Function to clean text (same as in training code)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# Set page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Title and description
st.title("📰 Fake News Detector")
st.markdown("""
This app uses a Linear SVM model to detect fake news articles.
Paste a news article or upload a .txt file to get a prediction.
""")

# Load the vectorizer
try:
    vectorizer = joblib.load("Models/vectorizer.pkl")
except FileNotFoundError:
    st.error("Vectorizer file not found. Please ensure the model files are in the correct directory.")
    st.stop()

# Load the Linear SVM model
try:
    model = joblib.load("Models/linear_svm.pkl")
except FileNotFoundError:
    st.error("Linear SVM model file not found. Please ensure the model file is in the correct directory.")
    st.stop()

# Create two columns for input options
col1, col2 = st.columns(2)

with col1:
    # Option 1: Text input
    user_input = st.text_area("Paste a news article here:", height=200)

with col2:
    # Option 2: File uploader
    uploaded_file = st.file_uploader("...or upload a .txt file", type=["txt"])
    if uploaded_file:
        try:
            file_content = uploaded_file.read().decode("utf-8")
            st.write("📄 Uploaded content preview:")
            st.write(file_content[:500] + "..." if len(file_content) > 500 else file_content)
            user_input = file_content  # override with file content
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Predict button
if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_input = clean_text(user_input)

        # Transform input using vectorizer
        transformed_input = vectorizer.transform([cleaned_input])

        # Make prediction
        prediction = model.predict(transformed_input)[0]

        # Display result with styling
        result = "REAL 🟢" if prediction == 1 else "FAKE 🔴"
        st.markdown(f"### Prediction: {result}")

        # Add decision score for confidence
        decision_score = model.decision_function(transformed_input)[0]
        confidence = 1 / (1 + np.exp(-decision_score))  # Convert to probability-like score
        st.markdown(f"**Confidence**: {confidence:.2%} {'(Real)' if prediction == 1 else '(Fake)'}")

        # Add some context about the prediction
        with st.expander("About this prediction"):
            st.write(f"""
            The Linear SVM model has classified this article as {result.lower()}.
            {'This indicates the article is likely to be genuine news.' if prediction == 1
            else 'This indicates the article is likely to be misleading or fabricated.'}
            Note: No model is 100% accurate, so consider this prediction alongside other sources.
            """)
    else:
        st.warning("Please enter some text or upload a file.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit | Linear SVM model trained using scikit-learn | © 2025")