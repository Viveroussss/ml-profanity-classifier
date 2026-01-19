import streamlit as st
import joblib
import re
import numpy as np
import os

# Load the TF-IDF vectorizer and the trained model
@st.cache_resource
def load_model_components():
    model_dir = '/content/saved_models'
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_dir, 'linear_svc_model.pkl')

    try:
        tfidf_vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return tfidf_vectorizer, model
    except FileNotFoundError:
        st.error("Error: Model components not found. Please ensure 'saved_models' directory and its contents are in the correct path.")
        st.stop()

tfidf_vectorizer, model = load_model_components()

# Target labels
target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Text cleaning function (must match preprocessing in training)
def clean_text(text):
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove remaining special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Streamlit app layout
st.title("Toxicity Classifier")
st.write("Enter a comment below to classify its toxicity types.")

user_input = st.text_area("Comment", "Type your comment here...")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Clean the input text
        cleaned_input = clean_text(user_input)

        # Vectorize the cleaned text
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # Make prediction
        prediction = model.predict(input_tfidf)

        st.subheader("Classification Results:")

        # Display results
        toxic_found = False
        for i, label in enumerate(target_labels):
            if prediction[0, i] == 1:
                st.markdown(f"- **{label.replace('_', ' ').title()}**: :red[Yes]")
                toxic_found = True
            else:
                st.markdown(f"- {label.replace('_', ' ').title()}: No")

        if not toxic_found:
            st.success("This comment appears to be non-toxic.")
