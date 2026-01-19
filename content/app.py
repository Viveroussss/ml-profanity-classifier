import streamlit as st
import re
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Toxicity Classifier",
    page_icon="🚨",
    layout="wide"
)

# Target labels
target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Text cleaning function (must match preprocessing in training)
def clean_text(text):
    """Clean text by lowercasing, removing punctuation, numbers, and special characters."""
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(r'[^\u001f-~]', '', text)  # Remove non-printable characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove remaining special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Load pretrained model and vectorizer
@st.cache_resource
def load_model_components():
    """Load the pretrained model and vectorizer."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible paths relative to script location
        possible_paths = [
            (os.path.join(script_dir, 'saved_models', 'tfidf_vectorizer.pkl'), 
             os.path.join(script_dir, 'saved_models', 'linear_svc_model.pkl')),
            ('saved_models/tfidf_vectorizer.pkl', 'saved_models/linear_svc_model.pkl'),
            ('content/saved_models/tfidf_vectorizer.pkl', 'content/saved_models/linear_svc_model.pkl'),
        ]
        
        vectorizer_path = None
        model_path = None
        
        for v_path, m_path in possible_paths:
            if os.path.exists(v_path) and os.path.exists(m_path):
                vectorizer_path = v_path
                model_path = m_path
                break
        
        if vectorizer_path is None or model_path is None:
            st.error("❌ Model files not found. Please ensure saved_models/tfidf_vectorizer.pkl and saved_models/linear_svc_model.pkl exist.")
            st.stop()
        
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model components
vectorizer, model = load_model_components()

# Main title
st.title("🚨 Toxicity Classification System")
st.markdown("Classify comments for toxicity using a pretrained multi-label classifier")

# Prediction interface
st.subheader("Enter Comment to Classify")
user_input = st.text_area(
    "Comment",
    "",
    height=150,
    placeholder="Type your comment here..."
)

if st.button("🔍 Classify", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Clean the input text
        cleaned_input = clean_text(user_input)
        
        # Vectorize the cleaned text
        input_tfidf = vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction = model.predict(input_tfidf)
        
        # Get prediction probabilities if available
        prediction_proba = None
        try:
            proba_list = model.predict_proba(input_tfidf)
            # OneVsRestClassifier returns a list of arrays, one per label
            if isinstance(proba_list, list) and len(proba_list) == len(target_labels):
                prediction_proba = [proba[0][1] for proba in proba_list]  # Get probability of positive class
        except:
            prediction_proba = None
        
        st.subheader("📊 Classification Results")
        
        # Display results in columns
        cols = st.columns(3)
        toxic_found = False
        
        for i, label in enumerate(target_labels):
            col_idx = i % 3
            with cols[col_idx]:
                if prediction[0, i] == 1:
                    st.markdown(f"**{label.replace('_', ' ').title()}**: :red[Yes]")
                    toxic_found = True
                    if prediction_proba is not None:
                        st.progress(prediction_proba[i])
                        st.caption(f"Confidence: {prediction_proba[i]:.2%}")
                else:
                    st.markdown(f"{label.replace('_', ' ').title()}: No")
                    if prediction_proba is not None:
                        st.progress(1 - prediction_proba[i])
                        st.caption(f"Confidence: {(1 - prediction_proba[i]):.2%}")
        
        if not toxic_found:
            st.success("✅ This comment appears to be non-toxic.")
        else:
            st.error("⚠️ This comment contains toxic content.")
        
        # Show balloons animation after classification
        st.balloons()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses a pretrained multi-label toxicity classifier with TF-IDF vectorization "
    "and OneVsRestClassifier with LogisticRegression. It can classify comments into "
    "six toxicity categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate."
)
