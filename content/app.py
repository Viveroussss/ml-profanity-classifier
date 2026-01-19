import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, hamming_loss
)
import zipfile
import tempfile

# Page configuration
st.set_page_config(
    page_title="Toxicity Classifier - Train & Predict",
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

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main title
st.title("🚨 Toxicity Classification System")
st.markdown("Train a multi-label toxicity classifier and test it on new comments")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Data Loading", "🎯 Model Training", "🔮 Prediction", "📈 Model Evaluation"]
)

# Page 1: Data Loading
if page == "📊 Data Loading":
    st.header("Load Training and Test Datasets")
    st.markdown("Upload your datasets or use the existing ones in the current directory.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data")
        train_option = st.radio(
            "Training data source",
            ["Use existing file (train.csv.zip)", "Upload new file"]
        )
        
        if train_option == "Upload new file":
            train_file = st.file_uploader(
                "Upload train.csv.zip",
                type=['zip', 'csv'],
                key='train_upload'
            )
            if train_file is not None:
                if train_file.name.endswith('.zip'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(train_file.read())
                        tmp_path = tmp_file.name
                    try:
                        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                            zip_ref.extractall(tempfile.gettempdir())
                            csv_path = os.path.join(tempfile.gettempdir(), 'train.csv')
                            if os.path.exists(csv_path):
                                st.session_state.train_df = pd.read_csv(csv_path)
                                st.success(f"✅ Training data loaded: {st.session_state.train_df.shape}")
                            else:
                                st.error("Could not find train.csv in the zip file")
                    except Exception as e:
                        st.error(f"Error loading zip file: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
                else:
                    st.session_state.train_df = pd.read_csv(train_file)
                    st.success(f"✅ Training data loaded: {st.session_state.train_df.shape}")
        else:
            if st.button("Load from train.csv.zip"):
                try:
                    if os.path.exists('train.csv.zip'):
                        with zipfile.ZipFile('train.csv.zip', 'r') as zip_ref:
                            zip_ref.extractall(tempfile.gettempdir())
                            csv_path = os.path.join(tempfile.gettempdir(), 'train.csv')
                            st.session_state.train_df = pd.read_csv(csv_path)
                            st.success(f"✅ Training data loaded: {st.session_state.train_df.shape}")
                    else:
                        st.error("File not found: train.csv.zip")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        if 'train_df' in st.session_state:
            st.dataframe(st.session_state.train_df.head(), use_container_width=True)
    
    with col2:
        st.subheader("Test Data")
        test_option = st.radio(
            "Test data source",
            ["Use existing files (raw_test.csv & test_labels.csv)", "Upload new files"]
        )
        
        if test_option == "Upload new files":
            raw_test_file = st.file_uploader(
                "Upload raw_test.csv",
                type=['csv'],
                key='raw_test_upload'
            )
            test_labels_file = st.file_uploader(
                "Upload test_labels.csv",
                type=['csv'],
                key='test_labels_upload'
            )
            
            if raw_test_file is not None and test_labels_file is not None:
                st.session_state.raw_test_df = pd.read_csv(raw_test_file)
                st.session_state.test_labels_df = pd.read_csv(test_labels_file)
                st.session_state.test_df = pd.merge(
                    st.session_state.raw_test_df,
                    st.session_state.test_labels_df,
                    on='id',
                    how='left'
                )
                st.success(f"✅ Test data loaded: {st.session_state.test_df.shape}")
        else:
            if st.button("Load from current directory"):
                try:
                    if os.path.exists('raw_test.csv') and os.path.exists('test_labels.csv'):
                        raw_test_df = pd.read_csv('raw_test.csv')
                        test_labels_df = pd.read_csv('test_labels.csv')
                        st.session_state.test_df = pd.merge(
                            raw_test_df,
                            test_labels_df,
                            on='id',
                            how='left'
                        )
                        st.success(f"✅ Test data loaded: {st.session_state.test_df.shape}")
                    else:
                        st.error("Files not found in current directory")
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")
        
        if 'test_df' in st.session_state:
            st.dataframe(st.session_state.test_df.head(), use_container_width=True)

# Page 2: Model Training
elif page == "🎯 Model Training":
    st.header("Train Multi-Label Toxicity Classifier")
    
    if 'train_df' not in st.session_state:
        st.warning("⚠️ Please load training data first in the 'Data Loading' page.")
        st.stop()
    
    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        max_features = st.slider("TF-IDF Max Features", 1000, 10000, 5000, 500)
        use_stopwords = st.checkbox("Use English stop words", value=True)
    
    with col2:
        solver = st.selectbox("Logistic Regression Solver", ['sag', 'lbfgs', 'liblinear'], index=0)
        max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                # Load and preprocess training data
                df_train = st.session_state.train_df.copy()
                
                # Filter out rows where all target labels are -1
                df_train_filtered = df_train[~(df_train[target_labels] == -1).all(axis=1)].copy()
                st.info(f"Training samples after filtering: {df_train_filtered.shape[0]:,}")
                
                # Clean text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Cleaning training text...")
                df_train_filtered['comment_text'] = df_train_filtered['comment_text'].apply(clean_text)
                progress_bar.progress(20)
                
                # Prepare target labels
                status_text.text("Preparing target labels...")
                y_train = df_train_filtered[target_labels]
                progress_bar.progress(30)
                
                # TF-IDF Vectorization
                status_text.text("Performing TF-IDF vectorization...")
                stop_words = 'english' if use_stopwords else None
                st.session_state.vectorizer = TfidfVectorizer(
                    stop_words=stop_words,
                    max_features=max_features
                )
                X_train_tfidf = st.session_state.vectorizer.fit_transform(df_train_filtered['comment_text'])
                progress_bar.progress(60)
                st.info(f"TF-IDF features shape: {X_train_tfidf.shape}")
                
                # Train model
                status_text.text("Training OneVsRestClassifier with LogisticRegression...")
                base_estimator = LogisticRegression(
                    solver=solver,
                    max_iter=max_iter,
                    random_state=42
                )
                st.session_state.model = OneVsRestClassifier(base_estimator)
                st.session_state.model.fit(X_train_tfidf, y_train)
                progress_bar.progress(100)
                
                st.session_state.model_trained = True
                status_text.text("✅ Training complete!")
                
                # Save model and vectorizer
                os.makedirs('saved_models', exist_ok=True)
                joblib.dump(st.session_state.vectorizer, 'saved_models/tfidf_vectorizer.pkl')
                joblib.dump(st.session_state.model, 'saved_models/linear_svc_model.pkl')
                
                st.success("🎉 Model trained successfully and saved!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                st.exception(e)
    
    if st.session_state.model_trained:
        st.success("✅ Model is ready for prediction!")
        st.info("You can now use the 'Prediction' page to test the model or 'Model Evaluation' to see performance metrics.")

# Page 3: Prediction
elif page == "🔮 Prediction":
    st.header("Predict Toxicity in Comments")
    
    if not st.session_state.model_trained:
        # Try to load existing model
        try:
            if os.path.exists('saved_models/tfidf_vectorizer.pkl') and \
               os.path.exists('saved_models/linear_svc_model.pkl'):
                st.session_state.vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
                st.session_state.model = joblib.load('saved_models/linear_svc_model.pkl')
                st.session_state.model_trained = True
                st.success("✅ Loaded pre-trained model from saved_models directory")
            else:
                st.warning("⚠️ No trained model found. Please train a model first in the 'Model Training' page.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    st.subheader("Enter Comment to Classify")
    user_input = st.text_area(
        "Comment",
        "Type your comment here...",
        height=150
    )
    
    if st.button("🔍 Classify", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            # Clean the input text
            cleaned_input = clean_text(user_input)
            
            # Vectorize the cleaned text
            input_tfidf = st.session_state.vectorizer.transform([cleaned_input])
            
            # Make prediction
            prediction = st.session_state.model.predict(input_tfidf)
            
            # Get prediction probabilities if available
            prediction_proba = None
            try:
                proba_list = st.session_state.model.predict_proba(input_tfidf)
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

# Page 4: Model Evaluation
elif page == "📈 Model Evaluation":
    st.header("Evaluate Model Performance")
    
    if not st.session_state.model_trained:
        # Try to load existing model
        try:
            if os.path.exists('saved_models/tfidf_vectorizer.pkl') and \
               os.path.exists('saved_models/linear_svc_model.pkl'):
                st.session_state.vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
                st.session_state.model = joblib.load('saved_models/linear_svc_model.pkl')
                st.session_state.model_trained = True
            else:
                st.warning("⚠️ No trained model found. Please train a model first.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    if 'test_df' not in st.session_state:
        st.warning("⚠️ Please load test data first in the 'Data Loading' page.")
        st.stop()
    
    if st.button("📊 Evaluate Model", type="primary"):
        with st.spinner("Evaluating model on test data..."):
            try:
                # Preprocess test data
                df_test = st.session_state.test_df.copy()
                
                # Filter out rows where all target labels are -1
                df_test_filtered = df_test[~(df_test[target_labels] == -1).all(axis=1)].copy()
                st.info(f"Test samples after filtering: {df_test_filtered.shape[0]:,}")
                
                # Clean text
                df_test_filtered['comment_text'] = df_test_filtered['comment_text'].apply(clean_text)
                
                # Prepare target labels
                y_test = df_test_filtered[target_labels]
                
                # Transform test data
                X_test_tfidf = st.session_state.vectorizer.transform(df_test_filtered['comment_text'])
                
                # Make predictions
                y_pred = st.session_state.model.predict(X_test_tfidf)
                
                # Calculate metrics
                st.subheader("📈 Evaluation Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.4f}")
                
                with col2:
                    micro_f1 = f1_score(y_test, y_pred, average='micro')
                    st.metric("Micro F1-Score", f"{micro_f1:.4f}")
                
                with col3:
                    macro_f1 = f1_score(y_test, y_pred, average='macro')
                    st.metric("Macro F1-Score", f"{macro_f1:.4f}")
                
                # Hamming loss
                hamming = hamming_loss(y_test, y_pred)
                st.metric("Hamming Loss", f"{hamming:.4f}")
                
                # ROC AUC (if probabilities available)
                try:
                    y_pred_proba = st.session_state.model.predict_proba(X_test_tfidf)
                    # OneVsRestClassifier returns a list of arrays, one per label
                    roc_aucs = []
                    for i in range(len(target_labels)):
                        try:
                            # Get probability of positive class for this label
                            if isinstance(y_pred_proba, list) and len(y_pred_proba) > i:
                                proba_positive = y_pred_proba[i][:, 1] if y_pred_proba[i].ndim > 1 else y_pred_proba[i]
                                auc = roc_auc_score(y_test.iloc[:, i], proba_positive)
                                roc_aucs.append(auc)
                        except Exception as e:
                            pass
                    if roc_aucs:
                        avg_roc_auc = np.mean(roc_aucs)
                        st.metric("Average ROC AUC", f"{avg_roc_auc:.4f}")
                except Exception as e:
                    st.info("ROC AUC not available (model doesn't support predict_proba or error occurred)")
                
                # Per-label metrics
                st.subheader("📋 Per-Label Metrics")
                metrics_data = []
                for i, label in enumerate(target_labels):
                    label_f1 = f1_score(y_test.iloc[:, i], y_pred[:, i])
                    label_acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
                    metrics_data.append({
                        'Label': label.replace('_', ' ').title(),
                        'F1-Score': f"{label_f1:.4f}",
                        'Accuracy': f"{label_acc:.4f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Classification report
                st.subheader("📄 Detailed Classification Report")
                report = classification_report(y_test, y_pred, target_names=target_labels)
                st.text(report)
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                st.exception(e)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app trains a multi-label toxicity classifier using TF-IDF vectorization "
    "and OneVsRestClassifier with LogisticRegression. It can classify comments into "
    "six toxicity categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate."
)
