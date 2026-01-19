"""
Standalone script to train the toxicity classification model.
This script loads the datasets, preprocesses them, trains the model, and saves it.
"""
import pandas as pd
import numpy as np
import re
import joblib
import os
import zipfile
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, hamming_loss

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

def main():
    print("=" * 60)
    print("Toxicity Classification Model Training")
    print("=" * 60)
    
    # Step 1: Load training data
    print("\n[1/5] Loading training data...")
    try:
        if os.path.exists('content/train.csv.zip'):
            with zipfile.ZipFile('content/train.csv.zip', 'r') as zip_ref:
                zip_ref.extractall(tempfile.gettempdir())
                csv_path = os.path.join(tempfile.gettempdir(), 'train.csv')
                df_train = pd.read_csv(csv_path)
                print(f"   [OK] Training data loaded: {df_train.shape}")
        elif os.path.exists('content/train.csv'):
            df_train = pd.read_csv('content/train.csv')
            print(f"   ✓ Training data loaded: {df_train.shape}")
        else:
            raise FileNotFoundError("Training data not found. Expected 'content/train.csv.zip' or 'content/train.csv'")
    except Exception as e:
        print(f"   [ERROR] Error loading training data: {str(e)}")
        return
    
    # Step 2: Preprocess training data
    print("\n[2/5] Preprocessing training data...")
    try:
        # Filter out rows where all target labels are -1
        df_train_filtered = df_train[~(df_train[target_labels] == -1).all(axis=1)].copy()
        print(f"   [OK] Filtered training samples: {df_train_filtered.shape[0]:,}")
        
        # Clean text
        print("   Cleaning text...")
        df_train_filtered['comment_text'] = df_train_filtered['comment_text'].apply(clean_text)
        
        # Prepare target labels
        y_train = df_train_filtered[target_labels]
        print(f"   [OK] Preprocessing complete")
    except Exception as e:
        print(f"   [ERROR] Error preprocessing data: {str(e)}")
        return
    
    # Step 3: TF-IDF Vectorization
    print("\n[3/5] Performing TF-IDF vectorization...")
    try:
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(df_train_filtered['comment_text'])
        print(f"   [OK] TF-IDF features shape: {X_train_tfidf.shape}")
    except Exception as e:
        print(f"   [ERROR] Error in vectorization: {str(e)}")
        return
    
    # Step 4: Train model
    print("\n[4/5] Training OneVsRestClassifier with LogisticRegression...")
    try:
        base_estimator = LogisticRegression(
            solver='sag',
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        model = OneVsRestClassifier(base_estimator)
        print("   Training in progress (this may take a few minutes)...")
        model.fit(X_train_tfidf, y_train)
        print("   [OK] Model training complete")
    except Exception as e:
        print(f"   [ERROR] Error training model: {str(e)}")
        return
    
    # Step 5: Save model and vectorizer
    print("\n[5/5] Saving model and vectorizer...")
    try:
        os.makedirs('content/saved_models', exist_ok=True)
        vectorizer_path = 'content/saved_models/tfidf_vectorizer.pkl'
        model_path = 'content/saved_models/linear_svc_model.pkl'
        
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        joblib.dump(model, model_path)
        
        print(f"   [OK] Vectorizer saved to: {vectorizer_path}")
        print(f"   [OK] Model saved to: {model_path}")
    except Exception as e:
        print(f"   [ERROR] Error saving model: {str(e)}")
        return
    
    # Optional: Evaluate on test data if available
    print("\n[Optional] Evaluating on test data...")
    try:
        if os.path.exists('content/raw_test.csv') and os.path.exists('content/test_labels.csv'):
            print("   Loading test data...")
            raw_test_df = pd.read_csv('content/raw_test.csv')
            test_labels_df = pd.read_csv('content/test_labels.csv')
            df_test = pd.merge(raw_test_df, test_labels_df, on='id', how='left')
            
            # Filter and preprocess
            df_test_filtered = df_test[~(df_test[target_labels] == -1).all(axis=1)].copy()
            df_test_filtered['comment_text'] = df_test_filtered['comment_text'].apply(clean_text)
            y_test = df_test_filtered[target_labels]
            
            # Transform test data
            X_test_tfidf = tfidf_vectorizer.transform(df_test_filtered['comment_text'])
            
            # Make predictions
            y_pred = model.predict(X_test_tfidf)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            hamming = hamming_loss(y_test, y_pred)
            
            print(f"\n   Test Set Metrics:")
            print(f"   - Accuracy: {accuracy:.4f}")
            print(f"   - Micro F1-Score: {micro_f1:.4f}")
            print(f"   - Macro F1-Score: {macro_f1:.4f}")
            print(f"   - Hamming Loss: {hamming:.4f}")
        else:
            print("   Test data not found, skipping evaluation")
    except Exception as e:
        print(f"   ⚠ Evaluation skipped: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Training complete! Model is ready to use.")
    print("=" * 60)

if __name__ == "__main__":
    main()
