# Toxicity Classification System

A Streamlit application for training and using a multi-label toxicity classifier. This app allows you to train a model using your datasets and then use it to classify comments for various types of toxicity.

## Features

- **Data Loading**: Upload or use existing datasets (train.csv.zip, raw_test.csv, test_labels.csv)
- **Model Training**: Train a multi-label classifier using TF-IDF vectorization and OneVsRestClassifier with LogisticRegression
- **Prediction**: Test the trained model on new comments
- **Model Evaluation**: Evaluate model performance with comprehensive metrics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate through the app using the sidebar:
   - **Data Loading**: Load your training and test datasets
   - **Model Training**: Configure and train the model
   - **Prediction**: Test the model on new comments
   - **Model Evaluation**: View performance metrics

## Dataset Format

### Training Data (train.csv.zip)
Should contain:
- `id`: Unique identifier
- `comment_text`: Text to classify
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: Binary labels (0 or 1, or -1 for not scored)

### Test Data
- `raw_test.csv`: Contains `id` and `comment_text`
- `test_labels.csv`: Contains `id` and the six toxicity labels

## Model Details

- **Text Preprocessing**: Lowercase conversion, removal of punctuation, numbers, and special characters
- **Vectorization**: TF-IDF with configurable max features (default: 5000)
- **Model**: OneVsRestClassifier with LogisticRegression
- **Target Labels**: toxic, severe_toxic, obscene, threat, insult, identity_hate

## Saved Models

Trained models are automatically saved to `content/saved_models/`:
- `tfidf_vectorizer.pkl`: The TF-IDF vectorizer
- `linear_svc_model.pkl`: The trained classifier (note: despite the filename, it uses LogisticRegression)

## Notes

- The app filters out rows where all target labels are -1
- Text cleaning matches the preprocessing used during training
- The model supports both training from scratch and loading pre-trained models
