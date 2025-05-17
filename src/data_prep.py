import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def download_kaggle_dataset():
    """Download dataset from Kaggle if kaggle.json is present."""
    try:
        import kaggle
        # Check if dataset is already downloaded
        if not os.path.exists('data/Amazon_Unlocked_Mobile.csv'):
            print("Downloading dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'PromptCloudHQ/amazon-reviews-unlocked-mobile-phones',
                path='data',
                unzip=True
            )
            print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Could not download from Kaggle: {str(e)}")
        print("Please ensure you have:")
        print("1. A kaggle.json file in ~/.kaggle/")
        print("2. The dataset 'PromptCloudHQ/amazon-reviews-unlocked-mobile-phones' is accessible")
        return False

def load_data(file_path, sample_size=None):
    """Load data from local file."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text data."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ''

def prepare_data(data_path='data/Amazon_Unlocked_Mobile.csv', sample_size=1000):
    """
    Prepare data for model training.
    
    Args:
        data_path (str): Path to the data file
        sample_size (int): Number of samples to use (None for all data)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df)
    """
    # Try to download from Kaggle first
    if not os.path.exists(data_path):
        download_kaggle_dataset()
    
    # Load data
    df = load_data(data_path, sample_size)
    if df is None:
        raise FileNotFoundError(f"Could not load data from {data_path}")
    
    print("Preprocessing data...")
    # Clean text data
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    
    # Create target variable (example: return risk based on rating)
    df['return_risk'] = (df['Rating'] <= 3).astype(int)
    
    # Prepare features
    X = df[['cleaned_review', 'Rating', 'Price']]
    y = df['return_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Data preparation complete!")
    return X_train, X_test, y_train, y_test, df

if __name__ == "__main__":
    # Test data preparation
    try:
        X_train, X_test, y_train, y_test, df = prepare_data()
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
    except Exception as e:
        print(f"Error in data preparation: {str(e)}") 