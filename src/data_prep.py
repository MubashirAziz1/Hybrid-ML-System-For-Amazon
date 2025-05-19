import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
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

def prepare_data(data_path='/kaggle/working/Amazon.txt', sample_size=10000):
    """
    Load and prepare the dataset for model training.
    
    Args:
        data_path (str): Path to the dataset file
        sample_size (int): Number of samples to use for training
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df)
    """
    try:
        # Load data
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Clean text
        print("Cleaning text...")
        df['cleaned_review'] = df['Reviews'].apply(clean_text)
        
        # Create features
        print("Creating features...")
        # Calculate review length
        df['review_length'] = df['Reviews'].str.len()
        
        # Normalize price
        df['normalized_price'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
        
        # Create target variable (return risk)
        df['return_risk'] = (df['Rating'] <= 3).astype(int)
        
        # Prepare all features
        X = df[['cleaned_review', 'normalized_price', 'review_length', 'Rating']]
        y = df['return_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, df
        
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Test data preparation
    try:
        X_train, X_test, y_train, y_test, df = prepare_data()
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print("\nFeatures available:")
        print(X_train.columns.tolist())
    except Exception as e:
        print(f"Error in data preparation: {str(e)}") 