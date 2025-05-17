import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import kaggle
from pathlib import Path

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def download_kaggle_dataset():
    """Download the dataset from Kaggle if not present."""
    data_dir = Path('data')
    data_file = data_dir / 'Amazon_Unlocked_Mobile.csv'
    
    if not data_file.exists():
        print("Dataset not found. Downloading from Kaggle...")
        try:
            # Create data directory if it doesn't exist
            data_dir.mkdir(exist_ok=True)
            
            # Download dataset
            kaggle.api.dataset_download_files(
                'PromptCloudHQ/amazon-reviews-unlocked-mobile-phones',
                path='data',
                unzip=True
            )
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            print("\nPlease download the dataset manually from:")
            print("https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones")
            print("\nAnd place it in the 'data' directory as 'Amazon_Unlocked_Mobile.csv'")
            raise

def load_and_sample_data(file_path, sample_size=1000):
    """Load data and take a sample for initial testing."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} rows")
        return df.sample(n=min(sample_size, len(df)), random_state=42)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {file_path}")
        print("Attempting to download from Kaggle...")
        download_kaggle_dataset()
        # Try loading again after download
        df = pd.read_csv(file_path)
        return df.sample(n=min(sample_size, len(df)), random_state=42)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_features(df):
    """Extract features from the dataset."""
    # Clean text
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    
    # Create return risk label
    return_keywords = ['return', 'defective', 'broken', 'damaged', 'not working', 'poor quality']
    df['return_risk'] = df.apply(lambda x: 1 if (
        x['Rating'] <= 2 or 
        any(keyword in x['cleaned_review'].lower() for keyword in return_keywords)
    ) else 0, axis=1)
    
    # Extract text length as a feature
    df['review_length'] = df['cleaned_review'].apply(len)
    
    # Normalize price
    df['normalized_price'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
    
    return df

def prepare_data(file_path='data/Amazon_Unlocked_Mobile.csv', test_size=0.2, sample_size=1000):
    """Main function to prepare the data."""
    # Ensure data file exists
    if not os.path.exists(file_path):
        download_kaggle_dataset()
    
    # Load and sample data
    df = load_and_sample_data(file_path, sample_size)
    
    # Extract features
    df = extract_features(df)
    
    # Split features and target
    X = df[['cleaned_review', 'Rating', 'normalized_price', 'review_length']]
    y = df['return_risk']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, df

if __name__ == "__main__":
    # Test the data preparation
    try:
        X_train, X_test, y_train, y_test, df = prepare_data(
            sample_size=1000
        )
        
        print("\nData preparation completed!")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Return risk distribution: {df['return_risk'].value_counts(normalize=True)}")
        
        # Display sample of processed data
        print("\nSample of processed data:")
        print(df[['Reviews', 'Rating', 'Price', 'return_risk']].head())
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Installed the Kaggle API: pip install kaggle")
        print("2. Configured your Kaggle credentials")
        print("3. Or manually downloaded the dataset to the 'data' directory") 