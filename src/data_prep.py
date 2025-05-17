import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_and_sample_data(file_path, sample_size=1000):
    """Load data and take a sample for initial testing."""
    df = pd.read_csv(file_path)
    return df.sample(n=min(sample_size, len(df)), random_state=42)

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

def prepare_data(file_path, test_size=0.2, sample_size=1000):
    """Main function to prepare the data."""
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
    X_train, X_test, y_train, y_test, df = prepare_data(
        '../Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    
    print("Data preparation completed!")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Return risk distribution: {df['return_risk'].value_counts(normalize=True)}") 