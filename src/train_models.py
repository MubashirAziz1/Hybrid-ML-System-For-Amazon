import pandas as pd
import numpy as np
import joblib
import os
from data_prep import prepare_data
from text_model import TextModel
from tabular_model import TabularModel
from hybrid_model import HybridPredictor

def train_and_save_models(data_path, sample_size=1000):
    """
    Train all models and save them to disk.
    
    Args:
        data_path (str): Path to the data file
        sample_size (int): Number of samples to use for training
    """
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, df = prepare_data(data_path, sample_size)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train and save text model
    print("Training text model...")
    text_model = TextModel()
    text_model.train(text_model.prepare_data(X_train, y_train, X_test, y_test)[0])
    joblib.dump(text_model, 'models/text_model.joblib')
    
    # Train and save tabular model
    print("Training tabular model...")
    tabular_model = TabularModel()
    tabular_model.train(X_train, y_train)
    joblib.dump(tabular_model, 'models/tabular_model.joblib')
    
    # Train and save hybrid model
    print("Training hybrid model...")
    hybrid_model = HybridPredictor()
    hybrid_model.train(X_train, y_train, X_test, y_test)
    joblib.dump(hybrid_model, 'models/hybrid_model.joblib')
    
    # Save test data for evaluation
    print("Saving test data...")
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'df': df
    }
    joblib.dump(test_data, 'models/test_data.joblib')
    
    # Calculate and save model metrics
    print("Calculating model metrics...")
    text_metrics = text_model.evaluate(text_model.prepare_data(X_train, y_train, X_test, y_test)[1])
    tabular_metrics = tabular_model.evaluate(X_test, y_test)
    hybrid_metrics = hybrid_model.evaluate(X_test, y_test)
    
    metrics = {
        'text_metrics': text_metrics,
        'tabular_metrics': tabular_metrics,
        'hybrid_metrics': hybrid_metrics
    }
    joblib.dump(metrics, 'models/model_metrics.joblib')
    
    print("Training complete! All models and metrics have been saved to the 'models' directory.")

if __name__ == "__main__":
    train_and_save_models('data/Amazon_Unlocked_Mobile.csv') 