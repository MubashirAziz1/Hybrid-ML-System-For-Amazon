import pandas as pd
import numpy as np
import joblib
import os
from data_prep import prepare_data
from text_model import TextModel
from tabular_model import TabularModel
from hybrid_model import HybridPredictor

def train_and_save_models(data_path='data/Amazon_Unlocked_Mobile.csv', sample_size=1000):
    """
    Train all models and save them to disk.
    
    Args:
        data_path (str): Path to the data file
        sample_size (int): Number of samples to use for training
    """
    print("Loading and preparing data...")
    try:
        X_train, X_test, y_train, y_test, df = prepare_data(data_path, sample_size)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure you have the dataset in one of these ways:")
        print("1. Place the dataset at 'data/Amazon_Unlocked_Mobile.csv'")
        print("2. Configure Kaggle API with your credentials")
        print("3. Download manually from: https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
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
        
        print("\nTraining complete! All models and metrics have been saved to the 'models' directory.")
        print("\nModel Performance Summary:")
        print(f"Text Model - F1 Score: {text_metrics['f1_score']:.3f}, Accuracy: {text_metrics['accuracy']:.3f}")
        print(f"Tabular Model - F1 Score: {tabular_metrics['f1_score']:.3f}, Accuracy: {tabular_metrics['accuracy']:.3f}")
        print(f"Hybrid Model - F1 Score: {hybrid_metrics['f1_score']:.3f}, Accuracy: {hybrid_metrics['accuracy']:.3f}")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Please check your data and model configurations.")

if __name__ == "__main__":
    train_and_save_models() 