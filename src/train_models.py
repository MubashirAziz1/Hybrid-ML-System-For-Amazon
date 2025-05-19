import pandas as pd
import numpy as np
import joblib
import os
import shutil
import json
from data_prep import prepare_data
from text_model import TextModel
from tabular_model import TabularModel
from hybrid_model import HybridModel

def train_and_save_models(sample_size=1000):
    """
    Train all models and save them to disk.
    
    Args:
        sample_size (int): Number of samples to use for training
    """
    print("\nLoading and preparing data...")
    try:
        X_train, X_test, y_train, y_test, df = prepare_data(
            data_path='Amazon_reviews.csv',
            sample_size=sample_size
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train and save tabular model
        print("\nTraining tabular model...")
        tabular_model = TabularModel()
        tabular_model.train(X_train, y_train)
        tabular_model.save('models/tabular_model.joblib')
        
        # Train and save text model
        print("\nTraining text model...")
        text_model = TextModel()
        text_model.train(X_train, y_train)
        text_model.save('models/text_model.joblib')
        
        # Train and save hybrid model
        print("\nTraining hybrid model...")
        hybrid_model = HybridModel()
        hybrid_model.train(X_train, y_train, tabular_model, text_model)
        hybrid_model.save('models/hybrid_model.joblib')
        
        # Calculate and save metrics
        print("\nCalculating metrics...")
        metrics = {
            'tabular': tabular_model.evaluate(X_test, y_test),
            'text': text_model.evaluate(X_test, y_test),
            'hybrid': hybrid_model.evaluate(X_test, y_test, tabular_model, text_model)
        }
        
        # Save metrics
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nTraining complete! Models and metrics saved to 'models/' directory.")
        
        # Print performance summary
        print("\nModel Performance Summary:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()} Model:")
            for metric, value in model_metrics.items():
                print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_models() 