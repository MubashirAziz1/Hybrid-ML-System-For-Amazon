import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from text_model import TextModel
from tabular_model import TabularModel
import xgboost as xgb
import joblib

class HybridModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.tabular_features = ['normalized_price', 'review_length', 'Rating']
        self.text_feature = 'cleaned_review'
    
    def train(self, X, y, tabular_model, text_model):
        """
        Train the hybrid model using both tabular and text features.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            y (pd.Series): Target variable
            tabular_model: Trained tabular model
            text_model: Trained text model
        """
        # Get predictions from both models
        tabular_preds = tabular_model.predict(X)
        text_preds = text_model.predict(X)
        
        # Combine predictions
        X_hybrid = np.column_stack([tabular_preds, text_preds])
        
        # Train the hybrid model
        self.model.fit(X_hybrid, y)
    
    def predict(self, X, tabular_model, text_model):
        """
        Make predictions using the hybrid model.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            tabular_model: Trained tabular model
            text_model: Trained text model
            
        Returns:
            np.array: Predicted probabilities
        """
        # Get predictions from both models
        tabular_preds = tabular_model.predict(X)
        text_preds = text_model.predict(X)
        
        # Combine predictions
        X_hybrid = np.column_stack([tabular_preds, text_preds])
        
        # Make final prediction
        return self.model.predict_proba(X_hybrid)[:, 1]
    
    def evaluate(self, X, y, tabular_model, text_model):
        """
        Evaluate the hybrid model performance.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            y (pd.Series): True labels
            tabular_model: Trained tabular model
            text_model: Trained text model
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.predict(X, tabular_model, text_model)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
    
    def save(self, path):
        """Save the model to disk."""
        model_state = {
            'model': self.model,
            'tabular_features': self.tabular_features,
            'text_feature': self.text_feature
        }
        joblib.dump(model_state, path, protocol=4)
    
    def load(self, path):
        """Load the model from disk."""
        model_state = joblib.load(path)
        self.model = model_state['model']
        self.tabular_features = model_state['tabular_features']
        self.text_feature = model_state['text_feature']

class HybridPredictor:
    def __init__(self):
        self.text_model = TextModel()
        self.tabular_model = TabularModel()
        self.hybrid_model = HybridModel()
        
    def train(self, X_train, y_train, X_test, y_test):
        """Train the hybrid model."""
        # Train text model
        print("Training text model...")
        self.text_model.train(X_train, y_train)
        
        # Train tabular model
        print("\nTraining tabular model...")
        self.tabular_model.train(X_train, y_train)
        
        # Train hybrid model
        print("\nTraining hybrid model...")
        self.hybrid_model.train(X_train, y_train, self.tabular_model, self.text_model)
        
    def evaluate(self, X_test, y_test):
        """Evaluate the hybrid model."""
        metrics = self.hybrid_model.evaluate(X_test, y_test, self.tabular_model, self.text_model)
        print("\nHybrid Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        return metrics
    
    def predict(self, X):
        """Make predictions using the hybrid model."""
        return self.hybrid_model.predict(X, self.tabular_model, self.text_model)

if __name__ == "__main__":
    from data_prep import prepare_data
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        'Amazon_reviews.csv',
        sample_size=1000
    )
    
    # Initialize and train hybrid model
    model = HybridPredictor()
    print("Training hybrid model...")
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nHybrid Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 