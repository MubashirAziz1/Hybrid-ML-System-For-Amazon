import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import shap
import joblib

class TabularModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.tabular_features = ['normalized_price', 'review_length', 'Rating']

    def train(self, X, y):
        """
        Train the tabular model using numerical features.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            y (pd.Series): Target variable
        """
        # Select only tabular features
        X_tabular = X[self.tabular_features]
        self.model.fit(X_tabular, y)

    def predict(self, X):
        """
        Make predictions using the tabular model.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            
        Returns:
            np.array: Predicted probabilities
        """
        # Select only tabular features
        X_tabular = X[self.tabular_features]
        return self.model.predict_proba(X_tabular)[:, 1]

    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            y (pd.Series): True labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Select only tabular features
        X_tabular = X[self.tabular_features]
        y_pred = self.model.predict(X_tabular)
        y_pred_proba = self.model.predict_proba(X_tabular)[:, 1]
        
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
            'tabular_features': self.tabular_features
        }
        joblib.dump(model_state, path, protocol=4)

    def load(self, path):
        """Load the model from disk."""
        model_state = joblib.load(path)
        self.model = model_state['model']
        self.tabular_features = model_state['tabular_features']

    def get_feature_importance(self, X):
        """Get feature importance using SHAP values."""
        X_features = X[self.tabular_features]
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_features)
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = pd.DataFrame({
            'feature': self.tabular_features,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        
        return feature_importance.sort_values('importance', ascending=False)

if __name__ == "__main__":
    from data_prep import prepare_data
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        '../Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    
    # Initialize and train model
    model = TabularModel()
    print("Training tabular model...")
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Performance:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Get feature importance
    feature_importance = model.get_feature_importance(X_test)
    print("\nFeature Importance:")
    print(feature_importance) 