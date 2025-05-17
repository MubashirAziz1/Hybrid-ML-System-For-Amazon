import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import shap

class TabularModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.feature_names = ['Rating', 'normalized_price', 'review_length']

    def prepare_features(self, X):
        """Prepare tabular features for the model."""
        return X[self.feature_names]

    def train(self, X_train, y_train):
        """Train the XGBoost model."""
        X_train_features = self.prepare_features(X_train)
        self.model.fit(X_train_features, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        X_test_features = self.prepare_features(X_test)
        predictions = self.model.predict(X_test_features)
        
        f1 = f1_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            'f1_score': f1,
            'accuracy': accuracy,
            'predictions': predictions
        }

    def get_feature_importance(self, X):
        """Get feature importance using SHAP values."""
        X_features = self.prepare_features(X)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_features)
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        
        return feature_importance.sort_values('importance', ascending=False)

    def predict(self, X):
        """Make predictions for new data."""
        X_features = self.prepare_features(X)
        return self.model.predict_proba(X_features)

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