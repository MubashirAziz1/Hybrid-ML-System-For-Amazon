import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from text_model import TextModel
from tabular_model import TabularModel

class HybridModel(nn.Module):
    def __init__(self, text_model, tabular_model):
        super().__init__()
        self.text_model = text_model
        self.tabular_model = tabular_model
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(4, 16),  # 2 (text) + 2 (tabular) probabilities
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, text_input, tabular_input):
        # Get predictions from both models
        text_probs = self.text_model.predict(text_input)
        tabular_probs = self.tabular_model.predict(tabular_input)
        
        # Combine predictions
        combined = np.concatenate([text_probs, tabular_probs], axis=1)
        combined = torch.FloatTensor(combined).to(self.text_model.device)
        
        # Pass through fusion layer
        output = self.fusion(combined)
        return output

class HybridPredictor:
    def __init__(self):
        self.text_model = TextModel()
        self.tabular_model = TabularModel()
        self.hybrid_model = HybridModel(self.text_model, self.tabular_model)
        
    def train(self, X_train, y_train, X_test, y_test, epochs=3):
        """Train the hybrid model."""
        # Train text model
        print("Training text model...")
        train_loader, test_loader = self.text_model.prepare_data(
            X_train, y_train, X_test, y_test
        )
        self.text_model.train(train_loader, epochs=epochs)
        
        # Train tabular model
        print("\nTraining tabular model...")
        self.tabular_model.train(X_train, y_train)
        
        # Train fusion layer
        print("\nTraining fusion layer...")
        self.train_fusion_layer(X_train, y_train, X_test, y_test)
        
    def train_fusion_layer(self, X_train, y_train, X_test, y_test):
        """Train the fusion layer of the hybrid model."""
        optimizer = torch.optim.Adam(self.hybrid_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Convert labels to tensor
        y_train_tensor = torch.LongTensor(y_train.values).to(self.text_model.device)
        
        self.hybrid_model.train()
        for epoch in range(3):
            total_loss = 0
            optimizer.zero_grad()
            
            # Get predictions from both models
            text_probs = self.text_model.predict(X_train['cleaned_review'].values)
            tabular_probs = self.tabular_model.predict(X_train)
            
            # Combine predictions
            combined = np.concatenate([text_probs, tabular_probs], axis=1)
            combined = torch.FloatTensor(combined).to(self.text_model.device)
            
            # Forward pass
            outputs = self.hybrid_model.fusion(combined)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f'Epoch {epoch + 1}/3, Loss: {total_loss:.4f}')
    
    def evaluate(self, X_test, y_test):
        """Evaluate the hybrid model."""
        self.hybrid_model.eval()
        
        with torch.no_grad():
            # Get predictions from both models
            text_probs = self.text_model.predict(X_test['cleaned_review'].values)
            tabular_probs = self.tabular_model.predict(X_test)
            
            # Combine predictions
            combined = np.concatenate([text_probs, tabular_probs], axis=1)
            combined = torch.FloatTensor(combined).to(self.text_model.device)
            
            # Get final predictions
            outputs = self.hybrid_model.fusion(combined)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            f1 = f1_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            
            return {
                'f1_score': f1,
                'accuracy': accuracy,
                'predictions': predictions
            }
    
    def predict(self, X):
        """Make predictions using the hybrid model."""
        self.hybrid_model.eval()
        
        with torch.no_grad():
            # Get predictions from both models
            text_probs = self.text_model.predict(X['cleaned_review'].values)
            tabular_probs = self.tabular_model.predict(X)
            
            # Combine predictions
            combined = np.concatenate([text_probs, tabular_probs], axis=1)
            combined = torch.FloatTensor(combined).to(self.text_model.device)
            
            # Get final predictions
            outputs = self.hybrid_model.fusion(combined)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            return probabilities

if __name__ == "__main__":
    from data_prep import prepare_data
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        '../Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    
    # Initialize and train hybrid model
    model = HybridPredictor()
    print("Training hybrid model...")
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nHybrid Model Performance:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}") 