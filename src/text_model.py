import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
from transformers import TrainingArguments, Trainer

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class TextModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        ).to(self.device)
        self.text_feature = 'cleaned_review'
    
    def train(self, X, y):
        """
        Train the text model using text features.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            y (pd.Series): Target variable
        """
        # Select only text features
        X_text = X[self.text_feature]
        
        # Prepare dataset
        train_dataset = self._prepare_dataset(X_text, y)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            no_cuda=not torch.cuda.is_available()  # Disable CUDA if not available
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset
        )
        
        # Train the model
        trainer.train()
    
    def predict(self, X):
        """
        Make predictions using the text model.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            
        Returns:
            np.array: Predicted probabilities
        """
        # Select only text features
        X_text = X[self.text_feature]
        
        # Prepare dataset
        dataset = self._prepare_dataset(X_text)
        
        # Get predictions
        predictions = []
        for batch in DataLoader(dataset, batch_size=32):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=1)
                predictions.extend(probs[:, 1].cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Args:
            X (pd.DataFrame): Input features including both text and numerical features
            y (pd.Series): True labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Select only text features
        X_text = X[self.text_feature]
        
        # Get predictions
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
    
    def _prepare_dataset(self, texts, labels=None):
        """Prepare dataset for the model."""
        # Convert texts to list if it's a pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Create dataset
        if labels is not None:
            # Convert labels to list if it's a pandas Series
            if isinstance(labels, pd.Series):
                labels = labels.tolist()
            return ReviewDataset(encodings, labels)
        return ReviewDataset(encodings)
    
    def save(self, path):
        """Save the model to disk."""
        # Save model state dict and configuration
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'device': str(self.device),
            'text_feature': self.text_feature
        }
        torch.save(model_state, path, _use_new_zipfile_serialization=True)
    
    def load(self, path):
        """Load the model from disk."""
        # Load model state dict and configuration
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.model.to(self.device)
        self.text_feature = model_state['text_feature']

if __name__ == "__main__":
    from data_prep import prepare_data
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        '../Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    
    # Initialize and train model
    model = TextModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 