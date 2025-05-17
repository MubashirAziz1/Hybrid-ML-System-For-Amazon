import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

    def prepare_data(self, X_train, y_train, X_test, y_test, batch_size=16):
        train_dataset = ReviewDataset(
            X_train['cleaned_review'].values,
            y_train.values,
            self.tokenizer
        )
        test_dataset = ReviewDataset(
            X_test['cleaned_review'].values,
            y_test.values,
            self.tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )

        return train_loader, test_loader

    def train(self, train_loader, epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.numpy())

        f1 = f1_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)

        return {
            'f1_score': f1,
            'accuracy': accuracy
        }

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.softmax(outputs.logits, dim=1)
            return predictions.cpu().numpy()[0]

if __name__ == "__main__":
    from data_prep import prepare_data
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        '../Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    
    # Initialize and train model
    model = TextModel()
    train_loader, test_loader = model.prepare_data(X_train, y_train, X_test, y_test)
    
    print("Training text model...")
    model.train(train_loader, epochs=3)
    
    # Evaluate model
    metrics = model.evaluate(test_loader)
    print("\nModel Performance:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}") 