import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_pii import PIIDataset


# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 10
MODEL_NAME = 'distilbert-base-uncased'  # smaller and faster than full BERT
RANDOM_SEED = 42
LOAD_TRAINED_MODEL = "/home/dsi/ohadico97/homework/pii_model.pt"


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def train_model(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions), predictions

def create_train(csv_path):
    # load data
    df = pd.read_csv(csv_path)
    texts = df['text'].values
    targets = df['target'].values
    
    # split data
    train_texts, val_texts, train_targets, val_targets = train_test_split(
        texts, targets, test_size=0.2, random_state=RANDOM_SEED, stratify=targets
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # True or False
    model = model.to(device)
    
    train_dataset = PIIDataset(train_texts, train_targets, tokenizer, MAX_LEN)
    val_dataset = PIIDataset(val_texts, val_targets, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    

    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_loss = train_model(model, train_loader, optimizer, device)
        val_accuracy, _ = evaluate_model(model, val_loader, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(model.state_dict(), LOAD_TRAINED_MODEL)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    

    model.load_state_dict(torch.load(LOAD_TRAINED_MODEL))
    return model, tokenizer