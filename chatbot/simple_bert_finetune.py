import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import json
from datetime import datetime
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    
    return avg_loss, accuracy

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    
    return avg_loss, accuracy, predictions, actual_labels

def main():
    print("🚀 Memulai Fine-Tuning BERT Sederhana...")
    
    # Load dataset
    print("📂 Memuat dataset...")
    df = pd.read_csv('dataset/dataset_training.csv')
    print(f"✅ Dataset loaded: {len(df)} rows")
    
    # Filter kelas dengan minimal 2 sampel
    intent_counts = df['intent'].value_counts()
    classes_to_keep = intent_counts[intent_counts >= 2].index
    df_filtered = df[df['intent'].isin(classes_to_keep)].copy()
    
    print(f"📊 Setelah filtering: {len(df_filtered)} rows")
    print(f"🎯 Kelas: {df_filtered['intent'].nunique()}")
    
    # Encode labels
    le = LabelEncoder()
    df_filtered['label'] = le.fit_transform(df_filtered['intent'])
    num_classes = len(le.classes_)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_filtered['pattern'].values,
        df_filtered['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df_filtered['label'].values if intent_counts.min() >= 2 else None
    )
    
    print(f"📊 Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Load model dan tokenizer
    model_name = "cahya/bert-base-indonesian-522M"
    print(f"📦 Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    
    model.to(device)
    
    # Create datasets
    train_dataset = ChatbotDataset(train_texts, train_labels, tokenizer)
    val_dataset = ChatbotDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training loop
    num_epochs = 3
    best_accuracy = 0
    
    print(f"\n🔥 Memulai training untuk {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss, val_acc, _, _ = eval_model(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'bert_simple_best.pth')
            print(f"💾 Saved best model with accuracy: {val_acc:.4f}")
    
    # Save final results
    print("\n💾 Menyimpan hasil akhir...")
    
    # Save tokenizer dan model
    os.makedirs('bert_simple_finetuned', exist_ok=True)
    tokenizer.save_pretrained('bert_simple_finetuned')
    model.save_pretrained('bert_simple_finetuned')
    
    # Save label encoder
    import pickle
    with open('bert_simple_finetuned/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save training info
    info = {
        "model_name": model_name,
        "num_classes": num_classes,
        "classes": le.classes_.tolist(),
        "best_accuracy": best_accuracy,
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_filtered)
    }
    
    with open('bert_simple_finetuned/info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✅ Fine-tuning selesai!")
    print(f"🎯 Best accuracy: {best_accuracy:.4f}")
    print("📁 Model disimpan di: bert_simple_finetuned/")

if __name__ == "__main__":
    main()