import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
from datetime import datetime
import json
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 Memulai Fine-Tuning IndoBERT...")

# === LOAD DATASET ===
print("📂 Memuat dataset...")
df = pd.read_csv('dataset/dataset_training.csv')
print(f"✅ Dataset loaded: {len(df)} rows")

# === PREPROCESSING ===
print("🔧 Preprocessing data...")

# Cek distribusi kelas
intent_counts = df['intent'].value_counts()
print("📊 Distribusi kelas:")
print(intent_counts)

# Identifikasi kelas dengan sampel sedikit
min_samples = intent_counts.min()
classes_with_one_sample = intent_counts[intent_counts == 1].index.tolist()

print(f"⚠️  Kelas dengan hanya 1 sampel: {len(classes_with_one_sample)}")
if classes_with_one_sample:
    print(f"   {classes_with_one_sample}")

# Filter kelas yang memiliki minimal 2 sampel
df_filtered = df[df['intent'].isin(intent_counts[intent_counts >= 2].index)].copy()
print(f"📊 Setelah filtering: {len(df_filtered)} rows dari {len(df)}")
print(f"🎯 Kelas tersisa: {df_filtered['intent'].nunique()} dari {df['intent'].nunique()}")

# Encode labels
le = LabelEncoder()
df_filtered['label'] = le.fit_transform(df_filtered['intent'])
num_classes = len(le.classes_)

print(f"🎯 Jumlah kelas setelah filtering: {num_classes}")
print(f"🏷️ Daftar intent: {list(le.classes_)}")

# Split data - tanpa stratify karena sudah di-filter
train_df, val_df = train_test_split(
    df_filtered, 
    test_size=0.2, 
    random_state=42
    # Tidak pakai stratify karena sudah handle imbalance
)

print(f"📊 Train: {len(train_df)}, Validation: {len(val_df)}")

# === LOAD TOKENIZER AND MODEL ===
print("📦 Loading IndoBERT model dan tokenizer...")

# Try different IndoBERT models in order of preference
model_candidates = [
    "indobenchmark/indobert-lite-base-p1",
    "cahya/bert-base-indonesian-522M",
    "indolem/indobert-base-uncased",
    "bert-base-multilingual-uncased"  # fallback
]

tokenizer = None
model_name = None

for candidate in model_candidates:
    try:
        print(f"🔄 Mencoba model: {candidate}")
        tokenizer = AutoTokenizer.from_pretrained(candidate)
        model_name = candidate
        print(f"✅ Berhasil load tokenizer: {model_name}")
        break
    except Exception as e:
        print(f"❌ Gagal load {candidate}: {str(e)[:100]}...")
        continue

if tokenizer is None:
    raise Exception("❌ Tidak ada model yang berhasil di-load!")

# Load model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    id2label={i: label for i, label in enumerate(le.classes_)},
    label2id={label: i for i, label in enumerate(le.classes_)}
)

print(f"✅ Model loaded: {model_name}")
print(f"📊 Model architecture: {model.config.model_type}")
print(f"🎯 Number of labels: {model.config.num_labels}")

# === TOKENIZE DATASET ===
print("🔤 Tokenizing dataset...")

def tokenize_function(examples):
    return tokenizer(
        examples['pattern'], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df[['pattern', 'label']])
val_dataset = Dataset.from_pandas(val_df[['pattern', 'label']])

# Tokenize
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_val = tokenized_val.rename_column("label", "labels")

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("✅ Dataset tokenized!")

# === TRAINING ARGUMENTS ===
print("⚙️ Setting up training arguments...")

training_args = TrainingArguments(
    output_dir="./bert_model",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Kurangi epoch untuk testing
    per_device_train_batch_size=8,  # Kurangi batch size
    per_device_eval_batch_size=8,
    warmup_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",  # Gunakan eval_strategy bukan evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    report_to=[],  # Gunakan list kosong
    push_to_hub=False,
    remove_unused_columns=False,  # Tambahan untuk kompatibilitas
)

# === METRICS FUNCTION ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# === TRAINER ===
print("🎯 Setting up trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# === TRAINING ===
print("\n🔥 Memulai training IndoBERT...")
print("⏰ Ini mungkin memakan waktu 5-15 menit...")

train_result = trainer.train()

print("✅ Training selesai!")

# === EVALUATION ===
print("\n📊 Evaluating model...")
eval_results = trainer.evaluate()
print(f"📈 Evaluation results: {eval_results}")

# === SAVE MODEL ===
print("💾 Menyimpan model...")

# Save fine-tuned model
trainer.save_model("./bert_fine_tuned")
tokenizer.save_pretrained("./bert_fine_tuned")

# Save label encoder
import pickle
with open('./bert_fine_tuned/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save training info
training_info = {
    "model_name": model_name,
    "fine_tuned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "original_dataset_size": len(df),
    "filtered_dataset_size": len(df_filtered),
    "original_classes": df['intent'].nunique(),
    "filtered_classes": num_classes,
    "excluded_classes": classes_with_one_sample,
    "training_epochs": 10,
    "final_loss": train_result.training_loss,
    "eval_accuracy": eval_results.get("eval_accuracy", 0),
    "classes": list(le.classes_)
}

with open('./bert_fine_tuned/training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("✅ Model disimpan di folder: bert_fine_tuned/")

# === TEST PREDICTION ===
print("\n🧪 Testing fine-tuned model...")

# Reload model untuk testing
test_model = AutoModelForSequenceClassification.from_pretrained("./bert_fine_tuned")
test_tokenizer = AutoTokenizer.from_pretrained("./bert_fine_tuned")

# Test dengan beberapa contoh
test_texts = [
    "kapan kantor pajak buka",
    "cara membuat nib",
    "izin operasional sekolah",
    "buat kartu kuning",
    "info bpjs kesehatan"
]

print("\n📋 Test Predictions:")
for text in test_texts:
    inputs = test_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = test_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    intent = le.inverse_transform([predicted_class])[0]
    
    print(f"💬 '{text}'")
    print(f"   🎯 Intent: {intent}")
    print(f"   📊 Confidence: {confidence:.3f}")
    print()

print("\n🎉 Fine-tuning IndoBERT selesai!")
print("📁 Model disimpan di: bert_fine_tuned/")
print(f"📊 Kelas yang digunakan: {num_classes} dari {df['intent'].nunique()} total")
if classes_with_one_sample:
    print(f"⚠️  Kelas yang di-exclude (hanya 1 sampel): {classes_with_one_sample}")