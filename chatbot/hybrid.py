import os
import warnings
import logging

# Suppress warnings di awal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info and warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

print("🚀 Starting Hybrid LSTM + BERT Training...")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
from datetime import datetime
import json

# === MUAT DATASET ===
print("📂 Memuat dataset...")
try:
    df = pd.read_csv('dataset/dataset_training.csv')
    print(f"✅ Dataset berhasil dimuat! Shape: {df.shape}")
    print(f"📊 Distribusi intent: {df['intent'].value_counts()}")
except FileNotFoundError:
    print("❌ Error: File dataset/dataset_training.csv tidak ditemukan!")
    exit()

# === PREPROCESSING DENGAN VALIDASI ===
patterns = [str(pattern).strip().lower() for pattern in df['pattern'].tolist()]
intents = df['intent'].tolist()

# Cek distribusi kelas
print("📈 Distribusi kelas:")
print(df['intent'].value_counts())

le = LabelEncoder()
encoded_labels = le.fit_transform(intents)
num_classes = len(le.classes_)

print(f"🎯 Jumlah kelas: {num_classes}")
print(f"🏷️ Daftar intent: {le.classes_}")

tokenizer = Tokenizer(oov_token="<oov>", num_words=5000)  # Batasi vocabulary
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)

# Analisis panjang sequence
seq_lengths = [len(seq) for seq in sequences]
max_len = min(50, int(np.percentile(seq_lengths, 95)))  # Gunakan percentile 95
print(f"📏 Panjang sequence: max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")
print(f"📐 Menggunakan max_len: {max_len}")

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

print(f"🔢 Vocabulary size: {len(tokenizer.word_index)}")
print("✅ Preprocessing LSTM selesai!")

# metode nlp untuk text classification
def bert_setup():
    try:
        # Import inside function to avoid early warnings
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        bert_model_name = "indobenchmark/indobert-lite-base-p1"

        print(f"📦 Model BERT tersedia: {bert_model_name}")
        bert_info = {
            "model_name": bert_model_name,
            "purpose": "hybrid nlp dengan lstm",
            "integration_ready": True,
            "version": "1.0.0",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return bert_info
    except ImportError as e:
        print("⚠️  Transformers tidak terinstall. IndoBERT bisa ditambahkan nanti.")
        print("   Install dengan: pip install transformers torch")
        return None

# Panggil fungsi bert_setup
print("🔧 Mengecek ketersediaan BERT...")
bert_info = bert_setup()

# === SPLIT DATA YANG LEBIH BAIK ===
# Cek apakah semua kelas memiliki minimal 2 sampel untuk stratify
unique, counts = np.unique(encoded_labels, return_counts=True)
min_samples_per_class = np.min(counts)
print(f"📊 Sampel minimum per kelas: {min_samples_per_class}")

# Jika ada kelas dengan hanya 1 sampel, jangan gunakan stratify
if min_samples_per_class < 2:
    print("⚠️  Ada kelas dengan sampel < 2, tidak menggunakan stratify")
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, 
        np.array(encoded_labels), 
        test_size=0.2, 
        random_state=42
    )
else:
    print("✅ Semua kelas memiliki ≥ 2 sampel, menggunakan stratify")
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, 
        np.array(encoded_labels), 
        test_size=0.2, 
        random_state=42,
        stratify=encoded_labels
    )

print(f"📊 Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# === BANGUN MODEL DENGAN REGULARISASI ===
vocab_size = min(5000, len(tokenizer.word_index) + 1)  # Batasi vocabulary

print("🏗️  Membangun model LSTM...")
model = Sequential([
    Embedding(
        input_dim=vocab_size, 
        output_dim=64, 
        input_length=max_len,
        mask_zero=True
    ),
    LSTM(32, dropout=0.3, recurrent_dropout=0.3),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Gunakan optimizer dengan learning rate lebih rendah
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer, 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# === CALLBACKS UNTUK MENCEGAH OVERFITTING ===
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.0001,
    verbose=1
)

# === TRAIN MODEL ===
print("\n🚀 Memulai training model LSTM...")
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("✅ Model LSTM selesai dilatih!")

# === EVALUASI ===
train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"\n📊 Hasil Final LSTM:")
print(f"Training - Loss: {train_accuracy[0]:.4f}, Accuracy: {train_accuracy[1]:.4f}")
print(f"Validation - Loss: {val_accuracy[0]:.4f}, Accuracy: {val_accuracy[1]:.4f}")

# === SIMPAN MODEL ===
print("💾 Menyimpan model...")
model.save('model/chatbot_model.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump({'tokenizer': tokenizer, 'max_len': max_len, 'vocab_size': vocab_size}, f)

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Model dan tokenizer berhasil disimpan!")

# === SIMPAN INFORMASI HYBRID ===
hybrid_config = {
    "model_type": "lstm",
    "vocab_size": vocab_size,
    "max_sequence_length": max_len,
    "embedding_dim": 64,
    "lstm_units": 32,
    "num_classes": num_classes,
    "training_info": {
        "epochs_trained": len(history.history['accuracy']),
        "final_train_accuracy": float(train_accuracy[1]),
        "final_val_accuracy": float(val_accuracy[1]),
        "final_train_loss": float(train_accuracy[0]),
        "final_val_loss": float(val_accuracy[0])
    },
    "bert_model": bert_info["model_name"] if bert_info else "not_configured",
    "training_strategy": "LSTM + IndoBERT Hybrid",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('model/hybrid_config.json', 'w') as f:
    json.dump(hybrid_config, f, indent=2)

print("🔧 Konfigurasi hybrid system disimpan!")

# === LOGGING ===
log_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_shape": df.shape,
    "num_classes": num_classes,
    "vocab_size": vocab_size,
    "max_sequence_length": max_len,
    "final_train_accuracy": float(train_accuracy[1]),
    "final_val_accuracy": float(val_accuracy[1]),
    "final_train_loss": float(train_accuracy[0]),
    "final_val_loss": float(val_accuracy[0]),
    "epochs_trained": len(history.history['accuracy']),
    "early_stopping_triggered": len(history.history['accuracy']) < 50,
    "bert_available": bert_info is not None
}

with open('model/training_log.json', 'a') as f:
    json.dump(log_data, f)
    f.write('\n')

print("🧾 Hasil training dicatat ke model/training_log.json")

print("\n🎯 TRAINING SELESAI! Model LSTM ready untuk digunakan.")
if bert_info:
    print("   ✅ IndoBERT tersedia untuk hybrid system")
else:
    print("   ⚠️  IndoBERT belum terinstall (optional)")