import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === BACA FILE LOG ===
try:
    with open('model/training_log.json', 'r') as f:
        logs = [json.loads(line) for line in f.readlines()]
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("✅ Data log berhasil dimuat!\n")
except FileNotFoundError:
    print("❌ Tidak ditemukan file 'model/training_log.json'. Jalankan train_model.py dulu.")
    exit()
except json.JSONDecodeError:
    print("⚠️ File 'training_log.json' rusak atau tidak valid JSON.")
    exit()

# === TAMPILKAN TABEL REKAP ===
print("📈 Riwayat Training Model:\n")
print(df[['timestamp', 'train_accuracy', 'val_accuracy', 'train_loss', 'val_loss']].to_string(index=False))

# === PLOT GRAFIK AKURASI DAN LOSS ===
plt.figure(figsize=(10, 5))
plt.suptitle('📊 Riwayat Training Model Chatbot', fontsize=14, fontweight='bold')

# --- Grafik Akurasi ---
plt.subplot(1, 2, 1)
plt.plot(df['timestamp'], df['train_accuracy'], marker='o', label='Train Accuracy')
plt.plot(df['timestamp'], df['val_accuracy'], marker='s', label='Validation Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Tanggal Training')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# --- Grafik Loss ---
plt.subplot(1, 2, 2)
plt.plot(df['timestamp'], df['train_loss'], marker='o', label='Train Loss')
plt.plot(df['timestamp'], df['val_loss'], marker='s', label='Validation Loss')
plt.title('Loss Model')
plt.xlabel('Tanggal Training')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
