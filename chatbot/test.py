# test_model.py
import os
import pickle

print("🔍 Testing model files...")

# Check files
files_to_check = [
    "model/chatbot_model.h5",
    "model/tokenizer.pkl", 
    "model/label_encoder.pkl",
    "dataset/dataset_training.csv"
]

for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{'✅' if exists else '❌'} {file_path}")

# Test tokenizer
try:
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer_data = pickle.load(f)
        print(f"✅ Tokenizer loaded: {len(tokenizer_data['tokenizer'].word_index)} words")
except Exception as e:
    print(f"❌ Tokenizer error: {e}")

# Test label encoder
try:
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        print(f"✅ Label encoder loaded: {len(le.classes_)} classes")
        print(f"   Classes: {le.classes_}")
except Exception as e:
    print(f"❌ Label encoder error: {e}")