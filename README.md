# 🤖 Chatbot Indonesia dengan BERT & LSTM

Repository ini berisi implementasi chatbot bahasa Indonesia yang menggunakan dua pendekatan: **BERT Fine-tuning** dan **LSTM** untuk Natural Language Understanding (NLU).

## 📊 Dataset

Dataset berisi **8,899 contoh percakapan** dengan **38 kategori intent** terkait layanan pemerintah dan umum:
- `dinsos_info`, `dlh_info`, `koperasi_info` 
- `kk_info`, `bpjs_sehat_info`, `samsat_info`
- `akta_lahir_info`, `nib_info`, `pdam_info`
- Dan 29 kategori lainnya

## 🚀 Fitur Utama

### 1. **BERT Fine-tuning** 
- Model: `cahya/bert-base-indonesian-522M`
- **Accuracy: 97.13%** 
- Training: 3 epochs dengan early stopping
- Script: `simple_bert_finetune.py`

### 2. **LSTM Model**
- Architecture: Embedding + LSTM + Dense layers
- Preprocessing dengan tokenizer
- Script: `chatbot_training.py`

### 3. **Data Processing**
- CSV validation dan cleaning: `scripts/fix_csv_malformed.py`
- Data splitting utilities: `scripts/data_splitter.py`
- Dataset preprocessing: `scripts/validate_csv.py`

### 4. **Testing & Evaluation**
- BERT model testing: `test_bert_model.py`
- Interactive chat mode
- Confidence scoring

## 🛠️ Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/aaaasuhuyy-dotcom/chatbot.git
   cd chatbot/chatbot
   ```

2. **Setup virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # atau .venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Fine-tune BERT Model
```bash
python simple_bert_finetune.py
```

### Train LSTM Model  
```bash
python chatbot_training.py
```

### Test Models
```bash
# Test BERT model (setelah fine-tuning)
python test_bert_model.py

# Test LSTM model
python test.py
```

### Run the API (development)
```bash
# from repository root
cd chatbot
source .venv/bin/activate
uvicorn chatbot.app:app --reload
```

### Environment variables
You can override default model paths with environment variables before starting the app:

- DATASET_PATH (default: `dataset/dataset_training.csv`)
- LSTM_MODEL_PATH (default: `model/chatbot_model.h5`)
- LSTM_TOKENIZER_PATH (default: `model/tokenizer.pkl`)
- LSTM_LABEL_ENCODER_PATH (default: `model/label_encoder.pkl`)
- BERT_MODEL_PATH (default: `bert_simple_finetuned`)

Example:
```bash
export BERT_MODEL_PATH=bert_simple_finetuned
export LSTM_MODEL_PATH=model/chatbot_model.h5
uvicorn chatbot.app:app --reload
```

### Debug endpoints
Two helpful debug endpoints were added to inspect predictions and response scoring:

- `/api/debug-prediction?text=...` — returns LSTM and BERT raw predictions and the fused hybrid result.
- `/api/debug-response-scores?text=...&intent=...&method=...` — returns candidate patterns/responses along with computed similarity and score (useful to tune `pattern_similarity_threshold`).

Example curl commands:
```bash
# Debug fused prediction
curl -G "http://127.0.0.1:8000/api/debug-prediction" --data-urlencode "text=halo"

# Show candidate response scores for an intent
curl -G "http://127.0.0.1:8000/api/debug-response-scores" --data-urlencode "text=halo" --data-urlencode "intent=salutation"
```

If the server is running remotely or behind a proxy, adjust the host/port accordingly.

### Fix Malformed CSV
```bash
python scripts/fix_csv_malformed.py
```

## 📁 Struktur Project

```
chatbot/
├── dataset/
│   ├── dataset_training.csv     # Dataset utama
│   ├── data_jadi.csv           # Data yang sudah diproses
│   └── data_mentah.csv         # Data mentah
├── scripts/
│   ├── fix_csv_malformed.py    # Perbaiki CSV yang rusak
│   ├── validate_csv.py         # Validasi dataset
│   └── data_splitter.py        # Split data
├── model/
│   ├── training_log.json       # Log training
│   └── hybrid_config.json      # Konfigurasi hybrid
├── simple_bert_finetune.py     # 🔥 BERT fine-tuning
├── chatbot_training.py         # LSTM training
├── test_bert_model.py          # Test BERT model
├── telegram_bot.py             # Telegram integration
├── hybrid_nlu_service.py       # Hybrid NLU service
└── requirements.txt            # Dependencies
```

## 🎨 Model Performance

### BERT Model Results:
- **Training Accuracy**: 97.61%
- **Validation Accuracy**: 97.13%
- **Training Loss**: 0.06
- **Validation Loss**: 0.10

### Example Predictions:
```
💬 'cara membuat kartu keluarga'
   🎯 Intent: kk_info
   📊 Confidence: 0.9972

💬 'info bpjs kesehatan'  
   🎯 Intent: bpjs_sehat_info
   📊 Confidence: 0.9963

💬 'buat nib online'
   🎯 Intent: nib_info
   📊 Confidence: 0.9942
```

## 📋 Intent Categories

Dataset mencakup 38 kategori layanan:

| Category | Examples |
|----------|----------|
| `kk_info` | Kartu keluarga, KK baru |
| `bpjs_sehat_info` | BPJS kesehatan |
| `samsat_info` | Pajak kendaraan, STNK |
| `akta_lahir_info` | Akta kelahiran |
| `nib_info` | NIB online |
| `pdam_info` | Air bersih, PDAM |

## 🔧 Development

### Requirements
- Python 3.8+
- PyTorch 
- Transformers
- TensorFlow/Keras
- pandas, numpy
- scikit-learn

### Training Tips
1. **Dataset Quality**: Pastikan CSV tidak ada yang malformed
2. **Class Balance**: Gunakan stratified split jika memungkinkan  
3. **Memory**: BERT model butuh RAM ~4GB untuk fine-tuning
4. **Time**: Fine-tuning BERT ~5-15 menit per epoch

## 🚧 Troubleshooting

### Common Issues:

**1. ValueError: The least populated class in y has only 1 member**
```bash
# Jalankan fix untuk stratified split
python scripts/fix_csv_malformed.py
```

**2. ModuleNotFoundError: transformers**
```bash
pip install transformers torch datasets
```

**3. File CSV malformed**
```bash
python scripts/fix_csv_malformed.py
python scripts/validate_csv.py
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes 
4. Push dan create Pull Request

## 📄 License

MIT License - lihat file LICENSE untuk detail.

## 🙏 Acknowledgments

- Dataset: Layanan pemerintah Indonesia
- Model: `cahya/bert-base-indonesian-522M`
- Libraries: Transformers, PyTorch, TensorFlow

---

**Made with ❤️ for Indonesian NLP Community**

🔗 **Repository**: https://github.com/aaaasuhuyy-dotcom/chatbot