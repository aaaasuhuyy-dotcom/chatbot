import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json

def load_fine_tuned_model():
    """Load model BERT yang sudah di-fine-tune"""
    print("📦 Loading fine-tuned BERT model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert_simple_finetuned')
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('bert_simple_finetuned')
    
    # Load label encoder
    with open('bert_simple_finetuned/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load info
    with open('bert_simple_finetuned/info.json', 'r') as f:
        info = json.load(f)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Number of classes: {info['num_classes']}")
    print(f"🎯 Best accuracy: {info['best_accuracy']:.4f}")
    
    return tokenizer, model, label_encoder, info

def predict_intent(text, tokenizer, model, label_encoder, max_length=128):
    """Prediksi intent dari text"""
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    
    # Decode prediction
    intent = label_encoder.inverse_transform([predicted_class])[0]
    
    return intent, confidence

def main():
    print("🧪 Testing Fine-Tuned BERT Model")
    print("=" * 50)
    
    # Load model
    tokenizer, model, label_encoder, info = load_fine_tuned_model()
    
    # Test examples
    test_texts = [
        "cara membuat kartu keluarga",
        "info jam buka bank jateng",
        "prosedur akta kelahiran",
        "buat nib online",
        "info bpjs kesehatan",
        "daftar pdam",
        "cara bayar pajak",
        "info samsat terdekat",
        "prosedur surat pindah",
        "info taspen",
        "cara membuat sim",
        "daftar sekolah"
    ]
    
    print("\n📋 Test Predictions:")
    print("-" * 50)
    
    for text in test_texts:
        intent, confidence = predict_intent(text, tokenizer, model, label_encoder)
        print(f"💬 '{text}'")
        print(f"   🎯 Intent: {intent}")
        print(f"   📊 Confidence: {confidence:.4f}")
        print()
    
    print("🎉 Testing completed!")
    
    # Interactive mode
    print("\n🔄 Interactive Mode (ketik 'exit' untuk keluar):")
    while True:
        user_input = input("\n💬 Masukkan pertanyaan: ")
        if user_input.lower() in ['exit', 'quit', 'keluar']:
            break
        
        if user_input.strip():
            intent, confidence = predict_intent(user_input, tokenizer, model, label_encoder)
            print(f"🎯 Intent: {intent}")
            print(f"📊 Confidence: {confidence:.4f}")
    
    print("👋 Selesai!")

if __name__ == "__main__":
    main()