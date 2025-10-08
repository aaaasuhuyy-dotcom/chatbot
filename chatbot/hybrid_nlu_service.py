import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridNLUService:
    def __init__(self):
        # LSTM Components
        self.lstm_model = None
        self.lstm_tokenizer = None
        self.lstm_max_len = None
        self.lstm_label_encoder = None
        
        # IndoBERT Components  
        self.bert_classifier = None
        self.bert_tokenizer = None
        
        # Data
        self.df = None
        self.intent_mappings = {}
        
        # Configuration
        self.confidence_thresholds = {
            'lstm_high': 0.8,
            'bert_high': 0.7,
            'fusion_threshold': 0.6
        }
    
    def load_lstm_model(self, model_path: str, tokenizer_path: str, label_encoder_path: str):
        """Load existing LSTM model"""
        try:
            print(f"🔄 Loading LSTM model from: {model_path}")
            
            # Check if files exist
            if not os.path.exists(model_path):
                print(f"❌ LSTM model file not found: {model_path}")
                return False
            if not os.path.exists(tokenizer_path):
                print(f"❌ Tokenizer file not found: {tokenizer_path}")
                return False
            if not os.path.exists(label_encoder_path):
                print(f"❌ Label encoder file not found: {label_encoder_path}")
                return False
            
            # Import tensorflow inside function to avoid early loading
            from tensorflow.keras.models import load_model
            
            print("📦 Loading TensorFlow model...")
            self.lstm_model = load_model(model_path)
            
            print("📦 Loading tokenizer...")
            with open(tokenizer_path, 'rb') as f:
                tokenizer_data = pickle.load(f)
                self.lstm_tokenizer = tokenizer_data['tokenizer']
                self.lstm_max_len = tokenizer_data['max_len']
                
            print("📦 Loading label encoder...")
            with open(label_encoder_path, 'rb') as f:
                self.lstm_label_encoder = pickle.load(f)
                
            print(f"✅ LSTM model loaded successfully!")
            print(f"   - Vocabulary size: {len(self.lstm_tokenizer.word_index)}")
            print(f"   - Max sequence length: {self.lstm_max_len}")
            print(f"   - Number of classes: {len(self.lstm_label_encoder.classes_)}")
            return True
            
        except Exception as e:
            print(f"❌ LSTM model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_bert_model(self, model_path: str = "./bert_fine_tuned"):
        """Load fine-tuned IndoBERT model"""
        try:
            print(f"🔄 Loading fine-tuned IndoBERT dari: {model_path}")
            
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import pickle
            
            # Load fine-tuned model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Create pipeline
            self.bert_classifier = pipeline(
                "text-classification",
                model=self.bert_model,
                tokenizer=self.bert_tokenizer,
                device=-1  # CPU
            )
            
            # Load label encoder
            with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
                self.bert_label_encoder = pickle.load(f)
            
            print("✅ Fine-tuned IndoBERT loaded successfully!")
            print(f"   - Classes: {len(self.bert_label_encoder.classes_)}")
            return True
            
        except Exception as e:
            print(f"❌ Fine-tuned BERT loading failed: {e}")
            return False
    
    def load_dataset(self, dataset_path: str):
        """Load dataset"""
        try:
            print(f"📂 Loading dataset from: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                print(f"❌ Dataset file not found: {dataset_path}")
                return False
            
            self.df = pd.read_csv(dataset_path, encoding='utf-8')
            print(f"📊 Dataset loaded: {len(self.df)} rows")
            
            # Create intent mappings
            self.intent_mappings = {}
            for intent in self.df['intent'].unique():
                intent_data = self.df[self.df['intent'] == intent]
                self.intent_mappings[intent] = {
                    'response_type': intent_data['response_type'].iloc[0],
                    'patterns': intent_data['pattern'].tolist(),
                    'responses': intent_data['response'].tolist()
                }
            
            print(f"✅ Intent mappings created: {len(self.intent_mappings)} intents")
            return True
            
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            return False
    
    def predict_with_lstm(self, text: str) -> Dict:
        """Predict intent menggunakan LSTM"""
        try:
            if self.lstm_model is None:
                return {
                    "intent": "lstm_unavailable",
                    "confidence": 0.0,
                    "method": "lstm",
                    "status": "unavailable"
                }
            
            # Preprocess
            processed_text = text.lower().strip()
            sequences = self.lstm_tokenizer.texts_to_sequences([processed_text])
            padded = pad_sequences(sequences, maxlen=self.lstm_max_len, padding='post')
            
            # Predict
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            prediction = self.lstm_model.predict(padded, verbose=0)[0]
            predicted_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_idx])
            
            intent = self.lstm_label_encoder.inverse_transform([predicted_idx])[0]
            
            return {
                "intent": intent,
                "confidence": confidence,
                "method": "lstm",
                "status": "success"
            }
            
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "method": "lstm",
                "status": "error"
            }
    
    def predict_with_bert(self, text: str) -> Dict:
        """Predict intent menggunakan fine-tuned IndoBERT"""
        try:
            if not self.bert_classifier:
                return {
                    "intent": "bert_unavailable",
                    "confidence": 0.0,
                    "method": "bert",
                    "status": "unavailable"
                }
            
            # Predict
            result = self.bert_classifier(text[:512])
            predicted_label = result[0]['label']
            confidence = result[0]['score']
            
            # Convert label back to original intent name
            # Note: BERT outputs label names from fine-tuning
            intent = predicted_label  # Sudah dalam format intent asli
            
            return {
                "intent": intent,
                "confidence": float(confidence),
                "method": "bert",
                "status": "success"
            }
            
        except Exception as e:
            print(f"BERT prediction failed: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "method": "bert",
                "status": "error"
            }
    
    def predict_intent_hybrid(self, text: str) -> Dict:
        """Main hybrid prediction method dengan fallback"""
        # Get predictions from both models
        lstm_pred = self.predict_with_lstm(text)
        bert_pred = self.predict_with_bert(text)
        
        # Jika BERT unavailable atau confidence rendah, use LSTM
        if bert_pred['status'] != 'success' or bert_pred['confidence'] < 0.6:
            return {
                **lstm_pred,
                "method": "lstm_fallback",
                "sources": ["lstm"],
                "fallback_reason": "bert_unavailable" if bert_pred['status'] != 'success' else "low_confidence"
            }
        
        # Jika LSTM unavailable, use BERT
        if lstm_pred['status'] != 'success':
            return {
                **bert_pred,
                "method": "bert_fallback", 
                "sources": ["bert"],
                "fallback_reason": "lstm_unavailable"
            }
        
        # Fusion logic: gunakan yang confidence lebih tinggi
        if bert_pred['confidence'] >= lstm_pred['confidence']:
            return {
                "intent": bert_pred['intent'],
                "confidence": bert_pred['confidence'],
                "method": "bert_higher_confidence", 
                "sources": ["bert"]
            }
        else:
            return {
                "intent": lstm_pred['intent'], 
                "confidence": lstm_pred['confidence'],
                "method": "lstm_higher_confidence",
                "sources": ["lstm"]
            }
    
    def get_best_response(self, intent: str, user_text: str) -> str:
        """Get response berdasarkan intent"""
        if intent not in self.intent_mappings:
            return "Maaf, saya belum memahami pertanyaan Anda."
        
        intent_data = self.intent_mappings[intent]
        user_words = set(user_text.lower().split())
        best_score = -1
        best_response = intent_data['responses'][0]
        
        for pattern, response in zip(intent_data['patterns'], intent_data['responses']):
            pattern_words = set(str(pattern).lower().split())
            common_words = len(user_words.intersection(pattern_words))
            total_words = len(user_words.union(pattern_words))
            
            if total_words > 0:
                score = common_words / total_words
                if score > best_score:
                    best_score = score
                    best_response = response
        
        return best_response

# Global instance
hybrid_nlu = HybridNLUService()

def initialize_hybrid_service(
    dataset_path: str = "dataset/dataset_training.csv",
    lstm_model_path: str = "model/chatbot_model.h5",
    lstm_tokenizer_path: str = "model/tokenizer.pkl", 
    lstm_label_encoder_path: str = "model/label_encoder.pkl",
    bert_model_path: str = "./bert_fine_tuned"  # New parameter
):
    """Initialize hybrid NLU service dengan fine-tuned BERT"""
    print("🎯 Initializing Hybrid NLU Service...")
    
    # Load dataset first
    if not hybrid_nlu.load_dataset(dataset_path):
        raise Exception(f"Dataset loading failed: {dataset_path}")
    
    # Load LSTM model
    if not hybrid_nlu.load_lstm_model(lstm_model_path, lstm_tokenizer_path, lstm_label_encoder_path):
        raise Exception(f"LSTM model loading failed")
    
    # Load Fine-tuned BERT
    hybrid_nlu.load_bert_model(bert_model_path)
    
    print("🚀 Hybrid NLU Service initialized successfully!")
    print(f"📊 Service Status:")
    print(f"   - LSTM Model: {'✅' if hybrid_nlu.lstm_model else '❌'}")
    print(f"   - BERT Model: {'✅' if hybrid_nlu.bert_classifier else '❌'}")
    print(f"   - Dataset: {'✅' if hybrid_nlu.df else '❌'}")
    print(f"   - Intents: {len(hybrid_nlu.intent_mappings)}")
    
    return hybrid_nlu