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
        self.bert_label_encoder = None
        
        # Data
        self.df = None
        self.intent_mappings = {}
        
        # Configuration
        self.confidence_thresholds = {
            'lstm_high': 0.8,
            'bert_high': 0.7,
            'fusion_threshold': 0.6,
            'pattern_similarity_threshold': 0.3
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
    
    def load_bert_model(self, model_path: str = "bert_simple_finetuned"):
        """Load fine-tuned IndoBERT model dengan mapping label yang benar"""
        try:
            print(f"🔄 Loading fine-tuned IndoBERT dari: {model_path}")

            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import pickle

            # Prefer local directory if it exists
            if os.path.isdir(model_path):
                local_path = model_path
            else:
                # try an alternative default local folder if the provided path looks invalid
                alt = "bert_simple_finetuned"
                if os.path.isdir(alt):
                    print(f"⚠️ Provided BERT path '{model_path}' not a local dir. Falling back to '{alt}'.")
                    local_path = alt
                else:
                    # fall back to provided path and let transformers decide (may attempt hub)
                    local_path = model_path

            # Load tokenizer and model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(local_path)

            # Create pipeline (CPU)
            self.bert_classifier = pipeline(
                "text-classification",
                model=self.bert_model,
                tokenizer=self.bert_tokenizer,
                device=-1  # CPU
            )

            # Load label encoder - CRITICAL untuk mapping yang benar
            le_path = os.path.join(local_path, 'label_encoder.pkl')
            if os.path.exists(le_path):
                with open(le_path, 'rb') as f:
                    self.bert_label_encoder = pickle.load(f)
                print(f"✅ BERT label encoder loaded: {len(self.bert_label_encoder.classes_)} classes")
            else:
                print(f"❌ label_encoder.pkl not found in {local_path}")
                print("⚠️ BERT predictions will use numeric labels. Creating fallback mapping...")
                # Fallback: create mapping from LSTM label encoder
                if self.lstm_label_encoder is not None:
                    self.bert_label_encoder = self.lstm_label_encoder
                    print("✅ Using LSTM label encoder as fallback for BERT")
                else:
                    print("❌ No label encoder available for BERT!")
                    self.bert_classifier = None
                    return False

            print("✅ Fine-tuned IndoBERT loaded successfully!")
            print(f"   - Classes: {len(self.bert_label_encoder.classes_)}")
            return True

        except Exception as e:
            print(f"❌ Fine-tuned BERT loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.bert_classifier = None
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

            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded = pad_sequences(sequences, maxlen=self.lstm_max_len, padding='post')

            # Predict
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
        """Predict intent menggunakan fine-tuned IndoBERT dengan mapping label yang benar"""
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
            
            # Convert label to original intent name
            if hasattr(self, 'bert_label_encoder') and self.bert_label_encoder is not None:
                # Handle both numeric labels (LABEL_0, LABEL_1) and string labels
                if predicted_label.startswith('LABEL_'):
                    try:
                        label_idx = int(predicted_label.split('_')[1])
                        intent = self.bert_label_encoder.inverse_transform([label_idx])[0]
                    except (ValueError, IndexError):
                        # Jika gagal, gunakan predicted_label langsung
                        intent = predicted_label
                else:
                    # Jika sudah string intent, gunakan langsung
                    intent = predicted_label
            else:
                # Fallback: use predicted label as is
                intent = predicted_label
            
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
    
    def check_pattern_similarity(self, text: str, intent: str) -> float:
        """Check similarity antara input text dengan patterns di intent"""
        if intent not in self.intent_mappings:
            return 0.0
        
        user_words = set(text.lower().split())
        max_similarity = 0.0
        
        for pattern in self.intent_mappings[intent]['patterns']:
            pattern_words = set(str(pattern).lower().split())
            common_words = len(user_words.intersection(pattern_words))
            total_words = len(user_words.union(pattern_words))
            
            if total_words > 0:
                similarity = common_words / total_words
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def predict_intent_hybrid(self, text: str) -> Dict:
        """
        Hybrid prediction dengan fallback strategy:
        1. Coba LSTM pertama
        2. Jika LSTM gagal atau pattern tidak match, coba BERT
        3. Gunakan LSTM untuk response matching
        4. Jika masih tidak match, gunakan BERT prediction langsung
        """
        
        print(f"🔍 Processing: '{text}'")
        
        # STEP 1: Coba LSTM pertama
        lstm_pred = self.predict_with_lstm(text)
        print(f"   LSTM Prediction: {lstm_pred['intent']} (conf: {lstm_pred['confidence']:.3f})")
        
        if lstm_pred['status'] == 'success':
            # Check pattern similarity untuk LSTM prediction
            similarity = self.check_pattern_similarity(text, lstm_pred['intent'])
            print(f"   LSTM Pattern Similarity: {similarity:.3f}")
            
            if similarity >= self.confidence_thresholds['pattern_similarity_threshold']:
                # LSTM berhasil dengan pattern yang cocok
                return {
                    **lstm_pred,
                    "method": "lstm_direct",
                    "sources": ["lstm"],
                    "pattern_similarity": similarity,
                    "fallback_reason": None
                }
            else:
                print(f"   ⚠️ LSTM pattern tidak cocok, mencoba BERT...")
        
        # STEP 2: LSTM gagal atau pattern tidak match, coba BERT
        bert_pred = self.predict_with_bert(text)
        print(f"   BERT Prediction: {bert_pred['intent']} (conf: {bert_pred['confidence']:.3f})")
        
        if bert_pred['status'] == 'success':
            # STEP 3: Gunakan intent dari BERT, tapi cari response dengan LSTM methodology
            bert_similarity = self.check_pattern_similarity(text, bert_pred['intent'])
            print(f"   BERT Pattern Similarity: {bert_similarity:.3f}")
            
            if bert_similarity >= self.confidence_thresholds['pattern_similarity_threshold']:
                # BERT + pattern matching berhasil
                return {
                    **bert_pred,
                    "method": "bert_with_lstm_pattern",
                    "sources": ["bert", "lstm_pattern"],
                    "pattern_similarity": bert_similarity,
                    "fallback_reason": "lstm_pattern_mismatch"
                }
            else:
                # STEP 4: Pattern masih tidak match, gunakan BERT prediction langsung
                print(f"   ⚠️ Pattern masih tidak cocok, menggunakan BERT langsung...")
                return {
                    **bert_pred,
                    "method": "bert_direct",
                    "sources": ["bert"],
                    "pattern_similarity": bert_similarity,
                    "fallback_reason": "both_patterns_mismatch"
                }
        
        # STEP 5: Fallback terakhir - semua method gagal
        print(f"   ❌ Semua method gagal, menggunakan fallback...")
        if lstm_pred['status'] == 'success':
            return {
                **lstm_pred,
                "method": "lstm_emergency_fallback",
                "sources": ["lstm"],
                "pattern_similarity": 0.0,
                "fallback_reason": "all_methods_failed"
            }
        else:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "method": "emergency_fallback",
                "status": "success",  # Tetap success agar bisa memberikan response
                "sources": [],
                "pattern_similarity": 0.0,
                "fallback_reason": "complete_failure"
            }
    
    def get_best_response(self, intent: str, user_text: str, method_used: str = "default", pattern_similarity: float = 0.0) -> str:
        """Get response berdasarkan intent dengan strategy yang berbeda berdasarkan method"""
        
        if intent not in self.intent_mappings:
            if method_used == "bert_direct":
                # Untuk BERT direct, berikan response generic yang lebih helpful
                return "Saya memahami pertanyaan Anda tentang informasi Bappenda Tegal. Untuk informasi lengkap mengenai jam operasional dan layanan, silakan hubungi Bappenda Tegal langsung atau kunjungi website resmi mereka."
            else:
                return "Maaf, saya belum memahami pertanyaan Anda. Bisakah Anda mengulangi pertanyaannya dengan kata-kata yang berbeda?"
        
        intent_data = self.intent_mappings[intent]
        user_words = set(user_text.lower().split())
        best_score = -1.0
        best_response = intent_data['responses'][0] if intent_data.get('responses') else "Maaf, saya belum memahami pertanyaan Anda."
        
        # Strategy berbeda berdasarkan method yang digunakan
        if method_used in ["lstm_direct", "lstm_with_bert_pattern"]:
            # Untuk LSTM-based methods, gunakan confidence-weighted similarity
            for pattern, response in zip(intent_data.get('patterns', []), intent_data.get('responses', [])):
                pattern_words = set(str(pattern).lower().split())
                common_words = len(user_words.intersection(pattern_words))
                total_words = len(user_words.union(pattern_words))
                
                if total_words > 0:
                    similarity = common_words / total_words
                    score = similarity * pattern_similarity
                    if score > best_score:
                        best_score = score
                        best_response = response
        else:
            # Untuk BERT-based methods, pilih response secara random atau berdasarkan context
            # atau gunakan pattern similarity biasa
            for pattern, response in zip(intent_data.get('patterns', []), intent_data.get('responses', [])):
                pattern_words = set(str(pattern).lower().split())
                common_words = len(user_words.intersection(pattern_words))
                total_words = len(user_words.union(pattern_words))
                
                if total_words > 0:
                    similarity = common_words / total_words
                    if similarity > best_score:
                        best_score = similarity
                        best_response = response
        
        # Jika similarity sangat rendah, tambahkan disclaimer
        if best_score < 0.2 and method_used == "bert_direct":
            best_response += " (Saya berusaha memahami pertanyaan Anda berdasarkan konteks yang tersedia)"
        
        return best_response

    def get_response_scores(self, intent: str, user_text: str, method_used: str = "default", pattern_similarity: float = 0.0) -> List[Dict]:
        """Return candidate responses with their similarity and computed score for debugging.

        Returns a list of dicts: { 'pattern': pattern, 'response': response, 'similarity': float, 'score': float }
        """
        results = []
        if intent not in self.intent_mappings:
            return results

        intent_data = self.intent_mappings[intent]
        user_words = set(user_text.lower().split())

        for pattern, response in zip(intent_data.get('patterns', []), intent_data.get('responses', [])):
            pattern_words = set(str(pattern).lower().split())
            common_words = len(user_words.intersection(pattern_words))
            total_words = len(user_words.union(pattern_words))
            similarity = (common_words / total_words) if total_words > 0 else 0.0

            # Compute score using same logic as get_best_response
            if method_used in ["lstm_direct", "lstm_with_bert_pattern", "lstm_only"]:
                score = similarity * float(pattern_similarity)
            else:
                score = similarity

            results.append({
                'pattern': pattern,
                'response': response,
                'similarity': round(similarity, 4),
                'score': round(score, 4)
            })

        # sort by score desc
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

# Global instance
hybrid_nlu = HybridNLUService()

def initialize_hybrid_service(
    dataset_path: str = "dataset/dataset_training.csv",
    lstm_model_path: str = "model/chatbot_model.h5",
    lstm_tokenizer_path: str = "model/tokenizer.pkl", 
    lstm_label_encoder_path: str = "model/label_encoder.pkl",
    bert_model_path: str = "./bert_fine_tuned"
):
    """Initialize hybrid NLU service dengan fine-tuned BERT"""
    print("🎯 Initializing Hybrid NLU Service dengan Fallback Strategy...")
    
    # Load dataset first
    if not hybrid_nlu.load_dataset(dataset_path):
        raise Exception(f"Dataset loading failed: {dataset_path}")
    
    # Load LSTM model
    if not hybrid_nlu.load_lstm_model(lstm_model_path, lstm_tokenizer_path, lstm_label_encoder_path):
        raise Exception(f"LSTM model loading failed")
    
    # Load Fine-tuned BERT
    hybrid_nlu.load_bert_model(bert_model_path)
    
    print("🚀 Hybrid NLU Service dengan Fallback Strategy initialized successfully!")
    print(f"📊 Service Status:")
    print(f"   - LSTM Model: {'✅' if hybrid_nlu.lstm_model is not None else '❌'}")
    print(f"   - BERT Model: {'✅' if hybrid_nlu.bert_classifier is not None else '❌'}")
    print(f"   - Dataset: {'✅' if hybrid_nlu.df is not None else '❌'}")
    print(f"   - Intents: {len(hybrid_nlu.intent_mappings) if hybrid_nlu.intent_mappings is not None else 0}")
    print(f"🎯 Fallback Strategy: LSTM → BERT → LSTM Pattern → BERT Direct")
    
    return hybrid_nlu