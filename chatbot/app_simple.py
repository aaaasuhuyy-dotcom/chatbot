# app_simple.py - Simple version without BERT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(title="Simple Chatbot API")

class UserInput(BaseModel):
    text: str

# Global variables
model = None
tokenizer = None
label_encoder = None
df = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, label_encoder, df
    
    print("🚀 Starting Simple Chatbot API...")
    
    try:
        # Load model
        print("📦 Loading LSTM model...")
        model = load_model('model/chatbot_model.h5', compile=False)
        
        # Recompile
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Load tokenizer
        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer_data = pickle.load(f)
            tokenizer = tokenizer_data['tokenizer']
            max_len = tokenizer_data['max_len']
            
        # Load label encoder
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Load dataset
        df = pd.read_csv('dataset/dataset_training.csv')
        
        print(f"✅ Model loaded: {model is not None}")
        print(f"✅ Tokenizer: {tokenizer is not None}")
        print(f"✅ Label encoder: {label_encoder is not None}")
        print(f"✅ Dataset: {len(df)} rows")
        print(f"✅ Classes: {len(label_encoder.classes_)}")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    return {"message": "Simple Chatbot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "classes_count": len(label_encoder.classes_) if label_encoder else 0
    }

@app.post("/chat")
async def chat(user_input: UserInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess
        text = user_input.text.lower().strip()
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=10, padding='post')  # Use max_len from training
        
        # Predict
        prediction = model.predict(padded, verbose=0)[0]
        predicted_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_idx])
        intent = label_encoder.inverse_transform([predicted_idx])[0]
        
        # Get response from dataset
        responses = df[df['intent'] == intent]['response'].tolist()
        response = responses[0] if responses else "Maaf, saya belum memahami pertanyaan Anda."
        
        return {
            "text": user_input.text,
            "intent": intent,
            "confidence": confidence,
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")