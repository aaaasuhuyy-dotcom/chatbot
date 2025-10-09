from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.responses import JSONResponse
import pickle
import uvicorn
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Pydantic models
class UserInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class ChatResponse(BaseModel):
    original_text: str
    predicted_intent: str
    confidence: float
    response: str
    method_used: str
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    bert_available: bool
    intents_count: int
    total_patterns: int

class StatsResponse(BaseModel):
    model_info: Dict
    performance: Dict
    system_info: Dict

# FastAPI app
app = FastAPI(
    title="Hybrid Chatbot API",
    description="LSTM + IndoBERT Hybrid Intent Classification API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global service instance
nlu_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize NLU service on startup"""
    global nlu_service
    try:
        print("🚀 Starting Hybrid Chatbot API...")
        
        # Import inside function to avoid circular imports
        from hybrid_nlu_service import initialize_hybrid_service

        # Use environment vars for pre-checks (same defaults used for initialization)
        dataset_path = os.environ.get('DATASET_PATH', 'dataset/dataset_training.csv')
        lstm_model_path = os.environ.get('LSTM_MODEL_PATH', 'model/chatbot_model.h5')
        lstm_tokenizer_path = os.environ.get('LSTM_TOKENIZER_PATH', 'model/tokenizer.pkl')
        lstm_label_encoder_path = os.environ.get('LSTM_LABEL_ENCODER_PATH', 'model/label_encoder.pkl')
        bert_model_path = os.environ.get('BERT_MODEL_PATH', 'bert_simple_finetuned')

        print("📂 Checking model files...")
        # Check if model files exist
        model_files = {
            "lstm_model": lstm_model_path,
            "tokenizer": lstm_tokenizer_path,
            "label_encoder": lstm_label_encoder_path,
            "dataset": dataset_path,
            "bert_folder": bert_model_path
        }

        for file_type, file_path in model_files.items():
            if os.path.exists(file_path):
                print(f"✅ {file_type}: {file_path} - EXISTS")
            else:
                print(f"❌ {file_type}: {file_path} - NOT FOUND")
        
        # Initialize with model paths (read from environment variables when provided)
        print("🔄 Initializing hybrid service...")

        dataset_path = os.environ.get('DATASET_PATH', 'dataset/dataset_training.csv')
        lstm_model_path = os.environ.get('LSTM_MODEL_PATH', 'model/chatbot_model.h5')
        lstm_tokenizer_path = os.environ.get('LSTM_TOKENIZER_PATH', 'model/tokenizer.pkl')
        lstm_label_encoder_path = os.environ.get('LSTM_LABEL_ENCODER_PATH', 'model/label_encoder.pkl')
        bert_model_path = os.environ.get('BERT_MODEL_PATH', 'bert_simple_finetuned')

        # Initialize and assign the fully-initialized service
        nlu_service = initialize_hybrid_service(
            dataset_path=dataset_path,
            lstm_model_path=lstm_model_path,
            lstm_tokenizer_path=lstm_tokenizer_path,
            lstm_label_encoder_path=lstm_label_encoder_path,
            bert_model_path=bert_model_path
        )

        print("✅ Hybrid service initialized!")
        print(f"📊 LSTM Model: {nlu_service.lstm_model is not None}")
        print(f"📊 BERT Model: {nlu_service.bert_classifier is not None}") 
        print(f"📊 Dataset: {nlu_service.df is not None}")
        print(f"📊 Intent Mappings: {len(nlu_service.intent_mappings) if nlu_service.intent_mappings else 0}")
        
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        import traceback
        traceback.print_exc()
        nlu_service = None
# Routes
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Hybrid LSTM + IndoBERT Chatbot API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return HealthResponse(
        status="healthy",
        model_loaded=nlu_service.lstm_model is not None,
        bert_available=nlu_service.bert_classifier is not None,
        intents_count=len(nlu_service.intent_mappings) if nlu_service.intent_mappings else 0,
        total_patterns=len(nlu_service.df) if nlu_service.df is not None else 0
    )

@app.get("/intents")
async def get_intents():
    """Get list of available intents"""
    if nlu_service is None or nlu_service.intent_mappings is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    intents = list(nlu_service.intent_mappings.keys())
    return {
        "available_intents": intents,
        "count": len(intents),
        "intents_details": [
            {
                "intent": intent,
                "response_type": data['response_type'],
                "patterns_count": len(data['patterns']),
                "responses_count": len(data['responses'])
            }
            for intent, data in nlu_service.intent_mappings.items()
        ]
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(user_input: UserInput):
    """Main chat endpoint - Hybrid LSTM + BERT prediction"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        # Get hybrid prediction
        prediction = nlu_service.predict_intent_hybrid(user_input.text)
        
        # Get response using method and pattern similarity from hybrid prediction
        response_text = nlu_service.get_best_response(
            prediction["intent"],
            user_input.text,
            method_used=prediction.get("method", "hybrid"),
            pattern_similarity=prediction.get("pattern_similarity", 0.0)
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
        
        return ChatResponse(
            original_text=user_input.text,
            predicted_intent=prediction["intent"],
            confidence=prediction["confidence"],
            response=response_text,
            method_used=prediction["method"],
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat-lstm")
async def chat_lstm_only(user_input: UserInput):
    """Chat using LSTM only (faster)"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        # Use LSTM only
        prediction = nlu_service.predict_with_lstm(user_input.text)
        # For LSTM-only, compute pattern similarity to use in response selection
        pattern_similarity = 0.0
        try:
            pattern_similarity = nlu_service.check_pattern_similarity(user_input.text, prediction["intent"])
        except Exception:
            pattern_similarity = 0.0

        response_text = nlu_service.get_best_response(
            prediction["intent"],
            user_input.text,
            method_used="lstm_only",
            pattern_similarity=pattern_similarity
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "original_text": user_input.text,
            "predicted_intent": prediction["intent"],
            "confidence": prediction["confidence"],
            "response": response_text,
            "method_used": "lstm_only",
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"LSTM chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat-bert")
async def chat_bert_only(user_input: UserInput):
    """Chat using BERT only (more accurate)"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if nlu_service.bert_classifier is None:
        raise HTTPException(status_code=503, detail="BERT model not available")
    
    start_time = datetime.now()
    
    try:
        # Use BERT only
        prediction = nlu_service.predict_with_bert(user_input.text)
        # For BERT-only, compute pattern similarity to use in response selection
        pattern_similarity = 0.0
        try:
            pattern_similarity = nlu_service.check_pattern_similarity(user_input.text, prediction["intent"])
        except Exception:
            pattern_similarity = 0.0

        response_text = nlu_service.get_best_response(
            prediction["intent"],
            user_input.text,
            method_used="bert_only",
            pattern_similarity=pattern_similarity
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "original_text": user_input.text,
            "predicted_intent": prediction["intent"],
            "confidence": prediction["confidence"],
            "response": response_text,
            "method_used": "bert_only",
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"BERT chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/batch-chat")
async def batch_chat(batch_input: BatchInput):
    """Batch prediction endpoint"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        results = []
        for text in batch_input.texts:
            prediction = nlu_service.predict_intent_hybrid(text)
            response = nlu_service.get_best_response(
                prediction["intent"],
                text,
                method_used=prediction.get("method", "hybrid"),
                pattern_similarity=prediction.get("pattern_similarity", 0.0)
            )
            
            results.append({
                "text": text,
                "predicted_intent": prediction["intent"],
                "confidence": prediction["confidence"],
                "response": response,
                "method_used": prediction["method"]
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "predictions": results,
            "count": len(results),
            "total_processing_time": round(processing_time, 2),
            "average_time_per_request": round(processing_time / len(results), 2)
        }
        
    except Exception as e:
        logger.error(f"Batch chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get service statistics and model information"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Model info
    model_info = {
        "lstm_loaded": nlu_service.lstm_model is not None,
        "bert_loaded": nlu_service.bert_classifier is not None,
        "dataset_loaded": nlu_service.df is not None,
        "intents_count": len(nlu_service.intent_mappings) if nlu_service.intent_mappings else 0,
        "total_patterns": len(nlu_service.df) if nlu_service.df is not None else 0
    }
    
    # Performance info (placeholder - bisa diisi dengan metrics actual)
    performance = {
        "prediction_methods": ["hybrid", "lstm_only", "bert_only"],
        "supported_strategies": nlu_service.confidence_thresholds
    }
    
    # System info
    system_info = {
        "service_started": True,
        "hybrid_mode": nlu_service.bert_classifier is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    return StatsResponse(
        model_info=model_info,
        performance=performance,
        system_info=system_info
    )

@app.get("/api/debug-prediction")
async def debug_prediction(text: str):
    """Debug endpoint to see detailed prediction information"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Get predictions from both models
        lstm_pred = nlu_service.predict_with_lstm(text)
        bert_pred = nlu_service.predict_with_bert(text)
        
        # Get fused prediction
        fused_pred = nlu_service.predict_intent_hybrid(text)
        
        return {
            "input_text": text,
            "lstm_prediction": lstm_pred,
            "bert_prediction": bert_pred,
            "fused_prediction": fused_pred,
            "final_response": nlu_service.get_best_response(
                fused_pred["intent"],
                text,
                method_used=fused_pred.get("method", "hybrid"),
                pattern_similarity=fused_pred.get("pattern_similarity", 0.0)
            ),
            "confidence_thresholds": nlu_service.confidence_thresholds
        }
        
    except Exception as e:
        logger.error(f"Debug prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/debug-response-scores")
async def debug_response_scores(text: str, intent: str, method: str = "hybrid"):
    """Return candidate responses and their computed similarity/scores for an intent."""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Determine pattern_similarity from method if possible
        pattern_similarity = 0.0
        try:
            pattern_similarity = nlu_service.check_pattern_similarity(text, intent)
        except Exception:
            pattern_similarity = 0.0

        scores = nlu_service.get_response_scores(intent, text, method_used=method, pattern_similarity=pattern_similarity)
        return {"intent": intent, "text": text, "method": method, "pattern_similarity": pattern_similarity, "candidates": scores}

    except Exception as e:
        logger.error(f"Debug response scores error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "available_endpoints": [
            "/docs", "/health", "/api/chat", "/intents", "/api/stats"
        ]}
    )

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Auto reload selama development
    )