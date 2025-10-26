#!/usr/bin/env python3
"""
🏥 Integrated Medical API - FLAN-T5 Enhanced
Complete FastAPI Gateway with auto-detection: FLAN-T5-Base (60-75%) or T5-Small (40%)
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Langchain service
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path
from src.models.medical_ai import LangchainMedicalService

# ============================================================================
# Configuration
# ============================================================================

# Auto-detect best model (FLAN-T5-Base or T5-Small)
# LangchainMedicalService now handles this automatically

VECTORSTORE_PATH = "data/vectorstore"
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/medical-qa"

# ============================================================================
# Pydantic Models
# ============================================================================

class QuestionRequest(BaseModel):
    """Request model for medical questions"""
    question: str = Field(..., min_length=1, description="Medical question")
    user_id: str = Field(default="anonymous", description="User identifier")
    use_langchain: bool = Field(default=True, description="Use Langchain RAG")
    use_n8n: bool = Field(default=False, description="Route through N8N workflow")
    max_length: int = Field(default=150, description="Max response length")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    answer: str
    sources: List[Dict[str, Any]] = []
    method: str  # langchain, n8n, direct
    processing_time: float
    timestamp: str
    disclaimer: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str

class SystemStatus(BaseModel):
    """Complete system status"""
    fastapi: Dict[str, Any]
    langchain: Dict[str, Any]
    n8n: Dict[str, Any]
    model: Dict[str, Any]
    vectorstore: Dict[str, Any]

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="🏥 Medical AI Q&A API",
    description="Complete integrated medical AI system with Langchain + N8N + Custom Models",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

langchain_service: Optional[LangchainMedicalService] = None
request_history: List[Dict] = []
MAX_HISTORY = 100

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global langchain_service
    
    print("\n" + "="*80)
    print("🚀 Starting Integrated Medical API (BioGPT Enhanced)")
    print("="*80 + "\n")
    
    # Initialize Langchain service (auto-detects BioGPT or T5)
    try:
        print("🔧 Initializing Langchain Medical Service...")
        print(f"   Vectorstore: {VECTORSTORE_PATH}\n")
        
        langchain_service = LangchainMedicalService(
            vectorstore_path=VECTORSTORE_PATH
        )
        
        print("✅ Langchain service initialized successfully!\n")
        
    except FileNotFoundError as e:
        print(f"❌ No trained model found!")
        print(f"   Error: {e}")
        print("\n   Train a model first:")
        print("   - FLAN-T5-Base (recommended): train-model.bat")
        print("   - Expected accuracy: 60-75% (vs 40% T5-Small)\n")
        langchain_service = None
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Please install: pip install -r requirements.txt\n")
        langchain_service = None
    except Exception as e:
        print(f"❌ Failed to initialize Langchain service: {e}")
        import traceback
        traceback.print_exc()
        print()
        langchain_service = None
    
    # Check N8N availability
    n8n_available = await check_n8n_status()
    if n8n_available:
        print(f"✅ N8N is available at: {N8N_WEBHOOK_URL}\n")
    else:
        print(f"⚠️ N8N is not available at: {N8N_WEBHOOK_URL}")
        print("   Start N8N with: start-n8n.bat\n")
    
    print("="*80)
    print("🎉 API is ready!")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"❤️ Health: http://localhost:8000/health")
    print(f"📊 Status: http://localhost:8000/status")
    print("="*80 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n👋 Shutting down Integrated Medical API...")

# ============================================================================
# Helper Functions
# ============================================================================

async def check_n8n_status() -> bool:
    """Check if N8N is available"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:5678/healthz")
            return response.status_code == 200
    except:
        return False

async def call_n8n_webhook(question: str, user_id: str) -> Dict[str, Any]:
    """Call N8N webhook with extended timeout for model processing"""
    try:
        # Extended timeout: 90 seconds (model loading can take 30-60s first time)
        timeout = httpx.Timeout(90.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"🌊 Calling N8N Webhook: {N8N_WEBHOOK_URL}")
            response = await client.post(
                N8N_WEBHOOK_URL,
                json={"question": question, "user_id": user_id}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"✅ N8N Response received: {result.get('method', 'unknown')}")
            return result
    except httpx.TimeoutException:
        logger.error(f"⏱️ N8N Timeout after 90 seconds")
        raise HTTPException(
            status_code=504, 
            detail="N8N workflow timeout. Model may be loading (first request takes 30-60s). Please try again."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ N8N HTTP Error: {e.response.status_code}")
        error_detail = f"N8N workflow error. This usually means:\n"
        error_detail += "1. Workflow not imported into N8N\n"
        error_detail += "2. Workflow not activated\n"
        error_detail += "3. Go to http://localhost:5678 and import: n8n/medical_qa_workflow_complete.json\n"
        error_detail += f"Original error: {e.response.text}"
        raise HTTPException(status_code=503, detail=error_detail)
    except Exception as e:
        logger.error(f"❌ N8N Error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"N8N service unavailable: {str(e)}")

def add_to_history(request_data: Dict, response_data: Dict):
    """Add request to history"""
    global request_history
    
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data
    }
    
    request_history.append(history_entry)
    
    # Keep only last MAX_HISTORY entries
    if len(request_history) > MAX_HISTORY:
        request_history = request_history[-MAX_HISTORY:]

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "🏥 Medical AI Q&A API",
        "version": "2.0.0",
        "status": "online",
        "architecture": "Frontend → FastAPI → N8N → Langchain → Custom Model",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "predict": "/predict (POST)",
            "langchain_direct": "/langchain/query (POST)",
            "n8n_direct": "/n8n/query (POST)",
            "history": "/history",
            "docs": "/docs"
        },
        "features": [
            "Custom trained medical model",
            "RAG with medical knowledge base",
            "N8N workflow orchestration",
            "Real-time system monitoring",
            "Request history tracking"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Quick health check"""
    return HealthResponse(
        status="healthy",
        service="Integrated Medical API",
        timestamp=datetime.now().isoformat()
    )

@app.get("/status", response_model=SystemStatus)
async def system_status():
    """Complete system status with detailed information"""
    
    # FastAPI status
    fastapi_status = {
        "online": True,
        "version": "2.0.0 (FLAN-T5 Enhanced)",
        "timestamp": datetime.now().isoformat()
    }
    
    # Langchain status with detailed info
    if langchain_service:
        try:
            langchain_status = langchain_service.get_status()
        except Exception as e:
            langchain_status = {
                "status": "error",
                "error": str(e),
                "initialized": False
            }
    else:
        langchain_status = {
            "status": "offline",
            "error": "Langchain service not initialized - check model path",
            "initialized": False
        }
    
    # N8N status
    try:
        n8n_online = await check_n8n_status()
    except Exception:
        n8n_online = False
        
    n8n_status = {
        "online": n8n_online,
        "webhook_url": N8N_WEBHOOK_URL,
        "status": "available" if n8n_online else "unavailable"
    }
    
    # Model status from langchain service
    if langchain_service:
        service_status = langchain_service.get_status()
        model_status = {
            "loaded": service_status.get("model_loaded", False),
            "path": service_status.get("model_path", "Not found"),
            "model_type": service_status.get("model_type", "Unknown"),
            "accuracy": service_status.get("model_accuracy", "N/A"),
            "device": service_status.get("device", "N/A"),
            "status": "ready" if service_status.get("model_loaded") else "not loaded",
            "emergency_rules": service_status.get("emergency_rules", 0)
        }
    else:
        model_status = {
            "loaded": False,
            "path": "No model found",
            "model_type": "None",
            "accuracy": "N/A",
            "device": "N/A",
            "status": "not loaded",
            "error": "Langchain service not initialized"
        }
    
    # Vector store status
    if langchain_service and hasattr(langchain_service, 'knowledge_base'):
        try:
            doc_count = langchain_service.knowledge_base.vectorstore._collection.count() if langchain_service.knowledge_base.vectorstore else 0
            vectorstore_status = {
                "available": langchain_service.knowledge_base.vectorstore is not None,
                "documents": doc_count,
                "path": VECTORSTORE_PATH
            }
        except Exception as e:
            vectorstore_status = {
                "available": False,
                "documents": 0,
                "path": VECTORSTORE_PATH,
                "error": str(e)
            }
    else:
        vectorstore_status = {
            "available": False,
            "documents": 0,
            "path": VECTORSTORE_PATH,
            "error": "Knowledge base not initialized"
        }
    
    return SystemStatus(
        fastapi=fastapi_status,
        langchain=langchain_status,
        n8n=n8n_status,
        model=model_status,
        vectorstore=vectorstore_status
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: QuestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Main prediction endpoint with smart routing
    
    Flow:
    1. If use_n8n=True: Route through N8N workflow
    2. Else if use_langchain=True: Use Langchain RAG
    3. Else: Direct model inference (not implemented)
    """
    
    start_time = datetime.now()
    
    try:
        # Route through N8N
        if request.use_n8n:
            print(f"🌊 Routing through N8N: {request.question[:50]}...")
            
            n8n_result = await call_n8n_webhook(request.question, request.user_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = PredictionResponse(
                success=True,
                answer=n8n_result.get("answer", "No answer from N8N"),
                sources=n8n_result.get("sources", []),
                method="n8n",
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                disclaimer="⚕️ This is general health information only. Always consult qualified healthcare professionals."
            )
        
        # Use Langchain RAG
        elif request.use_langchain:
            if not langchain_service:
                raise HTTPException(
                    status_code=503,
                    detail="Langchain service not available. Please check model path and initialization."
                )
            
            print(f"🧠 Processing with Langchain: {request.question[:50]}...")
            
            result = langchain_service.ask_question(
                question=request.question,
                user_id=request.user_id
            )
            
            response = PredictionResponse(
                success=result["success"],
                answer=result["answer"],
                sources=result.get("sources", []),
                method="langchain",
                processing_time=result["processing_time"],
                timestamp=result["timestamp"],
                disclaimer=result.get("disclaimer"),
                error=result.get("error")
            )
        
        else:
            # Direct model (not implemented in this version)
            raise HTTPException(
                status_code=400,
                detail="Direct model inference not available. Use use_langchain=True or use_n8n=True"
            )
        
        # Add to history (background task)
        background_tasks.add_task(
            add_to_history,
            request.dict(),
            response.dict()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            success=False,
            answer="I apologize, but I encountered an error processing your question.",
            sources=[],
            method="error",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.post("/langchain/query")
async def langchain_query(request: QuestionRequest):
    """Direct Langchain endpoint (bypasses N8N)"""
    
    if not langchain_service:
        raise HTTPException(
            status_code=503,
            detail="Langchain service not available"
        )
    
    result = langchain_service.ask_question(
        question=request.question,
        user_id=request.user_id
    )
    
    return result

@app.post("/n8n/query")
async def n8n_query(request: QuestionRequest):
    """Direct N8N endpoint (bypasses Langchain routing)"""
    
    n8n_available = await check_n8n_status()
    if not n8n_available:
        raise HTTPException(
            status_code=503,
            detail=f"N8N service not available at {N8N_WEBHOOK_URL}"
        )
    
    result = await call_n8n_webhook(request.question, request.user_id)
    return result

@app.get("/history")
async def get_history(limit: int = 20):
    """Get request history"""
    return {
        "total": len(request_history),
        "limit": limit,
        "history": request_history[-limit:]
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Use ASCII-safe characters for Windows console
    print("\n" + "="*80)
    print("INTEGRATED MEDICAL API SERVER (FLAN-T5 Enhanced)")
    print("="*80)
    print(f"\nModel: Auto-detecting (FLAN-T5-Base or T5-Small)")
    print(f"Vector Store: {VECTORSTORE_PATH}")
    print(f"N8N Webhook: {N8N_WEBHOOK_URL}\n")
    print("="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
