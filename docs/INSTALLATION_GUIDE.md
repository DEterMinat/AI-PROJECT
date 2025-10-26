# 📚 คู่มือการติดตั้งและใช้งานระบบ Medical AI

## 🎯 Overview

ระบบนี้เป็น Medical AI Q&A System ที่ใช้ **Langchain** และ **N8N** เป็นหลัก พร้อมด้วย **FastAPI** เป็น optional API layer

### ⭐ Core Components
- **Langchain** - หัวใจหลักของระบบ AI สำหรับการตอบคำถามทางการแพทย์
- **N8N** - Workflow orchestration และ automation
- **FastAPI** - Optional REST API wrapper
- **ChromaDB** - Vector database สำหรับเก็บความรู้ทางการแพทย์
- **SQLite** - Database สำหรับ logging และ analytics

---

## 🚀 การติดตั้งระบบ

### ขั้นตอนที่ 1: ตรวจสอบ Requirements

**System Requirements:**
- Python 3.8+
- Docker & Docker Compose
- Git
- อย่างน้อย 8GB RAM (สำหรับ AI models)
- อย่างน้อย 10GB disk space

**เช็คเวอร์ชัน:**
```bash
python --version
docker --version
docker-compose --version
git --version
```

### ขั้นตอนที่ 2: Clone และ Setup Environment

```bash
# Clone repository
git clone <your-repo-url> medical-ai-system
cd medical-ai-system

# สร้าง virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate

# Activate environment (Linux/Mac)
source .venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt
```

### ขั้นตอนที่ 3: การ Config เบื้องต้น

**สร้างโฟลเดอร์ที่จำเป็น:**
```bash
mkdir -p data/{raw,processed,exports}
mkdir -p models/{trained,cache}
mkdir -p logs
```

**ตั้งค่า environment variables:**
```bash
# สร้าง .env file
echo "ENVIRONMENT=development" > .env
echo "LOG_LEVEL=info" >> .env
echo "MODEL_CACHE_DIR=./models/cache" >> .env
echo "CHROMA_DB_PATH=./data/vectorstore" >> .env
```

---

## 🧠 Langchain Service Setup

### การติดตั้งและเริ่มต้นใช้งาน

**1. รัน Langchain Service แบบ Standalone:**
```bash
python run_langchain.py
```

**2. ตัวอย่าง Output:**
```
🏥 Initializing Langchain Medical AI Service...
🤖 Loading default medical model...
📚 Initializing ChromaDB vector store...
🔧 Setting up RetrievalQA chain...
✅ Langchain Medical Service ready!

💬 Interactive Mode - พิมพ์คำถาม หรือ 'quit' เพื่อออก
คำถาม: อาการของโรคเบาหวานคืออะไร?
🤖 AI: อาการของโรคเบาหวาน ได้แก่ กระหายน้ำมาก ปัสสาวะบ่อย น้ำหนักลด เหนื่อยง่าย
📊 Confidence: 0.87 | Sources: ['medical_knowledge_base']
```

### การใช้ Custom Model

**1. วาง Model Files:**
```
models/
├── my_medical_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── special_tokens_map.json
```

**2. รัน Service ด้วย Custom Model:**
```bash
python run_langchain.py --model-path ./models/my_medical_model
```

### การเพิ่มความรู้ใหม่

**ผ่าน Python API:**
```python
from langchain_service.medical_ai import LangchainMedicalService

service = LangchainMedicalService()

# เพิ่มความรู้เป็นชิ้นๆ
service.add_knowledge(
    content="โรคเบาหวานเป็นโรคที่มีระดับน้ำตาลในเลือดสูง...",
    metadata={"topic": "diabetes", "source": "medical_textbook"}
)

# เพิ่มจากไฟล์
service.add_knowledge_from_file("data/medical_articles.txt")
```

**ผ่าน Batch Script:**
```bash
# วางไฟล์ข้อมูลใน data/raw/
# จากนั้นรัน
python tools/data_processing/add_knowledge.py
```

---

## 🌐 FastAPI Integration

### การเริ่มต้น FastAPI Server

**1. รัน FastAPI แบบ Development:**
```bash
cd fastapi/app
python main.py
```

**2. รัน FastAPI แบบ Production:**
```bash
uvicorn fastapi.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints Documentation

**Base URL:** `http://localhost:8000`

#### 1. Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "langchain_service": "connected",
  "stats": {
    "questions_answered": 145,
    "knowledge_base_size": 1250,
    "avg_confidence": 0.82
  },
  "timestamp": "2025-09-15T14:30:15Z"
}
```

#### 2. Ask Medical Question
```bash
POST /api/medical-qa
Content-Type: application/json

{
  "question": "อาการของโรคเบาหวานคืออะไร?",
  "user_id": "user123"
}
```
**Response:**
```json
{
  "answer": "อาการของโรคเบาหวาน ได้แก่ กระหายน้ำมาก ปัสสาวะบ่อย น้ำหนักลด เหนื่อยง่าย",
  "confidence": 0.87,
  "sources": ["medical_knowledge_base"],
  "response_time": 0.45,
  "status": "success"
}
```

#### 3. Add Knowledge
```bash
POST /api/add-knowledge
Content-Type: application/json

{
  "content": "ข้อมูลทางการแพทย์ใหม่...",
  "topic": "diabetes",
  "category": "symptoms"
}
```

#### 4. Get Statistics
```bash
GET /api/stats
```

#### 5. Test Endpoint
```bash
POST /api/test?question=อาการของโรคเบาหวานคืออะไร?
```

### การใช้งาน API Documentation
เข้าไปที่ `http://localhost:8000/docs` เพื่อดู Interactive API Documentation (Swagger UI)

### การทดสอบ API

**ผ่าน cURL:**
```bash
# Health check
curl http://localhost:8000/health

# Ask question
curl -X POST "http://localhost:8000/api/medical-qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "อาการของโรคเบาหวานคืออะไร?", "user_id": "test"}'

# Add knowledge
curl -X POST "http://localhost:8000/api/add-knowledge" \
  -H "Content-Type: application/json" \
  -d '{"content": "โรคเบาหวานเป็นโรค...", "topic": "diabetes", "category": "info"}'
```

**ผ่าน Python:**
```python
import requests

# Ask question
response = requests.post(
    "http://localhost:8000/api/medical-qa",
    json={"question": "อาการของโรคเบาหวานคืออะไร?", "user_id": "python_test"}
)
print(response.json())

# Health check
health = requests.get("http://localhost:8000/health")
print(health.json())
```

---

## 🔄 N8N Workflow Integration

### การ Setup N8N

**1. รัน N8N Server:**
```bash
# แบบ Standalone
npx n8n

# หรือผ่าน Docker
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n
```

**2. เข้าใช้งาน N8N:**
- เปิด `http://localhost:5678`
- สร้าง account แรก
- Login เข้าสู่ระบบ

### Import Medical AI Workflow

**1. Import Workflow JSON:**
```bash
# Copy workflow file
cp n8n_workflows/medical_qa_workflow.json /path/to/n8n/workflows/

# หรือ import ผ่าน N8N UI
```

**2. Workflow Components:**
- **Webhook Node** - รับ HTTP requests
- **HTTP Request Node** - เรียก FastAPI endpoint
- **Code Node** - ประมวลผล response
- **Database Node** - บันทึกผลลัพธ์

### การสร้าง Custom Workflow

**1. Medical Q&A Workflow:**
```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "medical-qa",
        "httpMethod": "POST"
      }
    },
    {
      "name": "Call Langchain API",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/api/medical-qa",
        "method": "POST",
        "body": {
          "question": "={{ $json.question }}",
          "user_id": "={{ $json.user_id }}"
        }
      }
    },
    {
      "name": "Process Response",
      "type": "n8n-nodes-base.code",
      "parameters": {
        "jsCode": "return [{ json: { ...items[0].json, processed_at: new Date().toISOString() } }];"
      }
    }
  ]
}
```

**2. การทดสอบ Workflow:**
```bash
curl -X POST "http://localhost:5678/webhook/medical-qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "อาการของโรคเบาหวานคืออะไร?", "user_id": "n8n_test"}'
```

---

## 📊 Database และ Logging

### SQLite Database Structure

**Tables:**
```sql
-- Q&A Logging
CREATE TABLE medical_qa_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    confidence REAL,
    sources TEXT,  -- JSON array
    response_time REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Base Metadata
CREATE TABLE knowledge_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE,
    topic TEXT,
    category TEXT,
    source TEXT,
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### การดู Database

**1. ผ่าน SQLite CLI:**
```bash
sqlite3 data/medical_ai.db

-- ดูข้อมูล Q&A ล่าสุด
SELECT * FROM medical_qa_log ORDER BY created_at DESC LIMIT 10;

-- สถิติการใช้งาน
SELECT 
    COUNT(*) as total_questions,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT user_id) as unique_users
FROM medical_qa_log;
```

**2. ผ่าน Python:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/medical_ai.db')

# Load data เป็น DataFrame
df = pd.read_sql_query("""
    SELECT user_id, question, answer, confidence, created_at 
    FROM medical_qa_log 
    ORDER BY created_at DESC 
    LIMIT 100
""", conn)

print(df.head())
conn.close()
```

---

## 🐳 Docker Deployment

### Docker Compose Setup

**1. รัน Docker Services:**
```bash
# Start all services
docker-compose up -d

# ดู status
docker-compose ps

# ดู logs
docker-compose logs -f langchain-medical
```

**2. Services ที่จะรัน:**
- **langchain-medical** - Main Langchain service (port 8000)
- **n8n** - Workflow orchestration (port 5678)  
- **webapp** - Optional web interface (port 80)

### การ Build Custom Image

**1. Build Langchain Service:**
```bash
docker build -t medical-ai-langchain .
```

**2. Run Custom Container:**
```bash
docker run -d \
  --name medical-ai \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  medical-ai-langchain
```

### Production Deployment

**1. Environment Variables:**
```bash
export ENVIRONMENT=production
export LOG_LEVEL=warning
export MODEL_CACHE_DIR=/app/models/cache
export CHROMA_DB_PATH=/app/data/vectorstore
```

**2. Production Docker Compose:**
```yaml
version: '3.8'
services:
  langchain-medical:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=warning
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - langchain-medical
    restart: unless-stopped
```

---

## 🔧 Configuration และ Customization

### Langchain Configuration

**File:** `config/langchain_config.json`
```json
{
  "model": {
    "type": "huggingface",
    "model_name": "microsoft/DialoGPT-medium",
    "custom_model_path": null,
    "max_length": 512,
    "temperature": 0.7
  },
  "vectorstore": {
    "type": "chroma",
    "persist_directory": "./data/vectorstore",
    "collection_name": "medical_knowledge",
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "retrieval": {
    "search_type": "similarity",
    "k": 5,
    "score_threshold": 0.7
  },
  "logging": {
    "level": "INFO",
    "file": "logs/langchain.log",
    "max_size": "10MB",
    "backup_count": 5
  }
}
```

### FastAPI Configuration

**Environment Variables:**
```bash
# Server settings
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_WORKERS=4

# CORS settings
CORS_ORIGINS=["*"]
CORS_METHODS=["GET", "POST"]

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Model Performance Tuning

**1. Memory Optimization:**
```python
# ใน langchain_service/medical_ai.py
TORCH_SETTINGS = {
    "torch_dtype": "float16",  # ลด memory usage
    "device_map": "auto",      # automatic GPU allocation
    "low_cpu_mem_usage": True  # optimize CPU memory
}
```

**2. Batch Processing:**
```python
# ประมวลผลหลายคำถามพร้อมกัน
def batch_ask_questions(questions: List[str], batch_size: int = 4):
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_results = [service.ask_question(q) for q in batch]
        results.extend(batch_results)
    return results
```

---

## 🔍 Monitoring และ Analytics

### Log Analysis

**1. ดู Logs แบบ Real-time:**
```bash
# Langchain service logs
tail -f logs/langchain.log

# FastAPI logs
tail -f logs/fastapi.log

# Docker logs
docker-compose logs -f langchain-medical
```

**2. Log Analysis Script:**
```python
import pandas as pd
import json
from datetime import datetime, timedelta

def analyze_logs(log_file="logs/langchain.log", days=7):
    # Load และ analyze logs
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f if line.strip()]
    
    df = pd.DataFrame(logs)
    
    # Filter last N days
    cutoff = datetime.now() - timedelta(days=days)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    recent_df = df[df['timestamp'] >= cutoff]
    
    # Analysis
    stats = {
        "total_questions": len(recent_df),
        "avg_confidence": recent_df['confidence'].mean(),
        "avg_response_time": recent_df['response_time'].mean(),
        "top_topics": recent_df['topic'].value_counts().head(10).to_dict()
    }
    
    return stats
```

### Performance Metrics

**1. Response Time Tracking:**
```python
import time
from functools import wraps

def track_response_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log performance
        logger.info({
            "function": func.__name__,
            "response_time": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    return wrapper
```

**2. Memory Usage Monitoring:**
```bash
# ใน production script
import psutil
import GPUtil

def get_system_stats():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_utilization": GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0,
        "disk_usage": psutil.disk_usage('/').percent
    }
```

---

## ✅ Testing และ Quality Assurance

### Unit Tests

**1. Test Langchain Service:**
```python
# tests/test_langchain_service.py
import pytest
from langchain_service.medical_ai import LangchainMedicalService

def test_service_initialization():
    service = LangchainMedicalService()
    assert service is not None
    assert service.vector_store is not None

def test_ask_question():
    service = LangchainMedicalService()
    result = service.ask_question("What is diabetes?")
    
    assert "answer" in result
    assert "confidence" in result
    assert result["confidence"] > 0
    assert len(result["answer"]) > 0

def test_add_knowledge():
    service = LangchainMedicalService()
    service.add_knowledge(
        "Test medical knowledge", 
        {"topic": "test", "category": "test"}
    )
    # Test if knowledge was added successfully
```

**2. Test FastAPI Endpoints:**
```python
# tests/test_fastapi.py
from fastapi.testclient import TestClient
from fastapi.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_medical_qa():
    response = client.post(
        "/api/medical-qa",
        json={"question": "What is diabetes?", "user_id": "test"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```

### Integration Tests

**1. End-to-End Test:**
```bash
# tests/e2e_test.py
def test_full_pipeline():
    # 1. Start services
    # 2. Add knowledge
    # 3. Ask questions
    # 4. Verify responses
    # 5. Check database logs
    pass
```

**2. Load Testing:**
```python
# tests/load_test.py
import asyncio
import aiohttp

async def load_test(num_requests=100):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            task = session.post(
                "http://localhost:8000/api/medical-qa",
                json={"question": f"Test question {i}", "user_id": f"user{i}"}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
```

---

## 📈 Best Practices

### 1. Security
- ใช้ environment variables สำหรับ sensitive data
- ตั้งค่า CORS อย่างเหมาะสม
- ใช้ rate limiting
- Log access และ errors

### 2. Performance
- ใช้ model caching
- Implement batch processing สำหรับ bulk operations
- Monitor memory usage
- Use async operations ที่เหมาะสม

### 3. Maintainability  
- เขียน documentation ที่ครบถ้วน
- ใช้ type hints
- เขียน unit tests
- Use logging ที่เหมาะสม

### 4. Scalability
- ใช้ Docker สำหรับ deployment
- Setup load balancing สำหรับ production
- Use database connection pooling
- Monitor และ optimize ต่อเนื่อง

---

## 🆘 การแก้ปัญหาเบื้องต้น

### ปัญหาที่พบบ่อยและวิธีแก้ไข

**1. Langchain Service ไม่เริ่มต้น:**
```bash
# เช็ค dependencies
pip install -r requirements.txt

# เช็ค model files
ls -la models/

# เช็ค logs
tail -f logs/langchain.log
```

**2. FastAPI ไม่ตอบ:**
```bash
# เช็คว่า service running
curl http://localhost:8000/health

# เช็ค port usage
netstat -an | grep 8000

# Restart service
pkill -f "uvicorn"
python fastapi/app/main.py
```

**3. N8N Workflow ไม่ทำงาน:**
- เช็ค webhook URL
- ตรวจสอบ HTTP request settings
- ดู execution logs ใน N8N UI

**4. Memory Issues:**
```bash
# ลด model size
export TORCH_DTYPE=float16

# ใช้ CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# เพิ่ม swap space (Linux)
sudo swapon --show
```

ยังมีส่วนอื่นๆ ที่ต้องการให้เขียนเพิ่มเติมไหมครับ?