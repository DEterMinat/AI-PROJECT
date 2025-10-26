# 🚨 คู่มือแก้ไขปัญหา Medical AI System

## 📋 ปัญหาที่พบบ่อยและวิธีแก้ไข

---

## 🐍 Python และ Environment Issues

### 1. ModuleNotFoundError

**ปัญหา:**
```
ModuleNotFoundError: No module named 'langchain'
ImportError: cannot import name 'FastAPI' from 'fastapi'
```

**วิธีแก้:**
```bash
# 1. ตรวจสอบว่าใช้ virtual environment
python -c "import sys; print(sys.prefix)"

# 2. Activate virtual environment
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. ติดตั้ง dependencies ใหม่
pip install -r requirements.txt

# 4. ถ้ายังไม่ได้ ให้ reinstall
pip uninstall -y langchain fastapi
pip install langchain fastapi

# 5. เช็คเวอร์ชัน
pip list | findstr langchain
pip list | findstr fastapi
```

### 2. CUDA/GPU Issues

**ปัญหา:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**วิธีแก้:**
```bash
# 1. บังคับใช้ CPU
export CUDA_VISIBLE_DEVICES=""

# 2. ใช้ float16 เพื่อลด memory
export TORCH_DTYPE=float16

# 3. ลด batch size ใน config
# แก้ไขใน langchain_config.json
{
  "model": {
    "torch_dtype": "float16",
    "device_map": "cpu"
  }
}

# 4. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### 3. Encoding Issues

**ปัญหา:**
```
UnicodeDecodeError: 'charmap' codec can't decode byte
UnicodeEncodeError: 'ascii' codec can't encode character
```

**วิธีแก้:**
```bash
# 1. Set encoding environment variables
set PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

# 2. ใน Python code เพิ่ม encoding
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 3. ใน FastAPI main.py เพิ่ม
# -*- coding: utf-8 -*-
```

---

## 🌐 FastAPI Issues

### 1. Server Won't Start

**ปัญหา:**
```
Address already in use
[ERROR] Error loading ASGI app
```

**วิธีแก้:**
```bash
# 1. เช็คว่า port ถูกใช้แล้วหรือไม่
netstat -an | findstr :8000
# Linux: netstat -tulpn | grep :8000

# 2. Kill process ที่ใช้ port
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Linux
sudo lsof -ti:8000 | xargs kill -9

# 3. ใช้ port อื่น
uvicorn fastapi.app.main:app --port 8001

# 4. เช็ค Firewall/Antivirus
# อาจต้อง allow port 8000 ใน firewall
```

### 2. API Endpoints Not Working

**ปัญหา:**
```bash
curl http://localhost:8000/api/medical-qa
# 404 Not Found
```

**Debug Steps:**
```bash
# 1. เช็ค FastAPI docs
http://localhost:8000/docs

# 2. เช็ค health endpoint
curl http://localhost:8000/health

# 3. เช็ค logs
tail -f logs/fastapi.log

# 4. Test endpoint อื่น
curl http://localhost:8000/

# 5. เช็ค route registration
python -c "
from fastapi.app.main import app
for route in app.routes:
    print(f'{route.methods} {route.path}')
"
```

### 3. Slow API Response

**ปัญหา:**
- API ตอบช้า (> 10 วินาที)
- Timeout errors

**Optimization:**
```python
# 1. เพิ่ม async/await
@app.post("/api/medical-qa")
async def ask_medical_question(request: QuestionRequest):
    # ใช้ async operations
    
# 2. Connection pooling
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/api/medical-qa")
async def ask_medical_question(request: QuestionRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        medical_service.ask_question, 
        request.question
    )
    return result

# 3. Caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_ask_question(question: str):
    return medical_service.ask_question(question)

# 4. Background tasks
from fastapi import BackgroundTasks

@app.post("/api/medical-qa")
async def ask_medical_question(
    request: QuestionRequest, 
    background_tasks: BackgroundTasks
):
    # ตอบเร็วๆ แล้วประมวลผลใน background
    background_tasks.add_task(log_question, request.question)
    return quick_response
```

---

## 🧠 Langchain Service Issues

### 1. Model Loading Errors

**ปัญหา:**
```
OSError: ./models/trained does not appear to have a file named config.json
ValueError: Tokenizer class AutoTokenizer does not exist
```

**วิธีแก้:**
```bash
# 1. เช็ค model structure
ls -la models/trained/
# ต้องมี config.json, tokenizer.json, pytorch_model.bin

# 2. Download default model
python -c "
from transformers import AutoTokenizer, AutoModel
model_name = 'microsoft/DialoGPT-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.save_pretrained('./models/default')
model.save_pretrained('./models/default')
"

# 3. ใช้ HuggingFace model โดยตรง
# ใน langchain_service/medical_ai.py
model_name = "microsoft/DialoGPT-medium"  # แทน local path

# 4. Clear cache and retry
rm -rf ~/.cache/huggingface/
python run_langchain.py
```

### 2. Vector Database Issues

**ปัญหา:**
```
ChromaDB connection error
sqlite3.OperationalError: database is locked
```

**วิธีแก้:**
```bash
# 1. เช็ค ChromaDB directory
ls -la data/vectorstore/

# 2. Reset ChromaDB
rm -rf data/vectorstore/chroma.sqlite3*
mkdir -p data/vectorstore

# 3. ใช้ in-memory ChromaDB สำหรับ testing
# ใน langchain_service/medical_ai.py
vector_store = Chroma(
    collection_name="medical_knowledge",
    embedding_function=embeddings,
    # ลบ persist_directory สำหรับ in-memory
)

# 4. เช็ค file permissions
chmod -R 755 data/vectorstore/
```

### 3. Memory Issues with Large Models

**ปัญหา:**
```
RuntimeError: CUDA out of memory
killed (Out of memory)
```

**Solutions:**
```python
# 1. Model optimization
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # ลด precision
    device_map="auto",          # automatic device allocation
    low_cpu_mem_usage=True,     # ลด CPU memory
    load_in_8bit=True          # quantization
)

# 2. Batch processing
def process_questions_in_batches(questions, batch_size=4):
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_results = model.generate(batch)
        results.extend(batch_results)
        # Clear memory after each batch
        torch.cuda.empty_cache()
    return results

# 3. Model switching
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.models = {}
    
    def load_model(self, model_name):
        if self.current_model and self.current_model != model_name:
            # Unload current model
            del self.models[self.current_model]
            torch.cuda.empty_cache()
        
        if model_name not in self.models:
            self.models[model_name] = load_model(model_name)
        
        self.current_model = model_name
        return self.models[model_name]
```

---

## 🔄 N8N Workflow Issues

### 1. Webhook Not Receiving Requests

**ปัญหา:**
```bash
curl -X POST "http://localhost:5678/webhook/medical-qa"
# Connection refused or 404
```

**วิธีแก้:**
```bash
# 1. เช็ค N8N service
docker ps | grep n8n
# หรือ
curl http://localhost:5678/

# 2. เช็ค webhook configuration
# ใน N8N UI ไปที่ webhook node settings
# ตรวจสอบ Path และ HTTP Method

# 3. Test webhook directly ใน N8N
# กด "Listen for calls" button ใน webhook node

# 4. เช็ค N8N logs
docker logs n8n_container

# 5. Network issues
# ตรวจสอบว่า N8N และ FastAPI อยู่ใน network เดียวกัน
docker network ls
docker network inspect bridge
```

### 2. HTTP Request Node Failures

**ปัญหา:**
```
Error: getaddrinfo ENOTFOUND localhost
Request failed with status code 500
```

**วิธีแก้:**
```javascript
// 1. ใช้ container name แทน localhost (ใน Docker)
// แทน: http://localhost:8000/api/medical-qa
// ใช้: http://fastapi-service:8000/api/medical-qa

// 2. เพิ่ม error handling ใน HTTP Request node
{
  "url": "http://localhost:8000/api/medical-qa",
  "method": "POST",
  "timeout": 30000,
  "retry": {
    "count": 3,
    "delay": 1000
  },
  "ignoreHttpStatusErrors": true
}

// 3. ใน Code node เพิ่ม try-catch
try {
  const response = await this.helpers.httpRequest({
    method: 'POST',
    url: 'http://localhost:8000/api/medical-qa',
    body: { question: items[0].json.question },
    json: true
  });
  return [{ json: response }];
} catch (error) {
  return [{ 
    json: { 
      error: error.message,
      status: 'failed',
      timestamp: new Date().toISOString()
    }
  }];
}
```

### 3. Database Node Issues

**ปัญหา:**
```
SQLite database is locked
Connection timeout
```

**วิธีแก้:**
```bash
# 1. เช็ค database file permissions
ls -la data/medical_ai.db
chmod 666 data/medical_ai.db

# 2. เช็คว่าไม่มี process อื่นใช้ database
lsof data/medical_ai.db

# 3. ใช้ connection pooling
# ใน N8N Database node configuration
{
  "maxConnections": 5,
  "connectionTimeout": 30000,
  "acquireTimeout": 30000
}

# 4. Alternative: ใช้ API แทน direct DB access
# แทนที่จะเขียนตรงไป database
# ให้เรียก API endpoint เพื่อบันทึกข้อมูล
```

---

## 🐳 Docker Issues

### 1. Container Won't Start

**ปัญหา:**
```
docker-compose up
# Container exits immediately
```

**Debug Steps:**
```bash
# 1. เช็ค container logs
docker logs container_name

# 2. เช็ค docker-compose logs
docker-compose logs service_name

# 3. Run container interactively
docker run -it --entrypoint /bin/bash image_name

# 4. เช็ค Dockerfile syntax
docker build --no-cache .

# 5. เช็ค port conflicts
docker ps -a
netstat -an | findstr :8000
```

### 2. Volume Mount Issues

**ปัญหา:**
```
bind: no such file or directory
Permission denied
```

**วิธีแก้:**
```bash
# 1. สร้าง directories ก่อน mount
mkdir -p data/{raw,processed,vectorstore}
mkdir -p models/{trained,cache}
mkdir -p logs

# 2. ตั้งค่า permissions (Linux)
sudo chown -R 1000:1000 data/
sudo chmod -R 755 data/

# 3. ใน docker-compose.yml ใช้ absolute paths
volumes:
  - /absolute/path/to/data:/app/data
  - /absolute/path/to/models:/app/models

# 4. Windows: ตรวจสอบ Docker Desktop settings
# Settings > Resources > File Sharing
```

### 3. Network Connectivity Issues

**ปัญหา:**
- Services ไม่สามารถเชื่อมต่อกันได้
- External API calls fail

**วิธีแก้:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  langchain-service:
    networks:
      - medical-ai-network
  
  n8n:
    networks:
      - medical-ai-network
    environment:
      - N8N_HOST=0.0.0.0
      - WEBHOOK_URL=http://n8n:5678/

networks:
  medical-ai-network:
    driver: bridge
```

```bash
# Test connectivity
docker exec -it container_name ping other_container_name
docker exec -it container_name curl http://other_service:8000/health

# เช็ค network configuration
docker network inspect medical-ai_default
```

---

## 🔍 Debugging Tools และ Techniques

### 1. Logging Setup

**Enhanced Logging Configuration:**
```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'question'):
            log_obj['question'] = record.question
            
        return json.dumps(log_obj, ensure_ascii=False)

# Setup logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('logs/medical_ai.log')
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    return logger

# ใช้งาน
logger = setup_logging()
logger.info("System started", extra={"component": "langchain_service"})
```

### 2. Health Check Endpoints

```python
# health_check.py
@app.get("/health/detailed")
def detailed_health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check Langchain service
    try:
        if medical_service and medical_service.vector_store:
            health_status["components"]["langchain"] = {
                "status": "healthy",
                "vector_store_size": len(medical_service.vector_store._collection.get()["ids"])
            }
        else:
            health_status["components"]["langchain"] = {"status": "unhealthy"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["langchain"] = {
            "status": "error", 
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM medical_qa_log")
        count = cursor.fetchone()[0]
        conn.close()
        
        health_status["components"]["database"] = {
            "status": "healthy",
            "total_records": count
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check system resources
    import psutil
    health_status["components"]["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('.').percent
    }
    
    return health_status
```

### 3. Performance Monitoring

```python
# performance_monitor.py
import time
from functools import wraps
import threading
from collections import defaultdict, deque

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "call_count": 0,
            "total_time": 0,
            "avg_time": 0,
            "recent_times": deque(maxlen=100)
        })
        self.lock = threading.Lock()
    
    def track(self, func_name=None):
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    with self.lock:
                        metrics = self.metrics[name]
                        metrics["call_count"] += 1
                        metrics["total_time"] += duration
                        metrics["avg_time"] = metrics["total_time"] / metrics["call_count"]
                        metrics["recent_times"].append(duration)
                        
                        # Log slow operations
                        if duration > 5.0:  # 5 seconds
                            logger.warning(f"Slow operation detected: {name} took {duration:.2f}s")
            
            return wrapper
        return decorator
    
    def get_stats(self):
        with self.lock:
            return dict(self.metrics)

# Global monitor instance
performance_monitor = PerformanceMonitor()

# ใช้งาน
@performance_monitor.track("medical_qa")
def ask_question(question: str):
    return medical_service.ask_question(question)

@app.get("/metrics")
def get_metrics():
    return performance_monitor.get_stats()
```

### 4. Error Tracking

```python
# error_tracker.py
import traceback
from datetime import datetime, timedelta
from collections import defaultdict

class ErrorTracker:
    def __init__(self):
        self.errors = defaultdict(list)
    
    def track_error(self, error: Exception, context: dict = None):
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.errors[type(error).__name__].append(error_info)
        
        # Keep only recent errors (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        for error_type in self.errors:
            self.errors[error_type] = [
                err for err in self.errors[error_type]
                if datetime.fromisoformat(err["timestamp"]) > cutoff
            ]
    
    def get_error_summary(self):
        summary = {}
        for error_type, errors in self.errors.items():
            recent_errors = [
                err for err in errors
                if datetime.fromisoformat(err["timestamp"]) > datetime.now() - timedelta(hours=1)
            ]
            
            summary[error_type] = {
                "total_24h": len(errors),
                "recent_1h": len(recent_errors),
                "latest": errors[-1] if errors else None
            }
        
        return summary

error_tracker = ErrorTracker()

# ใช้งาน
def handle_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_tracker.track_error(e, {
                "function": func.__name__,
                "args": str(args)[:200],  # Truncate long args
                "kwargs": str(kwargs)[:200]
            })
            raise
    return wrapper

@app.get("/errors")
def get_errors():
    return error_tracker.get_error_summary()
```

---

## 🔧 Configuration Validation

### Automated System Check Script

```python
# system_check.py
#!/usr/bin/env python3
"""
System health and configuration check script
"""
import os
import sys
import json
import requests
import subprocess
from pathlib import Path

class SystemChecker:
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def check_python_environment(self):
        """Check Python version and packages"""
        print("🐍 Checking Python environment...")
        
        # Python version
        if sys.version_info < (3, 8):
            self.issues.append(f"Python {sys.version} is too old. Need 3.8+")
        
        # Required packages
        required_packages = [
            'langchain', 'fastapi', 'chromadb', 
            'transformers', 'torch', 'uvicorn'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                self.issues.append(f"Missing package: {package}")
                print(f"  ❌ {package}")
    
    def check_directories(self):
        """Check required directories exist"""
        print("\n📁 Checking directories...")
        
        required_dirs = [
            'data/raw', 'data/processed', 'data/vectorstore',
            'models/trained', 'logs', 'fastapi/app'
        ]
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print(f"  ✅ {dir_path}")
            else:
                self.warnings.append(f"Missing directory: {dir_path}")
                print(f"  ⚠️ {dir_path}")
    
    def check_configuration_files(self):
        """Check configuration files"""
        print("\n⚙️ Checking configuration files...")
        
        config_files = [
            ('requirements.txt', True),
            ('docker-compose.yml', False),
            ('Dockerfile', False),
            ('config/langchain_config.json', False)
        ]
        
        for file_path, required in config_files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
                
                # Validate JSON files
                if file_path.endswith('.json'):
                    try:
                        with open(file_path) as f:
                            json.load(f)
                        print(f"    ✅ Valid JSON")
                    except json.JSONDecodeError as e:
                        self.issues.append(f"Invalid JSON in {file_path}: {e}")
            else:
                if required:
                    self.issues.append(f"Missing required file: {file_path}")
                    print(f"  ❌ {file_path}")
                else:
                    self.warnings.append(f"Missing optional file: {file_path}")
                    print(f"  ⚠️ {file_path}")
    
    def check_services(self):
        """Check if services are running"""
        print("\n🌐 Checking services...")
        
        services = [
            ("FastAPI", "http://localhost:8000/health"),
            ("N8N", "http://localhost:5678"),
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ {service_name} is running")
                else:
                    self.warnings.append(f"{service_name} returned status {response.status_code}")
                    print(f"  ⚠️ {service_name} - Status {response.status_code}")
            except requests.exceptions.RequestException:
                self.warnings.append(f"{service_name} is not responding")
                print(f"  ❌ {service_name} is not running")
    
    def check_docker(self):
        """Check Docker setup"""
        print("\n🐳 Checking Docker...")
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Docker: {result.stdout.strip()}")
            else:
                self.warnings.append("Docker not found")
                print(f"  ❌ Docker not available")
        except FileNotFoundError:
            self.warnings.append("Docker not installed")
            print(f"  ❌ Docker not installed")
        
        # Check docker-compose
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Docker Compose: {result.stdout.strip()}")
            else:
                self.warnings.append("Docker Compose not found")
        except FileNotFoundError:
            self.warnings.append("Docker Compose not installed")
    
    def run_all_checks(self):
        """Run all system checks"""
        print("🔍 Medical AI System Health Check")
        print("=" * 50)
        
        self.check_python_environment()
        self.check_directories()
        self.check_configuration_files()
        self.check_services()
        self.check_docker()
        
        print("\n" + "=" * 50)
        print("📊 Summary")
        
        if not self.issues and not self.warnings:
            print("🎉 All checks passed! System is healthy.")
        else:
            if self.issues:
                print(f"\n❌ Issues found ({len(self.issues)}):")
                for issue in self.issues:
                    print(f"   • {issue}")
            
            if self.warnings:
                print(f"\n⚠️ Warnings ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"   • {warning}")
        
        return len(self.issues) == 0

if __name__ == "__main__":
    checker = SystemChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)
```

**วิธีใช้:**
```bash
# รัน system check
python system_check.py

# หรือเพิ่มใน batch script
./check-system.bat
```

---

## 🆘 Emergency Procedures

### 1. Service Recovery Script

```bash
#!/bin/bash
# emergency_restart.sh

echo "🚨 Emergency Service Recovery"
echo "=========================="

# Stop all services
echo "Stopping services..."
docker-compose down
pkill -f "uvicorn"
pkill -f "n8n"

# Clear temporary files
echo "Clearing temporary files..."
rm -f logs/*.log
rm -f data/*.lock
rm -f /tmp/langchain_*

# Restart services
echo "Starting services..."
docker-compose up -d

# Wait and test
sleep 30
echo "Testing services..."
curl -s http://localhost:8000/health || echo "❌ FastAPI failed"
curl -s http://localhost:5678 || echo "❌ N8N failed"

echo "Recovery complete!"
```

### 2. Data Backup Script

```bash
#!/bin/bash
# backup_data.sh

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🔄 Creating backup..."

# Backup database
cp data/medical_ai.db "$BACKUP_DIR/"

# Backup vector store
cp -r data/vectorstore "$BACKUP_DIR/"

# Backup configuration
cp -r config "$BACKUP_DIR/"

# Backup logs
cp -r logs "$BACKUP_DIR/"

echo "✅ Backup created: $BACKUP_DIR"

# Cleanup old backups (keep last 7 days)
find backups/ -type d -mtime +7 -exec rm -rf {} +
```

### 3. Quick Fixes Cheat Sheet

**Print this out and keep handy:**

```
🚨 EMERGENCY QUICK FIXES

1. Service Down:
   docker-compose restart
   OR
   ./emergency_restart.sh

2. Memory Full:
   docker system prune -f
   rm -rf data/vectorstore/chroma.sqlite3*
   
3. Port Conflicts:
   netstat -tulpn | grep :8000
   kill -9 <PID>
   
4. Database Locked:
   rm data/medical_ai.db-*
   service sqlite3 restart
   
5. Model Loading Error:
   rm -rf models/cache/
   python -c "import torch; torch.cuda.empty_cache()"
   
6. N8N Webhook Failed:
   curl -X POST http://localhost:5678/webhook-test/medical-qa
   
7. Complete Reset:
   docker-compose down -v
   rm -rf data/vectorstore/
   docker-compose up -d

8. Check System Health:
   python system_check.py
   
9. View Recent Logs:
   tail -f logs/medical_ai.log
   docker-compose logs -f --tail=50

10. Emergency Contacts:
    - System Admin: [phone/email]
    - Technical Lead: [phone/email]
```

คู่มือนี้ครอบคลุมปัญหาที่พบบ่อยแล้วครับ มีส่วนไหนที่ต้องการให้อธิบายเพิ่มเติมไหมครับ?