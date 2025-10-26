# 🏥 Medical AI - Docker Setup Guide

## 🚀 Quick Start Commands

```bash
# เริ่มระบบ
start-docker.bat

# หรือใช้ docker-compose ตรง ๆ
docker-compose up -d
```

## 📋 Services Overview

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Web App | 80 | http://localhost | หน้าเว็บหลัก |
| FastAPI | 8000 | http://localhost:8000 | REST API |
| N8N | 5678 | http://localhost:5678 | Workflow Engine |
| SQL Server | 1433 | localhost,1433 | Database |

## 🔧 Configuration

### docker-compose.yml Structure:
```yaml
services:
  fastapi:     # Medical AI API
  n8n:         # Workflow automation  
  sqlserver:   # Database
  webapp:      # Frontend
```

## 🧪 Health Checks

```bash
# FastAPI
curl http://localhost:8000/health

# N8N  
curl http://localhost:5678/healthz

# SQL Server
docker exec medical-ai-sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P YourPassword123 -Q "SELECT 1"
```

## 📊 Monitoring

```bash
# ดูสถานะ containers
docker-compose ps

# ดู resource usage
docker stats

# ดู logs real-time
docker-compose logs -f
```