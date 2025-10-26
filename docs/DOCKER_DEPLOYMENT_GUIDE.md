# 🐳 Docker Deployment Guide สำหรับ Medical AI System

## 📋 Overview

คู่มือนี้จะสอนการ deploy Medical AI System ด้วย Docker ตั้งแต่ development, staging จนถึง production environment

---

## 🚀 Quick Start

### 1. การ Deploy แบบง่าย

```bash
# Clone project
git clone <repository-url> medical-ai-system
cd medical-ai-system

# Start ด้วย Docker Compose
docker-compose up -d

# ตรวจสอบ services
docker-compose ps
docker-compose logs -f
```

### 2. การเข้าถึง Services

- **Langchain API**: http://localhost:8000
- **N8N Workflows**: http://localhost:5678
- **Web Interface**: http://localhost (optional)

---

## 📁 Docker Configuration Files

### 1. Dockerfile สำหรับ Langchain Service

**File:** `Dockerfile`
```dockerfile
# Multi-stage build สำหรับ production
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY langchain_service/ ./langchain_service/
COPY fastapi/ ./fastapi/
COPY run_langchain.py .
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p data/{raw,processed,vectorstore,exports} \
    && mkdir -p models/{trained,cache} \
    && mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "fastapi.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose Configuration

**File:** `docker-compose.yml`
```yaml
version: '3.8'

services:
  langchain-medical:
    build: .
    container_name: medical-ai-langchain
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=info
      - PYTHONPATH=/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    networks:
      - medical-ai-network

  n8n:
    image: n8nio/n8n:latest
    container_name: medical-ai-n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=Asia/Bangkok
      - N8N_LOG_LEVEL=info
      - N8N_SECURE_COOKIE=false
    volumes:
      - n8n_data:/home/node/.n8n
      - ./n8n_workflows:/home/node/.n8n/workflows
    depends_on:
      - langchain-medical
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5678"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    networks:
      - medical-ai-network

  webapp:
    build: 
      context: ./web_app
      dockerfile: Dockerfile
    container_name: medical-ai-webapp
    ports:
      - "80:80"
    environment:
      - API_BASE_URL=http://langchain-medical:8000
      - N8N_BASE_URL=http://n8n:5678
    depends_on:
      - langchain-medical
      - n8n
    restart: unless-stopped
    networks:
      - medical-ai-network

volumes:
  n8n_data:

networks:
  medical-ai-network:
    driver: bridge
```

### 3. Production Docker Compose

**File:** `docker-compose.prod.yml`
```yaml
version: '3.8'

services:
  langchain-medical:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    container_name: medical-ai-langchain-prod
    ports:
      - "8000:8000"
    volumes:
      - medical_data:/app/data
      - medical_models:/app/models
      - medical_logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=warning
      - WORKERS=4
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    restart: unless-stopped
    networks:
      - medical-ai-network

  n8n:
    image: n8nio/n8n:latest
    container_name: medical-ai-n8n-prod
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=${DOMAIN_NAME}
      - N8N_PORT=5678
      - N8N_PROTOCOL=https
      - WEBHOOK_URL=https://${DOMAIN_NAME}/
      - GENERIC_TIMEZONE=Asia/Bangkok
      - N8N_LOG_LEVEL=warn
      - N8N_SECURE_COOKIE=true
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      # Database configuration
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - langchain-medical
      - postgres
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1'
    restart: unless-stopped
    networks:
      - medical-ai-network

  postgres:
    image: postgres:13-alpine
    container_name: medical-ai-postgres
    environment:
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=n8n
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    restart: unless-stopped
    networks:
      - medical-ai-network

  nginx:
    image: nginx:alpine
    container_name: medical-ai-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - langchain-medical
      - n8n
    restart: unless-stopped
    networks:
      - medical-ai-network

  redis:
    image: redis:7-alpine
    container_name: medical-ai-redis
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'
    restart: unless-stopped
    networks:
      - medical-ai-network

volumes:
  medical_data:
  medical_models:
  medical_logs:
  n8n_data:
  postgres_data:
  redis_data:

networks:
  medical-ai-network:
    driver: bridge
```

---

## 🔧 Advanced Configuration

### 1. Production Dockerfile

**File:** `Dockerfile.prod`
```dockerfile
FROM python:3.11-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r medical && useradd -r -g medical medical

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/medical/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=medical:medical langchain_service/ ./langchain_service/
COPY --chown=medical:medical fastapi/ ./fastapi/
COPY --chown=medical:medical run_langchain.py .
COPY --chown=medical:medical config/ ./config/

# Create necessary directories
RUN mkdir -p data/{raw,processed,vectorstore,exports} \
    && mkdir -p models/{trained,cache} \
    && mkdir -p logs \
    && chown -R medical:medical /app

# Switch to non-root user
USER medical

# Set environment variables
ENV PATH=/home/medical/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "fastapi.app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 2. Nginx Configuration

**File:** `nginx/nginx.conf`
```nginx
events {
    worker_connections 1024;
}

http {
    upstream langchain_backend {
        server langchain-medical:8000;
    }

    upstream n8n_backend {
        server n8n:5678;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=n8n_limit:10m rate=20r/m;

    server {
        listen 80;
        server_name ${DOMAIN_NAME};

        # Redirect HTTP to HTTPS in production
        # return 301 https://$server_name$request_uri;

        # API routes
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://langchain_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://langchain_backend/health;
            access_log off;
        }

        # N8N routes
        location /n8n/ {
            limit_req zone=n8n_limit burst=10 nodelay;
            
            proxy_pass http://n8n_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for N8N
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Webhook endpoints
        location /webhook/ {
            proxy_pass http://n8n_backend/webhook/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Static files (if any)
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
    }

    # HTTPS server (production)
    # server {
    #     listen 443 ssl http2;
    #     server_name ${DOMAIN_NAME};
    #
    #     ssl_certificate /etc/nginx/ssl/cert.pem;
    #     ssl_certificate_key /etc/nginx/ssl/key.pem;
    #
    #     # SSL configuration
    #     ssl_protocols TLSv1.2 TLSv1.3;
    #     ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    #     ssl_prefer_server_ciphers off;
    #
    #     # Your location blocks here...
    # }
}
```

### 3. Environment Configuration

**File:** `.env.prod`
```bash
# Production environment variables
ENVIRONMENT=production
LOG_LEVEL=warning

# Domain configuration
DOMAIN_NAME=your-medical-ai-domain.com

# N8N configuration
N8N_USER=admin
N8N_PASSWORD=secure_password_here

# Database
POSTGRES_PASSWORD=secure_db_password_here

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# API Keys (if needed)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_hf_token_here

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_PASSWORD=grafana_password_here

# Backup
BACKUP_S3_BUCKET=your-backup-bucket
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

---

## 🚀 Deployment Strategies

### 1. Development Deployment

```bash
# Development environment
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f langchain-medical

# Scale services
docker-compose up -d --scale langchain-medical=2
```

### 2. Staging Deployment

```bash
# Staging environment
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Run tests
docker-compose exec langchain-medical python -m pytest tests/

# Check health
curl http://staging.yourdomain.com/health
```

### 3. Production Deployment

```bash
# Production environment
docker-compose -f docker-compose.prod.yml up -d

# Zero-downtime deployment
./scripts/deploy.sh

# Monitor deployment
watch docker-compose ps
```

### 4. Blue-Green Deployment Script

**File:** `scripts/deploy.sh`
```bash
#!/bin/bash

set -e

echo "🚀 Starting Blue-Green Deployment"

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
SERVICE_NAME="langchain-medical"
HEALTH_URL="http://localhost:8000/health"

# Build new image
echo "📦 Building new image..."
docker-compose -f $COMPOSE_FILE build $SERVICE_NAME

# Create new service with suffix
echo "🔵 Starting green service..."
docker-compose -f $COMPOSE_FILE up -d --scale ${SERVICE_NAME}=2 $SERVICE_NAME

# Wait for health check
echo "🏥 Waiting for health check..."
sleep 30

# Check if new service is healthy
if curl -f $HEALTH_URL > /dev/null 2>&1; then
    echo "✅ Green service is healthy"
    
    # Update load balancer to point to green
    echo "🔄 Switching traffic to green..."
    
    # Scale down blue service
    docker-compose -f $COMPOSE_FILE up -d --scale ${SERVICE_NAME}=1 $SERVICE_NAME
    
    echo "🎉 Deployment completed successfully!"
else
    echo "❌ Health check failed, rolling back..."
    
    # Stop green service
    docker-compose -f $COMPOSE_FILE up -d --scale ${SERVICE_NAME}=1 $SERVICE_NAME
    
    echo "🔙 Rollback completed"
    exit 1
fi
```

---

## 📊 Monitoring และ Logging

### 1. Monitoring Stack

**File:** `docker-compose.monitoring.yml`
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: medical-ai-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - medical-ai-network

  grafana:
    image: grafana/grafana:latest
    container_name: medical-ai-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources
    networks:
      - medical-ai-network

  node-exporter:
    image: prom/node-exporter:latest
    container_name: medical-ai-node-exporter
    ports:
      - "9100:9100"
    networks:
      - medical-ai-network

volumes:
  prometheus_data:
  grafana_data:
```

### 2. Centralized Logging

**File:** `docker-compose.logging.yml`
```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    container_name: medical-ai-elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - medical-ai-network

  kibana:
    image: docker.elastic.co/kibana/kibana:7.17.0
    container_name: medical-ai-kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - medical-ai-network

  logstash:
    image: docker.elastic.co/logstash/logstash:7.17.0
    container_name: medical-ai-logstash
    volumes:
      - ./logging/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
      - ./logs:/usr/share/logstash/logs
    depends_on:
      - elasticsearch
    networks:
      - medical-ai-network

volumes:
  elasticsearch_data:
```

---

## 🔐 Security Configuration

### 1. Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "secure_password" | docker secret create db_password -
echo "api_key_here" | docker secret create openai_api_key -

# Update docker-compose.yml
services:
  langchain-medical:
    secrets:
      - db_password
      - openai_api_key

secrets:
  db_password:
    external: true
  openai_api_key:
    external: true
```

### 2. Network Security

```yaml
# docker-compose.security.yml
version: '3.8'

services:
  langchain-medical:
    networks:
      - internal_network
      - web_network
    
  n8n:
    networks:
      - internal_network
    # Remove public port exposure in production

  nginx:
    networks:
      - web_network

networks:
  internal_network:
    driver: bridge
    internal: true
  web_network:
    driver: bridge
```

### 3. SSL/TLS Configuration

```bash
# Generate SSL certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem

# Or use Let's Encrypt
certbot certonly --webroot -w /var/www/certbot \
    -d your-medical-ai-domain.com
```

---

## 🏗️ CI/CD Pipeline

### 1. GitHub Actions Workflow

**File:** `.github/workflows/deploy.yml`
```yaml
name: Deploy Medical AI System

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
          
      - name: Run tests
        run: pytest tests/
        
      - name: Build Docker image
        run: docker build -t medical-ai:${{ github.sha }} .
        
      - name: Test Docker image
        run: |
          docker run -d --name test-container -p 8000:8000 medical-ai:${{ github.sha }}
          sleep 30
          curl -f http://localhost:8000/health
          docker stop test-container

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        env:
          SSH_PRIVATE_KEY: ${{ secrets.SSH_PRIVATE_KEY }}
          SERVER_HOST: ${{ secrets.SERVER_HOST }}
        run: |
          echo "$SSH_PRIVATE_KEY" > private_key
          chmod 600 private_key
          
          ssh -i private_key -o StrictHostKeyChecking=no user@$SERVER_HOST << 'EOF'
            cd /opt/medical-ai-system
            git pull origin main
            docker-compose -f docker-compose.prod.yml build
            ./scripts/deploy.sh
          EOF
```

### 2. Health Check Script

**File:** `scripts/health_check.sh`
```bash
#!/bin/bash

# Health check script for deployment validation
SERVICES=("langchain-medical" "n8n" "nginx")
HEALTH_ENDPOINTS=(
    "http://localhost:8000/health"
    "http://localhost:5678"
    "http://localhost/health"
)

for i in "${!SERVICES[@]}"; do
    SERVICE=${SERVICES[$i]}
    ENDPOINT=${HEALTH_ENDPOINTS[$i]}
    
    echo "Checking $SERVICE..."
    
    # Check if container is running
    if ! docker ps | grep -q "$SERVICE"; then
        echo "❌ $SERVICE container is not running"
        exit 1
    fi
    
    # Check health endpoint
    if curl -f --max-time 10 "$ENDPOINT" > /dev/null 2>&1; then
        echo "✅ $SERVICE is healthy"
    else
        echo "❌ $SERVICE health check failed"
        exit 1
    fi
done

echo "🎉 All services are healthy!"
```

---

## 📈 Performance Optimization

### 1. Resource Allocation

```yaml
# Optimized resource allocation
services:
  langchain-medical:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
    environment:
      - GUNICORN_WORKERS=4
      - GUNICORN_THREADS=2
      - MAX_MODEL_CACHE_SIZE=1GB

  n8n:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1'
    environment:
      - N8N_PAYLOAD_SIZE_MAX=16MB
      - N8N_EXECUTIONS_DATA_PRUNE_MAX_COUNT=1000
```

### 2. Caching Strategy

```yaml
# Redis for caching
services:
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    
  langchain-medical:
    environment:
      - CACHE_TYPE=redis
      - CACHE_REDIS_URL=redis://redis:6379/0
      - CACHE_DEFAULT_TIMEOUT=3600
```

### 3. Database Optimization

```yaml
# PostgreSQL optimization
services:
  postgres:
    environment:
      - POSTGRES_SHARED_BUFFERS=256MB
      - POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
      - POSTGRES_WORK_MEM=4MB
    command: >
      postgres
      -c shared_preload_libraries=pg_stat_statements
      -c pg_stat_statements.track=all
      -c max_connections=100
```

---

## 🔄 Backup และ Recovery

### 1. Automated Backup Script

**File:** `scripts/backup.sh`
```bash
#!/bin/bash

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🔄 Starting backup process..."

# Backup database
docker exec medical-ai-postgres pg_dump -U n8n n8n | gzip > "$BACKUP_DIR/database.sql.gz"

# Backup volumes
docker run --rm \
    -v medical_data:/source \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf /backup/medical_data.tar.gz -C /source .

docker run --rm \
    -v n8n_data:/source \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf /backup/n8n_data.tar.gz -C /source .

# Upload to S3 (optional)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    aws s3 sync "$BACKUP_DIR" "s3://$AWS_S3_BUCKET/backups/$(basename $BACKUP_DIR)"
fi

echo "✅ Backup completed: $BACKUP_DIR"

# Cleanup old backups (keep last 7 days)
find /backups -type d -mtime +7 -exec rm -rf {} +
```

### 2. Recovery Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

echo "🔄 Starting recovery process..."

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore database
gunzip -c "$BACKUP_DIR/database.sql.gz" | \
    docker exec -i medical-ai-postgres psql -U n8n n8n

# Restore volumes
docker run --rm \
    -v medical_data:/target \
    -v "$BACKUP_DIR":/backup \
    alpine tar xzf /backup/medical_data.tar.gz -C /target

# Restart services
docker-compose -f docker-compose.prod.yml up -d

echo "✅ Recovery completed"
```

---

## 🛠️ Maintenance Tasks

### 1. Regular Maintenance Script

**File:** `scripts/maintenance.sh`
```bash
#!/bin/bash

echo "🧹 Starting maintenance tasks..."

# Clean up Docker
echo "Cleaning Docker system..."
docker system prune -f
docker volume prune -f

# Rotate logs
echo "Rotating logs..."
find logs/ -name "*.log" -size +100M -exec logrotate {} \;

# Update images
echo "Updating Docker images..."
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Health check after maintenance
echo "Running health checks..."
./scripts/health_check.sh

echo "✅ Maintenance completed"
```

### 2. Monitoring Script

**File:** `scripts/monitor.sh`
```bash
#!/bin/bash

# System monitoring script
while true; do
    # Check disk space
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 85 ]; then
        echo "⚠️ Disk usage high: ${DISK_USAGE}%"
    fi
    
    # Check memory usage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ $MEMORY_USAGE -gt 90 ]; then
        echo "⚠️ Memory usage high: ${MEMORY_USAGE}%"
    fi
    
    # Check service health
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "🚨 API health check failed!"
    fi
    
    sleep 300  # Check every 5 minutes
done
```

คู่มือ Docker Deployment ครบถ้วนแล้วครับ! มีส่วนไหนที่ต้องการให้อธิบายเพิ่มเติมหรือปรับแต่งไหมครับ?