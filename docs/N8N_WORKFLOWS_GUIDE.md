# 🔄 N8N Workflows สำหรับ Medical AI System

## 📋 Overview

คู่มือนี้จะสอนการใช้งาน N8N เพื่อสร้าง automated workflows สำหรับระบบ Medical AI โดย N8N จะทำหน้าที่เป็น orchestrator ที่เชื่อมต่อระหว่าง Langchain Service, FastAPI, และ external systems

---

## 🚀 N8N Setup และ Configuration

### การติดตั้ง N8N

**แบบ Local Development:**
```bash
# ติดตั้งผ่าน npm
npm install n8n -g

# รัน N8N
n8n start

# หรือรันแบบ development mode
n8n start --tunnel
```

**แบบ Docker:**
```bash
# Run N8N container
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -e WEBHOOK_URL=http://localhost:5678/ \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n

# หรือใช้ docker-compose
docker-compose up n8n
```

### Initial Setup

1. เปิด `http://localhost:5678`
2. สร้าง owner account แรก
3. ตั้งค่า basic configuration:
   - Timezone: Asia/Bangkok
   - Default execution mode: Own process
   - Enable workflow statistics

---

## 🏥 Medical AI Workflows

### 1. Basic Medical Q&A Workflow

**Workflow Name:** `medical-qa-basic`

**Components:**
- **Webhook Trigger** - รับ HTTP requests
- **HTTP Request** - เรียก Langchain API
- **Code** - ประมวลผล response
- **Database** - บันทึกผลลัพธ์

**Workflow JSON:**
```json
{
  "name": "Medical Q&A Basic",
  "nodes": [
    {
      "parameters": {
        "path": "medical-qa",
        "options": {}
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [240, 300],
      "webhookId": "medical-qa-webhook"
    },
    {
      "parameters": {
        "url": "http://localhost:8000/api/medical-qa",
        "options": {
          "response": {
            "response": {
              "responseFormat": "json"
            }
          }
        },
        "requestMethod": "POST",
        "body": {
          "question": "={{$json.question}}",
          "user_id": "={{$json.user_id || 'n8n_user'}}"
        }
      },
      "name": "Call Langchain API",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 3,
      "position": [460, 300]
    },
    {
      "parameters": {
        "jsCode": "// Process และเพิ่มข้อมูล metadata\nconst response = items[0].json;\n\nreturn [{\n  json: {\n    ...response,\n    processed_at: new Date().toISOString(),\n    workflow_id: 'medical-qa-basic',\n    n8n_execution_id: $execution.id\n  }\n}];"
      },
      "name": "Process Response",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [680, 300]
    },
    {
      "parameters": {
        "operation": "insert",
        "table": "medical_qa_log",
        "columns": "user_id, question, answer, confidence, sources, response_time, workflow_execution_id",
        "additionalFields": {},
        "options": {}
      },
      "name": "Log to Database",
      "type": "n8n-nodes-base.sqlite",
      "typeVersion": 1,
      "position": [900, 300]
    }
  ],
  "connections": {
    "Webhook": {
      "main": [
        [
          {
            "node": "Call Langchain API",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Call Langchain API": {
      "main": [
        [
          {
            "node": "Process Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Process Response": {
      "main": [
        [
          {
            "node": "Log to Database",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### 2. Advanced Medical AI Workflow with Routing

**Workflow Name:** `medical-qa-advanced`

**Features:**
- Question classification
- Route ไปยัง specialized models
- Error handling
- Response validation
- Multi-language support

```json
{
  "name": "Medical Q&A Advanced",
  "nodes": [
    {
      "parameters": {
        "path": "medical-qa-advanced",
        "options": {}
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [240, 300]
    },
    {
      "parameters": {
        "jsCode": "// Classify คำถาม และกำหนด routing\nconst question = items[0].json.question.toLowerCase();\nlet category = 'general';\nlet priority = 'normal';\n\n// Classification logic\nif (question.includes('เบาหวาน') || question.includes('diabetes')) {\n  category = 'diabetes';\n} else if (question.includes('ความดัน') || question.includes('hypertension')) {\n  category = 'hypertension';\n} else if (question.includes('หัวใจ') || question.includes('heart')) {\n  category = 'cardiology';\n} else if (question.includes('ฉุกเฉิน') || question.includes('emergency')) {\n  category = 'emergency';\n  priority = 'high';\n}\n\nreturn [{\n  json: {\n    ...items[0].json,\n    category: category,\n    priority: priority,\n    classified_at: new Date().toISOString()\n  }\n}];"
      },
      "name": "Classify Question",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [460, 300]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{$json.category}}",
              "operation": "equal",
              "value2": "emergency"
            }
          ]
        }
      },
      "name": "Check Priority",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [680, 300]
    },
    {
      "parameters": {
        "url": "http://localhost:8000/api/medical-qa",
        "options": {},
        "requestMethod": "POST",
        "body": {
          "question": "={{$json.question}}",
          "user_id": "={{$json.user_id}}",
          "category": "={{$json.category}}",
          "priority": "high"
        }
      },
      "name": "Emergency Route",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 3,
      "position": [900, 200]
    },
    {
      "parameters": {
        "url": "http://localhost:8000/api/medical-qa",
        "options": {},
        "requestMethod": "POST",
        "body": {
          "question": "={{$json.question}}",
          "user_id": "={{$json.user_id}}",
          "category": "={{$json.category}}"
        }
      },
      "name": "Normal Route",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 3,
      "position": [900, 400]
    }
  ]
}
```

### 3. Batch Processing Workflow

**สำหรับประมวลผลคำถามจำนวนมาก:**

```json
{
  "name": "Medical Q&A Batch Processing",
  "nodes": [
    {
      "parameters": {
        "mode": "passThrough",
        "output": "input1"
      },
      "name": "Merge Results",
      "type": "n8n-nodes-base.merge",
      "typeVersion": 2,
      "position": [1120, 300]
    },
    {
      "parameters": {
        "batchSize": 5,
        "options": {}
      },
      "name": "Split Batch",
      "type": "n8n-nodes-base.splitInBatches",
      "typeVersion": 1,
      "position": [460, 300]
    },
    {
      "parameters": {
        "jsCode": "// ประมวลผล batch ของคำถาม\nconst results = [];\n\nfor (const item of items) {\n  const question = item.json.question;\n  const user_id = item.json.user_id;\n  \n  // เรียก API สำหรับแต่ละคำถาม\n  const response = await this.helpers.request({\n    method: 'POST',\n    url: 'http://localhost:8000/api/medical-qa',\n    body: {\n      question: question,\n      user_id: user_id\n    },\n    json: true\n  });\n  \n  results.push({\n    json: {\n      question: question,\n      user_id: user_id,\n      ...response,\n      processed_at: new Date().toISOString()\n    }\n  });\n}\n\nreturn results;"
      },
      "name": "Process Batch",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [680, 300]
    }
  ]
}
```

---

## 📧 Integration Workflows

### 1. Email-based Medical Consultation

**Scenario:** รับคำถามทางอีเมล และส่งคำตอบกลับ

```json
{
  "name": "Email Medical Consultation",
  "nodes": [
    {
      "parameters": {
        "pollTimes": {
          "item": [
            {
              "mode": "everyMinute"
            }
          ]
        },
        "options": {}
      },
      "name": "Email Trigger (IMAP)",
      "type": "n8n-nodes-base.emailReadImap",
      "typeVersion": 2,
      "position": [240, 300]
    },
    {
      "parameters": {
        "jsCode": "// Parse email content และแยกคำถาม\nconst email = items[0].json;\nconst subject = email.subject;\nconst body = email.text;\n\n// Extract medical question\nlet question = body;\nif (body.includes('คำถาม:')) {\n  question = body.split('คำถาม:')[1].trim();\n}\n\nreturn [{\n  json: {\n    email_from: email.from.value[0].address,\n    email_subject: subject,\n    question: question,\n    original_email: email\n  }\n}];"
      },
      "name": "Parse Email",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [460, 300]
    },
    {
      "parameters": {
        "url": "http://localhost:8000/api/medical-qa",
        "requestMethod": "POST",
        "body": {
          "question": "={{$json.question}}",
          "user_id": "={{$json.email_from}}"
        }
      },
      "name": "Get AI Response",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 3,
      "position": [680, 300]
    },
    {
      "parameters": {
        "fromEmail": "medical-ai@yourdomain.com",
        "toEmail": "={{$json.email_from}}",
        "subject": "RE: {{$json.email_subject}}",
        "text": "เรียน คุณผู้ป่วย,\n\nขอบคุณสำหรับคำถามของคุณ: {{$json.question}}\n\nคำตอบจาก AI Medical Assistant:\n{{$json.answer}}\n\n⚠️ คำแนะนำ: คำตอบนี้เป็นเพียงข้อมูลเบื้องต้น กรุณาปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ\n\nขอแสดงความนับถือ,\nMedical AI System"
      },
      "name": "Send Email Reply",
      "type": "n8n-nodes-base.emailSend",
      "typeVersion": 2,
      "position": [900, 300]
    }
  ]
}
```

### 2. Slack Integration

**สำหรับทีมแพทย์ใช้งานผ่าน Slack:**

```json
{
  "name": "Slack Medical Bot",
  "nodes": [
    {
      "parameters": {
        "channel": "#medical-ai",
        "options": {}
      },
      "name": "Slack Trigger",
      "type": "n8n-nodes-base.slackTrigger",
      "typeVersion": 1,
      "position": [240, 300]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{$json.text}}",
              "operation": "startsWith",
              "value2": "/ask"
            }
          ]
        }
      },
      "name": "Check Command",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [460, 300]
    },
    {
      "parameters": {
        "jsCode": "// Parse Slack command\nconst text = items[0].json.text;\nconst question = text.replace('/ask', '').trim();\n\nreturn [{\n  json: {\n    question: question,\n    user_id: items[0].json.user,\n    channel: items[0].json.channel,\n    slack_data: items[0].json\n  }\n}];"
      },
      "name": "Parse Command",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [680, 300]
    },
    {
      "parameters": {
        "authentication": "oAuth2",
        "select": "channel",
        "channelId": "={{$json.channel}}",
        "text": "🤖 *Medical AI Response*\n\n*คำถาม:* {{$json.question}}\n\n*คำตอบ:* {{$json.answer}}\n\n*ความมั่นใจ:* {{$json.confidence}}%\n\n⚠️ *หมายเหตุ:* ข้อมูลนี้เป็นเพียงคำแนะนำเบื้องต้น กรุณาปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ",
        "otherOptions": {}
      },
      "name": "Send Slack Reply",
      "type": "n8n-nodes-base.slack",
      "typeVersion": 2,
      "position": [1120, 300]
    }
  ]
}
```

---

## 📊 Analytics และ Monitoring Workflows

### 1. Daily Statistics Report

```json
{
  "name": "Daily Medical AI Stats",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "cronExpression",
              "expression": "0 9 * * *"
            }
          ]
        }
      },
      "name": "Daily Trigger (9 AM)",
      "type": "n8n-nodes-base.cron",
      "typeVersion": 1,
      "position": [240, 300]
    },
    {
      "parameters": {
        "operation": "select",
        "query": "SELECT \n  COUNT(*) as total_questions,\n  AVG(confidence) as avg_confidence,\n  COUNT(DISTINCT user_id) as unique_users,\n  DATE(created_at) as date\nFROM medical_qa_log \nWHERE DATE(created_at) = DATE('now', '-1 day')\nGROUP BY DATE(created_at)"
      },
      "name": "Get Daily Stats",
      "type": "n8n-nodes-base.sqlite",
      "typeVersion": 1,
      "position": [460, 300]
    },
    {
      "parameters": {
        "jsCode": "// Generate report\nconst stats = items[0].json;\n\nconst report = `📊 **Medical AI Daily Report**\n\n📅 **Date:** ${stats.date}\n\n📈 **Statistics:**\n- Total Questions: ${stats.total_questions}\n- Average Confidence: ${(stats.avg_confidence * 100).toFixed(1)}%\n- Unique Users: ${stats.unique_users}\n\n🎯 **Performance:**\n- Questions per User: ${(stats.total_questions / stats.unique_users).toFixed(1)}\n- System Health: ${stats.avg_confidence > 0.7 ? '✅ Good' : '⚠️ Needs Review'}`;\n\nreturn [{\n  json: {\n    report: report,\n    stats: stats\n  }\n}];"
      },
      "name": "Generate Report",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [680, 300]
    },
    {
      "parameters": {
        "authentication": "oAuth2",
        "select": "channel",
        "channelId": "#medical-ai-reports",
        "text": "={{$json.report}}"
      },
      "name": "Send to Slack",
      "type": "n8n-nodes-base.slack",
      "typeVersion": 2,
      "position": [900, 300]
    }
  ]
}
```

### 2. Error Monitoring Workflow

```json
{
  "name": "Error Monitoring",
  "nodes": [
    {
      "parameters": {
        "path": "error-webhook",
        "options": {}
      },
      "name": "Error Webhook",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [240, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{$json.error_count}}",
              "operation": "larger",
              "value2": 5
            }
          ]
        }
      },
      "name": "Check Error Threshold",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [460, 300]
    },
    {
      "parameters": {
        "fromEmail": "alerts@yourdomain.com",
        "toEmail": "admin@yourdomain.com",
        "subject": "🚨 Medical AI System Alert",
        "text": "⚠️ **High Error Rate Detected**\n\nError Count: {{$json.error_count}}\nTime Window: Last 10 minutes\nError Type: {{$json.error_type}}\n\nPlease investigate immediately."
      },
      "name": "Send Alert Email",
      "type": "n8n-nodes-base.emailSend",
      "typeVersion": 2,
      "position": [680, 300]
    }
  ]
}
```

---

## 🔧 Advanced Workflow Patterns

### 1. Circuit Breaker Pattern

```javascript
// ใน Code Node
const MAX_FAILURES = 3;
const RESET_TIMEOUT = 300000; // 5 minutes

// Get current circuit state
const circuitState = await this.helpers.request({
  method: 'GET',
  url: 'http://localhost:8000/api/circuit-state',
  json: true
}).catch(() => ({ failures: 0, lastFailure: 0, isOpen: false }));

if (circuitState.isOpen) {
  const timeSinceFailure = Date.now() - circuitState.lastFailure;
  if (timeSinceFailure < RESET_TIMEOUT) {
    return [{ json: { error: 'Circuit breaker is open', skipExecution: true } }];
  }
}

// Proceed with normal execution
return items;
```

### 2. Retry with Exponential Backoff

```javascript
// Retry logic
async function callAPIWithRetry(url, data, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await this.helpers.request({
        method: 'POST',
        url: url,
        body: data,
        json: true
      });
      return response;
    } catch (error) {
      if (attempt === maxRetries) throw error;
      
      // Exponential backoff
      const delay = Math.pow(2, attempt) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}
```

### 3. Rate Limiting

```javascript
// Rate limiting logic
const RATE_LIMIT = 100; // requests per minute
const TIME_WINDOW = 60000; // 1 minute

const userId = items[0].json.user_id;
const now = Date.now();

// Get user's request history
const userRequests = await this.helpers.request({
  method: 'GET',
  url: `http://localhost:8000/api/user-requests/${userId}`,
  json: true
}).catch(() => ({ requests: [] }));

// Count requests in current time window
const recentRequests = userRequests.requests.filter(
  req => now - req.timestamp < TIME_WINDOW
);

if (recentRequests.length >= RATE_LIMIT) {
  return [{ 
    json: { 
      error: 'Rate limit exceeded',
      retryAfter: TIME_WINDOW - (now - recentRequests[0].timestamp)
    }
  }];
}

// Log current request
await this.helpers.request({
  method: 'POST',
  url: `http://localhost:8000/api/user-requests/${userId}`,
  body: { timestamp: now },
  json: true
});

return items;
```

---

## 🚀 Deployment และ Production Setup

### N8N Production Configuration

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=${DOMAIN_NAME}
      - N8N_PORT=5678
      - N8N_PROTOCOL=https
      - WEBHOOK_URL=https://${DOMAIN_NAME}/
      - GENERIC_TIMEZONE=Asia/Bangkok
      - N8N_LOG_LEVEL=info
      - N8N_LOG_OUTPUT=console,file
      - N8N_LOG_FILE_LOCATION=/home/node/n8n.log
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - n8n_data:/home/node/.n8n
      - n8n_files:/files
    depends_on:
      - postgres
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=n8n
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  n8n_data:
  n8n_files:
  postgres_data:
```

### Monitoring Setup

**Prometheus Metrics:**
```javascript
// ใน N8N Code Node สำหรับ metrics
const metrics = {
  workflow_executions_total: 1,
  workflow_execution_duration: $executionData.finished - $executionData.started,
  workflow_success_rate: $executionData.success ? 1 : 0
};

// Send metrics to monitoring system
await this.helpers.request({
  method: 'POST',
  url: 'http://prometheus-pushgateway:9091/metrics/job/n8n-workflows',
  body: Object.entries(metrics)
    .map(([key, value]) => `${key} ${value}`)
    .join('\n'),
  headers: { 'Content-Type': 'text/plain' }
});
```

---

## 📋 Best Practices

### 1. Error Handling
- ใช้ try-catch ใน Code nodes
- ตั้งค่า timeout ที่เหมาะสม
- Implement retry logic สำหรับ external calls
- Log errors อย่างครบถ้วน

### 2. Performance Optimization
- ใช้ batch processing สำหรับข้อมูลจำนวนมาก
- Cache ผลลัพธ์ที่ใช้บ่อย
- Optimize database queries
- ใช้ async operations อย่างเหมาะสม

### 3. Security
- ใช้ credentials management
- Validate input data
- Implement rate limiting
- Use HTTPS สำหรับ webhooks

### 4. Monitoring
- Track execution metrics
- Monitor error rates
- Set up alerts สำหรับ critical failures
- Log important business events

---

## 🛠️ Troubleshooting N8N Issues

### ปัญหาที่พบบ่อย:

**1. Webhook ไม่ทำงาน:**
```bash
# เช็ค webhook URL
curl -X POST "http://localhost:5678/webhook/medical-qa" \
  -H "Content-Type: application/json" \
  -d '{"test": true}'

# เช็ค N8N logs
docker logs n8n_container
```

**2. Database connection issues:**
```bash
# เช็ค database connectivity
docker exec -it postgres_container psql -U n8n -d n8n -c "SELECT 1;"
```

**3. Memory issues:**
```bash
# เพิ่ม memory limit
docker run -m 2g n8nio/n8n

# หรือใน docker-compose
deploy:
  resources:
    limits:
      memory: 2G
```

**4. Execution timeouts:**
```javascript
// ตั้งค่า timeout ใน HTTP Request node
{
  "timeout": 30000,  // 30 seconds
  "followRedirect": true,
  "ignoreHttpStatusErrors": false
}
```

ต้องการให้อธิบายส่วนไหนเพิ่มเติมไหมครับ?