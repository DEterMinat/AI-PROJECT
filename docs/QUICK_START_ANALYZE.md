# 🚀 Quick Start Guide - /analyze Endpoint

## ⚡ Fast Setup (3 Steps)

### Step 1: Start the API
```bash
# Windows
start-api.bat

# Or manually
cd api_service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Test the Endpoint
```bash
# Run test suite
python test_analyze_api.py
```

### Step 3: Try It!
```bash
# Test with curl
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"symptom\": \"I have a fever and cough\"}"
```

---

## 🌐 Access Points

| Resource | URL |
|----------|-----|
| **API Root** | http://localhost:8000 |
| **Swagger Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |
| **/analyze Endpoint** | http://localhost:8000/analyze |

---

## 📝 Example Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"symptom": "I have a headache and feel dizzy"}
)

result = response.json()
print(f"Severity: {result['severity']}")
print(f"Category: {result['category']}")
print(f"Action: {result['next_action']}")
print(f"Advice: {result['advice']}")
```

### JavaScript (n8n)
```javascript
// In n8n HTTP Request node
{
  "method": "POST",
  "url": "http://api:8000/analyze",
  "body": {
    "symptom": "{{$json.user_symptom}}"
  }
}

// Process response
const severity = $json.severity;
const nextAction = $json.next_action;

if (nextAction === 'notify_doctor') {
  // Send urgent alert
} else if (nextAction === 'monitor') {
  // Set reminder
} else {
  // Save to database
}
```

### cURL
```bash
# Mild symptom
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symptom": "I have a runny nose"}'

# Moderate symptom
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symptom": "I have persistent stomach pain for 2 days"}'

# Severe symptom
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symptom": "I have severe chest pain and cant breathe"}'
```

---

## ✅ What You Get

### Request
```json
{
  "symptom": "I have a fever and body aches"
}
```

### Response
```json
{
  "advice": "I understand you're experiencing fever symptoms...\n\n**General Self-Care**:\n- Rest and stay hydrated\n- Monitor your temperature\n- Use cool compresses\n\n💡 If fever persists >3 days, see a doctor.",
  "severity": "mild",
  "category": "fever",
  "next_action": "save_to_db",
  "disclaimer": "⚠️ This is general health advice only. Please consult a qualified doctor for proper diagnosis and treatment."
}
```

---

## 🔧 n8n Workflow Example

```yaml
1. Webhook Trigger
   └─ Receives: {"symptom": "user input"}

2. HTTP Request to /analyze
   └─ POST http://api:8000/analyze
   
3. Switch on next_action
   ├─ save_to_db
   │  └─ PostgreSQL: INSERT symptom_log
   │
   ├─ monitor  
   │  ├─ PostgreSQL: INSERT + flag_for_followup
   │  └─ Email: Reminder in 24h
   │
   └─ notify_doctor
      ├─ PostgreSQL: INSERT + urgent_flag
      ├─ Email: Alert to doctor
      └─ SMS: Urgent notification
```

---

## 📊 Response Flow

```
User Symptom Input
        ↓
Categorize Symptom
        ↓
Determine Severity
        ↓
Generate Advice (LangChain)
        ↓
Decide Next Action
        ↓
Log to Database
        ↓
Return JSON Response
```

---

## 🎯 Severity Mapping

| Input Keywords | Severity | Next Action |
|----------------|----------|-------------|
| mild, slight, minor | `mild` | `save_to_db` |
| persistent, worsening, concerning | `moderate` | `monitor` |
| severe, intense, cant breathe, chest pain | `severe` | `notify_doctor` |

---

## 📦 Full Response Structure

```typescript
interface AnalysisResponse {
  advice: string;          // Detailed health advice
  severity: "mild" | "moderate" | "severe";
  category: string;        // flu, fever, digestion, etc.
  next_action: "save_to_db" | "notify_doctor" | "monitor";
  disclaimer: string;      // Medical disclaimer
}
```

---

## ✅ Status Checklist

- [x] API running on port 8000
- [x] `/analyze` endpoint responding
- [x] English-only processing
- [x] Symptom categorization working
- [x] Severity detection accurate
- [x] Next action logic correct
- [x] Database logging active
- [x] No diagnosis/prescription
- [x] Empathetic tone
- [x] n8n-ready JSON output

---

## 🔗 Related Files

- `api_service/app/main.py` - Main API implementation
- `langchain_service/medical_ai.py` - LangChain service
- `test_analyze_api.py` - Test suite
- `docs/ANALYZE_ENDPOINT.md` - Full documentation
- `docs/PROMPT_ANALYSIS.md` - Requirements analysis

---

## 🎉 You're Ready!

Your API now fully matches the prompt requirements:
- ✅ POST /analyze endpoint
- ✅ English-only symptom analysis
- ✅ Structured JSON output
- ✅ n8n workflow integration
- ✅ Cautious, empathetic responses
- ✅ No diagnosis or prescription

**Next Steps**:
1. Start API: `start-api.bat`
2. Test: `python test_analyze_api.py`
3. Integrate with n8n
4. Deploy to production!

🚀 Happy coding!
