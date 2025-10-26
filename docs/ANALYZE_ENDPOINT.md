# 🏥 AI Health Consultant API - /analyze Endpoint

## 📋 Overview

The `/analyze` endpoint provides AI-powered symptom analysis for health consultation. It matches the requirements from the project prompt exactly:

```
POST /analyze
Input:  {"symptom": "<user text in English>"}
Output: {
    "advice": "...",
    "severity": "mild | moderate | severe",
    "category": "flu | fever | digestion | ...",
    "next_action": "save_to_db | notify_doctor | monitor"
}
```

---

## ⚠️ Important Rules

This system follows strict ethical guidelines:

✅ **Does:**
- Provide general health information
- Suggest self-care measures
- Categorize symptoms
- Assess severity levels
- Recommend when to seek medical help
- Use empathetic, cautious language

❌ **Does NOT:**
- Diagnose diseases
- Prescribe medications
- Replace professional medical advice
- Make definitive medical statements

---

## 🚀 API Endpoint Details

### **POST /analyze**

**URL**: `http://localhost:8000/analyze`

**Request Body**:
```json
{
  "symptom": "I have a headache and feel dizzy"
}
```

**Response**:
```json
{
  "advice": "Thank you for sharing your symptoms...",
  "severity": "moderate",
  "category": "neurological",
  "next_action": "monitor",
  "disclaimer": "⚠️ This is general health advice only..."
}
```

---

## 📊 Response Fields

### 1. **advice** (string)
Empathetic, cautious health guidance:
- Acknowledges the symptom
- Provides general self-care suggestions
- Recommends when to see a doctor
- Never diagnoses or prescribes

**Example**:
```
I understand you're experiencing headache symptoms. Here's some general guidance:

**General Self-Care Suggestions**:
- Rest in a quiet, dark room
- Stay hydrated
- Avoid bright lights and loud noises

⚠️ IMPORTANT: Please schedule an appointment with a doctor soon...
```

### 2. **severity** (enum)
Classification of symptom intensity:
- `"mild"` - Minor discomfort, no immediate concern
- `"moderate"` - Persistent symptoms, should see doctor
- `"severe"` - Urgent, requires immediate medical attention

**Classification Logic**:
```python
severe_keywords = [
    'severe', 'intense', 'unbearable', 'emergency', 
    'cant breathe', 'chest pain', 'heart attack'
]

moderate_keywords = [
    'persistent', 'worsening', 'concerning', 
    'frequent', 'constant', 'ongoing'
]
```

### 3. **category** (string)
Symptom category for classification:

| Category | Keywords |
|----------|----------|
| `flu` | cough, runny nose, congestion, body aches |
| `fever` | fever, high temperature, chills |
| `digestion` | nausea, vomiting, diarrhea, stomach pain |
| `respiratory` | breathing, shortness of breath, wheezing |
| `cardiovascular` | chest pain, heart, palpitations |
| `neurological` | headache, migraine, dizziness |
| `musculoskeletal` | muscle pain, joint pain, back pain |
| `skin` | rash, itching, hives |
| `unknown` | No clear category match |

### 4. **next_action** (enum)
Action for n8n workflow automation:

| Action | When Used | Purpose |
|--------|-----------|---------|
| `save_to_db` | Mild symptoms | Log for tracking |
| `monitor` | Moderate symptoms | Watch and remind |
| `notify_doctor` | Severe or critical | Urgent alert |

**Decision Logic**:
```python
if severity == 'severe':
    return 'notify_doctor'
elif severity == 'moderate' and category in ['cardiovascular', 'respiratory']:
    return 'notify_doctor'
elif severity == 'moderate':
    return 'monitor'
else:
    return 'save_to_db'
```

### 5. **disclaimer** (string)
Medical disclaimer (always included):
```
⚠️ This is general health advice only. 
Please consult a qualified doctor for proper diagnosis and treatment.
```

---

## 🧪 Testing

### Quick Test
```bash
# Test the endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symptom": "I have a fever and cough"}'
```

### Run Test Suite
```bash
# Run comprehensive tests
python test_analyze_api.py
```

**Test Cases Included**:
1. ✅ Mild flu symptoms → `save_to_db`
2. ✅ Moderate fever → `monitor`
3. ✅ Severe chest pain → `notify_doctor`
4. ✅ Digestive issues → categorization
5. ✅ Headache → severity assessment
6. ✅ Respiratory emergency → urgent action

---

## 🌊 n8n Integration

### Webhook Setup

1. **Create n8n Webhook Node**
   - Method: POST
   - URL: `http://api:8000/analyze`
   - Authentication: None (or add Bearer token)

2. **Send Symptom Data**
   ```json
   {
     "symptom": "{{$json.user_input}}"
   }
   ```

3. **Process Response**
   - Parse JSON response
   - Route based on `next_action`
   - Log to database if `save_to_db`
   - Send alert if `notify_doctor`
   - Schedule reminder if `monitor`

### Example n8n Workflow

```
Webhook (receive symptom)
  ↓
HTTP Request (/analyze)
  ↓
Switch (next_action)
  ├─ save_to_db → PostgreSQL Insert
  ├─ monitor → Set Reminder + Email
  └─ notify_doctor → Urgent Email/SMS + Alert
```

---

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database
DB_PATH=data/medical_ai.db

# LangChain (optional)
MODEL_PATH=./models/simple_trained
```

### Starting the API

**Development**:
```bash
cd api_service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production**:
```bash
docker-compose up langchain-medical
```

---

## 📚 Example Requests & Responses

### Example 1: Mild Symptoms
**Request**:
```json
{
  "symptom": "I have a slight cough and runny nose"
}
```

**Response**:
```json
{
  "advice": "I understand you're experiencing flu-like symptoms...\n\n**General Self-Care**:\n- Rest and stay hydrated\n- Consider OTC flu medications\n- Avoid contact with others\n\n💡 If symptoms persist >3 days, see a doctor.",
  "severity": "mild",
  "category": "flu",
  "next_action": "save_to_db",
  "disclaimer": "⚠️ This is general health advice only..."
}
```

### Example 2: Moderate Symptoms
**Request**:
```json
{
  "symptom": "I have a persistent headache for 2 days and feel dizzy"
}
```

**Response**:
```json
{
  "advice": "I understand you're experiencing neurological symptoms...\n\n**General Self-Care**:\n- Rest in a quiet, dark room\n- Stay hydrated\n- Avoid bright screens\n\n⚠️ IMPORTANT: Please schedule a doctor appointment soon.",
  "severity": "moderate",
  "category": "neurological",
  "next_action": "monitor",
  "disclaimer": "⚠️ This is general health advice only..."
}
```

### Example 3: Severe Symptoms
**Request**:
```json
{
  "symptom": "I have severe chest pain and difficulty breathing"
}
```

**Response**:
```json
{
  "advice": "⚠️ **URGENT**: Your symptoms appear to be severe.\n\n**IMMEDIATE ACTION REQUIRED**:\n- Call emergency services NOW\n- Do NOT drive yourself\n- Stay calm and sit down\n- Loosen tight clothing\n\n🚨 **SEEK EMERGENCY MEDICAL ATTENTION IMMEDIATELY**",
  "severity": "severe",
  "category": "cardiovascular",
  "next_action": "notify_doctor",
  "disclaimer": "⚠️ This is general health advice only..."
}
```

---

## 🎯 Comparison with Prompt Requirements

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| POST /analyze | ✅ | Fully implemented |
| Input: {"symptom": "text"} | ✅ | SymptomRequest model |
| Output: advice, severity, category, next_action | ✅ | AnalysisResponse model |
| English-only | ✅ | All processing in English |
| No diagnosis | ✅ | Explicit in prompts & code |
| Cautious tone | ✅ | Empathetic advice generation |
| Empathetic | ✅ | Understanding language |
| LangChain integration | ✅ | Optional LLM enhancement |
| n8n ready | ✅ | Structured JSON output |
| FastAPI | ✅ | Framework used |
| Chroma/Vector DB | ✅ | Knowledge base support |

---

## 🔍 Database Logging

All symptom analyses are logged to SQLite:

**Table**: `symptom_analysis`

```sql
CREATE TABLE symptom_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symptom TEXT NOT NULL,
    advice TEXT,
    severity TEXT,
    category TEXT,
    next_action TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Query logs**:
```sql
-- Recent analyses
SELECT * FROM symptom_analysis 
ORDER BY created_at DESC 
LIMIT 10;

-- Severe cases
SELECT * FROM symptom_analysis 
WHERE severity = 'severe' 
ORDER BY created_at DESC;

-- By category
SELECT category, COUNT(*) as count 
FROM symptom_analysis 
GROUP BY category;
```

---

## 🚨 Error Handling

### 400 Bad Request
```json
{
  "detail": "Symptom description too short. Please provide more details."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error analyzing symptom: <error message>"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Medical service not available"
}
```

---

## 📖 API Documentation

**Swagger UI**: http://localhost:8000/docs
**ReDoc**: http://localhost:8000/redoc

---

## ✅ Checklist

- [x] `/analyze` endpoint created
- [x] English-only processing
- [x] Symptom categorization (flu, fever, digestion, etc.)
- [x] Severity detection (mild, moderate, severe)
- [x] Next action logic (save_to_db, notify_doctor, monitor)
- [x] Cautious, empathetic advice
- [x] No diagnosis disclaimer
- [x] Database logging
- [x] LangChain integration
- [x] n8n-ready JSON output
- [x] Test suite included
- [x] Documentation complete

---

## 🎉 Status

**✅ COMPLETE** - The `/analyze` endpoint fully matches the prompt requirements and is ready for production use with n8n workflows!

---

**Last Updated**: 2025-10-06
**API Version**: 2.0.0
**Status**: Production Ready 🚀
