# 📋 Prompt Analysis: Current vs Required Implementation

## 🎯 Prompt Requirements Summary

**Goal**: LangChain-based AI Health Consultant System
- Give general health advice based on symptoms
- NO disease diagnosis
- Structured JSON output
- Integration with n8n for workflow automation

---

## ✅ What Already Exists

### 1. **Stack Components** ✅

| Component | Required | Current Status | Location |
|-----------|----------|----------------|----------|
| **FastAPI** | ✅ Yes | ✅ Implemented | `api_service/app/main.py` |
| **LangChain** | ✅ Yes | ✅ Implemented | `langchain_service/medical_ai.py` |
| **Chroma Vector DB** | ✅ Yes | ✅ Implemented | `langchain_service/medical_ai.py` (MedicalKnowledgeBase) |
| **n8n** | ✅ Yes | ✅ Configured | `n8n/medical_qa_workflow.json`, `docker-compose.yml` |

### 2. **Current API Endpoints**

```python
# Existing endpoints in api_service/app/main.py
GET  /              # Root - service info
GET  /health        # Health check
POST /api/medical-qa   # Medical Q&A (close to required)
POST /api/add-knowledge  # Add knowledge to vector DB
GET  /api/stats     # Statistics
```

### 3. **Severity Classification** ✅

**Already implemented in `scripts/clean_data.py`:**

```python
self.severity_rules = {
    'critical': ['stroke', 'heart attack', 'cardiac arrest', ...],
    'high': ['severe', 'acute', 'unstable', ...],
    'moderate': ['chronic', 'stable', 'controlled', ...],
    'low': ['mild', 'minor', 'routine', ...]
}

def determine_severity_level(self, text):
    """Determine severity level based on content"""
    # Returns: 'critical', 'high', 'moderate', 'low'
```

### 4. **Category Classification** ⚠️ Partial

**Specialty categorization exists:**

```python
def categorize_medical_specialty(self, text):
    specialty_keywords = {
        'cardiology': ['heart', 'cardiac', ...],
        'neurology': ['brain', 'stroke', ...],
        'pulmonology': ['lung', 'respiratory', ...],
        'endocrinology': ['diabetes', 'thyroid', ...],
        # ... more specialties
    }
```

**But NOT symptom categories like:**
- 'flu', 'fever', 'digestion' (as per prompt requirement)

---

## ❌ What's Missing

### 1. **`/analyze` Endpoint** ❌

**Required:**
```python
POST /analyze
Input: {"symptom": "<user text>"}
Output: {
    "advice": "...",
    "severity": "mild | moderate | severe",
    "category": "flu | fever | digestion | unknown",
    "next_action": "save_to_db | notify_doctor"
}
```

**Current:**
```python
POST /api/medical-qa
Input: {"question": "...", "user_id": "..."}
Output: {
    "answer": "...",
    "confidence": 0.0,
    "sources": [],
    "response_time": 0.0,
    "status": "success"
}
```

### 2. **Symptom-Specific Categories** ❌

Need to add:
```python
symptom_categories = {
    'flu': ['cough', 'runny nose', 'body aches', 'fever', 'fatigue'],
    'fever': ['high temperature', 'chills', 'sweating'],
    'digestion': ['nausea', 'diarrhea', 'stomach pain', 'bloating'],
    'respiratory': ['shortness of breath', 'wheezing', 'chest tightness'],
    'cardiovascular': ['chest pain', 'palpitations', 'dizziness'],
    # ... more categories
}
```

### 3. **Next Action Logic** ❌

Need decision tree:
```python
def determine_next_action(severity, category):
    if severity in ['critical', 'severe']:
        return "notify_doctor"
    elif severity == 'moderate':
        return "save_to_db | monitor"
    else:  # mild
        return "save_to_db"
```

### 4. **Cautious & Empathetic Prompt Template** ⚠️ Needs Update

**Current prompt (Thai-focused):**
```python
formatted_prompt = f"คำถาม: {prompt}\nคำตอบ:"
```

**Needed (English, cautious, empathetic):**
```python
SYMPTOM_ANALYSIS_TEMPLATE = """
You are a helpful AI health assistant. Your role is to provide general health advice based on symptoms, 
but you MUST NOT diagnose diseases or prescribe medications.

User Symptom: {symptom}

Provide helpful, cautious advice following these rules:
1. Be empathetic and understanding
2. Suggest general self-care measures
3. Identify severity level
4. ALWAYS advise seeing a doctor if symptoms persist or worsen
5. Never claim to diagnose or prescribe

Response:
"""
```

---

## 🔧 Required Changes

### Priority 1: Create `/analyze` Endpoint

**File**: `api_service/app/main.py`

```python
from pydantic import BaseModel
from typing import Literal

class SymptomRequest(BaseModel):
    symptom: str

class AnalysisResponse(BaseModel):
    advice: str
    severity: Literal["mild", "moderate", "severe"]
    category: str  # "flu", "fever", "digestion", "unknown"
    next_action: Literal["save_to_db", "notify_doctor", "monitor"]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symptom(request: SymptomRequest):
    """
    Analyze user symptoms and return structured health advice.
    This endpoint is designed for n8n workflow integration.
    """
    try:
        # 1. Analyze symptom using LangChain
        analysis = medical_service.analyze_symptom(request.symptom)
        
        # 2. Determine severity
        severity = determine_severity(analysis)
        
        # 3. Categorize symptom
        category = categorize_symptom(request.symptom)
        
        # 4. Decide next action
        next_action = determine_next_action(severity, category)
        
        # 5. Log to database
        log_to_db(request.symptom, analysis, severity, category)
        
        return AnalysisResponse(
            advice=analysis["advice"],
            severity=severity,
            category=category,
            next_action=next_action
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Priority 2: Update LangChain Service

**File**: `langchain_service/medical_ai.py`

Add new method:

```python
class LangchainMedicalService:
    
    SYMPTOM_ANALYSIS_TEMPLATE = """
    You are a compassionate AI health assistant. Provide general health advice.
    IMPORTANT: You MUST NOT diagnose diseases or prescribe medications.
    
    User Symptom: {symptom}
    
    Provide:
    1. Empathetic acknowledgment
    2. General self-care suggestions
    3. When to see a doctor
    4. Reassurance
    
    Be cautious and caring in your response.
    """
    
    def analyze_symptom(self, symptom: str) -> Dict[str, Any]:
        """
        Analyze symptom and provide cautious health advice.
        Returns structured data for n8n workflow.
        """
        # Use LangChain with cautious prompt
        prompt = PromptTemplate(
            template=self.SYMPTOM_ANALYSIS_TEMPLATE,
            input_variables=["symptom"]
        )
        
        # Run through LLM
        chain = LLMChain(llm=self.llm, prompt=prompt)
        advice = chain.run(symptom=symptom)
        
        # Extract severity and category
        severity = self._extract_severity(symptom)
        category = self._categorize_symptom(symptom)
        
        return {
            "advice": advice,
            "severity": severity,
            "category": category,
            "raw_symptom": symptom
        }
```

### Priority 3: Add Symptom Categorization

```python
def categorize_symptom(symptom_text: str) -> str:
    """
    Categorize symptom into predefined categories.
    Returns: 'flu', 'fever', 'digestion', 'respiratory', 'unknown'
    """
    symptom_lower = symptom_text.lower()
    
    categories = {
        'flu': ['cough', 'runny nose', 'congestion', 'body aches', 'sore throat'],
        'fever': ['fever', 'high temperature', 'chills', 'hot', 'burning up'],
        'digestion': ['nausea', 'vomiting', 'diarrhea', 'stomach', 'abdominal pain'],
        'respiratory': ['breathing', 'shortness of breath', 'wheezing', 'chest tight'],
        'cardiovascular': ['chest pain', 'heart', 'palpitations'],
        'neurological': ['headache', 'dizziness', 'migraine', 'vertigo']
    }
    
    for category, keywords in categories.items():
        if any(keyword in symptom_lower for keyword in keywords):
            return category
    
    return 'unknown'
```

### Priority 4: Add Next Action Logic

```python
def determine_next_action(severity: str, category: str) -> str:
    """
    Determine what action n8n should take based on severity.
    """
    if severity == 'severe':
        return 'notify_doctor'
    elif severity == 'moderate' and category in ['cardiovascular', 'respiratory']:
        return 'notify_doctor'
    else:
        return 'save_to_db'
```

---

## 📊 Comparison Matrix

| Feature | Prompt Required | Current Status | Gap |
|---------|----------------|----------------|-----|
| FastAPI Server | ✅ | ✅ Exists | None |
| LangChain Integration | ✅ | ✅ Exists | Update prompt template |
| Chroma Vector DB | ✅ | ✅ Exists | None |
| n8n Ready | ✅ | ✅ Configured | Test integration |
| `/analyze` endpoint | ✅ | ❌ Missing | **Need to create** |
| Symptom input | ✅ | ⚠️ Different format | **Need to adapt** |
| Structured JSON output | ✅ | ⚠️ Different format | **Need to match** |
| Severity classification | ✅ | ✅ Exists (in cleaner) | **Need to integrate to API** |
| Category classification | ✅ | ⚠️ Different categories | **Need symptom categories** |
| Next action logic | ✅ | ❌ Missing | **Need to create** |
| Cautious tone | ✅ | ⚠️ Needs emphasis | **Update prompt** |
| No diagnosis rule | ✅ | ⚠️ Needs enforcement | **Update prompt** |
| Empathetic responses | ✅ | ⚠️ Needs emphasis | **Update prompt** |

---

## 🎯 Implementation Checklist

### Phase 1: Core Features (Priority)
- [ ] Create `/analyze` endpoint in FastAPI
- [ ] Add `SymptomRequest` and `AnalysisResponse` models
- [ ] Update LangChain prompt template (cautious, empathetic)
- [ ] Add symptom categorization function
- [ ] Add severity detection function
- [ ] Add next action logic

### Phase 2: Integration
- [ ] Test `/analyze` with sample symptoms
- [ ] Integrate with n8n workflow
- [ ] Add database logging
- [ ] Add error handling

### Phase 3: Safety & Polish
- [ ] Add "no diagnosis" disclaimer in all responses
- [ ] Add "see doctor" recommendations
- [ ] Test edge cases
- [ ] Add rate limiting
- [ ] Add input validation

---

## 📝 Sample Implementation Code

### Complete `/analyze` Endpoint

```python
# File: api_service/app/main.py

from pydantic import BaseModel, Field
from typing import Literal

class SymptomRequest(BaseModel):
    symptom: str = Field(..., min_length=3, description="User's symptom description")

class AnalysisResponse(BaseModel):
    advice: str = Field(..., description="Health advice for the symptom")
    severity: Literal["mild", "moderate", "severe"] = Field(..., description="Severity level")
    category: str = Field(..., description="Symptom category (flu, fever, digestion, etc)")
    next_action: Literal["save_to_db", "notify_doctor", "monitor"] = Field(..., description="Recommended action")
    disclaimer: str = Field(default="This is general advice. Please consult a doctor for proper diagnosis.")

@app.post("/analyze", response_model=AnalysisResponse, tags=["Health Analysis"])
async def analyze_symptom(request: SymptomRequest):
    """
    Analyze user symptoms and provide structured health advice.
    
    This endpoint:
    - Does NOT diagnose diseases
    - Provides general health advice
    - Returns structured JSON for n8n automation
    - Always recommends seeing a doctor when necessary
    """
    if not medical_service:
        raise HTTPException(status_code=503, detail="Medical service not available")
    
    try:
        # 1. Analyze symptom using LangChain
        analysis = medical_service.analyze_symptom(request.symptom)
        
        # 2. Determine severity (reuse existing logic from clean_data.py)
        severity = _determine_severity(request.symptom)
        
        # 3. Categorize symptom
        category = _categorize_symptom(request.symptom)
        
        # 4. Decide next action for n8n
        next_action = _determine_next_action(severity, category)
        
        # 5. Log to database
        _log_analysis(request.symptom, analysis, severity, category)
        
        return AnalysisResponse(
            advice=analysis,
            severity=severity,
            category=category,
            next_action=next_action
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Helper functions
def _determine_severity(symptom: str) -> str:
    """Determine severity: mild, moderate, or severe"""
    critical_keywords = ['severe', 'intense', 'unbearable', 'emergency', 'cant breathe']
    moderate_keywords = ['moderate', 'persistent', 'worsening', 'concerning']
    
    symptom_lower = symptom.lower()
    
    if any(kw in symptom_lower for kw in critical_keywords):
        return 'severe'
    elif any(kw in symptom_lower for kw in moderate_keywords):
        return 'moderate'
    else:
        return 'mild'

def _categorize_symptom(symptom: str) -> str:
    """Categorize into flu, fever, digestion, etc."""
    # Implementation as shown above
    pass

def _determine_next_action(severity: str, category: str) -> str:
    """Decide action for n8n automation"""
    if severity == 'severe':
        return 'notify_doctor'
    elif severity == 'moderate':
        return 'monitor'
    else:
        return 'save_to_db'

def _log_analysis(symptom, advice, severity, category):
    """Log to database for tracking"""
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO symptom_analysis (symptom, advice, severity, category, timestamp)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (symptom, advice, severity, category))
    conn.commit()
    conn.close()
```

---

## 🚀 Quick Start Implementation

**To fully match the prompt requirements:**

1. **Copy existing severity logic** from `scripts/clean_data.py` to API
2. **Create symptom categories** for flu/fever/digestion
3. **Update LangChain prompt** to be cautious and empathetic
4. **Create `/analyze` endpoint** as shown above
5. **Test with n8n** using webhook integration

---

## ✅ Summary

### What Works ✅
- ✅ FastAPI server running
- ✅ LangChain integration complete
- ✅ Chroma vector database operational
- ✅ n8n workflows configured
- ✅ Severity classification exists (in data cleaner)

### What Needs Work 🔧
- 🔧 Create `/analyze` endpoint
- 🔧 Adapt symptom categories (flu, fever, digestion)
- 🔧 Add next_action logic
- 🔧 Update prompt to be more cautious/empathetic
- 🔧 Integrate severity detection into API

### Effort Estimate ⏱️
- **Time**: 2-4 hours
- **Difficulty**: Medium (mostly adaptation, not new development)
- **Files to modify**: 2-3 files
- **Lines of code**: ~200 lines

---

**Status**: 70% Complete - Need to adapt existing components to match exact prompt requirements 🎯
