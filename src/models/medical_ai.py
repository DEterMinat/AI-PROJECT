#!/usr/bin/env python3
"""
🏥 Langchain Medical AI Service - Production Grade
=================================================

A comprehensive RAG-based medical Q&A system with:
- Custom T5-Small disease diagnosis model
- Rule-based emergency symptom detection
- Vector-based knowledge retrieval (ChromaDB)
- Conversation memory management
- SQLite logging for audit trails

Architecture:
    User Query → Emergency Check → T5 Model → Knowledge Base → Response

Safety Features:
    - Pre-screening for life-threatening symptoms
    - Medical disclaimer on all responses
    - Interaction logging for compliance

Author: Medical AI Team
Version: 2.0 (Refactored)
License: MIT
"""

import os
import json
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Langchain core
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ML/AI
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# ============================================================================
# EMERGENCY DETECTION SYSTEM
# ============================================================================

@dataclass
class EmergencyRule:
    """Data class representing an emergency symptom pattern and alert"""
    emergency_type: str
    symptom_patterns: List[List[str]]
    alert_message: str
    severity: str = "critical"  # critical, high, medium
    
    def matches(self, symptoms_text: str) -> bool:
        """
        Check if symptoms text matches any of this rule's patterns.
        
        Args:
            symptoms_text: Lowercased patient symptoms description
            
        Returns:
            True if ALL keywords in any pattern are found in symptoms_text
        """
        symptoms_lower = symptoms_text.lower()
        
        for pattern in self.symptom_patterns:
            if all(keyword in symptoms_lower for keyword in pattern):
                return True
        
        return False


class EmergencyDetector:
    """
    Rule-based emergency symptom detector.
    
    Catches life-threatening conditions that may be misdiagnosed by ML models.
    Uses pattern matching on symptom combinations to identify emergencies.
    
    Usage:
        detector = EmergencyDetector()
        alert = detector.check_symptoms("chest pain and arm pain")
        if alert:
            print(alert)  # Emergency alert message
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[EmergencyRule]:
        """Initialize emergency detection rules with medical patterns"""
        
        return [
            EmergencyRule(
                emergency_type="heart_attack",
                symptom_patterns=[
                    ["chest pain", "arm pain"],
                    ["chest pain", "shortness of breath"],
                    ["chest pressure", "sweating"],
                    ["chest pain", "nausea"],
                    ["crushing chest pain"],
                ],
                alert_message=(
                    "🚨 **EMERGENCY - POSSIBLE HEART ATTACK**\n\n"
                    "Your symptoms suggest a **potential heart attack (myocardial infarction)**.\n\n"
                    "**IMMEDIATE ACTION REQUIRED:**\n"
                    "• Call emergency services (911/1669) NOW\n"
                    "• Chew aspirin if available (unless allergic)\n"
                    "• Do NOT drive yourself\n"
                    "• Stay calm and rest while waiting\n\n"
                    "**Symptoms:** Chest pain/pressure, arm pain, shortness of breath, sweating, nausea.\n\n"
                    "⚠️ **DO NOT WAIT - Every minute matters for heart attacks!**"
                ),
                severity="critical"
            ),
            
            EmergencyRule(
                emergency_type="stroke",
                symptom_patterns=[
                    ["facial drooping", "arm weakness"],
                    ["slurred speech", "confusion"],
                    ["sudden severe headache", "vision problems"],
                    ["numbness", "weakness", "one side"],
                    ["sudden confusion", "trouble speaking"],
                ],
                alert_message=(
                    "🚨 **EMERGENCY - POSSIBLE STROKE**\n\n"
                    "Your symptoms suggest a **potential stroke (cerebrovascular accident)**.\n\n"
                    "**IMMEDIATE ACTION REQUIRED:**\n"
                    "• Call emergency services (911/1669) NOW\n"
                    "• Note the time symptoms started\n"
                    "• Do NOT give food, drink, or medication\n"
                    "• Keep person lying down\n\n"
                    "**FAST Test:**\n"
                    "• Face: Is one side drooping?\n"
                    "• Arms: Can they raise both arms?\n"
                    "• Speech: Is speech slurred?\n"
                    "• Time: Call for help immediately!\n\n"
                    "⚠️ **Time = Brain - Act within 3 hours for best outcome!**"
                ),
                severity="critical"
            ),
            
            EmergencyRule(
                emergency_type="severe_allergic_reaction",
                symptom_patterns=[
                    ["swelling", "throat", "difficulty breathing"],
                    ["hives", "swelling", "difficulty breathing"],
                    ["anaphylaxis"],
                    ["throat closing", "swelling"],
                ],
                alert_message=(
                    "🚨 **EMERGENCY - SEVERE ALLERGIC REACTION (ANAPHYLAXIS)**\n\n"
                    "Your symptoms suggest **anaphylaxis** - a life-threatening allergic reaction.\n\n"
                    "**IMMEDIATE ACTION REQUIRED:**\n"
                    "• Call emergency services (911/1669) NOW\n"
                    "• Use EpiPen/epinephrine if available\n"
                    "• Lie down and elevate legs\n"
                    "• Do NOT stand up suddenly\n\n"
                    "**Symptoms:** Throat swelling, difficulty breathing, hives, rapid pulse.\n\n"
                    "⚠️ **Can be fatal within minutes - Act immediately!**"
                ),
                severity="critical"
            ),
            
            EmergencyRule(
                emergency_type="meningitis",
                symptom_patterns=[
                    ["severe headache", "stiff neck", "fever"],
                    ["headache", "neck stiffness", "sensitivity to light"],
                    ["severe headache", "vomiting", "confusion"],
                ],
                alert_message=(
                    "🚨 **EMERGENCY - POSSIBLE MENINGITIS**\n\n"
                    "Your symptoms suggest **possible meningitis** - infection of brain/spinal cord membranes.\n\n"
                    "**IMMEDIATE ACTION REQUIRED:**\n"
                    "• Go to Emergency Room immediately\n"
                    "• Do NOT delay\n"
                    "• Avoid contact with others (may be contagious)\n\n"
                    "**Symptoms:** Severe headache, stiff neck, fever, sensitivity to light, confusion.\n\n"
                    "⚠️ **Can cause permanent brain damage or death if untreated!**"
                ),
                severity="critical"
            ),
            
            EmergencyRule(
                emergency_type="severe_bleeding",
                symptom_patterns=[
                    ["severe bleeding"],
                    ["heavy bleeding", "won't stop"],
                    ["bleeding profusely"],
                    ["blood won't stop"],
                ],
                alert_message=(
                    "🚨 **EMERGENCY - SEVERE BLEEDING**\n\n"
                    "You are experiencing **severe bleeding** that requires immediate attention.\n\n"
                    "**IMMEDIATE ACTION:**\n"
                    "• Call emergency services (911/1669) NOW\n"
                    "• Apply direct pressure to wound\n"
                    "• Elevate injured area above heart\n"
                    "• Do NOT remove embedded objects\n\n"
                    "⚠️ **Severe blood loss can be life-threatening!**"
                ),
                severity="critical"
            ),
        ]
    
    def check_symptoms(self, symptoms_text: str) -> Optional[str]:
        """
        Check if symptoms match any emergency patterns.
        
        Args:
            symptoms_text: Patient's description of symptoms
            
        Returns:
            Emergency alert message if pattern matched, None otherwise
        """
        for rule in self.rules:
            if rule.matches(symptoms_text):
                print(f"🚨 EMERGENCY DETECTED: {rule.emergency_type}")
                return rule.alert_message
        
        return None
    
    def get_rule_count(self) -> int:
        """Get total number of emergency rules loaded"""
        return len(self.rules)


# ============================================================================
# MEDICAL LLM MODELS
# ============================================================================

class CustomMedicalLLM(LLM):
    """
    T5-Family Medical Diagnosis Model (Encoder-Decoder Architecture).
    
    **Supported Models:**
    - FLAN-T5-Base (google/flan-t5-base) - 250M params - **RECOMMENDED** - 60-75% accuracy
    - FLAN-T5-Large (google/flan-t5-large) - 780M params - 70-80% accuracy  
    - T5-Small (t5-small) - 60M params - Legacy - 40% accuracy
    
    **Architecture:** Encoder-Decoder (perfect for Q&A tasks)
    **Task Format:** "diagnose disease: {symptoms}" → "{disease_name}"
    
    Attributes:
        model_path: Path to fine-tuned T5/FLAN-T5 model
        tokenizer: HuggingFace T5 tokenizer
        model: T5ForConditionalGeneration model
        device: Computation device (cuda/cpu)
        model_type: Model type (flan-t5-base, flan-t5-large, t5-small)
        emergency_detector: Rule-based emergency symptom checker
        knowledge_base: Medical knowledge vectorstore
    """
    
    model_path: str
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    device: str = "cpu"
    model_type: str = "unknown"
    max_length: int = 256
    emergency_detector: Optional[EmergencyDetector] = None
    knowledge_base: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, model_path: str, knowledge_base=None, **kwargs):
        """
        Initialize CustomMedicalLLM with T5/FLAN-T5 model.
        
        Args:
            model_path: Path to fine-tuned T5/FLAN-T5 model
            knowledge_base: Optional MedicalKnowledgeBase instance
            **kwargs: Additional LLM parameters
        """
        super().__init__(model_path=model_path, **kwargs)
        
        self.knowledge_base = knowledge_base
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emergency_detector = EmergencyDetector()
        
        # Detect model type from path
        model_path_lower = str(model_path).lower()
        if "flan-t5-large" in model_path_lower or "flan_t5_large" in model_path_lower:
            self.model_type = "flan-t5-large"
            print(f"🤖 Initializing FLAN-T5-Large (780M params, 70-80% accuracy)")
        elif "flan-t5-base" in model_path_lower or "flan_t5_base" in model_path_lower or "flan" in model_path_lower:
            self.model_type = "flan-t5-base"
            print(f"🤖 Initializing FLAN-T5-Base (250M params, 60-75% accuracy)")
        else:
            self.model_type = "t5-small"
            print(f"🤖 Initializing T5-Small (60M params, 40% accuracy - Legacy)")
        
        print(f"   Device: {self.device}")
        print(f"   Emergency Rules: {self.emergency_detector.get_rule_count()}")
        
        self._load_model()
    
    def _load_model(self):
        """
        Load T5/FLAN-T5 model and tokenizer from disk.
        
        Raises:
            FileNotFoundError: If model path doesn't exist
            Exception: If model loading fails
        """
        try:
            model_name = {
                "flan-t5-large": "FLAN-T5-Large",
                "flan-t5-base": "FLAN-T5-Base", 
                "t5-small": "T5-Small"
            }.get(self.model_type, "T5")
            
            print(f"📥 Loading {model_name} model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ {model_name} model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier"""
        return "custom_medical_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate medical diagnosis response.
        
        Pipeline:
        1. Check for emergency symptoms (rule-based)
        2. If emergency: Return immediate alert (bypass ML)
        3. If not emergency: Use T5 model for diagnosis
        4. Enrich response with knowledge base information
        
        Args:
            prompt: Patient symptoms description
            stop: Optional stop sequences (unused)
            run_manager: Optional callback manager (unused)
            **kwargs: Additional generation parameters
            
        Returns:
            Medical diagnosis response with recommendations
        """
        try:
            # STEP 1: Emergency Pre-screening
            emergency_alert = self.emergency_detector.check_symptoms(prompt)
            if emergency_alert:
                return emergency_alert
            
            # STEP 2: T5 Model Diagnosis
            formatted_prompt = f"diagnose disease: {prompt.lower()}"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate with T5
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode disease name
            disease = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # STEP 3: Build Response
            return self._format_diagnosis_response(disease)
            
        except Exception as e:
            print(f"❌ Diagnosis error: {e}")
            return (
                "I apologize, I'm having trouble processing your request right now. "
                "Please try describing your symptoms again, or consider consulting "
                "a medical professional directly."
            )
    
    def _format_diagnosis_response(self, disease: str) -> str:
        """
        Format T5 model output into user-friendly response.
        
        Args:
            disease: Disease name from T5 model (may be empty)
            
        Returns:
            Formatted diagnosis response with disclaimer
        """
        if not disease:
            return (
                "I'm having difficulty identifying a specific condition based on "
                "the symptoms you've described.\n\n"
                "Could you provide more details, such as:\n"
                "- When did the symptoms start?\n"
                "- How severe are they?\n"
                "- Are there any other symptoms?"
            )
        
        # Clean disease name
        disease_name = disease.replace('_', ' ').title()
        
        # Build response parts
        response_parts = [
            f"Based on your symptoms, you likely have **{disease_name}**.\n"
        ]
        
        # Add disease info from knowledge base
        disease_info = self._get_disease_info(disease_name)
        if disease_info:
            response_parts.append(f"\n**About this condition:**\n{disease_info}\n")
        
        # Add recommendation
        response_parts.append(
            "\n**Recommendation:**\n"
            "I strongly recommend consulting with a healthcare professional for "
            "an accurate diagnosis and appropriate treatment plan. "
            "Would you like to know more about your symptoms or this condition?"
        )
        
        return "".join(response_parts)
    
    def _get_disease_info(self, disease_name: str) -> str:
        """
        Retrieve disease information from vectorstore.
        
        Args:
            disease_name: Name of disease to search for
            
        Returns:
            Concise disease information (~200 chars) or empty string
        """
        try:
            if not self.knowledge_base or not self.knowledge_base.vectorstore:
                return ""
                return ""
            
            # Search for disease information
            docs = self.knowledge_base.vectorstore.similarity_search(
                f"{disease_name} symptoms causes treatment",
                k=3
            )
            
            if not docs:
                return ""
            
            # Extract meaningful medical info (skip Q&A format)
            info_parts = []
            
            for doc in docs:
                content = doc.page_content
                
                # Skip Q&A formatted content
                if 'Question:' in content or 'Answer:' in content:
                    continue
                
                # Extract informative lines
                lines = [
                    line.strip() 
                    for line in content.split('\n') 
                    if line.strip() and len(line) > 20 and not line.endswith(':')
                ]
                
                info_parts.extend(lines[:3])  # Max 3 lines per document
                
                if len(info_parts) >= 6:  # Stop after 6 total lines
                    break
            
            if not info_parts:
                return ""
            
            # Combine and truncate
            combined = ' '.join(info_parts[:2])  # Use top 2 sentences
            
            if len(combined) > 300:
                combined = combined[:297] + "..."
            
            return combined
            
        except Exception as e:
            print(f"⚠️ Knowledge retrieval error: {e}")
            return ""


class MedicalKnowledgeBase:
    """Medical Knowledge Base with Vector Database"""
    
    def __init__(self, persist_directory: str = "./data/vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self._setup_vectorstore()
    
    def _setup_vectorstore(self):
        """Setup Chroma vector database"""
        try:
            print("📚 Setting up Medical Knowledge Base...")
            
            # Create directory if not exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Load or create vectorstore
            if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
                print("📖 Loading existing knowledge base...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                doc_count = self.vectorstore._collection.count()
                
                # If empty, populate it
                if doc_count == 0:
                    print("📝 Knowledge base empty, populating...")
                    self._populate_initial_knowledge()
                else:
                    print(f"✅ Loaded {doc_count} documents from knowledge base")
            else:
                print("📝 Creating new knowledge base...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self._populate_initial_knowledge()
                
        except Exception as e:
            print(f"❌ Error setting up vectorstore: {e}")
            self.vectorstore = None
    
    def _populate_initial_knowledge(self):
        """Populate initial medical knowledge"""
        
        medical_documents = [
            {
                "content": """Diabetes Mellitus (โรคเบาหวาน) is a chronic metabolic disorder characterized by elevated blood glucose levels.

Common Symptoms:
- Excessive thirst (polydipsia)
- Frequent urination (polyuria)
- Unexplained weight loss
- Increased hunger
- Fatigue and weakness
- Blurred vision
- Slow-healing wounds
- Tingling or numbness in hands/feet

Types:
- Type 1: Autoimmune destruction of insulin-producing cells
- Type 2: Insulin resistance and relative insulin deficiency
- Gestational: Develops during pregnancy

Risk Factors:
- Family history
- Obesity
- Physical inactivity
- Age over 45
- High blood pressure
- Abnormal cholesterol levels""",
                "metadata": {"topic": "diabetes", "category": "symptoms_overview", "language": "en"}
            },
            {
                "content": """Diabetes Treatment and Management:

Lifestyle Modifications:
- Healthy diet: Low glycemic index foods, controlled carbohydrate intake
- Regular physical activity: 150 minutes per week
- Weight management
- Stress reduction
- Adequate sleep

Medical Management:
- Blood glucose monitoring
- Insulin therapy (Type 1 and some Type 2)
- Oral medications: Metformin, sulfonylureas, DPP-4 inhibitors
- GLP-1 receptor agonists
- SGLT2 inhibitors

Complications Prevention:
- Regular eye exams
- Foot care
- Kidney function tests
- Cardiovascular risk management
- A1C target: <7% for most adults

IMPORTANT: Always consult healthcare professionals for diagnosis and treatment plans.""",
                "metadata": {"topic": "diabetes", "category": "treatment", "language": "en"}
            },
            {
                "content": """Hypertension (High Blood Pressure) - Essential Information:

Definition: Blood pressure consistently ≥140/90 mmHg

Symptoms (often silent):
- Headaches (especially morning)
- Dizziness
- Shortness of breath
- Chest pain
- Visual changes
- Nosebleeds (rare)

Risk Factors:
- Age (>60 years)
- Family history
- Obesity
- High salt intake
- Sedentary lifestyle
- Smoking
- Excessive alcohol
- Chronic stress
- Kidney disease

Complications:
- Heart attack and stroke
- Heart failure
- Kidney damage
- Vision loss
- Peripheral artery disease

Management:
- DASH diet (low sodium, rich in fruits/vegetables)
- Regular exercise
- Weight reduction
- Limit alcohol
- Stress management
- Medications: ACE inhibitors, ARBs, beta-blockers, diuretics, calcium channel blockers

Target: <130/80 mmHg for most adults""",
                "metadata": {"topic": "hypertension", "category": "symptoms_treatment", "language": "en"}
            },
            {
                "content": """Influenza (Flu) - Comprehensive Guide:

Symptoms:
- Sudden onset high fever (38-40°C)
- Severe headache
- Muscle and body aches
- Extreme fatigue
- Dry cough
- Sore throat
- Runny or stuffy nose
- Sometimes nausea, vomiting, diarrhea (more common in children)

Transmission:
- Respiratory droplets from coughing/sneezing
- Direct contact with infected surfaces
- Contagious 1 day before symptoms to 5-7 days after

Prevention:
- Annual flu vaccination (most effective)
- Frequent handwashing
- Avoid touching face
- Cover coughs and sneezes
- Stay home when sick
- Maintain distance from sick people

Treatment:
- Rest and hydration
- Fever reducers (acetaminophen, ibuprofen)
- Antiviral medications (oseltamivir, zanamivir) within 48 hours
- Symptom relief medications

When to Seek Emergency Care:
- Difficulty breathing
- Chest pain
- Persistent dizziness
- Severe vomiting
- Confusion
- Symptoms improve then worsen""",
                "metadata": {"topic": "flu", "category": "symptoms_prevention_treatment", "language": "en"}
            },
            {
                "content": """Basic First Aid Guidelines:

For Minor Cuts and Wounds:
1. Wash hands thoroughly
2. Clean wound with water
3. Apply gentle pressure to stop bleeding
4. Apply antibiotic ointment
5. Cover with sterile bandage
6. Change bandage daily

For Burns:
1. Remove from heat source
2. Cool with running water (10-20 minutes)
3. Cover with sterile dressing
4. Do NOT apply ice, butter, or oils
5. Seek medical help for severe burns

For High Fever:
1. Check temperature regularly
2. Give fever reducers (appropriate dose)
3. Tepid sponge bath
4. Increase fluid intake
5. Light clothing
6. Rest in cool environment

For Sprains:
- R: Rest the injured area
- I: Ice (15-20 minutes every 2-3 hours)
- C: Compression with elastic bandage
- E: Elevation above heart level

When to Seek Emergency Care:
- Severe bleeding that won't stop
- Deep wounds
- Difficulty breathing
- Chest pain
- Severe burns
- Head injuries
- Loss of consciousness
- Severe allergic reactions

Remember: First aid is temporary care. Always seek professional medical attention for serious injuries.""",
                "metadata": {"topic": "first_aid", "category": "emergency_care", "language": "en"}
            },
            {
                "content": """Heart Disease - Warning Signs and Prevention:

Warning Signs:
- Chest discomfort or pain
- Shortness of breath
- Pain in arms, neck, jaw, back
- Lightheadedness or dizziness
- Nausea
- Cold sweat
- Irregular heartbeat

Risk Factors:
- High blood pressure
- High cholesterol
- Diabetes
- Smoking
- Obesity
- Physical inactivity
- Family history
- Age and gender

Prevention:
- Heart-healthy diet (low saturated fat, high fiber)
- Regular exercise (30 minutes/day)
- Maintain healthy weight
- Quit smoking
- Limit alcohol
- Manage stress
- Control blood pressure and cholesterol
- Regular health screenings

EMERGENCY: Call emergency services immediately if experiencing chest pain, shortness of breath, or other severe symptoms.""",
                "metadata": {"topic": "heart_disease", "category": "prevention_symptoms", "language": "en"}
            }
        ]
        
        # Convert to Document objects
        documents = [
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ) for doc in medical_documents
        ]
        
        # Add to vectorstore
        self.vectorstore.add_documents(documents)
        print(f"✅ Added {len(documents)} medical documents to knowledge base")
    
    def search_knowledge(self, query: str, k: int = 3) -> List[Document]:
        """Search relevant knowledge"""
        if self.vectorstore:
            try:
                docs = self.vectorstore.similarity_search(query, k=k)
                return docs
            except Exception as e:
                print(f"⚠️ Error searching knowledge: {e}")
                return []
        return []


class LangchainMedicalService:
    """
    Main Langchain Medical AI Service with FLAN-T5.
    
    Model Priority (auto-detection):
    1. FLAN-T5-Large (780M params, 70-80% accuracy) - Best
    2. FLAN-T5-Base (250M params, 60-75% accuracy) - Recommended  
    3. T5-Small (60M params, 40% accuracy) - Legacy fallback
    
    All models use Encoder-Decoder architecture (perfect for Q&A).
    """
    
    def __init__(
        self, 
        model_path: str = None,
        vectorstore_path: str = "./data/vectorstore"
    ):
        print("=" * 60)
        print("🏥 Initializing Langchain Medical AI Service")
        print("=" * 60)
        
        self.vectorstore_path = vectorstore_path
        
        # Auto-detect model path if not provided
        if model_path is None:
            model_path = self._auto_detect_model()
        
        self.model_path = model_path
        
        # Conversation memory
        self.conversation_history = {}
        self.max_history = 5
        
        # Initialize components
        print("\n[1/4] Loading Medical Knowledge Base...")
        self.knowledge_base = MedicalKnowledgeBase(persist_directory=vectorstore_path)
        
        print("\n[2/4] Loading Medical Model...")
        self.llm = self._load_model()
        
        print("\n[3/4] Creating QA Chain...")
        self._setup_qa_chain()
        
        print("\n[4/4] Setting up Logging Database...")
        self._setup_database()
        
        print("\n" + "=" * 60)
        model_name = getattr(self.llm, 'model_type', 'Unknown')
        print(f"✅ Langchain Medical Service Ready!")
        print(f"   Model: {model_name}")
        print("=" * 60 + "\n")
    
    def _auto_detect_model(self) -> str:
        """
        Auto-detect best available model.
        
        Priority:
        1. FLAN-T5-Large (780M, 70-80%)
        2. FLAN-T5-Base (250M, 60-75%) <- Recommended
        3. T5-Small (60M, 40%) <- Legacy
        
        Returns:
            str: Path to detected model
            
        Raises:
            FileNotFoundError: If no model found
        """
        from pathlib import Path
        
        models_dir = Path("models")
        
        # Try FLAN-T5-Large first (best accuracy)
        flan_large_models = sorted(
            models_dir.glob("flan_t5_large_diagnosis_*"), 
            key=lambda p: p.name, 
            reverse=True
        )
        if flan_large_models:
            print(f"✅ Found FLAN-T5-Large: {flan_large_models[0].name} (70-80% accuracy)")
            return str(flan_large_models[0])
        
        # Try FLAN-T5-Base (recommended)
        flan_base_models = sorted(
            models_dir.glob("flan_t5_diagnosis_*"), 
            key=lambda p: p.name, 
            reverse=True
        )
        if flan_base_models:
            print(f"✅ Found FLAN-T5-Base: {flan_base_models[0].name} (60-75% accuracy)")
            return str(flan_base_models[0])
        
        # Fallback to T5-Small (legacy)
        t5_models = sorted(
            models_dir.glob("t5_diagnosis_*"), 
            key=lambda p: p.name, 
            reverse=True
        )
        if t5_models:
            print(f"⚠️ Using legacy T5-Small: {t5_models[0].name} (40% accuracy)")
            print(f"   Recommend training FLAN-T5-Base for 60-75% accuracy")
            return str(t5_models[0])
        
        raise FileNotFoundError(
            "No trained model found!\n\n"
            "Train FLAN-T5-Base (recommended):\n"
            "  train-model.bat\n\n"
            "Expected accuracy: 60-75% (vs 40% T5-Small)\n"
            "Training time: 6-8 hours on CPU"
        )
    
    def _load_model(self):
        """Load T5/FLAN-T5 model"""
        try:
            return CustomMedicalLLM(
                model_path=self.model_path,
                knowledge_base=self.knowledge_base
            )
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
        
        print("\n[3/4] Creating QA Chain...")
        self._setup_qa_chain()
        
        print("\n[4/4] Setting up Logging Database...")
        self._setup_database()
        
        print("\n" + "=" * 60)
        print("✅ Langchain Medical Service Ready!")
        print("=" * 60 + "\n")
    
    def _setup_qa_chain(self):
        """Setup RetrievalQA chain with custom prompt"""
        
        # Medical QA prompt template - Natural conversation style
        prompt_template = """You are a friendly medical AI assistant having a natural conversation.

CONVERSATION STYLE:
- Reply naturally like a caring friend who knows medicine
- Keep responses SHORT and CONVERSATIONAL (2-3 sentences for simple questions)
- Use simple, easy-to-understand language
- Be warm and empathetic
- Only provide detailed info if specifically asked

SAFETY RULES:
- Never diagnose or prescribe
- Always suggest seeing a doctor for serious concerns
- Provide general health information only

Knowledge Available:
{context}

User: {question}

Assistant (reply naturally and briefly):"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        if self.knowledge_base.vectorstore:
            try:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.knowledge_base.vectorstore.as_retriever(
                        search_kwargs={"k": 2}
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": self.prompt}
                )
                print("✅ QA Chain created with RAG capabilities")
            except Exception as e:
                print(f"⚠️ Error creating QA chain: {e}")
                self.qa_chain = None
        else:
            print("⚠️ No vectorstore available, using LLM only mode")
            self.qa_chain = None
    
    def _setup_database(self):
        """Setup SQLite database for logging"""
        try:
            db_path = "data/medical_ai.db"
            os.makedirs("data", exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS medical_qa_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    sources TEXT,
                    processing_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
            print(f"✅ Database ready at: {db_path}")
            
        except Exception as e:
            print(f"⚠️ Database setup warning: {e}")
    def _get_conversation_context(self, user_id: str) -> str:
        """Get recent conversation history for context"""
        if user_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[user_id]
        if not history:
            return ""
        
        # Format last 2-3 exchanges
        context_parts = []
        for exchange in history[-3:]:
            context_parts.append(f"User: {exchange['question']}")
            context_parts.append(f"Assistant: {exchange['answer'][:100]}...")
        
        return "\n".join(context_parts)
    
    def _update_conversation_history(self, user_id: str, question: str, answer: str):
        """Update conversation history for user"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.conversation_history[user_id]) > self.max_history:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history:]
    
    def ask_question(self, question: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Process medical question with conversation memory
        
        Args:
            question: Patient's medical question
            user_id: User identifier for logging and conversation tracking
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        
        start_time = datetime.now()
        print(f"\n💬 Processing question from {user_id}: {question[:50]}...")
        
        try:
            # Get conversation context
            context = self._get_conversation_context(user_id)
            if context:
                print(f"📜 Using conversation history ({len(self.conversation_history.get(user_id, []))} messages)")
            
            # Use RAG chain if available
            if self.qa_chain:
                print("🔍 Searching knowledge base...")
                result = self.qa_chain({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
                sources = [
                    {
                        "topic": doc.metadata.get("topic", "unknown"),
                        "category": doc.metadata.get("category", "general"),
                        "excerpt": doc.page_content[:200] + "..."
                    }
                    for doc in source_docs
                ]
            else:
                # Fallback to direct LLM
                print("🤖 Using direct LLM (no RAG)...")
                answer = self.llm._call(question)
                sources = [{"topic": "direct_llm", "category": "no_rag"}]
            
            # Update conversation history
            self._update_conversation_history(user_id, question, answer)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log to database
            self._log_interaction(user_id, question, answer, sources, processing_time)
            
            print(f"✅ Response generated in {processing_time:.2f}s")
            
            # Return structured response
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "disclaimer": "⚕️ This is general health information only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment."
            }
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "answer": "I apologize, but I encountered an error processing your question. Please try again or consult a medical professional.",
                "error": error_msg,
                "sources": [],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _log_interaction(self, user_id: str, question: str, answer: str, 
                        sources: List[Dict], processing_time: float):
        """Log interaction to database"""
        try:
            conn = sqlite3.connect("data/medical_ai.db")
            conn.execute(
                """INSERT INTO medical_qa_log 
                   (user_id, question, answer, sources, processing_time) 
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, question, answer, json.dumps(sources), processing_time)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Logging warning: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status with accurate model information"""
        try:
            vectorstore_docs = 0
            vectorstore_available = False
            
            if self.knowledge_base and self.knowledge_base.vectorstore:
                try:
                    vectorstore_docs = self.knowledge_base.vectorstore._collection.count()
                    vectorstore_available = True
                except Exception as e:
                    print(f"⚠️ Vectorstore count error: {e}")
                    vectorstore_available = True
            
            # Get model type and accuracy from LLM
            model_type = getattr(self.llm, 'model_type', 'unknown')
            
            # Map model type to user-friendly name and accuracy
            model_info = {
                "flan-t5-large": ("FLAN-T5-Large (780M)", "70-80%"),
                "flan-t5-base": ("FLAN-T5-Base (250M)", "60-75%"),
                "t5-small": ("T5-Small (60M)", "40%")
            }
            
            model_name, accuracy = model_info.get(model_type, (model_type.title(), "N/A"))
            
            return {
                "service": "Langchain Medical AI",
                "status": "online",
                "initialized": True,
                "model_type": model_name,
                "model_accuracy": accuracy,
                "model_loaded": self.llm and (
                    hasattr(self.llm, 'model') and self.llm.model is not None
                ),
                "model_path": self.model_path,
                "vectorstore_available": vectorstore_available,
                "vectorstore_docs": vectorstore_docs,
                "rag_enabled": self.qa_chain is not None,
                "device": self.llm.device if self.llm else "unknown",
                "emergency_rules": self.llm.emergency_detector.get_rule_count() if hasattr(self.llm, 'emergency_detector') else 0
            }
        except Exception as e:
            return {
                "service": "Langchain Medical AI",
                "status": "error",
                "initialized": False,
                "error": str(e)
            }


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_service():
    """Test the Langchain Medical Service (auto-detects FLAN-T5 or T5-Small)"""
    
    print("\n" + "="*80)
    print("🧪 TESTING LANGCHAIN MEDICAL SERVICE")
    print("="*80 + "\n")
    
    # Initialize service (auto-detects FLAN-T5-Large > FLAN-T5-Base > T5-Small)
    service = LangchainMedicalService()
    
    # Test questions
    test_questions = [
        "fever, cough, sore throat",
        "severe headache, stiff neck, fever",
        "chest pain, shortness of breath",
        "frequent urination, excessive thirst"
    ]
    
    print("\n" + "-"*80)
    print("Testing with sample symptoms:")
    print("-"*80 + "\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {question}")
        print('='*80)
        
        result = service.ask_question(question, user_id=f"test_user_{i}")
        
        print(f"\n📋 Answer:\n{result['answer']}\n")
        
        if result.get('sources'):
            print("📚 Sources:")
            for j, source in enumerate(result['sources'], 1):
                print(f"  {j}. Topic: {source.get('topic', 'unknown')}")
                if 'excerpt' in source:
                    print(f"     {source['excerpt'][:100]}...\n")
        
        print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        print(f"📊 Status: {'✅ Success' if result['success'] else '❌ Failed'}")
    
    # Show service status
    print("\n" + "="*80)
    print("📊 SERVICE STATUS")
    print("="*80)
    status = service.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_service()
