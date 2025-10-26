#!/usr/bin/env python3
"""
Complete Unified Medical Data Cleaner - ALL-IN-ONE (ENGLISH-ONLY)
รวมกระบวนการ Clean data ทั้งหมดในไฟล์เดียว - English Only
Features:
- Load all raw datasets
- Advanced cleaning & deduplication
- Medical terminology normalization
- English-only processing (NO translation)
- Severity classification
- Quality optimization
- Output as English JSON/CSV only
"""

import json
import pandas as pd
import numpy as np
import re
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NO TRANSLATION - English-only mode
# Translation libraries removed to avoid dependency conflicts
logger.info("🇬🇧 English-only mode: No translation dependencies needed")

# Text processing libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic text processing")

# Similarity libraries
try:
    from difflib import SequenceMatcher
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    logger.warning("Text similarity libraries not available")

class CompleteUnifiedMedicalCleaner:
    def __init__(self, batch_size=10000, use_multiprocessing=True, max_workers=None):
        """Initialize the complete unified medical data cleaner with batch processing"""
        
        # Performance settings
        self.batch_size = batch_size
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        
        # Fast duplicate checking cache
        self.question_hashes = set()
        self.previous_questions_cache = set()
        
        # Pre-compile regex patterns for speed
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://[^\s]+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.multiple_space_pattern = re.compile(r'\s+')
        self.excessive_punct_pattern = re.compile(r'([!?.]){2,}')
        self.mcq_pattern = re.compile(r'\s*\(([A-Da-d])\)\s*')
        
        # Statistics tracking
        self.stats = {
            'original_count': 0,
            'duplicates_removed': 0,
            'irrelevant_removed': 0,
            'incomplete_removed': 0,
            'offensive_removed': 0,
            'formatting_cleaned': 0,
            'language_errors_fixed': 0,
            'normalized_count': 0,
            'missing_data_filled': 0,
            'abbreviations_expanded': 0,
            'statistical_normalized': 0,
            'answers_normalized': 0,
            'translated_count': 0,
            'chunked_count': 0,
            'severity_classified': 0,
            'final_english_count': 0,
            'final_thai_count': 0,
            'final_unified_count': 0,
            'batches_processed': 0
        }
        
        # Initialize translation clients
        self.translator = None
        self.pythainlp_translator = None
        self._init_translation()
        
        # Medical abbreviations dictionary (comprehensive)
        self.medical_abbreviations = {
            'HDL-C': 'High-Density Lipoprotein Cholesterol',
            'LDL-C': 'Low-Density Lipoprotein Cholesterol',
            'cIMT': 'carotid Intima-Media Thickness',
            'PCD': 'Percutaneous Coronary Disease',
            'MI': 'Myocardial Infarction',
            'CVD': 'Cardiovascular Disease',
            'HTN': 'Hypertension',
            'CAD': 'Coronary Artery Disease',
            'CHF': 'Congestive Heart Failure',
            'ECG': 'Electrocardiogram',
            'EKG': 'Electrocardiogram',
            'CBC': 'Complete Blood Count',
            'BUN': 'Blood Urea Nitrogen',
            'CRP': 'C-Reactive Protein',
            'ESR': 'Erythrocyte Sedimentation Rate',
            'HbA1c': 'Hemoglobin A1c',
            'TSH': 'Thyroid Stimulating Hormone',
            'T3': 'Triiodothyronine',
            'T4': 'Thyroxine',
            'PT': 'Prothrombin Time',
            'PTT': 'Partial Thromboplastin Time',
            'INR': 'International Normalized Ratio',
            'DM': 'Diabetes Mellitus',
            'COPD': 'Chronic Obstructive Pulmonary Disease',
            'GERD': 'Gastroesophageal Reflux Disease',
            'UTI': 'Urinary Tract Infection',
            'URI': 'Upper Respiratory Infection',
            'DVT': 'Deep Vein Thrombosis',
            'PE': 'Pulmonary Embolism',
            'IBS': 'Irritable Bowel Syndrome',
            'IBD': 'Inflammatory Bowel Disease',
            'mg/dL': 'milligrams per deciliter',
            'mmol/L': 'millimoles per liter',
            'mmHg': 'millimeters of mercury',
            'bpm': 'beats per minute',
            'kg/m²': 'kilograms per square meter',
            'mL/min': 'milliliters per minute',
            'ICU': 'Intensive Care Unit',
            'ER': 'Emergency Room',
            'OR': 'Operating Room',
            'IV': 'Intravenous',
            'IM': 'Intramuscular',
            'SC': 'Subcutaneous',
            'PO': 'Per Oral',
            'PRN': 'As Needed',
            'QID': 'Four Times Daily',
            'TID': 'Three Times Daily',
            'BID': 'Twice Daily',
            'QD': 'Once Daily',
            'NPO': 'Nothing by Mouth',
            'SOB': 'Shortness of Breath',
            'DOE': 'Dyspnea on Exertion',
            'CHD': 'Coronary Heart Disease',
            'AF': 'Atrial Fibrillation',
            'VT': 'Ventricular Tachycardia',
            'VF': 'Ventricular Fibrillation'
        }
        
        # Medical terms for relevance checking
        self.medical_terms = {
            'conditions': [
                'diabetes', 'hypertension', 'cancer', 'pneumonia', 'asthma', 'covid', 'flu', 'fever',
                'stroke', 'heart attack', 'kidney disease', 'liver disease', 'arthritis', 'depression',
                'anxiety', 'tuberculosis', 'hepatitis', 'hiv', 'aids', 'malaria', 'dengue',
                'cholera', 'typhoid', 'measles', 'mumps', 'rubella', 'chickenpox', 'shingles'
            ],
            'symptoms': [
                'pain', 'headache', 'nausea', 'fatigue', 'dizziness', 'cough', 'shortness of breath',
                'chest pain', 'abdominal pain', 'back pain', 'joint pain', 'muscle pain',
                'vomiting', 'diarrhea', 'constipation', 'fever', 'chills', 'sweating',
                'weakness', 'numbness', 'tingling', 'blurred vision', 'hearing loss'
            ],
            'treatments': [
                'medication', 'surgery', 'therapy', 'treatment', 'prescription', 'antibiotic',
                'antiviral', 'antifungal', 'chemotherapy', 'radiation', 'immunotherapy',
                'physical therapy', 'occupational therapy', 'speech therapy', 'dialysis',
                'transplant', 'bypass', 'angioplasty', 'stent', 'pacemaker'
            ],
            'body_parts': [
                'heart', 'lung', 'brain', 'liver', 'kidney', 'stomach', 'blood', 'bone',
                'muscle', 'skin', 'eye', 'ear', 'nose', 'throat', 'spine', 'joint',
                'artery', 'vein', 'nerve', 'gland', 'organ', 'tissue', 'cell'
            ],
            'medical_professionals': [
                'doctor', 'physician', 'surgeon', 'nurse', 'specialist', 'cardiologist',
                'neurologist', 'oncologist', 'psychiatrist', 'radiologist', 'pathologist',
                'anesthesiologist', 'dermatologist', 'ophthalmologist', 'otolaryngologist'
            ]
        }
        
        # Severity classification rules (enhanced)
        self.severity_rules = {
            'critical': [
                'stroke', 'heart attack', 'cardiac arrest', 'sepsis', 'coma', 'hemorrhage', 
                'emergency', 'life threatening', 'fatal', 'death', 'mortality', 'severe trauma',
                'organ failure', 'respiratory failure', 'kidney failure', 'liver failure',
                'brain death', 'cardiac death', 'sudden death', 'anaphylaxis', 'shock'
            ],
            'high': [
                'severe', 'acute', 'unstable', 'complicated', 'progression', 'advanced',
                'metastatic', 'malignant', 'invasive', 'aggressive', 'rapid', 'urgent',
                'hospitalization', 'intensive care', 'icu', 'emergency room', 'surgery'
            ],
            'moderate': [
                'chronic', 'stable', 'controlled', 'manageable', 'mild to moderate',
                'outpatient', 'clinic', 'regular follow-up', 'monitoring', 'observation'
            ],
            'low': [
                'mild', 'minor', 'routine', 'screening', 'prevention', 'check-up',
                'wellness', 'health maintenance', 'preventive care', 'education',
                'counseling', 'lifestyle', 'diet', 'exercise'
            ]
        }
        
        # Offensive content patterns
        self.offensive_patterns = [
            r'\b(spam|advertisement|ads|buy now|click here|promotional|marketing)\b',
            r'\b(xxx|porn|sex|adult|erotic|sexual)\b',
            r'\b(hate|racist|stupid|idiot|moron|dumb|retard)\b',
            r'\b(scam|fraud|fake|illegal|banned|prohibited)\b'
        ]

    def _init_translation(self):
        """Initialize translation services - DISABLED for English-only mode"""
        
        # English-only mode: No translation needed
        logger.info("🇬🇧 English-only mode: Translation services disabled")
        self.pythainlp_translator = None
        self.translator = None
        logger.info("✅ Ready for English medical data processing")

    def load_all_raw_data(self, max_records=None):
        """Load all available raw medical datasets"""
        print("📁 Loading all raw medical datasets...")
        if max_records:
            print(f"⚠️  Limiting to {max_records:,} records")
        
        # Try multiple possible data paths
        possible_paths = [
            Path("data/raw"),
            Path("./data/raw"),  
            Path("../data/raw"),
            Path("../AI-PROJECT/data/raw"),
            Path("./AI-PROJECT/data/raw")
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                print(f"✅ Found data directory: {path.absolute()}")
                break
        
        if not data_path:
            print(f"❌ Raw data directory not found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path.absolute()}")
            return []
        
        all_data = []
        
        # Look for combined datasets first
        combined_files = list(data_path.glob("medical_datasets_combined_*.json"))
        print(f"🔍 Found {len(combined_files)} combined dataset files")
        
        if combined_files:
            latest_combined = max(combined_files, key=lambda x: x.stat().st_mtime)
            print(f"📄 Loading combined dataset: {latest_combined.name}")
            
            try:
                print(f"   📦 Loading large dataset...")
                with open(latest_combined, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    print(f"   🔍 Data type: {type(data)}")
                    
                    # Handle different formats
                    if isinstance(data, list):
                        all_data.extend(data)
                        print(f"   ✅ Loaded {len(data):,} records from combined dataset")
                    elif isinstance(data, dict):
                        if 'data' in data and isinstance(data['data'], list):
                            all_data.extend(data['data'])
                            print(f"   ✅ Loaded {len(data['data']):,} records from combined dataset")
                        elif 'entries' in data and isinstance(data['entries'], list):
                            all_data.extend(data['entries'])
                            print(f"   ✅ Loaded {len(data['entries']):,} records from combined dataset")
                        else:
                            print(f"   ⚠️ Unknown combined dataset format")
                            
            except Exception as e:
                print(f"   ❌ Error loading {latest_combined.name}: {e}")
                import traceback
                traceback.print_exc()        # Look for individual dataset files
        individual_files = [
            "mega_medical_data_*.json",
            "medical_*.json",
            "*.json"
        ]
        
        for pattern in individual_files:
            files = list(data_path.glob(pattern))
            for file_path in files:
                if file_path.name.startswith('medical_datasets_combined'):
                    continue  # Skip combined files we already processed
                
                try:
                    print(f"📄 Loading individual dataset: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Handle different data formats
                        loaded_count = 0
                        if isinstance(data, list):
                            # Direct list of items
                            all_data.extend(data)
                            loaded_count = len(data)
                        elif isinstance(data, dict):
                            # Check for nested data structures
                            if 'data' in data and isinstance(data['data'], list):
                                all_data.extend(data['data'])
                                loaded_count = len(data['data'])
                            elif 'entries' in data and isinstance(data['entries'], list):
                                all_data.extend(data['entries'])
                                loaded_count = len(data['entries'])
                            elif 'questions' in data and isinstance(data['questions'], list):
                                all_data.extend(data['questions'])
                                loaded_count = len(data['questions'])
                            else:
                                # Single record or unknown structure - try as single item
                                all_data.append(data)
                                loaded_count = 1
                        
                        print(f"   ✅ Loaded {loaded_count:,} records")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not load {file_path.name}: {e}")
        
        # Remove duplicates based on question
        if all_data:
            print(f"🔄 Removing basic duplicates...")
            seen_questions = set()
            unique_data = []
            
            for item in all_data:
                question = str(item.get('question', '')).strip().lower()
                if question and question not in seen_questions:
                    seen_questions.add(question)
                    unique_data.append(item)
            
            removed_count = len(all_data) - len(unique_data)
            print(f"   🗑️ Removed {removed_count:,} basic duplicates")
            all_data = unique_data
        
        # Apply max_records limit if specified
        if max_records and len(all_data) > max_records:
            print(f"✂️ Limiting dataset from {len(all_data):,} to {max_records:,} records")
            all_data = all_data[:max_records]
        
        self.stats['original_count'] = len(all_data)
        print(f"📊 Total raw data loaded: {len(all_data):,} records")
        
        return all_data

    def clean_text(self, text):
        """Fast comprehensive text cleaning using pre-compiled patterns"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Use pre-compiled patterns for speed
        text = self.html_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        
        # Clean up medical formatting (MCQ options)
        text = self.mcq_pattern.sub(lambda m: f' ({m.group(1).upper()}) ', text)
        
        # Remove excessive punctuation and normalize whitespace
        text = self.excessive_punct_pattern.sub(r'\1', text)
        text = self.multiple_space_pattern.sub(' ', text)
        
        return text.strip()

    def fast_hash(self, text):
        """Create fast hash for duplicate detection"""
        if not text:
            return None
        # Normalize text for hashing (remove extra spaces, lowercase)
        normalized = re.sub(r'\s+', ' ', str(text).strip().lower())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def expand_medical_abbreviations(self, text):
        """Expand medical abbreviations"""
        if not text:
            return text
        
        expanded_text = text
        expansions_made = 0
        
        for abbrev, expansion in self.medical_abbreviations.items():
            # Case-sensitive replacement for abbreviations
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, expanded_text):
                expanded_text = re.sub(pattern, f"{abbrev} ({expansion})", expanded_text)
                expansions_made += 1
        
        if expansions_made > 0:
            self.stats['abbreviations_expanded'] += 1
        
        return expanded_text

    def normalize_statistical_notation(self, text):
        """Normalize statistical notation"""
        if not text:
            return text
        
        original_text = text
        
        # Normalize p-values
        text = re.sub(r'p\s*<\s*0\.05', 'p-value less than 0.05', text)
        text = re.sub(r'p\s*<\s*0\.01', 'p-value less than 0.01', text)
        text = re.sub(r'p\s*>\s*0\.05', 'p-value greater than 0.05', text)
        text = re.sub(r'p\s*=\s*(\d+\.?\d*)', r'p-value equals \1', text)
        
        # Normalize confidence intervals
        text = re.sub(r'95%\s*CI', '95% confidence interval', text)
        text = re.sub(r'CI\s*95%', '95% confidence interval', text)
        
        # Normalize ranges
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 to \2', text)
        
        # Normalize percentages
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        
        if text != original_text:
            self.stats['statistical_normalized'] += 1
        
        return text

    def normalize_answers(self, text):
        """Normalize answer formats"""
        if not text:
            return text
        
        original_text = text
        text_lower = text.lower().strip()
        
        # Standardize yes/no answers
        if text_lower in ['yes', 'y', 'true', 'correct', 'right', 'positive']:
            text = "Yes"
        elif text_lower in ['no', 'n', 'false', 'incorrect', 'wrong', 'negative']:
            text = "No"
        elif text_lower in ['maybe', 'possibly', 'uncertain', 'unclear', 'depends']:
            text = "Uncertain"
        elif text_lower in ['unknown', 'not known', 'not sure', 'unclear']:
            text = "Unknown"
        
        # Normalize common medical answers
        if 'not recommended' in text_lower:
            text = "Not recommended"
        elif 'recommended' in text_lower:
            text = "Recommended"
        
        if text != original_text:
            self.stats['answers_normalized'] += 1
        
        return text

    def detect_language(self, text):
        """Detect language of text"""
        if not text:
            return 'unknown'
        
        # Count Thai characters
        thai_chars = len([c for c in text if ord(c) >= 0x0E00 and ord(c) <= 0x0E7F])
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'unknown'
        
        thai_ratio = thai_chars / total_chars
        
        if thai_ratio > 0.1:
            return 'thai'
        else:
            return 'english'

    def translate_text_safe(self, text, target_lang='th', max_retries=3):
        """Translation disabled - English-only mode"""
        # No translation in English-only mode
        return text

    def translate_with_pythainlp(self, text, target_lang='th'):
        """Translation disabled - English-only mode"""
        # No translation in English-only mode
        return text
    
    def _basic_medical_translation(self, english_text):
        """Translation disabled - English-only mode"""
        # No translation in English-only mode
        return english_text

    def semantic_similarity(self, text1, text2, threshold=0.8):
        """Calculate semantic similarity between two texts"""
        if not SIMILARITY_AVAILABLE:
            return 0.0
        
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def remove_duplicates(self, data, similarity_threshold=0.85):
        """Remove duplicate questions using fast hash-based detection"""
        print("🔄 Removing duplicates with fast hash-based detection...")
        
        if not data:
            return data
        
        unique_data = []
        seen_hashes = set()
        duplicates_removed = 0
        
        for item in tqdm(data, desc="Fast duplicate checking"):
            question = str(item.get('question', '')).strip()
            if not question:
                continue
            
            # Fast hash-based duplicate detection
            question_hash = self.fast_hash(question)
            if not question_hash:
                continue
                
            if question_hash in seen_hashes:
                duplicates_removed += 1
                continue
            
            seen_hashes.add(question_hash)
            unique_data.append(item)
        
        self.stats['duplicates_removed'] = duplicates_removed
        print(f"   🗑️ Removed {duplicates_removed:,} hash-based duplicates")
        
        return unique_data

    def check_medical_relevance(self, text):
        """Check if text is medically relevant"""
        if not text:
            return False
        
        text_lower = text.lower()
        medical_score = 0
        
        # Check for medical terms
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in text_lower:
                    medical_score += 1
        
        # Check for medical patterns
        medical_patterns = [
            r'\b(symptom|diagnosis|treatment|medication|doctor|hospital|patient|disease|illness)\b',
            r'\b(mg|ml|dose|tablet|injection|prescription|therapy)\b',
            r'\b(blood pressure|heart rate|temperature|pulse|vital signs)\b',
            r'\b(medical|clinical|healthcare|health|medicine)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text_lower):
                medical_score += 1
        
        return medical_score > 0

    def is_offensive_content(self, text):
        """Check if content contains offensive material"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        for pattern in self.offensive_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def is_complete_qa_pair(self, question, answer):
        """Check if Q&A pair is complete"""
        if not question or not answer:
            return False
        
        question = str(question).strip()
        answer = str(answer).strip()
        
        # Check minimum length
        if len(question) < 10 or len(answer) < 3:
            return False
        
        # Check if it's actually a question
        if not any(marker in question.lower() for marker in ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'who']):
            return False
        
        return True

    def determine_severity_level(self, text):
        """Determine severity level based on content"""
        if not text:
            return 'moderate'
        
        text_lower = text.lower()
        
        # Check for severity keywords
        for severity, keywords in self.severity_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return severity
        
        return 'moderate'

    def categorize_medical_specialty(self, text):
        """Categorize medical specialty"""
        if not text:
            return 'general_medicine'
        
        text_lower = text.lower()
        
        specialty_keywords = {
            'cardiology': ['heart', 'cardiac', 'cardio', 'chest pain', 'blood pressure', 'cholesterol'],
            'neurology': ['brain', 'neuro', 'stroke', 'headache', 'seizure', 'memory'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'cough', 'asthma', 'pneumonia'],
            'endocrinology': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose'],
            'gastroenterology': ['stomach', 'intestine', 'liver', 'digestive', 'nausea', 'diarrhea'],
            'orthopedics': ['bone', 'joint', 'muscle', 'fracture', 'arthritis', 'spine'],
            'dermatology': ['skin', 'rash', 'eczema', 'acne', 'dermal'],
            'psychiatry': ['mental', 'depression', 'anxiety', 'psychiatric', 'mood'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
            'pediatrics': ['child', 'pediatric', 'infant', 'baby', 'adolescent']
        }
        
        for specialty, keywords in specialty_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return specialty
        
        return 'general_medicine'

    def fill_missing_data(self, item):
        """Fill missing data fields"""
        filled_fields = 0
        
        # Ensure essential fields exist
        if not item.get('question'):
            return item, 0
        
        if not item.get('answer'):
            return item, 0
        
        # Fill specialty if missing
        if not item.get('specialty'):
            combined_text = f"{item.get('question', '')} {item.get('answer', '')}"
            item['specialty'] = self.categorize_medical_specialty(combined_text)
            filled_fields += 1
        
        # Fill severity if missing
        if not item.get('severity_level'):
            combined_text = f"{item.get('question', '')} {item.get('answer', '')}"
            item['severity_level'] = self.determine_severity_level(combined_text)
            filled_fields += 1
        
        # Fill language if missing
        if not item.get('language'):
            question_lang = self.detect_language(item.get('question', ''))
            answer_lang = self.detect_language(item.get('answer', ''))
            item['language'] = question_lang if question_lang != 'unknown' else answer_lang
            filled_fields += 1
        
        # Fill type if missing
        if not item.get('type'):
            question = str(item.get('question', '')).lower()
            if any(marker in question for marker in ['(a)', '(b)', '(c)', '(d)', 'choose', 'select']):
                item['type'] = 'medical_mcq'
            else:
                item['type'] = 'medical_qa'
            filled_fields += 1
        
        # Add metadata
        if not item.get('source'):
            item['source'] = 'unified_cleaner'
            filled_fields += 1
        
        # Add processing timestamp
        item['processed_at'] = datetime.now().isoformat()
        item['cleaner_version'] = 'complete_unified_v1.0'
        
        return item, filled_fields

    def create_thai_version(self, item):
        """Create Thai translation - DISABLED in English-only mode"""
        # No translation in English-only mode
        return None
    
    def _create_thai_version_pythainlp(self, item):
        """Translation disabled - English-only mode"""
        return None
    
    def _create_thai_version_googletrans(self, item):
        """Translation disabled - English-only mode"""
        return None
        thai_item['translated_at'] = datetime.now().isoformat()
        
        self.stats['translated_count'] += 1
        
        return thai_item

    def process_batch(self, batch_data, batch_num, previous_batches=None):
        """Process a single batch with comprehensive cleaning and cross-batch duplicate checking"""
        print(f"🔧 Processing batch {batch_num} ({len(batch_data):,} records)...")
        
        # Use provided previous batches or load them if needed
        if previous_batches is None:
            if batch_num > 1:
                print(f"   📚 Loading batches 1-{batch_num-1} for cross-batch duplicate checking...")
                previous_batches = self.load_previous_batches(batch_num - 1)
                print(f"   📊 Loaded {len(previous_batches):,} records from {batch_num-1} previous batches")
            else:
                previous_batches = []
        else:
            print(f"   📊 Using provided {len(previous_batches):,} previous records for cross-batch duplicate checking")
        
        clean_batch = []
        batch_stats = {
            'processed': 0,
            'incomplete': 0,
            'irrelevant': 0,
            'offensive': 0,
            'cross_duplicates': 0,
            'cleaned': 0
        }
        
        for item in tqdm(batch_data, desc=f"Batch {batch_num}", leave=False):
            batch_stats['processed'] += 1
            
            # Extract Q&A with fallback field names
            question = item.get('question', '') or item.get('input', '') or item.get('prompt', '')
            answer = item.get('answer', '') or item.get('output', '') or item.get('response', '')
            
            # Check completeness
            if not self.is_complete_qa_pair(question, answer):
                batch_stats['incomplete'] += 1
                continue
            
            # Check medical relevance
            combined_text = f"{question} {answer}"
            if not self.check_medical_relevance(combined_text):
                batch_stats['irrelevant'] += 1
                continue
            
            # Check for offensive content
            if self.is_offensive_content(combined_text):
                batch_stats['offensive'] += 1
                continue
            
            # Comprehensive text cleaning
            clean_question = self.clean_text(question)
            clean_answer = self.clean_text(answer)
            
            # Medical abbreviation expansion
            clean_question = self.expand_medical_abbreviations(clean_question)
            clean_answer = self.expand_medical_abbreviations(clean_answer)
            
            # Statistical notation normalization
            clean_question = self.normalize_statistical_notation(clean_question)
            clean_answer = self.normalize_statistical_notation(clean_answer)
            
            # Answer normalization
            clean_answer = self.normalize_answers(clean_answer)
            
            # Final check after cleaning
            if not clean_question or not clean_answer:
                batch_stats['incomplete'] += 1
                continue
            
            # Create comprehensive clean record
            clean_item = {
                'question': clean_question,
                'answer': clean_answer,
                'language': 'english',
                'source': item.get('source', 'unified_cleaner'),
                'type': 'medical_qa',
                'processed_at': datetime.now().isoformat(),
                'batch_number': batch_num
            }
            
            # Add optional fields with cleaning
            if item.get('explanation'):
                clean_item['explanation'] = self.clean_text(item['explanation'])
                clean_item['explanation'] = self.expand_medical_abbreviations(clean_item['explanation'])
            
            if item.get('context'):
                clean_item['context'] = self.clean_text(item['context'])
            
            # Fill missing data
            clean_item, filled_count = self.fill_missing_data(clean_item)
            
            clean_batch.append(clean_item)
            batch_stats['cleaned'] += 1
        
        # Check for cross-batch duplicates
        if previous_batches:
            print(f"   🔍 Checking cross-batch duplicates against {len(previous_batches):,} previous records...")
            clean_batch, cross_duplicates = self.check_cross_batch_duplicates(clean_batch, previous_batches)
            batch_stats['cross_duplicates'] = cross_duplicates
            batch_stats['cleaned'] = len(clean_batch)  # Update after duplicate removal
        
        # Update global stats
        self.stats['incomplete_removed'] += batch_stats['incomplete']
        self.stats['irrelevant_removed'] += batch_stats['irrelevant']
        self.stats['offensive_removed'] += batch_stats['offensive']
        self.stats['duplicates_removed'] += batch_stats['cross_duplicates']
        self.stats['formatting_cleaned'] += batch_stats['cleaned']
        self.stats['normalized_count'] += batch_stats['cleaned']
        
        print(f"   ✅ Batch {batch_num}: {batch_stats['cleaned']:,}/{batch_stats['processed']:,} clean records")
        if batch_stats['cross_duplicates'] > 0:
            print(f"   🔗 Cross-batch duplicates: {batch_stats['cross_duplicates']:,}")
        print(f"   🗑️ Removed: {batch_stats['incomplete']} incomplete, {batch_stats['irrelevant']} irrelevant, {batch_stats['offensive']} offensive")
        
        return clean_batch

    def process_batch_fast(self, batch_data, batch_num, previous_batches=None):
        """Process batch with fast threading for maximum speed"""
        print(f"🚀 Processing batch {batch_num} with fast threading ({len(batch_data):,} records)...")
        
        # Use provided previous batches or use cached hashes
        if previous_batches is None:
            if batch_num > 1 and not self.previous_questions_cache:
                print(f"   📚 Loading previous batches for hash cache...")
                previous_batches = self.load_previous_batches(batch_num - 1)
                # Build cache
                for item in previous_batches:
                    question = str(item.get('question', '')).strip()
                    if question:
                        question_hash = self.fast_hash(question)
                        if question_hash:
                            self.previous_questions_cache.add(question_hash)
        
        batch_stats = {
            'processed': len(batch_data),
            'incomplete': 0,
            'irrelevant': 0,
            'offensive': 0,
            'cross_duplicates': 0,
            'cleaned': 0
        }
        
        # Process items with threading (faster than multiprocessing for I/O bound tasks)
        clean_batch = []
        
        # Use threading for better performance without pickling issues
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_single_item, item, batch_num): item 
                for item in batch_data
            }
            
            # Collect results with progress bar
            for future in tqdm(future_to_item, desc=f"Fast batch {batch_num}", leave=False):
                result, status = future.result()
                
                if status == 'cleaned' and result:
                    clean_batch.append(result)
                    batch_stats['cleaned'] += 1
                elif status == 'incomplete':
                    batch_stats['incomplete'] += 1
                elif status == 'irrelevant':
                    batch_stats['irrelevant'] += 1
                elif status == 'offensive':
                    batch_stats['offensive'] += 1
        
        # Fast cross-batch duplicate checking using hash cache
        if self.previous_questions_cache:
            print(f"   🔍 Fast cross-batch duplicate check against {len(self.previous_questions_cache):,} cached hashes...")
            
            unique_batch = []
            for item in clean_batch:
                question = str(item.get('question', '')).strip()
                if question:
                    question_hash = self.fast_hash(question)
                    if question_hash and question_hash not in self.previous_questions_cache:
                        unique_batch.append(item)
                        self.previous_questions_cache.add(question_hash)
                    else:
                        batch_stats['cross_duplicates'] += 1
            
            clean_batch = unique_batch
            batch_stats['cleaned'] = len(clean_batch)
        
        # Update global stats
        self.stats['incomplete_removed'] += batch_stats['incomplete']
        self.stats['irrelevant_removed'] += batch_stats['irrelevant']
        self.stats['offensive_removed'] += batch_stats['offensive']
        self.stats['duplicates_removed'] += batch_stats['cross_duplicates']
        self.stats['formatting_cleaned'] += batch_stats['cleaned']
        self.stats['normalized_count'] += batch_stats['cleaned']
        
        print(f"   ⚡ Fast batch {batch_num}: {batch_stats['cleaned']:,}/{batch_stats['processed']:,} clean records")
        if batch_stats['cross_duplicates'] > 0:
            print(f"   🔗 Cross-batch duplicates: {batch_stats['cross_duplicates']:,}")
        print(f"   🗑️ Removed: {batch_stats['incomplete']} incomplete, {batch_stats['irrelevant']} irrelevant, {batch_stats['offensive']} offensive")
        
        return clean_batch

    def process_single_item(self, item, batch_num):
        """Process a single item for parallel processing"""
        # Extract Q&A with fallback field names
        question = item.get('question', '') or item.get('input', '') or item.get('prompt', '')
        answer = item.get('answer', '') or item.get('output', '') or item.get('response', '')
        
        # Check completeness
        if not self.is_complete_qa_pair(question, answer):
            return None, 'incomplete'
        
        # Check medical relevance
        combined_text = f"{question} {answer}"
        if not self.check_medical_relevance(combined_text):
            return None, 'irrelevant'
        
        # Check for offensive content
        if self.is_offensive_content(combined_text):
            return None, 'offensive'
        
        # Comprehensive text cleaning
        clean_question = self.clean_text(question)
        clean_answer = self.clean_text(answer)
        
        # Medical abbreviation expansion
        clean_question = self.expand_medical_abbreviations(clean_question)
        clean_answer = self.expand_medical_abbreviations(clean_answer)
        
        # Statistical notation normalization
        clean_question = self.normalize_statistical_notation(clean_question)
        clean_answer = self.normalize_statistical_notation(clean_answer)
        
        # Answer normalization
        clean_answer = self.normalize_answers(clean_answer)
        
        # Final check after cleaning
        if not clean_question or not clean_answer:
            return None, 'incomplete'
        
        # Create comprehensive clean record
        clean_item = {
            'question': clean_question,
            'answer': clean_answer,
            'language': 'english',
            'source': item.get('source', 'unified_cleaner'),
            'type': 'medical_qa',
            'processed_at': datetime.now().isoformat(),
            'batch_number': batch_num
        }
        
        # Add optional fields with cleaning
        if item.get('explanation'):
            clean_item['explanation'] = self.clean_text(item['explanation'])
            clean_item['explanation'] = self.expand_medical_abbreviations(clean_item['explanation'])
        
        if item.get('context'):
            clean_item['context'] = self.clean_text(item['context'])
        
        # Fill missing data
        clean_item, filled_count = self.fill_missing_data(clean_item)
        
        return clean_item, 'cleaned'

    def save_batch_temp(self, batch_data, batch_num):
        """Save batch to temporary file"""
        temp_dir = Path("../data/temp_batches")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        batch_file = temp_dir / f"cleaned_batch_{batch_num:03d}.json"
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        return batch_file

    def load_previous_batches(self, current_batch_num):
        """Load all previous batches for cross-batch duplicate checking"""
        temp_dir = Path("../data/temp_batches")
        if not temp_dir.exists():
            return []
        
        previous_data = []
        
        # Load batches 1 to (current_batch_num - 1)
        for i in range(1, current_batch_num):
            batch_file = temp_dir / f"cleaned_batch_{i:03d}.json"
            if batch_file.exists():
                try:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        previous_data.extend(batch_data)
                except Exception as e:
                    print(f"⚠️ Error loading previous batch {i}: {e}")
        
        return previous_data

    def check_cross_batch_duplicates(self, current_batch, previous_batches):
        """Fast cross-batch duplicate checking using hash cache"""
        if not previous_batches:
            return current_batch, 0
        
        # Build hash cache from previous batches if not cached
        if not self.previous_questions_cache:
            print("   📦 Building hash cache from previous batches...")
            for item in previous_batches:
                question = str(item.get('question', '')).strip()
                if question:
                    question_hash = self.fast_hash(question)
                    if question_hash:
                        self.previous_questions_cache.add(question_hash)
        
        # Fast filter current batch using hash lookup
        unique_batch = []
        cross_duplicates = 0
        
        for item in current_batch:
            question = str(item.get('question', '')).strip()
            if question:
                question_hash = self.fast_hash(question)
                if question_hash and question_hash not in self.previous_questions_cache:
                    unique_batch.append(item)
                    self.previous_questions_cache.add(question_hash)  # Add to cache
                else:
                    cross_duplicates += 1
            else:
                cross_duplicates += 1
        
        return unique_batch, cross_duplicates

    def combine_all_batches(self):
        """Combine all batch results into final dataset"""
        print("\n🔗 Combining all batch results...")
        
        temp_dir = Path("../data/temp_batches")
        if not temp_dir.exists():
            print("❌ No batch results found!")
            return []
        
        all_clean_data = []
        batch_files = sorted(temp_dir.glob("cleaned_batch_*.json"))
        
        for batch_file in tqdm(batch_files, desc="Combining batches"):
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_clean_data.extend(batch_data)
            except Exception as e:
                print(f"⚠️ Error loading {batch_file}: {e}")
        
        print(f"✅ Combined {len(all_clean_data):,} clean records from {len(batch_files)} batches")
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary batch files")
        
        return all_clean_data

    def comprehensive_clean_all_data(self, data):
        """ULTRA-DETAILED comprehensive cleaning with maximum thoroughness"""
        print("\n🧹 Starting ULTRA-DETAILED comprehensive cleaning process...")
        print("🎯 Mode: MAXIMUM THOROUGHNESS - No shortcuts, highest quality")
        print("🔧 Enhanced Processing: Deep analysis, detailed validation, perfect cleaning")
        
        if not data:
            print("❌ No data to clean")
            return [], []
        
        # Step 1: Deep pre-analysis of raw dataset
        print("� Step 1: Deep pre-analysis of raw dataset...")
        self._analyze_raw_dataset(data)
        
        # Step 2: Advanced duplicate removal with semantic analysis
        print("🔍 Step 2: Advanced duplicate removal with semantic analysis...")
        data = self.advanced_remove_duplicates(data)
        
        # Step 3: Enhanced medical relevance filtering
        print("🏥 Step 3: Enhanced medical relevance filtering...")
        data = self.medical_relevance_filter(data)
        
        # Step 4: Quality assessment and scoring
        print("⭐ Step 4: Quality assessment and scoring...")
        data = self.quality_assessment_scoring(data)
        
        # Step 5: DETAILED batch processing (no fast mode - maximum quality)
        print(f"📦 Step 5: DETAILED batch processing ({self.batch_size:,} records per batch)...")
        print("🔧 Using THOROUGH processing mode - No fast shortcuts")
        
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        batch_files = []
        
        for i in range(0, len(data), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_data = data[i:i + self.batch_size]
            
            # Load previous batches for cross-duplicate checking (if any)
            previous_batches = []
            if batch_num > 1:
                previous_batches = self.load_previous_batches(batch_num - 1)
            
            # DETAILED thorough processing with comprehensive cleaning (NOT fast mode)
            clean_batch = self.thorough_process_batch(batch_data, batch_num, previous_batches)
            
            # Save batch temporarily
            if clean_batch:
                batch_file = self.save_batch_temp(clean_batch, batch_num)
                batch_files.append(batch_file)
            
            self.stats['batches_processed'] += 1
        
        print(f"✅ Processed {total_batches} batches with MAXIMUM THOROUGHNESS")
        
        # Step 6: Enhanced batch combination with validation
        english_data = self.combine_all_batches_enhanced()
        
        print(f"🇺🇸 Total ultra-detailed clean English records: {len(english_data):,}")
        
        # Step 7: Final comprehensive quality validation and enhancement
        print("🔍 Step 7: Final comprehensive quality validation...")
        english_data = self.final_quality_validation(english_data)
        
        # Step 8: English-only finalization (NO Thai translation)
        print("✅ Step 8: English-only dataset finalization...")
        print("🎯 Focus: MAXIMUM English quality - No Thai translation")
        print("🔬 Result: Ultra-detailed, highest-quality English medical dataset")
        thai_data = []  # Completely empty - English-only focus
        

        
        # Update final statistics (English-only focus)
        self.stats['final_english_count'] = len(english_data)
        self.stats['final_thai_count'] = 0  # NO Thai translation
        self.stats['final_unified_count'] = len(english_data)
        
        print(f"🇺🇸 Ultra-clean English records: {len(english_data):,}")
        print(f"📊 Total ultra-processed records: {self.stats['final_unified_count']:,}")
        print(f"✅ English-only dataset completed - Maximum quality achieved")
        print(f"💡 Ready for: Medical Q&A training, LangChain, N8N integration")
        
        return english_data, thai_data

    def old_thai_translation_logic(self):
        """Removed Thai translation logic - now handled separately"""
        pass

    def comprehensive_clean_all_data_old_end(self):
        # This method contains the old end of comprehensive_clean_all_data
        if False:  # Disabled for now
            # Batch translation for better performance and rate limit management
            batch_size = 50  # แปลทีละ 50 รายการ
            total_batches = (translation_limit + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, translation_limit)
                batch_items = english_data[start_idx:end_idx]
                
                print(f"   � PyThaiNLP แปลกลุ่มที่ {batch_num + 1}/{total_batches} ({len(batch_items)} รายการ)")
                
                for i, item in enumerate(tqdm(batch_items, desc=f"Batch {batch_num + 1}", leave=False)):
                    pass  # Placeholder, this code is disabled

    def save_unified_datasets(self, english_data, thai_data):
        """Save ultra-clean English-only datasets (focus on maximum quality)"""
        print("\n💾 Saving ultra-clean English datasets...")
        print("🎯 English-only focus: Maximum quality medical data")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = Path("../data/processed")
        output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save English data (ultra-clean version)
        if english_data:
            print(f"💾 Saving ultra-clean English dataset ({len(english_data):,} records)...")
            
            # 1. Ultra-clean English JSON (primary format)
            english_json = output_dir / f"english_medical_qa_ultraclean_{timestamp}.json"
            with open(english_json, 'w', encoding='utf-8') as f:
                json.dump(english_data, f, ensure_ascii=False, indent=2)
            saved_files.append(str(english_json))
            print(f"   📄 Ultra-clean English JSON: {english_json}")
            
            # 2. Ultra-clean English CSV (for analysis/training)
            english_csv = output_dir / f"english_medical_qa_ultraclean_{timestamp}.csv"
            df_english = pd.DataFrame(english_data)
            df_english.to_csv(english_csv, index=False, encoding='utf-8')
            saved_files.append(str(english_csv))
            print(f"   📊 Ultra-clean English CSV: {english_csv}")
            
            # 3. Processing statistics and quality metrics
            stats_json = output_dir / f"processing_stats_ultraclean_{timestamp}.json"
            with open(stats_json, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            saved_files.append(str(stats_json))
            print(f"   📈 Processing statistics: {stats_json}")
        
        # NO Thai data processing (English-only focus)
        if thai_data and len(thai_data) > 0:
            print("⚠️ Thai data detected but ignored (English-only processing)")
        else:
            print("✅ English-only processing - No Thai translation")
        
        print(f"🎯 English-only focus: Maximum quality achieved")
        print(f"✅ Saved {len(saved_files)} ultra-clean English files")
        print(f"💡 Ready for: Medical Q&A, LangChain, N8N, API development")
        
        return saved_files

    def _analyze_raw_dataset(self, data):
        """Pre-analysis of raw dataset to understand data quality"""
        print("   📊 Analyzing raw dataset characteristics...")
        
        analysis = {
            'total_records': len(data),
            'languages_detected': {},
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'medical_terms_coverage': 0,
            'quality_issues': []
        }
        
        question_lengths = []
        answer_lengths = []
        
        for item in data[:1000]:  # Sample first 1000 for analysis
            question = str(item.get('question', ''))
            answer = str(item.get('answer', ''))
            
            if question:
                question_lengths.append(len(question))
                lang = self.detect_language(question)
                analysis['languages_detected'][lang] = analysis['languages_detected'].get(lang, 0) + 1
            
            if answer:
                answer_lengths.append(len(answer))
        
        analysis['avg_question_length'] = np.mean(question_lengths) if question_lengths else 0
        analysis['avg_answer_length'] = np.mean(answer_lengths) if answer_lengths else 0
        
        print(f"      📈 Records: {analysis['total_records']:,}")
        print(f"      📝 Avg question length: {analysis['avg_question_length']:.0f} chars")
        print(f"      💬 Avg answer length: {analysis['avg_answer_length']:.0f} chars")
        print(f"      🌍 Languages: {dict(analysis['languages_detected'])}")
    
    def advanced_remove_duplicates(self, data):
        """Advanced duplicate removal with semantic similarity checking"""
        print("   🔍 Advanced duplicate detection with semantic analysis...")
        
        if not data:
            return data
        
        unique_data = []
        question_embeddings = {}  # Simple similarity cache
        duplicates_found = 0
        
        for item in tqdm(data, desc="Advanced duplicate checking"):
            question = str(item.get('question', '')).strip().lower()
            if not question:
                continue
            
            # Normalize question for better similarity detection
            normalized_question = re.sub(r'\s+', ' ', question)
            normalized_question = re.sub(r'[^\w\s]', '', normalized_question)
            
            # Check against existing questions with similarity
            is_duplicate = False
            for existing_q in question_embeddings.keys():
                similarity = self.semantic_similarity(normalized_question, existing_q)
                if similarity > 0.9:  # High similarity threshold
                    duplicates_found += 1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                question_embeddings[normalized_question] = True
                unique_data.append(item)
        
        removed = len(data) - len(unique_data)
        print(f"      🗑️ Removed {removed:,} advanced duplicates")
        self.stats['duplicates_removed'] += removed
        
        return unique_data
    
    def medical_relevance_filter(self, data):
        """Enhanced medical relevance filtering"""
        print("   🏥 Enhanced medical relevance filtering...")
        
        relevant_data = []
        irrelevant_count = 0
        
        for item in tqdm(data, desc="Medical relevance check"):
            question = str(item.get('question', ''))
            answer = str(item.get('answer', ''))
            combined_text = f"{question} {answer}".lower()
            
            # Enhanced medical relevance scoring
            medical_score = 0
            
            # Check for medical terms (weighted scoring)
            for category, terms in self.medical_terms.items():
                for term in terms:
                    count = combined_text.count(term.lower())
                    if category == 'conditions':
                        medical_score += count * 3  # Higher weight for conditions
                    elif category == 'symptoms':
                        medical_score += count * 2  # Medium weight for symptoms
                    else:
                        medical_score += count  # Base weight
            
            # Check for medical abbreviations
            for abbrev in self.medical_abbreviations.keys():
                if abbrev.lower() in combined_text:
                    medical_score += 2
            
            # Check for medical patterns
            medical_patterns = [
                r'\b(patient|diagnosis|treatment|therapy|medication|prescription)\b',
                r'\b(hospital|clinic|doctor|physician|nurse|medical)\b',
                r'\b(symptom|disease|illness|condition|syndrome)\b',
                r'\b(mg|ml|dose|tablet|capsule|injection)\b',
                r'\b(blood pressure|heart rate|temperature|vital signs)\b'
            ]
            
            for pattern in medical_patterns:
                matches = len(re.findall(pattern, combined_text))
                medical_score += matches * 1.5
            
            # Minimum medical relevance threshold
            if medical_score >= 2.0:  # Adjusted threshold
                relevant_data.append(item)
            else:
                irrelevant_count += 1
        
        print(f"      ✅ Kept {len(relevant_data):,} medically relevant records")
        print(f"      🗑️ Filtered out {irrelevant_count:,} non-medical records")
        self.stats['irrelevant_removed'] += irrelevant_count
        
        return relevant_data
    
    def quality_assessment_scoring(self, data):
        """Quality assessment and scoring for each record"""
        print("   ⭐ Quality assessment and scoring...")
        
        scored_data = []
        
        for item in tqdm(data, desc="Quality scoring"):
            quality_score = self._calculate_quality_score(item)
            item['quality_score'] = quality_score
            
            # Only keep high and medium quality records
            if quality_score >= 0.6:  # Threshold for acceptable quality
                scored_data.append(item)
        
        removed = len(data) - len(scored_data)
        print(f"      ⭐ Quality scoring completed")
        print(f"      ✅ Kept {len(scored_data):,} high-quality records")
        print(f"      🗑️ Filtered out {removed:,} low-quality records")
        
        return scored_data
    
    def _calculate_quality_score(self, item):
        """Calculate quality score for a single item"""
        score = 0.0
        
        question = str(item.get('question', '')).strip()
        answer = str(item.get('answer', '')).strip()
        
        # Question quality (40% of total score)
        if question:
            # Length check
            if 20 <= len(question) <= 500:
                score += 0.15
            # Question markers
            if any(marker in question.lower() for marker in ['?', 'what', 'how', 'why', 'when', 'where']):
                score += 0.15
            # Medical terminology
            if any(term in question.lower() for term in ['medical', 'health', 'disease', 'symptom', 'treatment']):
                score += 0.10
        
        # Answer quality (40% of total score)
        if answer:
            # Length check
            if 10 <= len(answer) <= 2000:
                score += 0.15
            # Complete sentences
            if answer.count('.') >= 1 or len(answer) > 50:
                score += 0.10
            # Informative content
            if any(word in answer.lower() for word in ['because', 'due to', 'caused by', 'treatment', 'recommended']):
                score += 0.15
        
        # Completeness (20% of total score)
        if question and answer:
            score += 0.10
        if len(question) > 0 and len(answer) > 0:
            score += 0.10
        
        return min(score, 1.0)  # Cap at 1.0
    
    def enhanced_process_batch(self, batch_data, batch_num, previous_batches=None):
        """Enhanced batch processing with comprehensive cleaning"""
        print(f"🔧 Enhanced processing batch {batch_num} ({len(batch_data):,} records)...")
        
        clean_batch = []
        batch_stats = {
            'processed': 0,
            'enhanced_cleaned': 0,
            'terminology_expanded': 0,
            'formats_normalized': 0
        }
        
        for item in tqdm(batch_data, desc=f"Enhanced batch {batch_num}", leave=False):
            batch_stats['processed'] += 1
            
            # Extract and validate Q&A
            question = str(item.get('question', '') or item.get('input', '') or item.get('prompt', '')).strip()
            answer = str(item.get('answer', '') or item.get('output', '') or item.get('response', '')).strip()
            
            if not question or not answer:
                continue
            
            # Enhanced text cleaning
            clean_question = self.enhanced_text_cleaning(question)
            clean_answer = self.enhanced_text_cleaning(answer)
            
            if not clean_question or not clean_answer:
                continue
            
            # Create enhanced clean record
            clean_item = {
                'question': clean_question,
                'answer': clean_answer,
                'language': self.detect_language(clean_question),
                'quality_score': item.get('quality_score', 0.8),
                'source': item.get('source', 'enhanced_cleaner'),
                'type': 'medical_qa',
                'processed_at': datetime.now().isoformat(),
                'batch_number': batch_num,
                'enhanced_cleaning': True
            }
            
            # Enhanced metadata
            clean_item['specialty'] = self.categorize_medical_specialty(f"{clean_question} {clean_answer}")
            clean_item['severity_level'] = self.determine_severity_level(f"{clean_question} {clean_answer}")
            clean_item['medical_terms_count'] = self._count_medical_terms(f"{clean_question} {clean_answer}")
            
            clean_batch.append(clean_item)
            batch_stats['enhanced_cleaned'] += 1
        
        print(f"   ✅ Enhanced batch {batch_num}: {batch_stats['enhanced_cleaned']:,}/{batch_stats['processed']:,} records")
        
        return clean_batch

    def thorough_process_batch(self, batch_data, batch_num, previous_batches=None):
        """THOROUGH batch processing with MAXIMUM detailed cleaning - No shortcuts"""
        print(f"🔧 THOROUGH processing batch {batch_num} ({len(batch_data):,} records)...")
        print(f"   🎯 Mode: MAXIMUM DETAIL - Every record perfectly cleaned")
        
        clean_batch = []
        batch_stats = {
            'processed': 0,
            'thorough_cleaned': 0,
            'terminology_expanded': 0,
            'formats_normalized': 0,
            'deep_validated': 0,
            'cross_checked': 0,
            'quality_enhanced': 0
        }
        
        # Process each item with MAXIMUM thoroughness
        for item in tqdm(batch_data, desc=f"Thorough batch {batch_num}", leave=False):
            batch_stats['processed'] += 1
            
            # Extract Q&A with comprehensive field checking
            question = str(item.get('question', '') or item.get('input', '') or item.get('prompt', '') or item.get('text', '')).strip()
            answer = str(item.get('answer', '') or item.get('output', '') or item.get('response', '') or item.get('target', '')).strip()
            
            # THOROUGH completeness validation
            if not self.thorough_completeness_check(question, answer):
                continue
                
            # DETAILED medical relevance validation
            if not self.detailed_medical_relevance_check(question, answer):
                continue
                
            # COMPREHENSIVE text cleaning (multiple passes)
            clean_question = self.comprehensive_text_cleaning(question)
            clean_answer = self.comprehensive_text_cleaning(answer)
            
            # Validate cleaning results
            if not clean_question or not clean_answer or len(clean_question) < 10 or len(clean_answer) < 3:
                continue
            
            # DETAILED cross-batch duplicate checking
            if previous_batches:
                question_hash = self.fast_hash(clean_question)
                if question_hash in self.previous_questions_cache:
                    continue
                self.previous_questions_cache.add(question_hash)
                batch_stats['cross_checked'] += 1
            
            # Create COMPREHENSIVE clean record with all enhancements
            clean_item = {
                'question': clean_question,
                'answer': clean_answer,
                'language': 'english',
                'source': item.get('source', 'thorough_cleaner'),
                'type': 'medical_qa',
                'quality_score': self._calculate_detailed_quality_score(clean_question, clean_answer),
                'medical_terms_count': self._count_medical_terms(f"{clean_question} {clean_answer}"),
                'medical_specialty': self.categorize_medical_specialty(f"{clean_question} {clean_answer}"),
                'severity_level': self.determine_severity_level(f"{clean_question} {clean_answer}"),
                'completeness_score': self._calculate_completeness_score(clean_question, clean_answer),
                'processed_at': datetime.now().isoformat(),
                'batch_number': batch_num,
                'processing_method': 'thorough_maximum_quality',
                'validation_passed': True
            }
            
            # Add comprehensive optional fields with thorough cleaning
            if item.get('explanation'):
                clean_item['explanation'] = self.comprehensive_text_cleaning(item['explanation'])
                batch_stats['terminology_expanded'] += 1
            
            if item.get('context'):
                clean_item['context'] = self.comprehensive_text_cleaning(item['context'])
            
            # Final thorough validation
            if self._final_thorough_validation(clean_item):
                clean_batch.append(clean_item)
                batch_stats['thorough_cleaned'] += 1
                batch_stats['quality_enhanced'] += 1
                batch_stats['deep_validated'] += 1
        
        print(f"   ✅ Thorough batch {batch_num}: {batch_stats['thorough_cleaned']:,}/{batch_stats['processed']:,} ultra-clean records")
        print(f"   🔍 Deep validation: {batch_stats['deep_validated']:,} passed")
        print(f"   ⭐ Quality enhanced: {batch_stats['quality_enhanced']:,} records")
        print(f"   📚 Terminology expanded: {batch_stats['terminology_expanded']:,} items")
        
        return clean_batch
    
    def enhanced_text_cleaning(self, text):
        """Enhanced text cleaning with medical focus"""
        if not text:
            return ""
        
        # Basic cleaning
        text = self.clean_text(text)
        
        # Medical abbreviation expansion
        text = self.expand_medical_abbreviations(text)
        
        # Statistical notation normalization
        text = self.normalize_statistical_notation(text)
        
        # Medical format standardization
        text = self._standardize_medical_formats(text)
        
        # Remove redundant information
        text = self._remove_redundant_info(text)
        
        return text.strip()

    def comprehensive_text_cleaning(self, text):
        """COMPREHENSIVE multi-pass text cleaning for maximum quality"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Pass 1: Basic cleaning
        text = self.clean_text(text)
        
        # Pass 2: Medical abbreviation expansion (detailed)
        text = self.expand_medical_abbreviations(text)
        
        # Pass 3: Statistical notation normalization
        text = self.normalize_statistical_notation(text)
        
        # Pass 4: Medical format standardization
        text = self._standardize_medical_formats(text)
        
        # Pass 5: Advanced grammar and punctuation correction
        text = self._advanced_grammar_correction(text)
        
        # Pass 6: Medical terminology enhancement
        text = self._enhance_medical_terminology(text)
        
        # Pass 7: Remove redundant information
        text = self._remove_redundant_info(text)
        
        # Pass 8: Final polish and validation
        text = self._final_text_polish(text)
        
        return text.strip()

    def thorough_completeness_check(self, question, answer):
        """Thorough check for Q&A pair completeness"""
        if not question or not answer:
            return False
        
        question = str(question).strip()
        answer = str(answer).strip()
        
        # Minimum length requirements (stricter)
        if len(question) < 15 or len(answer) < 5:
            return False
        
        # Must be actual questions
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'can', 'will', 'should', 'would', 'could']
        if not any(indicator in question.lower() for indicator in question_indicators):
            return False
        
        # Answer should not be too generic
        generic_answers = ['yes', 'no', 'maybe', 'i don\'t know', 'unclear', 'unknown']
        if answer.lower().strip() in generic_answers:
            return False
        
        # Check for meaningful content
        if len(set(question.lower().split())) < 4 or len(set(answer.lower().split())) < 3:
            return False
        
        return True

    def detailed_medical_relevance_check(self, question, answer):
        """Detailed medical relevance checking with scoring"""
        combined_text = f"{question} {answer}".lower()
        medical_score = 0
        
        # Check for medical terms (detailed scoring)
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in combined_text:
                    medical_score += 2 if category == 'conditions' else 1
        
        # Check for medical patterns (enhanced)
        medical_patterns = [
            r'\b(symptom|diagnosis|treatment|medication|doctor|hospital|patient|disease|illness|condition)\b',
            r'\b(mg|ml|dose|dosage|tablet|injection|prescription|therapy|pharmaceutical)\b',
            r'\b(blood pressure|heart rate|temperature|pulse|vital signs|lab results)\b',
            r'\b(medical|clinical|healthcare|health|medicine|therapeutic|diagnostic)\b',
            r'\b(surgery|operation|procedure|examination|test|scan|x-ray|mri|ct)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, combined_text):
                medical_score += 3
        
        # Minimum medical score threshold (stricter)
        return medical_score >= 5

    def _calculate_detailed_quality_score(self, question, answer):
        """Calculate detailed quality score with comprehensive metrics"""
        score = 0.0
        
        # Question quality (30%)
        if question:
            # Length appropriateness
            q_len = len(question)
            if 20 <= q_len <= 500:
                score += 0.10
            elif 15 <= q_len <= 600:
                score += 0.05
            
            # Question complexity
            if len(set(question.lower().split())) >= 5:
                score += 0.10
            
            # Medical terminology
            if self._count_medical_terms(question) >= 2:
                score += 0.10
        
        # Answer quality (40%)
        if answer:
            # Length appropriateness
            a_len = len(answer)
            if 30 <= a_len <= 1000:
                score += 0.15
            elif 20 <= a_len <= 1200:
                score += 0.10
            
            # Answer completeness
            if len(set(answer.lower().split())) >= 8:
                score += 0.15
            
            # Medical content
            if self._count_medical_terms(answer) >= 3:
                score += 0.10
        
        # Coherence and relevance (30%)
        combined = f"{question} {answer}"
        
        # Medical focus
        if self.check_medical_relevance(combined):
            score += 0.15
        
        # Information quality
        informative_words = ['because', 'due to', 'treatment', 'recommended', 'indicates', 'suggests']
        if any(word in answer.lower() for word in informative_words):
            score += 0.15
        
        return min(score, 1.0)

    def _calculate_completeness_score(self, question, answer):
        """Calculate completeness score for Q&A pair"""
        score = 0.0
        
        # Basic completeness
        if question and answer:
            score += 0.3
        
        # Length adequacy
        if len(question) >= 15 and len(answer) >= 20:
            score += 0.3
        
        # Content variety
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        if len(q_words) >= 5 and len(a_words) >= 8:
            score += 0.2
        
        # Medical completeness
        if self._count_medical_terms(f"{question} {answer}") >= 3:
            score += 0.2
        
        return min(score, 1.0)

    def _final_thorough_validation(self, item):
        """Final thorough validation before accepting item"""
        try:
            # Check required fields
            required_fields = ['question', 'answer', 'quality_score']
            for field in required_fields:
                if not item.get(field):
                    return False
            
            # Validate quality thresholds
            if item.get('quality_score', 0) < 0.6:
                return False
            
            # Validate completeness
            if item.get('completeness_score', 0) < 0.5:
                return False
            
            # Validate medical relevance
            if item.get('medical_terms_count', 0) < 2:
                return False
            
            # Final content validation
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if len(question) < 15 or len(answer) < 20:
                return False
            
            return True
            
        except Exception:
            return False

    def _advanced_grammar_correction(self, text):
        """Advanced grammar and punctuation correction"""
        if not text:
            return ""
        
        # Fix common grammar issues
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Capitalize sentences
        sentences = text.split('. ')
        sentences = [s.strip().capitalize() for s in sentences if s.strip()]
        text = '. '.join(sentences)
        
        return text

    def _enhance_medical_terminology(self, text):
        """Enhance medical terminology for clarity"""
        if not text:
            return ""
        
        # Medical enhancement patterns
        enhancements = {
            r'\bhigh bp\b': 'high blood pressure',
            r'\blow bp\b': 'low blood pressure',
            r'\bheart attack\b': 'myocardial infarction',
            r'\bstroke\b': 'cerebrovascular accident',
            r'\bsugar\b': 'blood glucose',
            r'\bsugar level\b': 'blood glucose level',
            r'\bpills\b': 'medication',
            r'\bmeds\b': 'medications'
        }
        
        for pattern, replacement in enhancements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _final_text_polish(self, text):
        """Final text polishing and cleanup"""
        if not text:
            return ""
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Ensure proper sentence ending
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _standardize_medical_formats(self, text):
        """Standardize medical formats and units"""
        if not text:
            return text
        
        # Standardize blood pressure format
        text = re.sub(r'(\d+)/(\d+)\s*mmhg', r'\1/\2 mmHg', text, flags=re.IGNORECASE)
        
        # Standardize temperature
        text = re.sub(r'(\d+\.?\d*)\s*°?c', r'\1°C', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*°?f', r'\1°F', text, flags=re.IGNORECASE)
        
        # Standardize weight and height
        text = re.sub(r'(\d+\.?\d*)\s*kg', r'\1 kg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*lb', r'\1 lb', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*cm', r'\1 cm', text, flags=re.IGNORECASE)
        
        # Standardize dosages
        text = re.sub(r'(\d+\.?\d*)\s*mg', r'\1 mg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*ml', r'\1 ml', text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_redundant_info(self, text):
        """Remove redundant or unnecessary information"""
        if not text:
            return text
        
        # Remove common redundant phrases
        redundant_patterns = [
            r'\b(please note that|it should be noted that|it is important to note)\b',
            r'\b(in conclusion|to summarize|in summary)\b',
            r'\b(as mentioned before|as stated earlier)\b'
        ]
        
        for pattern in redundant_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _count_medical_terms(self, text):
        """Count medical terms in text"""
        if not text:
            return 0
        
        text_lower = text.lower()
        term_count = 0
        
        for category, terms in self.medical_terms.items():
            for term in terms:
                term_count += text_lower.count(term.lower())
        
        return term_count
    
    def combine_all_batches_enhanced(self):
        """Enhanced batch combination with final validation"""
        print("\n🔗 Enhanced batch combination with final validation...")
        
        temp_dir = Path("../data/temp_batches")
        if not temp_dir.exists():
            print("❌ No temp batch directory found")
            return []
        
        all_clean_data = []
        batch_files = sorted(temp_dir.glob("cleaned_batch_*.json"))
        
        for batch_file in tqdm(batch_files, desc="Combining enhanced batches"):
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_clean_data.extend(batch_data)
            except Exception as e:
                print(f"❌ Error loading {batch_file}: {e}")
        
        # Final validation and sorting
        print("   🔍 Final quality validation...")
        validated_data = []
        
        for item in all_clean_data:
            if self._final_validation_check(item):
                validated_data.append(item)
        
        # Sort by quality score (highest first)
        validated_data.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        print(f"✅ Enhanced combination: {len(validated_data):,} validated records")
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files")
        
        return validated_data
    
    def _final_validation_check(self, item):
        """Final validation check for each item"""
        try:
            # Essential fields check
            if not item.get('question') or not item.get('answer'):
                return False
            
            # Length validation
            question_len = len(str(item.get('question', '')))
            answer_len = len(str(item.get('answer', '')))
            
            if question_len < 10 or question_len > 1000:
                return False
            
            if answer_len < 5 or answer_len > 5000:
                return False
            
            # Quality score check
            if item.get('quality_score', 0) < 0.6:
                return False
            
            return True
        
        except Exception:
            return False
    
    def final_quality_validation(self, data):
        """Final comprehensive quality validation"""
        print("   🔍 Final comprehensive quality validation...")
        
        validated_data = []
        validation_stats = {
            'passed': 0,
            'failed_length': 0,
            'failed_quality': 0,
            'failed_medical': 0
        }
        
        for item in tqdm(data, desc="Final validation"):
            # Comprehensive validation
            if self._comprehensive_validation(item):
                validated_data.append(item)
                validation_stats['passed'] += 1
            else:
                # Count failure reasons
                if not self._length_validation(item):
                    validation_stats['failed_length'] += 1
                elif item.get('quality_score', 0) < 0.7:
                    validation_stats['failed_quality'] += 1
                else:
                    validation_stats['failed_medical'] += 1
        
        print(f"      ✅ Validation passed: {validation_stats['passed']:,}")
        print(f"      ❌ Failed - Length: {validation_stats['failed_length']:,}")
        print(f"      ❌ Failed - Quality: {validation_stats['failed_quality']:,}")
        print(f"      ❌ Failed - Medical: {validation_stats['failed_medical']:,}")
        
        return validated_data
    
    def _comprehensive_validation(self, item):
        """Comprehensive validation for final quality"""
        return (self._length_validation(item) and 
                self._quality_validation(item) and 
                self._medical_validation(item))
    
    def _length_validation(self, item):
        """Validate text lengths"""
        question = str(item.get('question', ''))
        answer = str(item.get('answer', ''))
        return 10 <= len(question) <= 800 and 5 <= len(answer) <= 3000
    
    def _quality_validation(self, item):
        """Validate quality score"""
        return item.get('quality_score', 0) >= 0.7
    
    def _medical_validation(self, item):
        """Validate medical relevance"""
        combined = f"{item.get('question', '')} {item.get('answer', '')}".lower()
        medical_terms = sum(1 for category in self.medical_terms.values() 
                          for term in category if term.lower() in combined)
        return medical_terms >= 1

    def generate_comprehensive_report(self):
        """Generate comprehensive processing report"""
        report = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'cleaner_version': 'complete_unified_enhanced_v2.0',
                'translation_enabled': self.translator is not None
            },
            'statistics': self.stats,
            'quality_metrics': {
                'data_retention_rate': (self.stats['final_unified_count'] / max(self.stats['original_count'], 1)) * 100,
                'english_percentage': (self.stats['final_english_count'] / max(self.stats['final_unified_count'], 1)) * 100,
                'thai_percentage': (self.stats['final_thai_count'] / max(self.stats['final_unified_count'], 1)) * 100,
                'cleaning_efficiency': {
                    'duplicates_removed_percentage': (self.stats['duplicates_removed'] / max(self.stats['original_count'], 1)) * 100,
                    'irrelevant_removed_percentage': (self.stats['irrelevant_removed'] / max(self.stats['original_count'], 1)) * 100,
                    'offensive_removed_percentage': (self.stats['offensive_removed'] / max(self.stats['original_count'], 1)) * 100
                }
            },
            'feature_summary': {
                'medical_abbreviation_expansion': self.stats['abbreviations_expanded'] > 0,
                'statistical_notation_normalization': self.stats['statistical_normalized'] > 0,
                'answer_format_standardization': self.stats['answers_normalized'] > 0,
                'bilingual_translation': self.stats['translated_count'] > 0,
                'semantic_deduplication': self.stats['duplicates_removed'] > 0,
                'medical_relevance_filtering': self.stats['irrelevant_removed'] > 0,
                'offensive_content_removal': self.stats['offensive_removed'] > 0
            }
        }
        
        return report

    def print_final_summary(self, saved_files):
        """Print final English-only processing summary"""
        print("\n" + "="*80)
        print("🎯 ENGLISH-ONLY MEDICAL DATA CLEANING COMPLETED!")
        print("�🇧 Maximum Quality English-Only Strategy")
        print("="*80)
        
        print(f"\n📊 Processing Statistics:")
        print(f"   📈 Original records loaded: {self.stats['original_count']:,}")
        print(f"   🗑️ Duplicates removed: {self.stats['duplicates_removed']:,}")
        print(f"   🚫 Irrelevant content removed: {self.stats['irrelevant_removed']:,}")
        print(f"   ❌ Incomplete pairs removed: {self.stats['incomplete_removed']:,}")
        print(f"   🔒 Offensive content removed: {self.stats['offensive_removed']:,}")
        print(f"   🔧 Records with formatting cleaned: {self.stats['formatting_cleaned']:,}")
        print(f"   🏥 Medical abbreviations expanded: {self.stats['abbreviations_expanded']:,}")
        print(f"   📊 Statistical notations normalized: {self.stats['statistical_normalized']:,}")
        print(f"   ✅ Answer formats normalized: {self.stats['answers_normalized']:,}")
        print(f"   📋 Missing data fields filled: {self.stats['missing_data_filled']:,}")
        print(f"   🇬🇧 English-only focus: Maximum quality achieved")
        
        print(f"\n�🇧 Final English-Only Dataset:")
        print(f"   ✅ Clean English records: {self.stats['final_english_count']:,}")
        print(f"   📊 Total processed records: {self.stats['final_unified_count']:,}")
        print(f"   🚫 Thai translation: DISABLED (English-only mode)")
        
        retention_rate = (self.stats['final_english_count'] / max(self.stats['original_count'], 1)) * 100
        print(f"   📈 Data retention rate: {retention_rate:.1f}%")
        print(f"   ⭐ Quality focus: Maximum English quality achieved")
        
        print(f"\n📁 Output Files ({len(saved_files)} files):")
        for i, file_path in enumerate(saved_files, 1):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   {i}. {os.path.basename(file_path)} ({file_size:.1f} MB)")
        
        print(f"\n✨ Processing Workflow:")
        print(f"   🏥 Step 1: Load all raw medical datasets")
        print(f"   🧹 Step 2: Clean English data thoroughly (dedup, filter, normalize)")
        print(f"   🚫 Step 3: Translation SKIPPED (English-only mode)")
        print(f"   💾 Step 4: Save English-only files (JSON + CSV)")
        
        print(f"\n🔧 Advanced Features Applied:")
        print(f"   ✅ Medical terminology expansion (HDL-C → High-Density Lipoprotein Cholesterol)")
        print(f"   ✅ Statistical notation normalization (p<0.05 → p-value less than 0.05)")
        print(f"   ✅ Answer format standardization (yes/no/uncertain/unknown)")
        print(f"   ✅ Semantic duplicate detection and removal")
        print(f"   ✅ Medical relevance filtering")
        print(f"   ✅ Offensive content removal")
        print(f"   🚫 Translation: DISABLED (English-only focus)")
        print(f"   ✅ Comprehensive metadata enrichment")
        
        print(f"\n🚀 Ready for:")
        print(f"   • LangChain vector database import (use JSON files)")
        print(f"   • Medical Q&A model training (use CSV files)")
        print(f"   • English medical assistant deployment")
        print(f"   • API development and data analysis")
        
        print(f"\n🎯 Quality Strategy Success:")
        print(f"   ✅ English-only focus ensures maximum quality")
        print(f"   ✅ No translation overhead = faster processing")
        print(f"   ✅ Clean files ready for immediate use")
        print(f"   ✅ Perfect for English medical applications")
        
        print("\n" + "="*80)

def main(max_records=None):
    """Main execution function - English-only processing"""
    print("🏥 Medical Data Cleaner - ENGLISH-ONLY MODE")
    print("ระบบทำความสะอาดข้อมูลการแพทย์ (ภาษาอังกฤษเท่านั้น)")
    print("🎯 Focus: ENGLISH ONLY - Maximum quality, no translation")
    print("🇬🇧 English-only medical data cleaning and enhancement")
    print("📊 Detailed cleaning and quality optimization")
    print("⚠️  Note: Thai translation DISABLED")
    if max_records:
        print(f"📦 Processing limit: {max_records:,} records")
    print("="*80)
    
    # Initialize cleaner
    print("🔧 Initializing English-only cleaner...")
    print("   • Batch processing enabled")
    print("   • Medical terminology expansion")
    print("   • Quality assessment & validation")
    print("   • English-only focus: No translation overhead")
    
    # Use all available cores for maximum speed
    cpu_count = mp.cpu_count()
    print(f"   • CPU cores available: {cpu_count}")
    
    cleaner = CompleteUnifiedMedicalCleaner(
        batch_size=10000,
        use_multiprocessing=True,
        max_workers=max(1, cpu_count - 1)
    )
    
    # Load all available data
    print("\n📂 Step 1: Loading all raw medical datasets...")
    all_raw_data = cleaner.load_all_raw_data(max_records=max_records)
    
    if not all_raw_data:
        print("❌ No raw data found to process!")
        print("💡 Please ensure you have medical datasets in the data/raw/ directory")
        return
    
    # Show processing plan
    if all_raw_data:
        total_batches = (len(all_raw_data) + cleaner.batch_size - 1) // cleaner.batch_size
        print(f"\n📊 Processing Plan:")
        print(f"   • Input records: {len(all_raw_data):,}")
        print(f"   • Batch size: {cleaner.batch_size:,}")
        print(f"   • Total batches: {total_batches}")
        print(f"   • Processing: English-only cleaning (no translation)")
    
    # Run comprehensive cleaning
    start_time = datetime.now()
    print(f"\n🧹 Step 2: Running comprehensive English-only cleaning...")
    english_data, thai_data = cleaner.comprehensive_clean_all_data(all_raw_data)
    
    processing_time = datetime.now() - start_time
    print(f"⏱️ Processing completed in: {processing_time}")
    
    # Save datasets
    print(f"\n💾 Step 3: Saving English-only datasets...")
    saved_files = cleaner.save_unified_datasets(english_data, thai_data)
    
    # Generate report
    print(f"\n📋 Step 4: Generating performance report...")
    report = cleaner.generate_comprehensive_report()
    
    # Add performance metrics
    report['performance_metrics'] = {
        'processing_time_seconds': processing_time.total_seconds(),
        'records_per_second': len(all_raw_data) / max(processing_time.total_seconds(), 1),
        'batches_processed': cleaner.stats['batches_processed'],
        'batch_size': cleaner.batch_size,
        'english_only_mode': True
    }
    
    # Save report
    report_dir = Path("../data/exports")
    report_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"english_only_cleaning_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Report saved: {report_file}")
    
    # Print final summary
    cleaner.print_final_summary(saved_files)
    
    # Performance summary
    print(f"\n🔬 English-Only Processing Performance:")
    print(f"   ⏱️ Processing time: {processing_time}")
    print(f"   🚀 Speed: {len(all_raw_data) / max(processing_time.total_seconds(), 1):.0f} records/second")
    print(f"   📦 Batches processed: {cleaner.stats['batches_processed']}")
    print(f"   🎯 Result: {cleaner.stats['original_count']:,} → {cleaner.stats['final_english_count']:,} clean English records")
    print(f"   📁 Output: {len(saved_files)} English files")
    print(f"   ⭐ Quality: Multi-level validation, English-only focus")
    print(f"   🎯 Ready for: Medical Q&A training, LangChain, API development")
    
    print(f"\n✅ English-only cleaning completed successfully!")
    print(f"🏆 Maximum quality English medical dataset achieved!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Data Cleaner - English Only")
    parser.add_argument("--max-records", type=int, default=None, help="Maximum number of records to process (e.g., 50000)")
    
    args = parser.parse_args()
    
    print("🇬🇧 ENGLISH-ONLY MODE: No translation, maximum quality")
    
    main(max_records=args.max_records)
