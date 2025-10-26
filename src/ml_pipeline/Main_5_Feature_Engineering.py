#!/usr/bin/env python3
"""
Step 5: Feature Engineering
แปลงข้อมูลให้อยู่ในรูปที่เหมาะสมกับโมเดล
- Tokenization
- Embeddings (Word2Vec, GloVe, or Transformer)
- TF-IDF
- Data Augmentation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLP Libraries
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available")

class MedicalFeatureEngineering:
    """Feature Engineering for Medical Q&A"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.stop_words = set()
        
        # Initialize tokenizer
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"✅ Loaded tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load tokenizer: {e}")
        
        # Initialize stop words
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
        
        self.stats = {
            'original_count': 0,
            'tokenized_count': 0,
            'augmented_count': 0,
            'final_count': 0
        }
    
    def tokenize_text(self, text, max_length=512):
        """Tokenize text using transformer tokenizer"""
        if not self.tokenizer or not text:
            return None
        
        try:
            tokens = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return None
    
    def extract_ngrams(self, text, n=2):
        """Extract n-grams from text"""
        if not text:
            return []
        
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def calculate_tfidf_features(self, texts):
        """Calculate TF-IDF features (simplified version)"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix, vectorizer
        except Exception as e:
            logger.error(f"TF-IDF error: {e}")
            return None, None
    
    def augment_paraphrase(self, text):
        """Simple paraphrasing using synonym replacement"""
        # Simple augmentation: question variation
        variations = []
        
        # Original
        variations.append(text)
        
        # Add "please" for politeness
        if "?" in text and "please" not in text.lower():
            variations.append(text.replace("?", ", please?"))
        
        # Change question format
        if text.lower().startswith("what is"):
            variations.append(text.replace("What is", "Can you explain what is"))
        elif text.lower().startswith("how to"):
            variations.append(text.replace("How to", "What is the best way to"))
        
        return variations
    
    def augment_data(self, data, augment_ratio=0.2):
        """Augment data with paraphrasing"""
        logger.info(f"🔄 Augmenting data (ratio: {augment_ratio})...")
        
        original_count = len(data)
        augmented_data = data.copy()
        
        # Calculate how many to augment
        num_to_augment = int(original_count * augment_ratio)
        
        # Select high-quality samples for augmentation
        # Prefer critical/high severity and specialized fields
        candidates = [
            item for item in data 
            if item.get('severity_level') in ['critical', 'high']
            and item.get('specialty') != 'general_medicine'
        ]
        
        if len(candidates) > num_to_augment:
            candidates = candidates[:num_to_augment]
        
        for item in tqdm(candidates, desc="Augmenting"):
            question = item.get('question', '')
            variations = self.augment_paraphrase(question)
            
            for var in variations[1:]:  # Skip original
                augmented_item = item.copy()
                augmented_item['question'] = var
                augmented_item['augmented'] = True
                augmented_item['augmentation_method'] = 'paraphrase'
                augmented_data.append(augmented_item)
        
        self.stats['augmented_count'] = len(augmented_data) - original_count
        logger.info(f"   ✅ Added {self.stats['augmented_count']:,} augmented samples")
        
        return augmented_data
    
    def extract_advanced_features(self, item):
        """Extract advanced features from Q&A pair"""
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        features = {}
        
        # 1. Length features
        features['q_word_count'] = len(question.split())
        features['a_word_count'] = len(answer.split())
        features['q_char_count'] = len(question)
        features['a_char_count'] = len(answer)
        features['qa_length_ratio'] = features['a_word_count'] / max(features['q_word_count'], 1)
        
        # 2. Medical term features
        medical_terms = ['patient', 'treatment', 'diagnosis', 'symptom', 'disease', 
                        'medication', 'doctor', 'hospital', 'therapy', 'clinical']
        features['q_medical_terms'] = sum(1 for term in medical_terms if term in question.lower())
        features['a_medical_terms'] = sum(1 for term in medical_terms if term in answer.lower())
        
        # 3. Question type features
        features['is_what_question'] = 1 if question.lower().startswith('what') else 0
        features['is_how_question'] = 1 if question.lower().startswith('how') else 0
        features['is_why_question'] = 1 if question.lower().startswith('why') else 0
        features['is_yes_no_question'] = 1 if any(question.lower().startswith(w) for w in ['is', 'are', 'do', 'does', 'can', 'could', 'should']) else 0
        
        # 4. N-grams
        features['q_bigrams'] = self.extract_ngrams(question, 2)[:10]  # Top 10
        features['a_bigrams'] = self.extract_ngrams(answer, 2)[:10]
        
        # 5. Numeric features
        features['q_has_numbers'] = 1 if re.search(r'\d', question) else 0
        features['a_has_numbers'] = 1 if re.search(r'\d', answer) else 0
        features['q_number_count'] = len(re.findall(r'\d+', question))
        features['a_number_count'] = len(re.findall(r'\d+', answer))
        
        # 6. Sentiment/Urgency indicators
        urgent_words = ['urgent', 'emergency', 'severe', 'critical', 'immediately', 'acute']
        features['urgency_score'] = sum(1 for word in urgent_words if word in (question + answer).lower())
        
        return features
    
    def process_dataset(self, data, augment=True, tokenize=True):
        """Process entire dataset with feature engineering"""
        logger.info("🔧 Starting Feature Engineering...")
        
        self.stats['original_count'] = len(data)
        processed_data = []
        
        # Step 1: Extract advanced features
        logger.info("📊 Extracting advanced features...")
        for item in tqdm(data, desc="Feature extraction"):
            features = self.extract_advanced_features(item)
            item['features'] = features
            processed_data.append(item)
        
        # Step 2: Data augmentation
        if augment:
            processed_data = self.augment_data(processed_data, augment_ratio=0.1)
        
        # Step 3: Tokenization
        if tokenize and self.tokenizer:
            logger.info("🔤 Tokenizing texts...")
            for item in tqdm(processed_data, desc="Tokenization"):
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                # Tokenize question and answer
                q_tokens = self.tokenize_text(question)
                a_tokens = self.tokenize_text(answer)
                
                if q_tokens:
                    item['question_tokens'] = {
                        'input_ids': q_tokens['input_ids'],
                        'attention_mask': q_tokens['attention_mask']
                    }
                
                if a_tokens:
                    item['answer_tokens'] = {
                        'input_ids': a_tokens['input_ids'],
                        'attention_mask': a_tokens['attention_mask']
                    }
            
            self.stats['tokenized_count'] = len(processed_data)
        
        self.stats['final_count'] = len(processed_data)
        
        return processed_data
    
    def save_processed_data(self, data, output_dir="data/model_ready"):
        """Save feature-engineered data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"feature_engineered_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print feature engineering summary"""
        print("\n" + "="*60)
        print("🔧 FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"📥 Original count:     {self.stats['original_count']:,}")
        print(f"➕ Augmented:          +{self.stats['augmented_count']:,}")
        print(f"🔤 Tokenized:          {self.stats['tokenized_count']:,}")
        print(f"✅ Final count:        {self.stats['final_count']:,}")
        print(f"📈 Growth:             {(self.stats['final_count']/max(self.stats['original_count'],1)-1)*100:.1f}%")
        print("="*60)

def main():
    """Main feature engineering process"""
    print("🔧 STEP 5: FEATURE ENGINEERING")
    print("=" * 60)
    print("📌 Tokenization, Embeddings, TF-IDF, Data Augmentation")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Step 5: Feature Engineering")
    parser.add_argument("--input", type=str, 
                       default=None,
                       help="Input data file (auto-detect if not specified)")
    parser.add_argument("--augment", action="store_true", default=False,
                       help="Enable data augmentation")
    parser.add_argument("--tokenize", action="store_true", default=False,
                       help="Enable tokenization")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                       help="Tokenizer model name")
    args = parser.parse_args()
    
    # Auto-detect latest file
    if args.input is None:
        # Try processed directory first
        processed_path = Path("data/processed")
        cleaned_files = list(processed_path.glob("chatdoctor_cleaned_*.json"))
        if cleaned_files:
            args.input = str(max(cleaned_files, key=lambda x: x.stat().st_mtime))
            logger.info(f"✅ Auto-detected: {args.input}")
        else:
            print(f"❌ No cleaned data found in data/processed/")
            return 1
    
    # Load data
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return 1
    
    print(f"📂 Loading: {input_file.name}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded: {len(data):,} records")
    
    # Feature engineering
    engineer = MedicalFeatureEngineering(model_name=args.model)
    processed_data = engineer.process_dataset(
        data,
        augment=args.augment,
        tokenize=args.tokenize
    )
    
    # Save
    output_file = engineer.save_processed_data(processed_data)
    
    # Summary
    engineer.print_summary()
    
    # Show sample
    if processed_data:
        print(f"\n📝 Sample with Features:")
        sample = processed_data[0]
        print(f"   Question: {sample['question'][:100]}...")
        print(f"   Answer: {sample['answer'][:100]}...")
        if 'features' in sample:
            print(f"   Features: {list(sample['features'].keys())[:10]}...")
        if 'question_tokens' in sample:
            print(f"   Tokenized: ✅ Yes")
    
    print(f"\n✅ Feature engineering completed!")
    print(f"📁 Output: {output_file}")
    print(f"🚀 Ready for data splitting!")
    
    return 0

if __name__ == "__main__":
    exit(main())
