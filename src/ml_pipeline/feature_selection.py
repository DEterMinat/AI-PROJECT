#!/usr/bin/env python3
"""
Feature Selection for Medical Q&A Model
เลือก Features ที่สำคัญที่สุดก่อน Train Model
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalFeatureSelector:
    """Feature Selection for Medical Q&A Data"""
    
    def __init__(self):
        self.stats = {
            'original_count': 0,
            'after_selection': 0,
            'features_extracted': 0
        }
        
        # Feature weights (importance scores)
        self.feature_weights = {
            'specialty': {
                'cardiology': 1.2,      # สำคัญมาก
                'neurology': 1.2,       # สำคัญมาก
                'oncology': 1.1,        # สำคัญ
                'endocrinology': 1.1,   # สำคัญ
                'pulmonology': 1.0,
                'gastroenterology': 1.0,
                'nephrology': 1.0,
                'pediatrics': 0.9,
                'orthopedics': 0.9,
                'psychiatry': 0.9,
                'dermatology': 0.8,
                'urology': 0.8,
                'general_medicine': 0.7  # น้อยสุด
            },
            'severity': {
                'critical': 1.3,    # สำคัญที่สุด
                'high': 1.1,        # สำคัญ
                'moderate': 0.9,    # ปานกลาง
                'low': 0.7          # น้อย
            },
            'type': {
                'medical_qa': 1.0,
                'advice': 0.9
            }
        }
    
    def extract_text_features(self, text):
        """Extract text-based features"""
        if not text:
            return {}
        
        features = {}
        
        # 1. Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # 2. Medical term density
        medical_terms = [
            'patient', 'treatment', 'diagnosis', 'symptom', 'disease', 'medication',
            'doctor', 'hospital', 'surgery', 'therapy', 'clinical', 'medical'
        ]
        text_lower = text.lower()
        features['medical_term_count'] = sum(1 for term in medical_terms if term in text_lower)
        features['medical_term_density'] = features['medical_term_count'] / max(features['word_count'], 1)
        
        # 3. Technical complexity (medical abbreviations)
        abbreviations = ['MI', 'CVD', 'HTN', 'DM', 'COPD', 'ECG', 'CBC', 'TSH', 'CT', 'MRI']
        features['abbreviation_count'] = sum(1 for abbr in abbreviations if abbr in text)
        
        # 4. Question quality indicators
        features['has_question_mark'] = 1 if '?' in text else 0
        features['starts_with_wh'] = 1 if any(text.lower().startswith(w) for w in ['what', 'how', 'why', 'when', 'where', 'who', 'which']) else 0
        
        # 5. Numerical content
        features['has_numbers'] = 1 if bool(re.search(r'\d', text)) else 0
        features['number_count'] = len(re.findall(r'\d+', text))
        
        return features
    
    def calculate_feature_score(self, item):
        """Calculate overall feature importance score"""
        score = 1.0
        
        # 1. Specialty weight
        specialty = item.get('specialty', 'general_medicine')
        score *= self.feature_weights['specialty'].get(specialty, 0.7)
        
        # 2. Severity weight
        severity = item.get('severity_level', 'moderate')
        score *= self.feature_weights['severity'].get(severity, 0.9)
        
        # 3. Type weight
        qa_type = item.get('type', 'medical_qa')
        score *= self.feature_weights['type'].get(qa_type, 1.0)
        
        # 4. Text features
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        q_features = self.extract_text_features(question)
        a_features = self.extract_text_features(answer)
        
        # Quality score based on text features
        quality_score = 1.0
        
        # Prefer longer, more detailed Q&A
        if q_features['word_count'] >= 10:
            quality_score *= 1.1
        if a_features['word_count'] >= 20:
            quality_score *= 1.2
        
        # Prefer high medical term density
        if q_features['medical_term_density'] > 0.2:
            quality_score *= 1.1
        if a_features['medical_term_density'] > 0.15:
            quality_score *= 1.1
        
        # Prefer technical content
        if q_features['abbreviation_count'] > 0:
            quality_score *= 1.05
        if a_features['abbreviation_count'] > 0:
            quality_score *= 1.05
        
        # Penalize very short answers
        if a_features['word_count'] < 5:
            quality_score *= 0.7
        
        score *= quality_score
        
        return score, {
            'specialty_score': self.feature_weights['specialty'].get(specialty, 0.7),
            'severity_score': self.feature_weights['severity'].get(severity, 0.9),
            'quality_score': quality_score,
            'q_word_count': q_features['word_count'],
            'a_word_count': a_features['word_count'],
            'q_medical_density': q_features['medical_term_density'],
            'a_medical_density': a_features['medical_term_density']
        }
    
    def select_features(self, data, target_count=None, min_score=0.8):
        """Select best features based on importance scores"""
        logger.info("🎯 Performing Feature Selection...")
        
        self.stats['original_count'] = len(data)
        
        # Calculate scores for all items
        scored_items = []
        
        for item in tqdm(data, desc="Calculating feature scores"):
            score, details = self.calculate_feature_score(item)
            scored_items.append({
                'item': item,
                'score': score,
                'details': details
            })
        
        # Sort by score (descending)
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter by minimum score
        filtered_items = [si for si in scored_items if si['score'] >= min_score]
        
        logger.info(f"   ✅ {len(filtered_items):,} items passed min_score >= {min_score}")
        
        # Select top N if specified
        if target_count and len(filtered_items) > target_count:
            selected_items = filtered_items[:target_count]
            logger.info(f"   ✅ Selected top {target_count:,} items")
        else:
            selected_items = filtered_items
        
        self.stats['after_selection'] = len(selected_items)
        
        # Extract just the items
        result = [si['item'] for si in selected_items]
        
        # Add feature metadata
        for i, si in enumerate(selected_items):
            result[i]['feature_score'] = round(si['score'], 3)
            result[i]['feature_details'] = si['details']
        
        return result, selected_items
    
    def analyze_distribution(self, selected_items):
        """Analyze distribution after feature selection"""
        logger.info("\n📊 Distribution Analysis:")
        
        # Specialty distribution
        specialties = [si['item']['specialty'] for si in selected_items]
        specialty_counts = Counter(specialties)
        
        print("\n🏥 Specialty Distribution:")
        for specialty, count in specialty_counts.most_common():
            percentage = (count / len(selected_items)) * 100
            print(f"   {specialty:20s}: {count:6,} ({percentage:5.2f}%)")
        
        # Severity distribution
        severities = [si['item']['severity_level'] for si in selected_items]
        severity_counts = Counter(severities)
        
        print("\n⚡ Severity Distribution:")
        for severity, count in severity_counts.most_common():
            percentage = (count / len(selected_items)) * 100
            print(f"   {severity:20s}: {count:6,} ({percentage:5.2f}%)")
        
        # Score statistics
        scores = [si['score'] for si in selected_items]
        print(f"\n📈 Score Statistics:")
        print(f"   Min:     {min(scores):.3f}")
        print(f"   Max:     {max(scores):.3f}")
        print(f"   Mean:    {np.mean(scores):.3f}")
        print(f"   Median:  {np.median(scores):.3f}")
        print(f"   Std Dev: {np.std(scores):.3f}")
        
        # Quality distribution
        q_word_counts = [si['details']['q_word_count'] for si in selected_items]
        a_word_counts = [si['details']['a_word_count'] for si in selected_items]
        
        print(f"\n📝 Text Length Statistics:")
        print(f"   Question words - Mean: {np.mean(q_word_counts):.1f}, Median: {np.median(q_word_counts):.1f}")
        print(f"   Answer words   - Mean: {np.mean(a_word_counts):.1f}, Median: {np.median(a_word_counts):.1f}")
    
    def save_selected_data(self, data, output_dir="data/model_ready"):
        """Save selected features data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"feature_selected_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 Saved: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print selection summary"""
        print("\n" + "="*60)
        print("🎯 FEATURE SELECTION SUMMARY")
        print("="*60)
        print(f"📥 Original count:    {self.stats['original_count']:,}")
        print(f"✅ After selection:   {self.stats['after_selection']:,}")
        print(f"📊 Retention:         {(self.stats['after_selection']/max(self.stats['original_count'],1))*100:.1f}%")
        print("="*60)

def main():
    """Main feature selection process"""
    print("🎯 MEDICAL FEATURE SELECTION")
    print("=" * 60)
    print("📌 Purpose: Select best features for training")
    print("📌 Criteria: Specialty, Severity, Quality, Medical Density")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Medical Feature Selection")
    parser.add_argument("--input", type=str, default="data/processed/detailed_cleaned_20251006_183315.json",
                       help="Input cleaned data file")
    parser.add_argument("--target", type=int, default=50000,
                       help="Target number of samples (default: 50,000)")
    parser.add_argument("--min-score", type=float, default=0.8,
                       help="Minimum feature score (default: 0.8)")
    args = parser.parse_args()
    
    # Load cleaned data
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return 1
    
    print(f"📂 Loading: {input_file.name}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded: {len(data):,} records")
    
    # Feature selection
    selector = MedicalFeatureSelector()
    selected_data, selected_items = selector.select_features(
        data, 
        target_count=args.target,
        min_score=args.min_score
    )
    
    # Analyze distribution
    selector.analyze_distribution(selected_items)
    
    # Save
    output_file = selector.save_selected_data(selected_data)
    
    # Summary
    selector.print_summary()
    
    # Show samples
    print("\n📝 Top 3 Samples (Highest Scores):")
    for i, si in enumerate(selected_items[:3], 1):
        item = si['item']
        print(f"\n{i}. Score: {si['score']:.3f}")
        print(f"   Specialty: {item['specialty']}, Severity: {item['severity_level']}")
        print(f"   Q: {item['question'][:100]}...")
        print(f"   A: {item['answer'][:100]}...")
    
    print(f"\n✅ Feature selection completed!")
    print(f"📁 Output: {output_file}")
    print(f"🚀 Ready for model training!")
    
    return 0

if __name__ == "__main__":
    exit(main())
