#!/usr/bin/env python3
"""
Step 6: Data Splitting
แบ่งข้อมูลเป็น Train / Validation / Test sets
- Train: 70% (สำหรับเทรนโมเดล)
- Validation: 15% (สำหรับปรับพารามิเตอร์)
- Test: 15% (สำหรับทดสอบสุดท้าย)
- ใช้ Stratified split ตาม specialty และ severity
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using manual split")

class MedicalDataSplitter:
    """Split medical Q&A data with stratification"""
    
    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        self.stats = {
            'total': 0,
            'train': 0,
            'val': 0,
            'test': 0
        }
    
    def create_stratify_key(self, item):
        """Create stratification key from specialty and severity"""
        specialty = item.get('specialty', 'general_medicine')
        severity = item.get('severity_level', 'moderate')
        return f"{specialty}_{severity}"
    
    def stratified_split(self, data):
        """Perform stratified split"""
        logger.info("🔀 Performing stratified split...")
        
        self.stats['total'] = len(data)
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Create stratification key
        df['stratify_key'] = df.apply(self.create_stratify_key, axis=1)
        
        # Check if stratification is possible
        key_counts = Counter(df['stratify_key'])
        min_count = min(key_counts.values())
        
        if min_count < 3:
            logger.warning(f"⚠️ Some stratify keys have < 3 samples, using random split")
            return self.random_split(data)
        
        if SKLEARN_AVAILABLE:
            # Split train + (val + test)
            train_df, temp_df = train_test_split(
                df,
                train_size=self.train_ratio,
                stratify=df['stratify_key'],
                random_state=self.random_state
            )
            
            # Split val and test from temp
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_ratio_adjusted,
                stratify=temp_df['stratify_key'],
                random_state=self.random_state
            )
            
            # Remove stratify_key column
            train_df = train_df.drop('stratify_key', axis=1)
            val_df = val_df.drop('stratify_key', axis=1)
            test_df = test_df.drop('stratify_key', axis=1)
            
            # Convert back to list of dicts
            train_data = train_df.to_dict('records')
            val_data = val_df.to_dict('records')
            test_data = test_df.to_dict('records')
            
        else:
            # Manual stratified split
            train_data, val_data, test_data = self.manual_stratified_split(data)
        
        self.stats['train'] = len(train_data)
        self.stats['val'] = len(val_data)
        self.stats['test'] = len(test_data)
        
        logger.info(f"   ✅ Train: {self.stats['train']:,} ({self.stats['train']/self.stats['total']*100:.1f}%)")
        logger.info(f"   ✅ Val:   {self.stats['val']:,} ({self.stats['val']/self.stats['total']*100:.1f}%)")
        logger.info(f"   ✅ Test:  {self.stats['test']:,} ({self.stats['test']/self.stats['total']*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def manual_stratified_split(self, data):
        """Manual stratified split (fallback)"""
        # Group by stratify key
        groups = {}
        for item in data:
            key = self.create_stratify_key(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        train_data = []
        val_data = []
        test_data = []
        
        # Split each group proportionally
        for key, items in groups.items():
            n = len(items)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            # Shuffle
            np.random.seed(self.random_state)
            indices = np.random.permutation(n)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            
            train_data.extend([items[i] for i in train_indices])
            val_data.extend([items[i] for i in val_indices])
            test_data.extend([items[i] for i in test_indices])
        
        return train_data, val_data, test_data
    
    def random_split(self, data):
        """Simple random split (fallback)"""
        logger.info("🔀 Using random split...")
        
        n = len(data)
        indices = np.random.permutation(n)
        
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, val_data, test_data
    
    def analyze_split_distribution(self, train_data, val_data, test_data):
        """Analyze distribution across splits"""
        print("\n" + "="*60)
        print("📊 SPLIT DISTRIBUTION ANALYSIS")
        print("="*60)
        
        def get_distribution(data, name):
            print(f"\n{name} Set ({len(data):,} samples):")
            
            # Specialty distribution
            if data and 'specialty' in data[0]:
                specialties = Counter([item['specialty'] for item in data])
                print(f"  Specialties:")
                for spec, count in specialties.most_common(5):
                    print(f"    {spec:20s}: {count:5,} ({count/len(data)*100:5.2f}%)")
            
            # Severity distribution
            if data and 'severity_level' in data[0]:
                severities = Counter([item['severity_level'] for item in data])
                print(f"  Severities:")
                for sev, count in severities.most_common():
                    print(f"    {sev:20s}: {count:5,} ({count/len(data)*100:5.2f}%)")
        
        get_distribution(train_data, "Train")
        get_distribution(val_data, "Validation")
        get_distribution(test_data, "Test")
    
    def save_splits(self, train_data, val_data, test_data, output_dir="data/model_ready"):
        """Save split datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files = {}
        
        # Save train set
        train_file = output_path / f"train_{timestamp}.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        files['train'] = train_file
        logger.info(f"💾 Saved train: {train_file}")
        
        # Save validation set
        val_file = output_path / f"val_{timestamp}.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        files['val'] = val_file
        logger.info(f"💾 Saved val: {val_file}")
        
        # Save test set
        test_file = output_path / f"test_{timestamp}.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        files['test'] = test_file
        logger.info(f"💾 Saved test: {test_file}")
        
        # Save split summary
        summary_file = output_path / f"split_summary_{timestamp}.json"
        summary = {
            'timestamp': timestamp,
            'ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'counts': self.stats,
            'files': {
                'train': str(train_file),
                'val': str(val_file),
                'test': str(test_file)
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved summary: {summary_file}")
        
        return files
    
    def print_summary(self):
        """Print split summary"""
        print("\n" + "="*60)
        print("🔀 DATA SPLITTING SUMMARY")
        print("="*60)
        print(f"📊 Total samples:      {self.stats['total']:,}")
        print(f"🎯 Train set:          {self.stats['train']:,} ({self.train_ratio*100:.0f}%)")
        print(f"✅ Validation set:     {self.stats['val']:,} ({self.val_ratio*100:.0f}%)")
        print(f"🧪 Test set:           {self.stats['test']:,} ({self.test_ratio*100:.0f}%)")
        print("="*60)

def main():
    """Main data splitting process"""
    print("🔀 STEP 6: DATA SPLITTING")
    print("=" * 60)
    print("📌 Split: Train 80% / Val 10% / Test 10%")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Step 6: Data Splitting")
    parser.add_argument("--input", type=str, 
                       default=None,
                       help="Input data file (auto-detect if not specified)")
    parser.add_argument("--train", type=float, default=0.80, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.10, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.10, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Auto-detect latest file
    if args.input is None:
        # Try model_ready directory first
        model_ready_path = Path("data/model_ready")
        engineered_files = list(model_ready_path.glob("feature_engineered_*.json"))
        if not engineered_files:
            # Fall back to processed directory
            processed_path = Path("data/processed")
            engineered_files = list(processed_path.glob("chatdoctor_cleaned_*.json"))
        
        if engineered_files:
            args.input = str(max(engineered_files, key=lambda x: x.stat().st_mtime))
            logger.info(f"✅ Auto-detected: {args.input}")
        else:
            print(f"❌ No data found in data/model_ready/ or data/processed/")
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
    
    # Split data
    splitter = MedicalDataSplitter(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_state=args.seed
    )
    
    train_data, val_data, test_data = splitter.stratified_split(data)
    
    # Analyze distribution
    splitter.analyze_split_distribution(train_data, val_data, test_data)
    
    # Save splits
    files = splitter.save_splits(train_data, val_data, test_data)
    
    # Summary
    splitter.print_summary()
    
    print(f"\n✅ Data splitting completed!")
    print(f"📁 Train: {files['train']}")
    print(f"📁 Val:   {files['val']}")
    print(f"📁 Test:  {files['test']}")
    print(f"🚀 Ready for model training!")
    
    return 0

if __name__ == "__main__":
    exit(main())
