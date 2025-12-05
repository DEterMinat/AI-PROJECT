#!/usr/bin/env python3
"""
Step 10: Hyperparameter Tuning
ปรับค่าพารามิเตอร์ให้โมเดลดีที่สุด
- Learning Rate
- Batch Size
- Number of Epochs
- Dropout Rate
- Weight Decay
- Warmup Steps
ใช้ Optuna สำหรับ Bayesian Optimization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - install with: pip install optuna")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

class MedicalQADataset(Dataset):
    """Simple dataset for hyperparameter tuning"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Tokenize
        encoding = self.tokenizer(
            f"question: {question}",
            answer,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, model_name="t5-small", n_trials=20):
        self.model_name = model_name
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float('inf')
        self.study = None
        
        self.param_space = {
            'learning_rate': (1e-5, 1e-3),
            'per_device_train_batch_size': [4, 8, 16],
            'num_train_epochs': [2, 3, 5],
            'weight_decay': (0.0, 0.3),
            'warmup_ratio': (0.0, 0.2),
            'gradient_accumulation_steps': [1, 2, 4]
        }
    
    def objective(self, trial, train_data, val_data, tokenizer):
        """Objective function for Optuna"""
        
        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', *self.param_space['learning_rate'], log=True)
        batch_size = trial.suggest_categorical('per_device_train_batch_size', self.param_space['per_device_train_batch_size'])
        num_epochs = trial.suggest_categorical('num_train_epochs', self.param_space['num_train_epochs'])
        weight_decay = trial.suggest_float('weight_decay', *self.param_space['weight_decay'])
        warmup_ratio = trial.suggest_float('warmup_ratio', *self.param_space['warmup_ratio'])
        grad_accum = trial.suggest_categorical('gradient_accumulation_steps', self.param_space['gradient_accumulation_steps'])
        
        print(f"\n🧪 Trial {trial.number + 1}/{self.n_trials}")
        print(f"   LR: {learning_rate:.2e}, BS: {batch_size}, Epochs: {num_epochs}")
        print(f"   Weight Decay: {weight_decay:.3f}, Warmup: {warmup_ratio:.3f}")
        
        try:
            # Load model
            if 't5' in self.model_name.lower() or 'bart' in self.model_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Prepare datasets
            train_dataset = MedicalQADataset(train_data, tokenizer)
            val_dataset = MedicalQADataset(val_data, tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./optuna_trial_{trial.number}",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                gradient_accumulation_steps=grad_accum,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                logging_dir=f"./logs/trial_{trial.number}",
                logging_steps=50,
                save_total_limit=1,
                report_to="none",
                fp16=torch.cuda.is_available()
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train
            train_result = trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            
            eval_loss = eval_result['eval_loss']
            
            print(f"   ✅ Eval Loss: {eval_loss:.4f}")
            
            # Clean up
            import shutil
            shutil.rmtree(f"./optuna_trial_{trial.number}", ignore_errors=True)
            
            return eval_loss
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return float('inf')
    
    def tune(self, train_data, val_data):
        """Run hyperparameter tuning"""
        
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna not available")
            print("💡 Install with: pip install optuna")
            return None
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available")
            return None
        
        print("\n" + "="*60)
        print("🔧 HYPERPARAMETER TUNING")
        print("="*60)
        print(f"📊 Model: {self.model_name}")
        print(f"🎯 Trials: {self.n_trials}")
        print(f"📈 Optimization: Minimize validation loss")
        print()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            study_name='medical_qa_tuning',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, train_data, val_data, tokenizer),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def analyze_results(self):
        """Analyze tuning results"""
        
        if not self.study:
            print("❌ No study results available")
            return
        
        print("\n" + "="*60)
        print("📊 HYPERPARAMETER TUNING RESULTS")
        print("="*60)
        
        print(f"\n🏆 Best Trial:")
        print(f"   Trial Number: {self.study.best_trial.number}")
        print(f"   Best Score (Eval Loss): {self.best_score:.4f}")
        
        print(f"\n⚙️ Best Hyperparameters:")
        for param, value in self.best_params.items():
            if 'learning' in param:
                print(f"   {param:30s}: {value:.2e}")
            else:
                print(f"   {param:30s}: {value}")
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            print(f"\n📊 Parameter Importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"   {param:30s}: {imp:.4f}")
        except:
            pass
        
        # Trials dataframe
        df = self.study.trials_dataframe()
        print(f"\n📈 All Trials Summary:")
        print(f"   Total trials: {len(df)}")
        print(f"   Successful: {len(df[df['state'] == 'COMPLETE'])}")
        print(f"   Failed: {len(df[df['state'] == 'FAIL'])}")
        print(f"   Best score: {df['value'].min():.4f}")
        print(f"   Worst score: {df['value'].max():.4f}")
        print(f"   Mean score: {df['value'].mean():.4f}")
    
    def save_results(self, output_dir="data/model_ready"):
        """Save tuning results"""
        
        if not self.study:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best parameters
        params_file = output_path / f"best_hyperparameters_{timestamp}.json"
        results = {
            'timestamp': timestamp,
            'model_name': self.model_name,
            'n_trials': self.n_trials,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'best_trial_number': self.study.best_trial.number
        }
        
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Saved: {params_file}")
        
        # Save trials dataframe
        df = self.study.trials_dataframe()
        csv_file = output_path / f"tuning_trials_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Saved: {csv_file}")
        
        return params_file

def main():
    """Main hyperparameter tuning process"""
    print("🔧 STEP 10: HYPERPARAMETER TUNING")
    print("=" * 60)
    print("📌 Using Optuna Bayesian Optimization")
    print("📌 Parameters: LR, Batch Size, Epochs, Weight Decay, etc.")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Step 10: Hyperparameter Tuning")
    parser.add_argument("--train", type=str, help="Train data file")
    parser.add_argument("--val", type=str, help="Validation data file")
    parser.add_argument("--model", type=str, default="t5-small",
                       help="Model name (default: t5-small)")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of trials (default: 10)")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Max samples to use (default: 5000)")
    args = parser.parse_args()
    
    # Find data files if not specified
    model_ready_path = Path("data/model_ready")
    
    if not args.train:
        train_files = list(model_ready_path.glob("train_*.json"))
        if train_files:
            args.train = str(train_files[-1])
    
    if not args.val:
        val_files = list(model_ready_path.glob("val_*.json"))
        if val_files:
            args.val = str(val_files[-1])
    
    if not args.train or not Path(args.train).exists():
        print(f"❌ Train file not found")
        print(f"💡 Please run Step 6 (Data Splitting) first")
        return 1
    
    if not args.val or not Path(args.val).exists():
        print(f"❌ Validation file not found")
        print(f"💡 Please run Step 6 (Data Splitting) first")
        return 1
    
    # Load data
    print(f"📂 Loading train: {Path(args.train).name}")
    with open(args.train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"📂 Loading val: {Path(args.val).name}")
    with open(args.val, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Limit samples for faster tuning
    if len(train_data) > args.samples:
        import random
        random.seed(42)
        train_data = random.sample(train_data, args.samples)
    
    if len(val_data) > args.samples // 5:
        import random
        random.seed(42)
        val_data = random.sample(val_data, args.samples // 5)
    
    print(f"📊 Train: {len(train_data):,} samples")
    print(f"📊 Val: {len(val_data):,} samples")
    
    # Run tuning
    tuner = HyperparameterTuner(
        model_name=args.model,
        n_trials=args.trials
    )
    
    best_params = tuner.tune(train_data, val_data)
    
    if best_params:
        # Analyze and save results
        tuner.analyze_results()
        output_file = tuner.save_results()
        
        print(f"\n✅ Hyperparameter tuning completed!")
        print(f"📁 Results: {output_file}")
        print(f"🚀 Next: Use best parameters for training (Step 8)")
    else:
        print(f"\n❌ Tuning failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
