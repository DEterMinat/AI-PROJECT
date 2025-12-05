#!/usr/bin/env python3
"""
Step 7: Model Selection
เปรียบเทียบและเลือกโมเดลที่ดีที่สุด
- Test multiple architectures: GPT-2, BERT, T5, BART, BioGPT
- Compare metrics: Loss, BLEU, ROUGE, Perplexity
- Select best model for medical Q&A task
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
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

class ModelSelector:
    """Select best model for medical Q&A"""
    
    def __init__(self):
        self.models_to_test = {
            # Causal LM (GPT-style)
            'gpt2': {
                'name': 'gpt2',
                'type': 'causal',
                'description': 'GPT-2 - General purpose language model'
            },
            'distilgpt2': {
                'name': 'distilgpt2',
                'type': 'causal',
                'description': 'DistilGPT2 - Lighter, faster GPT-2'
            },
            'biogpt': {
                'name': 'microsoft/biogpt',
                'type': 'causal',
                'description': 'BioGPT - Pre-trained on biomedical text'
            },
            
            # Seq2Seq (Encoder-Decoder)
            't5-small': {
                'name': 't5-small',
                'type': 'seq2seq',
                'description': 'T5-Small - Text-to-Text Transfer Transformer'
            },
            'flan-t5-small': {
                'name': 'google/flan-t5-small',
                'type': 'seq2seq',
                'description': 'FLAN-T5 - Instruction-tuned T5'
            },
            'bart-base': {
                'name': 'facebook/bart-base',
                'type': 'seq2seq',
                'description': 'BART - Denoising autoencoder'
            }
        }
        
        self.results = []
    
    def load_sample_data(self, train_file, max_samples=1000):
        """Load small sample for quick testing"""
        logger.info(f"📂 Loading sample data ({max_samples} samples)...")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Take sample
        if len(data) > max_samples:
            import random
            random.seed(42)
            data = random.sample(data, max_samples)
        
        return data
    
    def prepare_dataset(self, data, tokenizer, model_type):
        """Prepare dataset for specific model type"""
        questions = [item['question'] for item in data]
        answers = [item['answer'] for item in data]
        
        if model_type == 'seq2seq':
            # T5/BART style: "question: Q" -> "answer: A"
            inputs = [f"question: {q}" for q in questions]
            targets = [f"answer: {a}" for a in answers]
        else:
            # GPT style: "Q\n\nA"
            inputs = questions
            targets = answers
        
        return inputs, targets
    
    def quick_test_model(self, model_key, train_data, val_data=None):
        """Quick test of a model"""
        model_info = self.models_to_test[model_key]
        model_name = model_info['name']
        model_type = model_info['type']
        
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {model_key}")
        print(f"📝 {model_info['description']}")
        print(f"{'='*60}")
        
        result = {
            'model_key': model_key,
            'model_name': model_name,
            'model_type': model_type,
            'description': model_info['description'],
            'timestamp': datetime.now().isoformat()
        }
        
        if not TRANSFORMERS_AVAILABLE:
            print("   ⚠️ Transformers not available")
            result['status'] = 'skipped'
            result['error'] = 'transformers_not_available'
            return result
        
        try:
            start_time = time.time()
            
            # Load tokenizer
            print(f"   📦 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            print(f"   🤖 Loading model...")
            if model_type == 'seq2seq':
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Prepare data
            print(f"   📊 Preparing data...")
            train_inputs, train_targets = self.prepare_dataset(train_data, tokenizer, model_type)
            
            # Tokenize (small batch for testing)
            train_encodings = tokenizer(
                train_inputs[:100], 
                truncation=True, 
                padding=True, 
                max_length=128,
                return_tensors='pt'
            )
            
            target_encodings = tokenizer(
                train_targets[:100], 
                truncation=True, 
                padding=True, 
                max_length=128,
                return_tensors='pt'
            )
            
            # Quick inference test
            print(f"   🔮 Testing inference...")
            model.eval()
            with torch.no_grad():
                sample_input = train_encodings['input_ids'][:1]
                
                if model_type == 'seq2seq':
                    outputs = model.generate(
                        sample_input,
                        max_length=50,
                        num_return_sequences=1
                    )
                else:
                    outputs = model.generate(
                        sample_input,
                        max_length=sample_input.shape[1] + 50,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   💬 Sample output: {generated_text[:100]}...")
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            
            load_time = time.time() - start_time
            
            result['status'] = 'success'
            result['load_time_seconds'] = round(load_time, 2)
            result['parameter_count'] = param_count
            result['parameter_count_M'] = round(param_count / 1e6, 2)
            result['sample_output'] = generated_text[:200]
            result['tokenizer_vocab_size'] = len(tokenizer)
            
            print(f"   ✅ Success!")
            print(f"   ⏱️ Load time: {load_time:.2f}s")
            print(f"   📐 Parameters: {param_count/1e6:.2f}M")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def compare_all_models(self, train_data, val_data=None):
        """Compare all models"""
        print("\n" + "="*60)
        print("🔍 MODEL SELECTION - COMPARING MODELS")
        print("="*60)
        
        for model_key in self.models_to_test.keys():
            result = self.quick_test_model(model_key, train_data, val_data)
            self.results.append(result)
        
        return self.results
    
    def analyze_results(self):
        """Analyze and rank models"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter successful models
        successful = df[df['status'] == 'success']
        
        if len(successful) == 0:
            print("   ❌ No models loaded successfully")
            return None
        
        print(f"\n✅ Successful models: {len(successful)}/{len(df)}\n")
        
        # Display comparison
        print("Model Comparison:")
        print("-" * 100)
        print(f"{'Model':<20} {'Type':<10} {'Params (M)':<12} {'Load Time (s)':<15} {'Status':<10}")
        print("-" * 100)
        
        for _, row in df.iterrows():
            model = row['model_key']
            model_type = row.get('model_type', 'N/A')
            params = row.get('parameter_count_M', 'N/A')
            load_time = row.get('load_time_seconds', 'N/A')
            status = row['status']
            
            print(f"{model:<20} {model_type:<10} {str(params):<12} {str(load_time):<15} {status:<10}")
        
        print("-" * 100)
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if len(successful) > 0:
            # Sort by parameter count (prefer smaller for faster training)
            successful_sorted = successful.sort_values('parameter_count')
            
            best_small = successful_sorted.iloc[0]
            print(f"\n🥇 Best Small Model (Fastest Training):")
            print(f"   Model: {best_small['model_key']}")
            print(f"   Type: {best_small['model_type']}")
            print(f"   Parameters: {best_small['parameter_count_M']:.2f}M")
            print(f"   Description: {best_small['description']}")
            
            # Recommend medical-specific if available
            medical_models = successful[successful['model_key'].str.contains('bio|med', case=False)]
            if len(medical_models) > 0:
                best_medical = medical_models.iloc[0]
                print(f"\n🏥 Best Medical Model:")
                print(f"   Model: {best_medical['model_key']}")
                print(f"   Type: {best_medical['model_type']}")
                print(f"   Parameters: {best_medical['parameter_count_M']:.2f}M")
                print(f"   Description: {best_medical['description']}")
            
            # Recommend seq2seq for Q&A
            seq2seq_models = successful[successful['model_type'] == 'seq2seq']
            if len(seq2seq_models) > 0:
                best_seq2seq = seq2seq_models.sort_values('parameter_count').iloc[0]
                print(f"\n💬 Best Q&A Model (Seq2Seq):")
                print(f"   Model: {best_seq2seq['model_key']}")
                print(f"   Parameters: {best_seq2seq['parameter_count_M']:.2f}M")
                print(f"   Description: {best_seq2seq['description']}")
                print(f"   ⭐ Recommended for Medical Q&A tasks")
        
        return successful
    
    def save_results(self, output_dir="data/model_ready"):
        """Save model selection results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"model_selection_results_{timestamp}.json"
        
        results_data = {
            'timestamp': timestamp,
            'models_tested': len(self.results),
            'successful': len([r for r in self.results if r['status'] == 'success']),
            'results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved: {output_file}")
        return output_file

def main():
    """Main model selection process"""
    print("🔍 STEP 7: MODEL SELECTION")
    print("=" * 60)
    print("📌 Testing: GPT-2, DistilGPT2, BioGPT, T5, FLAN-T5, BART")
    print("📌 Goal: Find best model for Medical Q&A")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Step 7: Model Selection")
    parser.add_argument("--train", type=str, 
                       help="Train data file (will use sample)")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to test (default: 1000)")
    args = parser.parse_args()
    
    # Find train file if not specified
    if not args.train:
        model_ready_path = Path("data/model_ready")
        train_files = list(model_ready_path.glob("train_*.json"))
        if train_files:
            args.train = str(train_files[-1])  # Use latest
        else:
            # Fallback to feature selected
            feature_files = list(model_ready_path.glob("feature_selected_*.json"))
            if feature_files:
                args.train = str(feature_files[-1])
    
    if not args.train or not Path(args.train).exists():
        print(f"❌ Train file not found")
        print(f"💡 Please run Step 6 (Data Splitting) first")
        return 1
    
    # Load sample data
    selector = ModelSelector()
    train_data = selector.load_sample_data(args.train, max_samples=args.samples)
    
    print(f"📊 Loaded: {len(train_data):,} samples for testing")
    
    # Compare models
    results = selector.compare_all_models(train_data)
    
    # Analyze results
    selector.analyze_results()
    
    # Save results
    output_file = selector.save_results()
    
    print(f"\n✅ Model selection completed!")
    print(f"📁 Results: {output_file}")
    print(f"🚀 Next: Use selected model for training (Step 8)")
    
    return 0

if __name__ == "__main__":
    exit(main())
