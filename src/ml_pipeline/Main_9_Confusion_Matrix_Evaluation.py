#!/usr/bin/env python3
"""
📊 FLAN-T5 Medical Model Evaluation with Confusion Matrix
🎯 Disease classification evaluation with detailed confusion matrix
💡 Evaluates diagnostic accuracy and creates visual confusion matrix
"""

import json
import torch
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict, Counter
import logging
import re

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("⚠️ reportlab not available. Install: pip install reportlab")

# For PNG generation and Confusion Matrix
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.metrics import precision_recall_fscore_support
    plt.style.use('seaborn-v0_8')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("⚠️ matplotlib/sklearn not available. Install: pip install matplotlib seaborn scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLANT5ConfusionMatrixEvaluator:
    """FLAN-T5 model evaluation with confusion matrix for disease classification"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"data/exports/evaluation/flan_t5_confusion_matrix_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        
        # Disease categories for classification
        self.disease_categories = {
            # Common Diseases
            'Infection': ['infection', 'bacterial', 'viral', 'flu', 'cold', 'fever', 'pneumonia'],
            'Cardiovascular': ['heart', 'cardiac', 'blood pressure', 'hypertension', 'heart attack', 'stroke', 'chest pain'],
            'Respiratory': ['asthma', 'copd', 'bronchitis', 'breathing', 'cough', 'shortness of breath'],
            'Neurological': ['headache', 'migraine', 'seizure', 'epilepsy', 'stroke', 'dizziness'],
            'Gastrointestinal': ['stomach', 'nausea', 'vomiting', 'diarrhea', 'abdominal', 'digestive'],
            'Musculoskeletal': ['arthritis', 'joint pain', 'back pain', 'muscle', 'bone', 'osteoporosis'],
            'Endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin', 'blood sugar'],
            'Mental_Health': ['depression', 'anxiety', 'stress', 'panic', 'bipolar', 'ptsd'],
            'Dermatological': ['skin', 'rash', 'burn', 'allergy', 'eczema'],
            'Emergency': ['emergency', 'urgent', 'hospital', 'call 911', 'life threatening', 'unconscious'],
            'Preventive': ['prevention', 'screening', 'vaccine', 'checkup', 'healthy lifestyle'],
            'Other': ['unknown', 'unclear', 'various', 'multiple', 'general']
        }
        
        # Evaluation results
        self.results = {
            'model_info': {},
            'classification_metrics': {},
            'confusion_matrix_data': {},
            'test_results': [],
            'category_performance': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def load_model(self):
        """Load FLAN-T5 model and tokenizer"""
        logger.info(f"🔄 Loading FLAN-T5 model from {self.model_path}...")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            
            self.results['model_info'] = {
                'path': str(self.model_path),
                'model_type': 'FLAN-T5',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'size_mb': model_size_mb,
                'device': str(self.device),
                'pytorch_version': torch.__version__
            }
            
            logger.info(f"✅ Model loaded: {total_params:,} parameters ({model_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def prepare_classification_test_cases(self):
        """Prepare test cases with expected disease categories"""
        test_cases = [
            # Infection Cases
            {'question': 'I have fever, cough, and sore throat', 'expected': 'Infection'},
            {'question': 'What are symptoms of pneumonia?', 'expected': 'Infection'},
            {'question': 'How to treat viral infection?', 'expected': 'Infection'},
            {'question': 'Bacterial vs viral infection differences', 'expected': 'Infection'},
            {'question': 'When to use antibiotics for infection?', 'expected': 'Infection'},
            
            # Cardiovascular Cases
            {'question': 'I have chest pain and shortness of breath', 'expected': 'Cardiovascular'},
            {'question': 'What causes high blood pressure?', 'expected': 'Cardiovascular'},
            {'question': 'Signs of heart attack in women', 'expected': 'Cardiovascular'},
            {'question': 'How to prevent stroke?', 'expected': 'Cardiovascular'},
            {'question': 'Managing hypertension naturally', 'expected': 'Cardiovascular'},
            
            # Respiratory Cases
            {'question': 'How to control asthma attacks?', 'expected': 'Respiratory'},
            {'question': 'What is COPD treatment?', 'expected': 'Respiratory'},
            {'question': 'Chronic cough causes and treatment', 'expected': 'Respiratory'},
            {'question': 'Breathing difficulty at night', 'expected': 'Respiratory'},
            {'question': 'Bronchitis vs pneumonia differences', 'expected': 'Respiratory'},
            
            # Neurological Cases
            {'question': 'Severe headache with stiff neck', 'expected': 'Neurological'},
            {'question': 'What triggers migraine attacks?', 'expected': 'Neurological'},
            {'question': 'How to manage epilepsy seizures?', 'expected': 'Neurological'},
            {'question': 'Dizziness and balance problems', 'expected': 'Neurological'},
            {'question': 'Early signs of stroke recognition', 'expected': 'Neurological'},
            
            # Gastrointestinal Cases
            {'question': 'Persistent nausea and vomiting', 'expected': 'Gastrointestinal'},
            {'question': 'Chronic diarrhea causes', 'expected': 'Gastrointestinal'},
            {'question': 'Stomach pain after eating', 'expected': 'Gastrointestinal'},
            {'question': 'How to treat acid reflux?', 'expected': 'Gastrointestinal'},
            {'question': 'Inflammatory bowel disease symptoms', 'expected': 'Gastrointestinal'},
            
            # Musculoskeletal Cases
            {'question': 'Joint pain and stiffness in morning', 'expected': 'Musculoskeletal'},
            {'question': 'Lower back pain treatment options', 'expected': 'Musculoskeletal'},
            {'question': 'What is rheumatoid arthritis?', 'expected': 'Musculoskeletal'},
            {'question': 'How to prevent osteoporosis?', 'expected': 'Musculoskeletal'},
            {'question': 'Muscle cramps and weakness', 'expected': 'Musculoskeletal'},
            
            # Endocrine Cases
            {'question': 'Type 2 diabetes management', 'expected': 'Endocrine'},
            {'question': 'Thyroid disorder symptoms', 'expected': 'Endocrine'},
            {'question': 'High blood sugar treatment', 'expected': 'Endocrine'},
            {'question': 'Insulin resistance prevention', 'expected': 'Endocrine'},
            {'question': 'Hormonal imbalance signs', 'expected': 'Endocrine'},
            
            # Mental Health Cases
            {'question': 'Clinical depression symptoms', 'expected': 'Mental_Health'},
            {'question': 'How to manage anxiety attacks?', 'expected': 'Mental_Health'},
            {'question': 'PTSD treatment approaches', 'expected': 'Mental_Health'},
            {'question': 'Bipolar disorder management', 'expected': 'Mental_Health'},
            {'question': 'Chronic stress health effects', 'expected': 'Mental_Health'},
            
            # Emergency Cases
            {'question': 'Severe chest pain call 911?', 'expected': 'Emergency'},
            {'question': 'Someone is unconscious what to do?', 'expected': 'Emergency'},
            {'question': 'Signs of life threatening emergency', 'expected': 'Emergency'},
            {'question': 'When to go to hospital urgently?', 'expected': 'Emergency'},
            {'question': 'Severe allergic reaction treatment', 'expected': 'Emergency'},
            
            # Preventive Cases
            {'question': 'Adult vaccination schedule', 'expected': 'Preventive'},
            {'question': 'Cancer screening guidelines', 'expected': 'Preventive'},
            {'question': 'Healthy lifestyle habits', 'expected': 'Preventive'},
            {'question': 'Regular checkup importance', 'expected': 'Preventive'},
            {'question': 'Disease prevention strategies', 'expected': 'Preventive'},
        ]
        
        logger.info(f"📋 Prepared {len(test_cases)} classification test cases")
        return test_cases
    
    def classify_response(self, response):
        """Classify model response into disease category"""
        response_lower = response.lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.disease_categories.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'Other' if no matches
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'Other'
    
    def evaluate_classification(self, question, expected_category):
        """Evaluate single question for classification"""
        # Format question for FLAN-T5
        input_text = f"Diagnose the medical condition: {question}"
        
        # Performance metrics
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Classify response
            predicted_category = self.classify_response(response)
            
            # Calculate metrics
            inference_time = time.time() - start_time
            is_correct = predicted_category == expected_category
            
            result = {
                'question': question,
                'response': response,
                'expected_category': expected_category,
                'predicted_category': predicted_category,
                'is_correct': is_correct,
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"{'✅' if is_correct else '❌'} {expected_category} → {predicted_category}: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                'question': question,
                'response': f"Error: {str(e)}",
                'expected_category': expected_category,
                'predicted_category': 'Other',
                'is_correct': False,
                'inference_time': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_confusion_matrix_evaluation(self):
        """Run classification evaluation and generate confusion matrix"""
        logger.info("🚀 Starting FLAN-T5 Confusion Matrix Evaluation...")
        
        # Load model
        if not self.load_model():
            logger.error("❌ Failed to load model. Aborting evaluation.")
            return False
        
        # Prepare test cases
        test_cases = self.prepare_classification_test_cases()
        logger.info(f"📋 Total test cases: {len(test_cases)}")
        
        # Run evaluation
        results = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"📝 [{i}/{len(test_cases)}] Processing: {test_case['expected']}")
            result = self.evaluate_classification(test_case['question'], test_case['expected'])
            results.append(result)
            self.results['test_results'].append(result)
        
        # Calculate metrics and confusion matrix
        self.calculate_classification_metrics(results)
        
        # Generate reports
        self.generate_confusion_matrix_reports()
        
        logger.info(f"✅ Evaluation completed! Results saved to: {self.output_dir}")
        return True
    
    def calculate_classification_metrics(self, results):
        """Calculate classification metrics and confusion matrix data with detailed TP/FP/TN/FN"""
        # Extract true and predicted labels
        y_true = [r['expected_category'] for r in results]
        y_pred = [r['predicted_category'] for r in results]
        
        # Get unique labels
        labels = sorted(list(set(y_true + y_pred)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        # Calculate detailed TP/FP/TN/FN for each class
        detailed_metrics = {}
        total_samples = len(results)
        
        for i, label in enumerate(labels):
            # True Positives: correctly predicted as this class
            tp = cm[i, i]
            
            # False Positives: incorrectly predicted as this class
            fp = cm[:, i].sum() - tp
            
            # False Negatives: should be this class but predicted as other
            fn = cm[i, :].sum() - tp
            
            # True Negatives: correctly predicted as not this class
            tn = total_samples - tp - fp - fn
            
            # Calculate metrics manually for verification
            manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            manual_f1 = 2 * (manual_precision * manual_recall) / (manual_precision + manual_recall) if (manual_precision + manual_recall) > 0 else 0.0
            manual_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            manual_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            detailed_metrics[label] = {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn), 
                'fn': int(fn),
                'precision': float(manual_precision),
                'recall': float(manual_recall),
                'f1_score': float(manual_f1),
                'specificity': float(manual_specificity),
                'accuracy': float(manual_accuracy),
                'support': int(support[i]),
                'sklearn_precision': float(precision[i]),
                'sklearn_recall': float(recall[i]),
                'sklearn_f1': float(f1[i])
            }
        
        # Calculate macro and micro averages
        macro_precision = np.mean([m['precision'] for m in detailed_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in detailed_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in detailed_metrics.values()])
        
        # Micro averages (weighted by support)
        total_tp = sum([m['tp'] for m in detailed_metrics.values()])
        total_fp = sum([m['fp'] for m in detailed_metrics.values()])
        total_fn = sum([m['fn'] for m in detailed_metrics.values()])
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Store results
        self.results['confusion_matrix_data'] = {
            'labels': labels,
            'confusion_matrix': cm.tolist(),
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        self.results['classification_metrics'] = {
            'overall_accuracy': accuracy,
            'total_predictions': len(results),
            'correct_predictions': sum(r['is_correct'] for r in results),
            'avg_inference_time': np.mean([r['inference_time'] for r in results if r['inference_time'] > 0]),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1),
            'total_tp': int(total_tp),
            'total_fp': int(total_fp),
            'total_tn': int(sum([m['tn'] for m in detailed_metrics.values()])),
            'total_fn': int(total_fn)
        }
        
        # Store detailed per-category metrics
        self.results['category_performance'] = detailed_metrics
        
        logger.info(f"📊 Overall Accuracy: {accuracy:.3f}")
        logger.info(f"📊 Correct Predictions: {sum(r['is_correct'] for r in results)}/{len(results)}")
        logger.info(f"📊 Macro F1-Score: {macro_f1:.3f}")
        logger.info(f"📊 Micro F1-Score: {micro_f1:.3f}")
        logger.info(f"📊 Total TP: {total_tp}, FP: {total_fp}, TN: {sum([m['tn'] for m in detailed_metrics.values()])}, FN: {total_fn}")
    
    def generate_confusion_matrix_reports(self):
        """Generate confusion matrix visualizations and reports"""
        # Save JSON results
        json_file = self.output_dir / "confusion_matrix_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 JSON results saved: {json_file}")
        
        # Generate PNG visualizations
        if MATPLOTLIB_AVAILABLE:
            self.generate_confusion_matrix_plots()
        
        # Generate PDF report
        if REPORTLAB_AVAILABLE:
            self.generate_confusion_matrix_pdf()
    
    def generate_confusion_matrix_plots(self):
        """Generate confusion matrix and related plots"""
        logger.info("📊 Generating confusion matrix plots...")
        
        cm_data = self.results['confusion_matrix_data']
        cm = np.array(cm_data['confusion_matrix'])
        labels = cm_data['labels']
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title('FLAN-T5 Medical Classification Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
        plt.ylabel('True Category', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Proportion of Predictions'})
        plt.title('FLAN-T5 Medical Classification Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
        plt.ylabel('True Category', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance Metrics by Category
        categories = list(self.results['category_performance'].keys())
        precision_scores = [self.results['category_performance'][cat]['precision'] for cat in categories]
        recall_scores = [self.results['category_performance'][cat]['recall'] for cat in categories]
        f1_scores = [self.results['category_performance'][cat]['f1_score'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Disease Category', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance Metrics by Disease Category', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. TP/FP/TN/FN Analysis Chart
        tp_values = [self.results['category_performance'][cat]['tp'] for cat in categories]
        fp_values = [self.results['category_performance'][cat]['fp'] for cat in categories]
        tn_values = [self.results['category_performance'][cat]['tn'] for cat in categories]
        fn_values = [self.results['category_performance'][cat]['fn'] for cat in categories]
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(categories))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, tp_values, width, label='TP (True Positive)', alpha=0.8, color='green')
        bars2 = ax.bar(x - 0.5*width, fp_values, width, label='FP (False Positive)', alpha=0.8, color='red')
        bars3 = ax.bar(x + 0.5*width, tn_values, width, label='TN (True Negative)', alpha=0.8, color='blue')
        bars4 = ax.bar(x + 1.5*width, fn_values, width, label='FN (False Negative)', alpha=0.8, color='orange')
        
        ax.set_xlabel('Disease Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Count', fontweight='bold', fontsize=12)
        ax.set_title('TP/FP/TN/FN Analysis by Disease Category', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_count_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        add_count_labels(bars1)
        add_count_labels(bars2) 
        add_count_labels(bars3)
        add_count_labels(bars4)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "tp_fp_tn_fn_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Enhanced Classification Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall Accuracy Pie Chart with detailed metrics
        accuracy = self.results['classification_metrics']['overall_accuracy']
        correct = self.results['classification_metrics']['correct_predictions']
        total = self.results['classification_metrics']['total_predictions']
        macro_f1 = self.results['classification_metrics']['macro_f1']
        micro_f1 = self.results['classification_metrics']['micro_f1']
        
        ax1.pie([correct, total-correct], labels=['Correct', 'Incorrect'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Overall Accuracy: {accuracy:.1%}\nMacro F1: {macro_f1:.3f} | Micro F1: {micro_f1:.3f}', 
                     fontweight='bold')
        
        # TP vs FP Comparison
        tp_totals = [self.results['category_performance'][cat]['tp'] for cat in categories]
        fp_totals = [self.results['category_performance'][cat]['fp'] for cat in categories]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, tp_totals, width, label='True Positive (TP)', 
                       color='green', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, fp_totals, width, label='False Positive (FP)', 
                       color='red', alpha=0.7)
        
        ax2.set_title('True Positive vs False Positive by Category', fontweight='bold')
        ax2.set_xlabel('Disease Category')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Precision vs Recall Scatter with F1-Score color mapping
        scatter = ax3.scatter(recall_scores, precision_scores, s=150, c=f1_scores, 
                             cmap='RdYlGn', alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax3.set_ylabel('Precision (PPV)', fontweight='bold')
        ax3.set_title('Precision vs Recall (Color = F1-Score)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('F1-Score', fontweight='bold')
        
        # Add category labels
        for i, cat in enumerate(categories):
            ax3.annotate(cat[:6], (recall_scores[i], precision_scores[i]), 
                        xytext=(8, 8), textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # F1-Score vs Support Analysis
        supports = [self.results['category_performance'][cat]['support'] for cat in categories]
        
        bars = ax4.bar(categories, f1_scores, color='gold', alpha=0.8, edgecolor='orange')
        ax4.set_title('F1-Score by Category', fontweight='bold')
        ax4.set_xlabel('Disease Category')
        ax4.set_ylabel('F1-Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add F1 value labels and support info
        for i, (bar, f1_val, support) in enumerate(zip(bars, f1_scores, supports)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'n={support}', ha='center', va='center', fontsize=8, color='darkblue')
        
        # Add mean F1 line
        mean_f1 = np.mean(f1_scores)
        ax4.axhline(y=mean_f1, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean F1: {mean_f1:.3f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "enhanced_classification_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. F1-Score Breakdown Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # F1-Score Components Breakdown
        categories_short = [cat[:8] for cat in categories]  # Shorten for better display
        
        ax1.barh(categories_short, precision_scores, alpha=0.7, label='Precision', color='skyblue')
        ax1.barh(categories_short, recall_scores, alpha=0.7, label='Recall', color='lightcoral', left=0)
        
        # Add F1 scores as text
        for i, (cat, f1_val) in enumerate(zip(categories_short, f1_scores)):
            ax1.text(1.05, i, f'F1: {f1_val:.3f}', va='center', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Score', fontweight='bold')
        ax1.set_title('Precision and Recall by Category', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.3)
        
        # Confusion Matrix Summary (Aggregated TP/FP/TN/FN)
        total_tp = self.results['classification_metrics']['total_tp']
        total_fp = self.results['classification_metrics']['total_fp']
        total_tn = self.results['classification_metrics']['total_tn']
        total_fn = self.results['classification_metrics']['total_fn']
        
        confusion_data = [total_tp, total_fp, total_tn, total_fn]
        confusion_labels = ['True\nPositive', 'False\nPositive', 'True\nNegative', 'False\nNegative']
        colors = ['green', 'red', 'blue', 'orange']
        
        bars = ax2.bar(confusion_labels, confusion_data, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Overall Confusion Matrix Components', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Count', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, confusion_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(confusion_data)*0.02,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add percentage labels
        total_all = sum(confusion_data)
        for bar, value in zip(bars, confusion_data):
            percentage = (value / total_all) * 100 if total_all > 0 else 0
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{percentage:.1f}%', ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "f1_measure_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Confusion matrix plots generated successfully!")
    
    def generate_confusion_matrix_pdf(self):
        """Generate PDF report with confusion matrix analysis"""
        logger.info("📄 Generating confusion matrix PDF report...")
        
        pdf_file = self.output_dir / "FLAN_T5_Confusion_Matrix_Report.pdf"
        doc = SimpleDocTemplate(str(pdf_file), pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkblue
        )
        
        story = []
        
        # Title Page
        story.append(Paragraph("FLAN-T5 Medical AI", title_style))
        story.append(Paragraph("Confusion Matrix Analysis Report", title_style))
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        accuracy = self.results['classification_metrics']['overall_accuracy']
        total_cases = self.results['classification_metrics']['total_predictions']
        correct_cases = self.results['classification_metrics']['correct_predictions']
        
        summary_text = f"""
        This report presents a comprehensive confusion matrix analysis of the FLAN-T5 medical AI model's 
        diagnostic classification performance. The model was evaluated on {total_cases} test cases across 
        {len(self.results['category_performance'])} disease categories.
        
        <b>Key Findings:</b><br/>
        • Overall Classification Accuracy: {accuracy:.1%}<br/>
        • Correct Predictions: {correct_cases}/{total_cases}<br/>
        • Average Inference Time: {self.results['classification_metrics']['avg_inference_time']:.3f} seconds<br/>
        • Best Performing Category: {max(self.results['category_performance'].items(), key=lambda x: x[1]['f1_score'])[0]}<br/>
        • Lowest Performing Category: {min(self.results['category_performance'].items(), key=lambda x: x[1]['f1_score'])[0]}
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Model Information
        story.append(Paragraph("Model Information", heading_style))
        
        model_data = [
            ["Property", "Value"],
            ["Model Type", self.results['model_info']['model_type']],
            ["Model Path", self.results['model_info']['path']],
            ["Total Parameters", f"{self.results['model_info']['total_parameters']:,}"],
            ["Model Size", f"{self.results['model_info']['size_mb']:.1f} MB"],
            ["Device", self.results['model_info']['device']],
            ["PyTorch Version", self.results['model_info']['pytorch_version']]
        ]
        
        model_table = Table(model_data, colWidths=[2*inch, 4*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(model_table)
        story.append(Spacer(1, 20))
        
        # Classification Results Summary
        story.append(Paragraph("Classification Performance Summary", heading_style))
        
        # Enhanced per-category performance table with TP/FP/TN/FN
        perf_data = [["Category", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1-Score", "Specificity", "Support"]]
        
        for category, metrics in self.results['category_performance'].items():
            perf_data.append([
                category,
                str(metrics['tp']),
                str(metrics['fp']),
                str(metrics['tn']),
                str(metrics['fn']),
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['specificity']:.3f}",
                str(metrics['support'])
            ])
        
        perf_table = Table(perf_data, colWidths=[1*inch, 0.4*inch, 0.4*inch, 0.4*inch, 0.4*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.5*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
        
        # F1-Score Analysis Summary
        story.append(Paragraph("F1-Score and Classification Metrics Summary", heading_style))
        
        metrics_summary = f"""
        <b>Overall Performance Metrics:</b><br/>
        • Overall Accuracy: {self.results['classification_metrics']['overall_accuracy']:.3f}<br/>
        • Macro-averaged F1-Score: {self.results['classification_metrics']['macro_f1']:.3f}<br/>
        • Micro-averaged F1-Score: {self.results['classification_metrics']['micro_f1']:.3f}<br/>
        • Macro-averaged Precision: {self.results['classification_metrics']['macro_precision']:.3f}<br/>
        • Macro-averaged Recall: {self.results['classification_metrics']['macro_recall']:.3f}<br/>
        
        <b>Aggregate Confusion Matrix Components:</b><br/>
        • True Positives (TP): {self.results['classification_metrics']['total_tp']}<br/>
        • False Positives (FP): {self.results['classification_metrics']['total_fp']}<br/>
        • True Negatives (TN): {self.results['classification_metrics']['total_tn']}<br/>
        • False Negatives (FN): {self.results['classification_metrics']['total_fn']}<br/>
        
        <b>F1-Score Interpretation:</b><br/>
        • F1-Score is the harmonic mean of Precision and Recall<br/>
        • F1 = 2 × (Precision × Recall) / (Precision + Recall)<br/>
        • Range: 0.0 (worst) to 1.0 (perfect)<br/>
        • Macro F1: Unweighted average across all classes<br/>
        • Micro F1: Weighted by class frequency<br/>
        """
        
        story.append(Paragraph(metrics_summary, styles['Normal']))
        story.append(PageBreak())
        
        # Confusion Matrix Analysis
        story.append(Paragraph("Detailed Confusion Matrix Analysis", heading_style))
        
        cm_analysis = """
        The confusion matrix reveals how well the model distinguishes between different disease categories. 
        Diagonal elements represent correct classifications, while off-diagonal elements show misclassifications.
        
        <b>Key Observations:</b><br/>
        """
        
        # Find most confused categories
        cm = np.array(self.results['confusion_matrix_data']['confusion_matrix'])
        labels = self.results['confusion_matrix_data']['labels']
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i in range(len(labels)):
            if cm[i].sum() > 0:
                accuracy = cm[i, i] / cm[i].sum()
                class_accuracies.append((labels[i], accuracy))
        
        class_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        cm_analysis += f"• Best classified category: {class_accuracies[0][0]} ({class_accuracies[0][1]:.1%})<br/>"
        cm_analysis += f"• Most challenging category: {class_accuracies[-1][0]} ({class_accuracies[-1][1]:.1%})<br/>"
        
        # Find most common misclassification
        max_confusion = 0
        max_confusion_pair = None
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > max_confusion:
                    max_confusion = cm[i, j]
                    max_confusion_pair = (labels[i], labels[j])
        
        if max_confusion_pair:
            cm_analysis += f"• Most common misclassification: {max_confusion_pair[0]} → {max_confusion_pair[1]} ({max_confusion} cases)<br/>"
        
        story.append(Paragraph(cm_analysis, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Sample Predictions
        story.append(Paragraph("Sample Predictions Analysis", heading_style))
        
        # Get correct and incorrect predictions
        correct_preds = [r for r in self.results['test_results'] if r['is_correct']]
        incorrect_preds = [r for r in self.results['test_results'] if not r['is_correct']]
        
        if correct_preds:
            story.append(Paragraph("✅ Correct Classifications (Sample):", styles['Heading3']))
            for i, pred in enumerate(correct_preds[:5], 1):
                story.append(Paragraph(f"<b>{i}. Question:</b> {pred['question']}", styles['Normal']))
                story.append(Paragraph(f"<b>Response:</b> {pred['response'][:100]}...", styles['Normal']))
                story.append(Paragraph(f"<b>Category:</b> {pred['expected_category']} ✅", styles['Normal']))
                story.append(Spacer(1, 8))
        
        if incorrect_preds:
            story.append(Paragraph("❌ Misclassifications (Sample):", styles['Heading3']))
            for i, pred in enumerate(incorrect_preds[:5], 1):
                story.append(Paragraph(f"<b>{i}. Question:</b> {pred['question']}", styles['Normal']))
                story.append(Paragraph(f"<b>Response:</b> {pred['response'][:100]}...", styles['Normal']))
                story.append(Paragraph(f"<b>Expected:</b> {pred['expected_category']} | <b>Predicted:</b> {pred['predicted_category']} ❌", styles['Normal']))
                story.append(Spacer(1, 8))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"✅ Confusion matrix PDF report generated: {pdf_file}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FLAN-T5 Medical Model Confusion Matrix Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to FLAN-T5 model directory")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FLANT5ConfusionMatrixEvaluator(args.model_path)
    
    # Run evaluation
    success = evaluator.run_confusion_matrix_evaluation()
    
    if success:
        print("✅ Confusion Matrix Evaluation completed successfully!")
        print(f"📁 Results saved to: {evaluator.output_dir}")
        print("📊 Generated files:")
        print(f"  - confusion_matrix_results.json")
        print(f"  - FLAN_T5_Confusion_Matrix_Report.pdf")
        print(f"  - confusion_matrix.png")
        print(f"  - confusion_matrix_normalized.png") 
        print(f"  - performance_by_category.png")
        print(f"  - classification_dashboard.png")
    else:
        print("❌ Evaluation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())