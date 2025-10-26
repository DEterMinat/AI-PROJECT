#!/usr/bin/env python3
"""
📊 FLAN-T5 Medical Model Evaluation Suite
🎯 Comprehensive testing with PDF & PNG reports
💡 Evaluates FLAN-T5 quality, performance, and medical accuracy
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

# For PNG generation
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("⚠️ matplotlib not available. Install: pip install matplotlib seaborn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLANT5MedicalEvaluator:
    """Comprehensive FLAN-T5 model evaluation with visual reports"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"data/exports/evaluation/flan_t5_evaluation_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        
        # Evaluation results
        self.results = {
            'model_info': {},
            'performance_metrics': {},
            'quality_metrics': {},
            'test_results': [],
            'error_analysis': {},
            'category_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Emergency keywords for detection
        self.emergency_keywords = [
            'chest pain', 'heart attack', 'stroke', 'choking', 'bleeding',
            'unconscious', 'breathing difficulty', 'severe pain', 'emergency',
            'call 911', 'hospital', 'urgent', 'critical', 'life threatening'
        ]
    
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
    
    def prepare_test_cases(self):
        """Prepare comprehensive medical test cases"""
        test_cases = [
            # Basic Symptoms (15 cases)
            {
                'category': 'Basic Symptoms',
                'questions': [
                    "What are the common symptoms of fever and how to treat it?",
                    "How to treat a persistent headache naturally?",
                    "What causes chronic cough and when to see a doctor?",
                    "When should I worry about nausea and vomiting?",
                    "How to stop diarrhea quickly and safely?",
                    "What helps with sore throat and swollen glands?",
                    "How long does flu typically last and recovery tips?",
                    "What causes extreme fatigue and weakness?",
                    "When to see doctor for lower back pain?",
                    "How to reduce inflammation and swelling naturally?",
                    "What are signs of dehydration?",
                    "How to treat muscle cramps?",
                    "What causes dizziness and vertigo?",
                    "When is stomach pain serious?",
                    "How to manage joint pain?"
                ]
            },
            # Chronic Diseases (15 cases)
            {
                'category': 'Chronic Diseases',
                'questions': [
                    "What is Type 2 diabetes and how to manage it?",
                    "How to control high blood pressure naturally?",
                    "What are early signs of heart disease?",
                    "How to prevent stroke in high-risk patients?",
                    "What is the best arthritis treatment approach?",
                    "How to control asthma attacks effectively?",
                    "What causes chronic kidney disease progression?",
                    "How to manage COPD and improve breathing?",
                    "What is hypothyroidism and treatment options?",
                    "How to treat and prevent osteoporosis?",
                    "What is inflammatory bowel disease?",
                    "How to manage chronic migraines?",
                    "What causes fibromyalgia pain?",
                    "How to control epilepsy seizures?",
                    "What is multiple sclerosis treatment?"
                ]
            },
            # Emergency Situations (15 cases)
            {
                'category': 'Emergency',
                'questions': [
                    "What to do immediately for chest pain?",
                    "How to recognize stroke symptoms quickly?",
                    "What are heart attack warning signs?",
                    "When to call 911 for breathing problems?",
                    "How to help someone who is choking?",
                    "What to do for severe bleeding control?",
                    "Signs of severe allergic reaction?",
                    "How to treat second-degree burns?",
                    "What to do for serious head injury?",
                    "Signs of drug poisoning emergency?",
                    "How to handle severe asthma attack?",
                    "What to do for broken bone injury?",
                    "Signs of internal bleeding?",
                    "How to help unconscious person?",
                    "What to do for seizure emergency?"
                ]
            },
            # Preventive Care (15 cases)
            {
                'category': 'Preventive Care',
                'questions': [
                    "How often should adults get health checkups?",
                    "What vaccines are recommended for adults?",
                    "How to maintain healthy weight long-term?",
                    "What is considered normal blood pressure?",
                    "How much exercise is needed weekly?",
                    "What constitutes a heart-healthy diet?",
                    "How to effectively reduce daily stress?",
                    "When should cancer screening begin?",
                    "How to prevent Type 2 diabetes?",
                    "What are healthy cholesterol levels?",
                    "How to maintain bone health?",
                    "What supplements are beneficial?",
                    "How to prevent heart disease?",
                    "When to get eye exams?",
                    "How to maintain mental wellness?"
                ]
            },
            # Mental Health (15 cases)
            {
                'category': 'Mental Health',
                'questions': [
                    "What are clinical depression symptoms?",
                    "How to manage anxiety disorders effectively?",
                    "What happens during a panic attack?",
                    "How to improve sleep quality naturally?",
                    "What are major stress causes?",
                    "How to cope with grief and loss?",
                    "What is PTSD and treatment options?",
                    "How to recover from job burnout?",
                    "What is bipolar disorder management?",
                    "When to seek professional mental help?",
                    "How to deal with seasonal depression?",
                    "What are eating disorder signs?",
                    "How to manage OCD symptoms?",
                    "What causes social anxiety?",
                    "How to support someone with depression?"
                ]
            }
        ]
        
        return test_cases
    
    def evaluate_single_question(self, question, category):
        """Evaluate single question with FLAN-T5"""
        # Format question for FLAN-T5
        input_text = f"Answer the medical question: {question}"
        
        # Performance metrics
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metrics
            inference_time = time.time() - start_time
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0])
            
            # Quality assessment
            quality_score = self.assess_response_quality(question, response, category)
            emergency_detected = self.detect_emergency(response)
            
            result = {
                'question': question,
                'response': response,
                'category': category,
                'inference_time': inference_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'quality_score': quality_score,
                'emergency_detected': emergency_detected,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Processed: {question[:50]}... (Quality: {quality_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                'question': question,
                'response': f"Error: {str(e)}",
                'category': category,
                'inference_time': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'quality_score': 0.0,
                'emergency_detected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def assess_response_quality(self, question, response, category):
        """Assess response quality (0-1 score)"""
        score = 0.0
        
        # Basic response checks
        if len(response.strip()) > 10:
            score += 0.2
        
        # Medical relevance keywords
        medical_keywords = [
            'symptom', 'treatment', 'doctor', 'medical', 'health', 'disease',
            'condition', 'diagnosis', 'therapy', 'medication', 'prevention'
        ]
        
        response_lower = response.lower()
        keyword_matches = sum(1 for kw in medical_keywords if kw in response_lower)
        score += min(keyword_matches / len(medical_keywords), 0.3)
        
        # Category-specific scoring
        if category == 'Emergency':
            emergency_words = ['urgent', 'emergency', 'hospital', 'call', '911', 'immediately']
            emergency_matches = sum(1 for ew in emergency_words if ew in response_lower)
            score += min(emergency_matches / 3, 0.3)
        elif category == 'Preventive Care':
            prevention_words = ['prevent', 'healthy', 'diet', 'exercise', 'lifestyle']
            prevention_matches = sum(1 for pw in prevention_words if pw in response_lower)
            score += min(prevention_matches / 3, 0.3)
        else:
            score += 0.2  # Base score for other categories
        
        # Length and completeness
        if 50 <= len(response) <= 300:
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_emergency(self, response):
        """Detect if response indicates emergency"""
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in self.emergency_keywords)
    
    def run_evaluation(self):
        """Run complete evaluation"""
        logger.info("🚀 Starting FLAN-T5 Medical Model Evaluation...")
        
        # Load model
        if not self.load_model():
            logger.error("❌ Failed to load model. Aborting evaluation.")
            return False
        
        # Prepare test cases
        test_cases = self.prepare_test_cases()
        total_questions = sum(len(tc['questions']) for tc in test_cases)
        
        logger.info(f"📋 Total test cases: {total_questions} across {len(test_cases)} categories")
        
        # Run evaluation
        question_count = 0
        category_results = defaultdict(list)
        
        for test_category in test_cases:
            category = test_category['category']
            questions = test_category['questions']
            
            logger.info(f"🔄 Processing category: {category} ({len(questions)} questions)")
            
            for question in questions:
                question_count += 1
                logger.info(f"📝 [{question_count}/{total_questions}] {category}")
                
                result = self.evaluate_single_question(question, category)
                self.results['test_results'].append(result)
                category_results[category].append(result)
        
        # Calculate aggregate metrics
        self.calculate_metrics(category_results)
        
        # Generate reports
        self.generate_reports()
        
        logger.info(f"✅ Evaluation completed! Results saved to: {self.output_dir}")
        return True
    
    def calculate_metrics(self, category_results):
        """Calculate performance and quality metrics"""
        all_results = self.results['test_results']
        
        # Performance metrics
        inference_times = [r['inference_time'] for r in all_results if r['inference_time'] > 0]
        quality_scores = [r['quality_score'] for r in all_results]
        
        self.results['performance_metrics'] = {
            'total_questions': len(all_results),
            'successful_responses': len([r for r in all_results if 'error' not in r]),
            'failed_responses': len([r for r in all_results if 'error' in r]),
            'success_rate': len([r for r in all_results if 'error' not in r]) / len(all_results),
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'median_inference_time': np.median(inference_times) if inference_times else 0,
            'min_inference_time': np.min(inference_times) if inference_times else 0,
            'max_inference_time': np.max(inference_times) if inference_times else 0
        }
        
        # Quality metrics
        self.results['quality_metrics'] = {
            'avg_quality_score': np.mean(quality_scores),
            'median_quality_score': np.median(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'quality_std': np.std(quality_scores)
        }
        
        # Category-specific metrics
        for category, results in category_results.items():
            cat_quality_scores = [r['quality_score'] for r in results]
            cat_inference_times = [r['inference_time'] for r in results if r['inference_time'] > 0]
            
            self.results['category_scores'][category] = {
                'questions_count': len(results),
                'avg_quality_score': np.mean(cat_quality_scores),
                'avg_inference_time': np.mean(cat_inference_times) if cat_inference_times else 0,
                'emergency_detected': len([r for r in results if r['emergency_detected']]),
                'success_rate': len([r for r in results if 'error' not in r]) / len(results)
            }
        
        # Error analysis
        errors = [r for r in all_results if 'error' in r]
        error_types = Counter([r['error'] for r in errors])
        
        self.results['error_analysis'] = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(all_results),
            'error_types': dict(error_types)
        }
        
        logger.info(f"📊 Calculated metrics for {len(all_results)} test cases")
    
    def generate_reports(self):
        """Generate PNG charts and PDF report"""
        # Save JSON results
        json_file = self.output_dir / "evaluation_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 JSON results saved: {json_file}")
        
        # Generate PNG charts
        if MATPLOTLIB_AVAILABLE:
            self.generate_png_charts()
        
        # Generate PDF report
        if REPORTLAB_AVAILABLE:
            self.generate_pdf_report()
    
    def generate_png_charts(self):
        """Generate PNG visualization charts"""
        logger.info("📊 Generating PNG charts...")
        
        # 1. Quality Scores by Category
        plt.figure(figsize=(12, 8))
        categories = list(self.results['category_scores'].keys())
        quality_scores = [self.results['category_scores'][cat]['avg_quality_score'] for cat in categories]
        
        bars = plt.bar(categories, quality_scores, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Average Quality Score by Category', fontsize=16, fontweight='bold')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Quality Score (0-1)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Inference Time Distribution
        plt.figure(figsize=(10, 6))
        inference_times = [r['inference_time'] for r in self.results['test_results'] if r['inference_time'] > 0]
        
        plt.hist(inference_times, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
        plt.title('Inference Time Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Inference Time (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(np.mean(inference_times), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(inference_times):.2f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "inference_time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance Summary Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success Rate Pie Chart
        success_rate = self.results['performance_metrics']['success_rate']
        ax1.pie([success_rate, 1-success_rate], labels=['Success', 'Failed'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Success Rate', fontweight='bold')
        
        # Quality Score Distribution
        quality_scores = [r['quality_score'] for r in self.results['test_results']]
        ax2.hist(quality_scores, bins=15, color='gold', edgecolor='orange', alpha=0.7)
        ax2.set_title('Quality Score Distribution', fontweight='bold')
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Frequency')
        
        # Response Length Distribution
        response_lengths = [len(r['response']) for r in self.results['test_results']]
        ax3.hist(response_lengths, bins=20, color='mediumpurple', edgecolor='purple', alpha=0.7)
        ax3.set_title('Response Length Distribution', fontweight='bold')
        ax3.set_xlabel('Response Length (characters)')
        ax3.set_ylabel('Frequency')
        
        # Category Performance Comparison
        categories = list(self.results['category_scores'].keys())
        cat_scores = [self.results['category_scores'][cat]['avg_quality_score'] for cat in categories]
        cat_times = [self.results['category_scores'][cat]['avg_inference_time'] for cat in categories]
        
        ax4.scatter(cat_times, cat_scores, s=100, c=range(len(categories)), 
                   cmap='viridis', alpha=0.7, edgecolor='black')
        ax4.set_title('Quality vs Speed by Category', fontweight='bold')
        ax4.set_xlabel('Average Inference Time (s)')
        ax4.set_ylabel('Average Quality Score')
        
        # Add category labels
        for i, cat in enumerate(categories):
            ax4.annotate(cat[:10], (cat_times[i], cat_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ PNG charts generated successfully!")
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        logger.info("📄 Generating PDF report...")
        
        pdf_file = self.output_dir / "FLAN_T5_Evaluation_Report.pdf"
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
        story.append(Paragraph("FLAN-T5 Medical Model", title_style))
        story.append(Paragraph("Comprehensive Evaluation Report", title_style))
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
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
        
        # Performance Metrics
        story.append(Paragraph("Performance Metrics", heading_style))
        
        perf_metrics = self.results['performance_metrics']
        perf_data = [
            ["Metric", "Value"],
            ["Total Questions", str(perf_metrics['total_questions'])],
            ["Successful Responses", str(perf_metrics['successful_responses'])],
            ["Failed Responses", str(perf_metrics['failed_responses'])],
            ["Success Rate", f"{perf_metrics['success_rate']:.2%}"],
            ["Avg Inference Time", f"{perf_metrics['avg_inference_time']:.3f} seconds"],
            ["Median Inference Time", f"{perf_metrics['median_inference_time']:.3f} seconds"],
            ["Min Inference Time", f"{perf_metrics['min_inference_time']:.3f} seconds"],
            ["Max Inference Time", f"{perf_metrics['max_inference_time']:.3f} seconds"]
        ]
        
        perf_table = Table(perf_data, colWidths=[3*inch, 2*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
        
        # Quality Metrics
        story.append(Paragraph("Quality Metrics", heading_style))
        
        quality_metrics = self.results['quality_metrics']
        quality_data = [
            ["Metric", "Value"],
            ["Average Quality Score", f"{quality_metrics['avg_quality_score']:.3f}"],
            ["Median Quality Score", f"{quality_metrics['median_quality_score']:.3f}"],
            ["Min Quality Score", f"{quality_metrics['min_quality_score']:.3f}"],
            ["Max Quality Score", f"{quality_metrics['max_quality_score']:.3f}"],
            ["Quality Std Deviation", f"{quality_metrics['quality_std']:.3f}"]
        ]
        
        quality_table = Table(quality_data, colWidths=[3*inch, 2*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(quality_table)
        story.append(PageBreak())
        
        # Category Performance
        story.append(Paragraph("Performance by Category", heading_style))
        
        cat_data = [["Category", "Questions", "Avg Quality", "Avg Time (s)", "Emergency Detected", "Success Rate"]]
        
        for category, metrics in self.results['category_scores'].items():
            cat_data.append([
                category,
                str(metrics['questions_count']),
                f"{metrics['avg_quality_score']:.3f}",
                f"{metrics['avg_inference_time']:.3f}",
                str(metrics['emergency_detected']),
                f"{metrics['success_rate']:.2%}"
            ])
        
        cat_table = Table(cat_data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(cat_table)
        story.append(Spacer(1, 20))
        
        # Sample Responses
        story.append(Paragraph("Sample Responses", heading_style))
        
        # Get best and worst responses
        sorted_results = sorted(self.results['test_results'], key=lambda x: x['quality_score'], reverse=True)
        best_responses = sorted_results[:3]
        worst_responses = sorted_results[-3:]
        
        story.append(Paragraph("Best Quality Responses:", styles['Heading3']))
        for i, result in enumerate(best_responses, 1):
            story.append(Paragraph(f"<b>{i}. Question:</b> {result['question']}", styles['Normal']))
            story.append(Paragraph(f"<b>Response:</b> {result['response'][:200]}...", styles['Normal']))
            story.append(Paragraph(f"<b>Quality Score:</b> {result['quality_score']:.3f} | <b>Time:</b> {result['inference_time']:.3f}s", styles['Normal']))
            story.append(Spacer(1, 10))
        
        story.append(Paragraph("Lowest Quality Responses:", styles['Heading3']))
        for i, result in enumerate(worst_responses, 1):
            story.append(Paragraph(f"<b>{i}. Question:</b> {result['question']}", styles['Normal']))
            story.append(Paragraph(f"<b>Response:</b> {result['response'][:200]}...", styles['Normal']))
            story.append(Paragraph(f"<b>Quality Score:</b> {result['quality_score']:.3f} | <b>Time:</b> {result['inference_time']:.3f}s", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"✅ PDF report generated: {pdf_file}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FLAN-T5 Medical Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to FLAN-T5 model directory")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FLANT5MedicalEvaluator(args.model_path)
    
    # Run evaluation
    success = evaluator.run_evaluation()
    
    if success:
        print("✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {evaluator.output_dir}")
        print("📊 Generated files:")
        print(f"  - evaluation_results.json")
        print(f"  - FLAN_T5_Evaluation_Report.pdf")
        print(f"  - quality_by_category.png")
        print(f"  - inference_time_distribution.png")
        print(f"  - performance_dashboard.png")
    else:
        print("❌ Evaluation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())