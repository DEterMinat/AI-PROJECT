#!/usr/bin/env python3
"""
📊 Medical Model Evaluation Suite
🎯 Comprehensive testing with PDF & PNG reports
💡 Evaluates model quality, performance, and medical accuracy
"""

import json
import torch
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import logging

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("⚠️ reportlab not available. Install: pip install reportlab")

# For PNG generation
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("⚠️ matplotlib not available. Install: pip install matplotlib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalModelEvaluator:
    """Comprehensive model evaluation with visual reports"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"data/exports/evaluation/evaluation_{timestamp}")
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
            'timestamp': datetime.now().isoformat()
        }
    
    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"🔄 Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        self.results['model_info'] = {
            'path': str(self.model_path),
            'parameters': total_params,
            'size_mb': model_size_mb,
            'device': str(self.device)
        }
        
        logger.info(f"✅ Model loaded: {total_params:,} parameters ({model_size_mb:.1f} MB)")
    
    def prepare_test_cases(self):
        """Prepare comprehensive test cases"""
        test_cases = [
            # Basic Symptoms (10 cases)
            {
                'category': 'Basic Symptoms',
                'questions': [
                    "What are the symptoms of fever?",
                    "How to treat a headache?",
                    "What causes cough?",
                    "When to worry about nausea?",
                    "How to stop diarrhea?",
                    "What helps with sore throat?",
                    "How long does flu last?",
                    "What causes fatigue?",
                    "When to see doctor for back pain?",
                    "How to reduce inflammation?"
                ]
            },
            # Chronic Diseases (10 cases)
            {
                'category': 'Chronic Diseases',
                'questions': [
                    "What is diabetes?",
                    "How to manage high blood pressure?",
                    "What are signs of heart disease?",
                    "How to prevent stroke?",
                    "What is arthritis treatment?",
                    "How to control asthma?",
                    "What causes kidney disease?",
                    "How to manage COPD?",
                    "What is thyroid disorder?",
                    "How to treat osteoporosis?"
                ]
            },
            # Emergency Situations (10 cases)
            {
                'category': 'Emergency',
                'questions': [
                    "What to do for chest pain?",
                    "How to recognize stroke?",
                    "What are signs of heart attack?",
                    "When to call 911 for breathing?",
                    "How to help choking person?",
                    "What to do for severe bleeding?",
                    "Signs of allergic reaction?",
                    "How to treat burn injury?",
                    "What to do for head injury?",
                    "Signs of poisoning emergency?"
                ]
            },
            # Preventive Care (10 cases)
            {
                'category': 'Preventive Care',
                'questions': [
                    "How often to get checkup?",
                    "What vaccines are needed?",
                    "How to maintain healthy weight?",
                    "What is good blood pressure?",
                    "How much exercise needed?",
                    "What is healthy diet?",
                    "How to reduce stress?",
                    "When to screen for cancer?",
                    "How to prevent diabetes?",
                    "What is good cholesterol level?"
                ]
            },
            # Mental Health (10 cases)
            {
                'category': 'Mental Health',
                'questions': [
                    "What are signs of depression?",
                    "How to manage anxiety?",
                    "What is panic attack?",
                    "How to improve sleep?",
                    "What causes stress?",
                    "How to cope with grief?",
                    "What is PTSD?",
                    "How to help with burnout?",
                    "What is bipolar disorder?",
                    "When to seek mental help?"
                ]
            }
        ]
        
        return test_cases
    
    def evaluate_single_question(self, question, max_retries=3):
        """Evaluate single question with retries"""
        input_text = f"Human: {question}\nAssistant:"
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        # Performance metrics
        start_time = time.time()
        
        try:
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
            
            generation_time = time.time() - start_time
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(input_text, "").strip()
            
            # Clean up
            if answer.endswith("Human:"):
                answer = answer[:-6].strip()
            
            # Quality checks
            quality_score = self.calculate_quality_score(question, answer)
            
            return {
                'question': question,
                'answer': answer,
                'generation_time': generation_time,
                'answer_length': len(answer),
                'quality_score': quality_score,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {
                'question': question,
                'answer': f"ERROR: {str(e)}",
                'generation_time': 0,
                'answer_length': 0,
                'quality_score': 0,
                'success': False
            }
    
    def calculate_quality_score(self, question, answer):
        """Calculate answer quality score (0-100)"""
        score = 0
        
        # Length check (0-20 points)
        if len(answer) >= 50:
            score += 20
        elif len(answer) >= 30:
            score += 15
        elif len(answer) >= 20:
            score += 10
        
        # Medical relevance (0-20 points)
        medical_terms = ['symptom', 'treatment', 'doctor', 'health', 'medical', 'patient', 
                        'disease', 'condition', 'diagnosis', 'medication', 'therapy', 'care']
        medical_count = sum(1 for term in medical_terms if term.lower() in answer.lower())
        score += min(20, medical_count * 5)
        
        # Completeness (0-20 points)
        if not answer.lower().startswith('error'):
            score += 10
        if len(answer.split('.')) >= 2:  # Multiple sentences
            score += 10
        
        # Professionalism (0-20 points)
        if not any(word in answer.lower() for word in ['stupid', 'dumb', 'idiot']):
            score += 10
        if any(word in answer.lower() for word in ['please', 'should', 'recommend', 'important']):
            score += 10
        
        # Relevance to question (0-20 points)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words)
        score += min(20, overlap * 3)
        
        return min(100, score)
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation"""
        logger.info("🧪 Starting comprehensive evaluation...")
        
        test_cases = self.prepare_test_cases()
        all_results = []
        category_results = defaultdict(list)
        
        total_questions = sum(len(cat['questions']) for cat in test_cases)
        logger.info(f"📊 Total questions: {total_questions}")
        
        question_num = 0
        for test_category in test_cases:
            category = test_category['category']
            logger.info(f"\n📂 Category: {category}")
            
            for question in test_category['questions']:
                question_num += 1
                logger.info(f"  [{question_num}/{total_questions}] Testing...")
                
                result = self.evaluate_single_question(question)
                result['category'] = category
                
                all_results.append(result)
                category_results[category].append(result)
                
                # Log result
                status = "✅" if result['success'] else "❌"
                logger.info(f"    {status} Q: {question}")
                logger.info(f"       A: {result['answer'][:100]}...")
                logger.info(f"       Quality: {result['quality_score']}/100, Time: {result['generation_time']:.2f}s")
        
        self.results['test_results'] = all_results
        
        # Calculate metrics
        self.calculate_metrics(all_results, category_results)
        
        logger.info("✅ Evaluation completed!")
    
    def calculate_metrics(self, all_results, category_results):
        """Calculate performance and quality metrics"""
        logger.info("\n📊 Calculating metrics...")
        
        successful_results = [r for r in all_results if r['success']]
        
        if not successful_results:
            logger.error("❌ No successful results!")
            return
        
        # Performance metrics
        generation_times = [r['generation_time'] for r in successful_results]
        answer_lengths = [r['answer_length'] for r in successful_results]
        quality_scores = [r['quality_score'] for r in successful_results]
        
        self.results['performance_metrics'] = {
            'total_questions': len(all_results),
            'successful_answers': len(successful_results),
            'success_rate': len(successful_results) / len(all_results) * 100,
            'avg_generation_time': np.mean(generation_times),
            'min_generation_time': np.min(generation_times),
            'max_generation_time': np.max(generation_times),
            'std_generation_time': np.std(generation_times)
        }
        
        # Quality metrics
        self.results['quality_metrics'] = {
            'avg_answer_length': np.mean(answer_lengths),
            'min_answer_length': np.min(answer_lengths),
            'max_answer_length': np.max(answer_lengths),
            'avg_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'std_quality_score': np.std(quality_scores)
        }
        
        # Category breakdown
        category_metrics = {}
        for category, results in category_results.items():
            successful = [r for r in results if r['success']]
            if successful:
                category_metrics[category] = {
                    'total': len(results),
                    'successful': len(successful),
                    'success_rate': len(successful) / len(results) * 100,
                    'avg_quality': np.mean([r['quality_score'] for r in successful]),
                    'avg_time': np.mean([r['generation_time'] for r in successful])
                }
        
        self.results['category_metrics'] = category_metrics
        
        logger.info("✅ Metrics calculated!")
    
    def generate_png_charts(self):
        """Generate PNG charts"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("⚠️ Matplotlib not available, skipping PNG generation")
            return []
        
        logger.info("🎨 Generating PNG charts...")
        
        png_files = []
        
        # Chart 1: Quality Scores by Category
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            categories = list(self.results['category_metrics'].keys())
            quality_scores = [self.results['category_metrics'][cat]['avg_quality'] for cat in categories]
            
            bars = ax.bar(categories, quality_scores, color='skyblue', edgecolor='navy')
            ax.set_xlabel('Category', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Quality Score', fontsize=12, fontweight='bold')
            ax.set_title('Model Quality Scores by Category', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart1_file = self.output_dir / "quality_by_category.png"
            plt.savefig(chart1_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            png_files.append(chart1_file)
            logger.info(f"  ✅ Saved: {chart1_file.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Chart 1 error: {e}")
        
        # Chart 2: Generation Time Distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            times = [r['generation_time'] for r in self.results['test_results'] if r['success']]
            
            ax.hist(times, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            ax.set_xlabel('Generation Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Response Generation Time Distribution', fontsize=14, fontweight='bold')
            ax.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.2f}s')
            ax.legend()
            
            plt.tight_layout()
            
            chart2_file = self.output_dir / "generation_time_distribution.png"
            plt.savefig(chart2_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            png_files.append(chart2_file)
            logger.info(f"  ✅ Saved: {chart2_file.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Chart 2 error: {e}")
        
        # Chart 3: Success Rate by Category
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            categories = list(self.results['category_metrics'].keys())
            success_rates = [self.results['category_metrics'][cat]['success_rate'] for cat in categories]
            
            bars = ax.bar(categories, success_rates, color='lightcoral', edgecolor='darkred')
            ax.set_xlabel('Category', fontsize=12, fontweight='bold')
            ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
            ax.set_title('Model Success Rate by Category', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 100)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart3_file = self.output_dir / "success_rate_by_category.png"
            plt.savefig(chart3_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            png_files.append(chart3_file)
            logger.info(f"  ✅ Saved: {chart3_file.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Chart 3 error: {e}")
        
        # Chart 4: Overall Performance Summary
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Quality distribution
            qualities = [r['quality_score'] for r in self.results['test_results'] if r['success']]
            ax1.hist(qualities, bins=15, color='purple', alpha=0.7, edgecolor='black')
            ax1.set_title('Quality Score Distribution', fontweight='bold')
            ax1.set_xlabel('Quality Score')
            ax1.set_ylabel('Frequency')
            ax1.axvline(np.mean(qualities), color='red', linestyle='--', label=f'Mean: {np.mean(qualities):.1f}')
            ax1.legend()
            
            # Answer length distribution
            lengths = [r['answer_length'] for r in self.results['test_results'] if r['success']]
            ax2.hist(lengths, bins=15, color='orange', alpha=0.7, edgecolor='black')
            ax2.set_title('Answer Length Distribution', fontweight='bold')
            ax2.set_xlabel('Answer Length (characters)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
            ax2.legend()
            
            # Success vs Failure
            success_count = sum(1 for r in self.results['test_results'] if r['success'])
            failure_count = len(self.results['test_results']) - success_count
            ax3.pie([success_count, failure_count], labels=['Success', 'Failure'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
            ax3.set_title('Overall Success Rate', fontweight='bold')
            
            # Average metrics by category
            categories = list(self.results['category_metrics'].keys())
            avg_times = [self.results['category_metrics'][cat]['avg_time'] for cat in categories]
            ax4.barh(categories, avg_times, color='steelblue')
            ax4.set_xlabel('Average Time (seconds)')
            ax4.set_title('Average Response Time by Category', fontweight='bold')
            
            plt.tight_layout()
            
            chart4_file = self.output_dir / "performance_summary.png"
            plt.savefig(chart4_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            png_files.append(chart4_file)
            logger.info(f"  ✅ Saved: {chart4_file.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Chart 4 error: {e}")
        
        logger.info(f"✅ Generated {len(png_files)} PNG charts")
        return png_files
    
    def generate_pdf_report(self, png_files):
        """Generate comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            logger.warning("⚠️ ReportLab not available, skipping PDF generation")
            return None
        
        logger.info("📄 Generating PDF report...")
        
        pdf_file = self.output_dir / "evaluation_report.pdf"
        doc = SimpleDocTemplate(str(pdf_file), pagesize=letter,
                               topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Title
        story.append(Paragraph("Medical AI Model Evaluation Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Timestamp
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Model Information
        story.append(Paragraph("1. Model Information", heading_style))
        model_data = [
            ['Parameter', 'Value'],
            ['Model Path', str(self.results['model_info']['path'])],
            ['Parameters', f"{self.results['model_info']['parameters']:,}"],
            ['Size', f"{self.results['model_info']['size_mb']:.1f} MB"],
            ['Device', self.results['model_info']['device']]
        ]
        
        model_table = Table(model_data, colWidths=[2*inch, 4*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Performance Metrics
        story.append(Paragraph("2. Performance Metrics", heading_style))
        perf = self.results['performance_metrics']
        perf_data = [
            ['Metric', 'Value'],
            ['Total Questions', str(perf['total_questions'])],
            ['Successful Answers', str(perf['successful_answers'])],
            ['Success Rate', f"{perf['success_rate']:.2f}%"],
            ['Avg Generation Time', f"{perf['avg_generation_time']:.3f}s"],
            ['Min Generation Time', f"{perf['min_generation_time']:.3f}s"],
            ['Max Generation Time', f"{perf['max_generation_time']:.3f}s"]
        ]
        
        perf_table = Table(perf_data, colWidths=[3*inch, 3*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Quality Metrics
        story.append(Paragraph("3. Quality Metrics", heading_style))
        qual = self.results['quality_metrics']
        qual_data = [
            ['Metric', 'Value'],
            ['Avg Answer Length', f"{qual['avg_answer_length']:.1f} chars"],
            ['Min Answer Length', f"{qual['min_answer_length']} chars"],
            ['Max Answer Length', f"{qual['max_answer_length']} chars"],
            ['Avg Quality Score', f"{qual['avg_quality_score']:.2f}/100"],
            ['Min Quality Score', f"{qual['min_quality_score']:.2f}/100"],
            ['Max Quality Score', f"{qual['max_quality_score']:.2f}/100"]
        ]
        
        qual_table = Table(qual_data, colWidths=[3*inch, 3*inch])
        qual_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(qual_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Category Performance
        story.append(Paragraph("4. Performance by Category", heading_style))
        cat_data = [['Category', 'Success Rate', 'Avg Quality', 'Avg Time']]
        
        for cat, metrics in self.results['category_metrics'].items():
            cat_data.append([
                cat,
                f"{metrics['success_rate']:.1f}%",
                f"{metrics['avg_quality']:.1f}/100",
                f"{metrics['avg_time']:.2f}s"
            ])
        
        cat_table = Table(cat_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lavender, colors.lightgrey])
        ]))
        story.append(cat_table)
        story.append(PageBreak())
        
        # Add PNG charts
        if png_files:
            story.append(Paragraph("5. Visual Analysis", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            for png_file in png_files:
                try:
                    img = Image(str(png_file), width=6.5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
                except Exception as e:
                    logger.error(f"❌ Could not add image {png_file.name}: {e}")
        
        # Sample Results
        story.append(PageBreak())
        story.append(Paragraph("6. Sample Test Results", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Show first 5 successful results
        successful_results = [r for r in self.results['test_results'] if r['success']][:5]
        
        for i, result in enumerate(successful_results, 1):
            story.append(Paragraph(f"<b>Test {i}:</b>", styles['Normal']))
            story.append(Paragraph(f"<b>Q:</b> {result['question']}", styles['Normal']))
            story.append(Paragraph(f"<b>A:</b> {result['answer'][:200]}...", styles['Normal']))
            story.append(Paragraph(f"<b>Quality:</b> {result['quality_score']:.1f}/100 | "
                                 f"<b>Time:</b> {result['generation_time']:.2f}s", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"✅ PDF report saved: {pdf_file}")
        return pdf_file
    
    def save_json_results(self):
        """Save raw results as JSON"""
        import numpy as np
        
        json_file = self.output_dir / "evaluation_results.json"
        
        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 JSON results saved: {json_file}")
        return json_file
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📊 MEDICAL MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n🤖 Model: {self.results['model_info']['path']}")
        print(f"   Parameters: {self.results['model_info']['parameters']:,}")
        print(f"   Size: {self.results['model_info']['size_mb']:.1f} MB")
        
        perf = self.results['performance_metrics']
        print(f"\n⚡ Performance:")
        print(f"   Total Questions: {perf['total_questions']}")
        print(f"   Success Rate: {perf['success_rate']:.1f}%")
        print(f"   Avg Time: {perf['avg_generation_time']:.3f}s")
        
        qual = self.results['quality_metrics']
        print(f"\n⭐ Quality:")
        print(f"   Avg Quality Score: {qual['avg_quality_score']:.1f}/100")
        print(f"   Avg Answer Length: {qual['avg_answer_length']:.0f} chars")
        
        print(f"\n📁 Output Directory: {self.output_dir}")
        print("="*60)

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Model Evaluation Suite")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    args = parser.parse_args()
    
    print("📊 MEDICAL MODEL EVALUATION SUITE")
    print("="*60)
    
    try:
        # Check dependencies
        if not REPORTLAB_AVAILABLE:
            print("⚠️ Warning: reportlab not installed. Install with: pip install reportlab")
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Warning: matplotlib not installed. Install with: pip install matplotlib")
        
        # Initialize evaluator
        evaluator = MedicalModelEvaluator(args.model)
        
        # Load model
        evaluator.load_model()
        
        # Run evaluation
        evaluator.run_comprehensive_evaluation()
        
        # Generate visualizations
        png_files = evaluator.generate_png_charts()
        
        # Generate PDF report
        pdf_file = evaluator.generate_pdf_report(png_files)
        
        # Save JSON
        json_file = evaluator.save_json_results()
        
        # Print summary
        evaluator.print_summary()
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"\n📦 Generated Files:")
        if pdf_file:
            print(f"   📄 PDF Report: {pdf_file}")
        print(f"   💾 JSON Results: {json_file}")
        for png_file in png_files:
            print(f"   🖼️ PNG Chart: {png_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
