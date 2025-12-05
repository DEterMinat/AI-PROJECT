#!/usr/bin/env python3
"""
Step 4: Exploratory Data Analysis (EDA)
วิเคราะห์ข้อมูลเบื้องต้นเพื่อเข้าใจ pattern และ distribution
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class MedicalDataEDA:
    """Exploratory Data Analysis for Medical Q&A Data"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None
        self.df = None
        
    def load_data(self):
        """Load cleaned medical data"""
        logger.info(f"📂 Loading data from: {self.data_path.name}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"✅ Loaded {len(self.data):,} records")
        
        # Convert to DataFrame for analysis
        self.df = pd.DataFrame(self.data)
        
        return self.df
    
    def basic_statistics(self):
        """Display basic statistics"""
        print("\n" + "="*60)
        print("📊 BASIC STATISTICS")
        print("="*60)
        
        print(f"\n📏 Dataset Size:")
        print(f"   Total records: {len(self.df):,}")
        print(f"   Total columns: {len(self.df.columns)}")
        print(f"\n📝 Columns: {', '.join(self.df.columns.tolist())}")
        
        # Missing values
        print(f"\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values!")
        else:
            print(missing[missing > 0])
        
        # Data types
        print(f"\n📋 Data Types:")
        print(self.df.dtypes)
    
    def text_length_analysis(self):
        """Analyze text length distribution"""
        print("\n" + "="*60)
        print("📝 TEXT LENGTH ANALYSIS")
        print("="*60)
        
        # Calculate word counts
        self.df['question_words'] = self.df['question'].apply(lambda x: len(str(x).split()))
        self.df['answer_words'] = self.df['answer'].apply(lambda x: len(str(x).split()))
        self.df['question_chars'] = self.df['question'].apply(lambda x: len(str(x)))
        self.df['answer_chars'] = self.df['answer'].apply(lambda x: len(str(x)))
        
        # Statistics
        print(f"\n📏 Question Length:")
        print(f"   Words  - Min: {self.df['question_words'].min()}, Max: {self.df['question_words'].max()}, Mean: {self.df['question_words'].mean():.1f}, Median: {self.df['question_words'].median():.1f}")
        print(f"   Chars  - Min: {self.df['question_chars'].min()}, Max: {self.df['question_chars'].max()}, Mean: {self.df['question_chars'].mean():.1f}, Median: {self.df['question_chars'].median():.1f}")
        
        print(f"\n📏 Answer Length:")
        print(f"   Words  - Min: {self.df['answer_words'].min()}, Max: {self.df['answer_words'].max()}, Mean: {self.df['answer_words'].mean():.1f}, Median: {self.df['answer_words'].median():.1f}")
        print(f"   Chars  - Min: {self.df['answer_chars'].min()}, Max: {self.df['answer_chars'].max()}, Mean: {self.df['answer_chars'].mean():.1f}, Median: {self.df['answer_chars'].median():.1f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Question word count histogram
        axes[0, 0].hist(self.df['question_words'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Question Word Count Distribution')
        axes[0, 0].set_xlabel('Word Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['question_words'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["question_words"].mean():.1f}')
        axes[0, 0].legend()
        
        # Answer word count histogram
        axes[0, 1].hist(self.df['answer_words'], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Answer Word Count Distribution')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(self.df['answer_words'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["answer_words"].mean():.1f}')
        axes[0, 1].legend()
        
        # Question char count boxplot
        axes[1, 0].boxplot(self.df['question_chars'], vert=False)
        axes[1, 0].set_title('Question Character Count Boxplot')
        axes[1, 0].set_xlabel('Character Count')
        
        # Answer char count boxplot
        axes[1, 1].boxplot(self.df['answer_chars'], vert=False)
        axes[1, 1].set_title('Answer Character Count Boxplot')
        axes[1, 1].set_xlabel('Character Count')
        
        plt.tight_layout()
        output_path = Path('data/exports/evaluation') / f'eda_text_length_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved: {output_path}")
        plt.close()
    
    def specialty_distribution(self):
        """Analyze specialty distribution"""
        print("\n" + "="*60)
        print("🏥 SPECIALTY DISTRIBUTION")
        print("="*60)
        
        if 'specialty' not in self.df.columns:
            print("   ⚠️ No specialty column found")
            return
        
        specialty_counts = self.df['specialty'].value_counts()
        
        print(f"\n📊 Specialty Counts:")
        for specialty, count in specialty_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {specialty:20s}: {count:6,} ({percentage:5.2f}%)")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        specialty_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_title('Specialty Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Specialty')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        axes[1].pie(specialty_counts.values, labels=specialty_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Specialty Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = Path('data/exports/evaluation') / f'eda_specialty_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved: {output_path}")
        plt.close()
    
    def severity_distribution(self):
        """Analyze severity distribution"""
        print("\n" + "="*60)
        print("⚡ SEVERITY DISTRIBUTION")
        print("="*60)
        
        if 'severity_level' not in self.df.columns:
            print("   ⚠️ No severity_level column found")
            return
        
        severity_counts = self.df['severity_level'].value_counts()
        
        print(f"\n📊 Severity Counts:")
        for severity, count in severity_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {severity:20s}: {count:6,} ({percentage:5.2f}%)")
        
        # Visualization
        severity_order = ['critical', 'high', 'moderate', 'low']
        severity_counts = severity_counts.reindex(severity_order, fill_value=0)
        
        colors = {'critical': '#d32f2f', 'high': '#f57c00', 'moderate': '#fbc02d', 'low': '#388e3c'}
        color_list = [colors.get(s, 'gray') for s in severity_order]
        
        plt.figure(figsize=(10, 6))
        severity_counts.plot(kind='bar', color=color_list, edgecolor='black')
        plt.title('Severity Level Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = Path('data/exports/evaluation') / f'eda_severity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved: {output_path}")
        plt.close()
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numerical columns
        numerical_cols = ['question_words', 'answer_words', 'question_chars', 'answer_chars']
        
        if not all(col in self.df.columns for col in numerical_cols):
            print("   ⚠️ Missing numerical columns for correlation")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        print(f"\n📊 Correlation Matrix:")
        print(corr_matrix)
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path('data/exports/evaluation') / f'eda_correlation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved: {output_path}")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "="*70)
        print("📋 GENERATING COMPREHENSIVE EDA REPORT")
        print("="*70)
        
        # Run all analyses
        self.load_data()
        self.basic_statistics()
        self.text_length_analysis()
        self.specialty_distribution()
        self.severity_distribution()
        self.correlation_analysis()
        
        # Generate summary report
        report_path = Path('data/exports/evaluation') / f'eda_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MEDICAL Q&A DATASET - EDA REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path.name}\n")
            f.write(f"Total Records: {len(self.df):,}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Question Length (words): {self.df['question_words'].describe()}\n\n")
            f.write(f"Answer Length (words): {self.df['answer_words'].describe()}\n\n")
            
            if 'specialty' in self.df.columns:
                f.write("SPECIALTY DISTRIBUTION:\n")
                f.write("-" * 70 + "\n")
                for specialty, count in self.df['specialty'].value_counts().items():
                    f.write(f"{specialty:20s}: {count:6,} ({count/len(self.df)*100:5.2f}%)\n")
                f.write("\n")
            
            if 'severity_level' in self.df.columns:
                f.write("SEVERITY DISTRIBUTION:\n")
                f.write("-" * 70 + "\n")
                for severity, count in self.df['severity_level'].value_counts().items():
                    f.write(f"{severity:20s}: {count:6,} ({count/len(self.df)*100:5.2f}%)\n")
        
        logger.info(f"💾 Saved report: {report_path}")
        
        print("\n✅ EDA Complete!")
        print(f"📁 All outputs saved to: data/exports/evaluation/")

def main():
    """Main EDA process"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 4: Exploratory Data Analysis")
    parser.add_argument("--input", type=str, 
                       default=None,
                       help="Input data file (auto-detect if not specified)")
    args = parser.parse_args()
    
    # Auto-detect latest cleaned file
    if args.input is None:
        processed_path = Path("data/processed")
        cleaned_files = list(processed_path.glob("chatdoctor_cleaned_*.json"))
        if cleaned_files:
            args.input = str(max(cleaned_files, key=lambda x: x.stat().st_mtime))
            logger.info(f"✅ Auto-detected: {args.input}")
        else:
            logger.error("❌ No cleaned data found in data/processed/")
            return 1
    
    # Run EDA
    eda = MedicalDataEDA(args.input)
    eda.generate_report()
    
    return 0

if __name__ == "__main__":
    exit(main())
