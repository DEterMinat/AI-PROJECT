#!/usr/bin/env python3
"""
Medical Dataset Downloader
Downloads large medical datasets (10,000+ samples) for Medical Q&A system
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback

def install_requirements():
    """Install required packages"""
    import subprocess
    
    packages = [
        'datasets',
        'huggingface-hub',
        'pandas',
        'openpyxl',
        'tqdm'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def download_medical_datasets():
    """Download large medical datasets"""
    print("🏥 Starting Medical Dataset Download...")
    print("=" * 50)
    
    # Create directories
    base_path = Path("data")
    raw_path = base_path / "raw"
    exports_path = base_path / "exports"
    
    for path in [raw_path, exports_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Try to import datasets
    try:
        from datasets import load_dataset
        from tqdm import tqdm
        print("✅ Datasets library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import datasets: {e}")
        print("Installing datasets...")
        install_requirements()
        from datasets import load_dataset
        from tqdm import tqdm
    
    collected_data = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dataset configurations - ONLY ChatDoctor for academic presentation
    datasets_config = [
        {
            'name': 'chatdoctor',
            'hf_name': 'lavita/ChatDoctor-HealthCareMagic-100k',
            'config': None,
            'description': 'ChatDoctor Medical Q&A (112K high-quality samples with long answers)'
        }
    ]
    
    total_samples = 0
    
    for dataset_config in datasets_config:
        print(f"\\n📥 Downloading {dataset_config['description']}...")
        print("-" * 40)
        
        try:
            # Load dataset
            if dataset_config['config']:
                dataset = load_dataset(
                    dataset_config['hf_name'], 
                    dataset_config['config'],
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    dataset_config['hf_name'],
                    trust_remote_code=True
                )
            
            processed_data = []
            
            # Process based on dataset type
            if dataset_config['name'] == 'medqa':
                for split in ['train', 'test']:
                    if split in dataset:
                        for item in tqdm(dataset[split], desc=f"Processing {split}"):
                            processed_data.append({
                                'question': item.get('question', ''),
                                'answer': item.get('answer', [''])[0] if item.get('answer') else '',
                                'source': 'MedQA',
                                'split': split,
                                'type': 'medical_qa'
                            })
            
            elif dataset_config['name'] == 'pubmedqa':
                for item in tqdm(dataset['train'], desc="Processing PubMedQA"):
                    context = ' '.join(item['context']['contexts']) if item['context']['contexts'] else ''
                    processed_data.append({
                        'question': item.get('question', ''),
                        'answer': item.get('final_decision', ''),
                        'context': context,
                        'source': 'PubMedQA',
                        'type': 'biomedical_qa'
                    })
            
            elif dataset_config['name'] == 'medmcqa':
                for split in ['train', 'validation']:
                    if split in dataset:
                        for item in tqdm(dataset[split], desc=f"Processing {split}"):
                            options = [item['opa'], item['opb'], item['opc'], item['opd']]
                            correct_answer = options[item['cop']] if item['cop'] < len(options) else ''
                            
                            processed_data.append({
                                'question': item.get('question', ''),
                                'answer': correct_answer,
                                'options': options,
                                'subject': item.get('subject_name', ''),
                                'explanation': item.get('exp', ''),
                                'source': 'MedMCQA',
                                'split': split,
                                'type': 'medical_mcq'
                            })
            
            elif dataset_config['name'] == 'chatdoctor':
                for item in tqdm(dataset['train'], desc="Processing ChatDoctor"):
                    processed_data.append({
                        'question': item.get('input', ''),
                        'answer': item.get('output', ''),
                        'instruction': item.get('instruction', ''),
                        'source': 'ChatDoctor',
                        'type': 'medical_conversation'
                    })
            
            if processed_data:
                collected_data[dataset_config['name']] = processed_data
                dataset_size = len(processed_data)
                total_samples += dataset_size
                
                print(f"✅ {dataset_config['description']}: {dataset_size:,} samples")
                
                # Save individual dataset
                dataset_file = raw_path / f"{dataset_config['name']}_{timestamp}.json"
                with open(dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                # Save as CSV
                csv_file = raw_path / f"{dataset_config['name']}_{timestamp}.csv"
                df = pd.DataFrame(processed_data)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                
            else:
                print(f"⚠️ No data collected for {dataset_config['name']}")
                
        except Exception as e:
            print(f"❌ Error downloading {dataset_config['name']}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Save combined dataset
    if collected_data:
        print(f"\\n💾 Saving combined dataset...")
        
        # Combine all data
        all_data = []
        for dataset_name, data in collected_data.items():
            all_data.extend(data)
        
        # Save combined JSON
        combined_file = raw_path / f"medical_datasets_combined_{timestamp}.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Save combined CSV
        combined_csv = raw_path / f"medical_datasets_combined_{timestamp}.csv"
        df_combined = pd.DataFrame(all_data)
        df_combined.to_csv(combined_csv, index=False, encoding='utf-8')
        
        # Create summary CSV instead of Excel
        summary_data = []
        for name, data in collected_data.items():
            summary_data.append({
                'Dataset': name,
                'Samples': len(data),
                'Description': [d['description'] for d in datasets_config if d['name'] == name][0]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file_csv = exports_path / f"dataset_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file_csv, index=False, encoding='utf-8')
        
        excel_file = summary_file_csv  # Use CSV as main report file
        
        # Create summary report
        summary = {
            'download_timestamp': datetime.now().isoformat(),
            'total_datasets': len(collected_data),
            'total_samples': total_samples,
            'datasets': {
                name: {
                    'samples': len(data),
                    'file': f"{name}_{timestamp}.json"
                }
                for name, data in collected_data.items()
            },
            'files': {
                'combined_json': str(combined_file),
                'combined_csv': str(combined_csv),
                'excel_report': str(excel_file)
            }
        }
        
        summary_file = exports_path / f"download_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\\n🎉 MEDICAL DATASET DOWNLOAD COMPLETED!")
        print("=" * 50)
        print(f"📊 Total Samples: {total_samples:,}")
        print(f"📁 Combined File: {combined_file}")
        print(f"📊 CSV File: {combined_csv}")
        print(f"📈 Summary Report: {excel_file}")
        print(f"📋 Summary: {summary_file}")
        
        if total_samples >= 10000:
            print(f"\\n🎯 ✅ SUCCESS: Downloaded {total_samples:,} medical samples (target: 10,000+)")
        else:
            print(f"\\n⚠️ Downloaded {total_samples:,} samples (target was 10,000+)")
        
        return summary
    
    else:
        print("❌ No datasets were successfully downloaded")
        return None

if __name__ == "__main__":
    try:
        print("🚀 Medical Dataset Downloader Starting...")
        summary = download_medical_datasets()
        
        if summary:
            print("\\n✅ Download completed successfully!")
        else:
            print("\\n❌ Download failed!")
            
    except Exception as e:
        print(f"\\n💥 Fatal error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def main(config=None):
    "Main execution function"
    return download_medical_datasets()

if __name__ == "__main__":
    main()
