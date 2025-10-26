# 🚀 Quick Medical AI Workflow

Quick scripts for rapid testing and evaluation.

---

## 📋 Quick Scripts Overview

### 1. ⚡ Quick Clean (`quick-clean.bat`)
**Fast data cleaning for rapid testing**

```bash
# Clean 1000 samples (default)
quick-clean.bat

# Clean custom amount
quick-clean.bat --samples 500
quick-clean.bat --samples 2000
```

**What it does:**
- ✅ Fast loading from multiple data sources
- ✅ Basic text cleaning (HTML, URLs, spaces)
- ✅ Quick duplicate detection (hash-based)
- ✅ Simple validation (length, format)
- ✅ Output: `data/processed/quick_cleaned_*.json`

**Speed:** ~1000 samples in 10-30 seconds

---

### 2. ⚡ Quick Train (`quick-train.bat`)
**Fast model training for rapid testing**

```bash
# Train with 100 samples, 1 epoch (default)
quick-train.bat

# Train with custom settings
quick-train.bat --samples 100 --epochs 2 --batch-size 2
quick-train.bat --samples 500 --epochs 3
```

**What it does:**
- ✅ Loads DialoGPT-small model
- ✅ Quick data loading
- ✅ Simple training loop
- ✅ Basic testing with 3 questions
- ✅ Output: `models/quick_trained_*/`

**Speed:** ~100 samples in 2-5 minutes (CPU)

---

### 3. 📊 Evaluate Model (`evaluate-model.bat`)
**Comprehensive evaluation with PDF & PNG reports**

```bash
# Evaluate trained model
evaluate-model.bat models\quick_trained_20251006_120000

# Or use latest model
evaluate-model.bat models\simple_trained_20251006_170947
```

**What it does:**
- ✅ Tests 50 questions across 5 categories:
  - Basic Symptoms (10 questions)
  - Chronic Diseases (10 questions)
  - Emergency Situations (10 questions)
  - Preventive Care (10 questions)
  - Mental Health (10 questions)
- ✅ Calculates quality scores (0-100)
- ✅ Measures generation time & performance
- ✅ Generates **4 PNG charts**:
  1. Quality by Category
  2. Generation Time Distribution
  3. Success Rate by Category
  4. Performance Summary (4-panel)
- ✅ Creates **comprehensive PDF report**:
  - Model information
  - Performance metrics
  - Quality metrics
  - Category breakdown
  - Visual charts
  - Sample results
- ✅ Saves JSON results
- ✅ Output: `data/exports/evaluation/evaluation_*/`

**Speed:** ~50 questions in 2-5 minutes

**Generated Files:**
- 📄 `evaluation_report.pdf` - Full report
- 🖼️ `quality_by_category.png`
- 🖼️ `generation_time_distribution.png`
- 🖼️ `success_rate_by_category.png`
- 🖼️ `performance_summary.png`
- 💾 `evaluation_results.json`

---

## 🎯 Complete Workflow Example

### Step-by-Step: Clean → Train → Evaluate

```bash
# Step 1: Quick clean data
quick-clean.bat --samples 200

# Step 2: Quick train model
quick-train.bat --samples 200 --epochs 2

# Step 3: Evaluate model (use the path from step 2)
evaluate-model.bat models\quick_trained_20251006_143022
```

**Total time:** ~10-15 minutes for complete cycle

---

## 📊 Evaluation Metrics Explained

### Performance Metrics
- **Success Rate** - % of questions answered successfully
- **Avg Generation Time** - Average time to generate response
- **Min/Max Time** - Fastest and slowest responses

### Quality Metrics (0-100 Score)
Quality score calculated from:
- **Length (20 pts)** - Answer completeness
- **Medical Relevance (20 pts)** - Medical terminology usage
- **Completeness (20 pts)** - Multiple sentences, no errors
- **Professionalism (20 pts)** - Appropriate language
- **Question Relevance (20 pts)** - Answer matches question

### Category Performance
Each category evaluated separately:
- Basic Symptoms
- Chronic Diseases
- Emergency Situations
- Preventive Care
- Mental Health

---

## 🔧 Dependencies

### For Evaluation (PDF & PNG)
```bash
pip install reportlab matplotlib numpy
```

**Auto-installed** when you run `evaluate-model.bat`

---

## 📁 Output Structure

```
data/
├── processed/
│   └── quick_cleaned_20251006_143000.json
└── exports/
    └── evaluation/
        └── evaluation_20251006_143500/
            ├── evaluation_report.pdf           # 📄 Main report
            ├── quality_by_category.png         # 🖼️ Chart 1
            ├── generation_time_distribution.png # 🖼️ Chart 2
            ├── success_rate_by_category.png    # 🖼️ Chart 3
            ├── performance_summary.png          # 🖼️ Chart 4
            └── evaluation_results.json          # 💾 Raw data

models/
└── quick_trained_20251006_143022/
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer_config.json
    ├── stats.json                              # Training stats
    └── (other model files)
```

---

## 💡 Tips & Best Practices

### For Quick Testing
```bash
# Minimal setup (fastest)
quick-clean.bat --samples 50
quick-train.bat --samples 50 --epochs 1
evaluate-model.bat models\quick_trained_*
```

### For Better Quality
```bash
# More data, more training
quick-clean.bat --samples 500
quick-train.bat --samples 500 --epochs 3
evaluate-model.bat models\quick_trained_*
```

### For Production Testing
```bash
# Use full clean_data.py and train_model.py
python scripts\clean_data.py --max-records 5000
python scripts\train_model.py --samples 5000 --epochs 3
evaluate-model.bat models\simple_trained_*
```

---

## 🐛 Troubleshooting

### Issue: "reportlab not found"
```bash
pip install reportlab
```

### Issue: "matplotlib not found"
```bash
pip install matplotlib
```

### Issue: "Model not found"
Check the exact path:
```bash
dir models\quick_trained_*
# Use the full path from output
```

### Issue: "Out of memory" during training
Reduce batch size:
```bash
quick-train.bat --samples 50 --batch-size 1
```

---

## 📈 Understanding PDF Report

### Section 1: Model Information
- Model architecture details
- Parameter count
- Model size

### Section 2: Performance Metrics
- Total questions tested
- Success rate
- Generation time statistics

### Section 3: Quality Metrics
- Answer quality scores
- Answer length statistics

### Section 4: Category Performance
- Success rate per category
- Quality score per category
- Time per category

### Section 5: Visual Analysis
- 4 comprehensive charts
- Distribution graphs
- Category comparisons

### Section 6: Sample Results
- 5 example Q&A pairs
- Quality scores
- Generation times

---

## 🎯 Quick Reference

| Task | Command | Time | Output |
|------|---------|------|--------|
| Clean 1000 samples | `quick-clean.bat` | 10-30s | JSON file |
| Train 100 samples | `quick-train.bat` | 2-5m | Model folder |
| Evaluate model | `evaluate-model.bat [path]` | 2-5m | PDF + PNG + JSON |

---

## ✨ Features

### Quick Clean
- ⚡ Hash-based duplicate detection
- 🧹 Basic text cleaning
- ✅ Simple validation
- 💾 JSON output

### Quick Train
- 🤖 DialoGPT-small model
- 🚀 Fast training loop
- 🧪 Automatic testing
- 💾 Model + stats saved

### Evaluate
- 📊 50 diverse test questions
- 🎨 4 visualization charts
- 📄 Professional PDF report
- 💾 JSON raw results
- 🔍 Quality scoring (0-100)
- ⏱️ Performance metrics
- 📂 Category analysis

---

## 🚀 Ready to Use!

All scripts are ready to run immediately:

```bash
# 1. Clean some data
quick-clean.bat

# 2. Train a model  
quick-train.bat

# 3. Evaluate it
evaluate-model.bat models\quick_trained_[timestamp]
```

**That's it!** 🎉

You'll get a complete evaluation with PDF report and PNG charts showing your model's performance across multiple medical categories.

---

## 📞 Need Help?

Check the console output - scripts provide detailed logging:
- ✅ Success indicators
- ⚠️ Warnings
- ❌ Error messages
- 📊 Progress updates

All scripts pause at the end so you can read the results!
