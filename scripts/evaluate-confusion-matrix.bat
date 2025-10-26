@echo off
REM ===============================================================================
REM 📊 FLAN-T5 Confusion Matrix Evaluation Script
REM 🎯 Disease classification with detailed confusion matrix analysis
REM ===============================================================================

echo.
echo ===============================================================================
echo 📊 FLAN-T5 CONFUSION MATRIX EVALUATION
echo ===============================================================================
echo.
echo This script will evaluate FLAN-T5 disease classification and generate:
echo   📊 Confusion Matrix (Raw & Normalized)
echo   📈 Performance Metrics by Category (Precision, Recall, F1)
echo   📋 Classification Dashboard (4 visualizations)
echo   📄 Comprehensive PDF Report
echo   💾 JSON Results with detailed metrics
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo.
)

REM Check Python packages
echo [STEP 1/4] Checking required packages...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || (
    echo ❌ PyTorch not found! Installing...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

python -c "import transformers; print('✅ Transformers available')" 2>nul || (
    echo ❌ Transformers not found! Installing...
    pip install transformers
)

python -c "import matplotlib; print('✅ Matplotlib available')" 2>nul || (
    echo ❌ Matplotlib not found! Installing...
    pip install matplotlib seaborn
)

python -c "import sklearn; print('✅ Scikit-learn available')" 2>nul || (
    echo ❌ Scikit-learn not found! Installing...
    pip install scikit-learn
)

python -c "import reportlab; print('✅ ReportLab available')" 2>nul || (
    echo ❌ ReportLab not found! Installing...
    pip install reportlab
)

echo.

REM Find latest FLAN-T5 model
echo [STEP 2/4] Finding latest FLAN-T5 model...
set MODEL_PATH=""

REM Check for flan_t5_diagnosis models
for /f "delims=" %%i in ('dir /b /ad models\flan_t5_diagnosis_* 2^>nul') do (
    set MODEL_PATH=models\%%i
)

if "%MODEL_PATH%"=="" (
    echo ❌ No FLAN-T5 model found in models\ directory!
    echo.
    echo Expected model directories:
    echo   - models\flan_t5_diagnosis_YYYYMMDD_HHMMSS\
    echo.
    echo Please train a model first using:
    echo   train-model.bat
    pause
    exit /b 1
)

echo ✅ Found model: %MODEL_PATH%
echo.

REM Check model files
echo [STEP 3/4] Verifying model files...
if not exist "%MODEL_PATH%\config.json" (
    echo ❌ Model config.json not found!
    pause
    exit /b 1
)

if not exist "%MODEL_PATH%\model.safetensors" (
    echo ❌ Model weights not found!
    pause
    exit /b 1
)

if not exist "%MODEL_PATH%\tokenizer_config.json" (
    echo ❌ Tokenizer files not found!
    pause
    exit /b 1
)

echo ✅ Model files verified
echo.

REM Display evaluation configuration
echo [STEP 4/4] Starting confusion matrix evaluation...
echo ===============================================================================
echo 🔧 CONFUSION MATRIX EVALUATION CONFIGURATION
echo ===============================================================================
echo Model Path:           %MODEL_PATH%
echo Disease Categories:   12 (Infection, Cardiovascular, Respiratory, etc.)
echo Test Cases:           45 questions (classified by disease type)
echo Metrics Generated:    Precision, Recall, F1-Score, Support
echo Visualization:        Confusion Matrix (Raw + Normalized)
echo Performance Analysis: Per-category accuracy and error analysis
echo Expected Duration:    3-5 minutes
echo GPU Support:          Auto-detected
echo ===============================================================================
echo.
echo 🎯 Disease Categories Being Evaluated:
echo   • Infection (Bacterial, Viral, Pneumonia)
echo   • Cardiovascular (Heart Disease, Hypertension, Stroke)
echo   • Respiratory (Asthma, COPD, Breathing Issues)
echo   • Neurological (Headache, Seizure, Dizziness)
echo   • Gastrointestinal (Stomach, Nausea, Digestive)
echo   • Musculoskeletal (Arthritis, Joint Pain, Muscle)
echo   • Endocrine (Diabetes, Thyroid, Hormones)
echo   • Mental Health (Depression, Anxiety, PTSD)
echo   • Dermatological (Skin, Allergic Reactions)
echo   • Emergency (Life-threatening conditions)
echo   • Preventive (Vaccines, Screening, Checkups)
echo   • Other (General or unclear conditions)
echo.

REM Run confusion matrix evaluation
python scripts\Main_9_Confusion_Matrix_Evaluation.py --model-path %MODEL_PATH%

if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo ❌ CONFUSION MATRIX EVALUATION FAILED!
    echo ===============================================================================
    echo.
    echo Check the error messages above and:
    echo   1. Make sure the model is properly trained
    echo   2. Verify all dependencies are installed (sklearn, matplotlib, seaborn)
    echo   3. Check GPU memory availability
    echo   4. Ensure model files are not corrupted
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ===============================================================================
    echo ✅ CONFUSION MATRIX EVALUATION COMPLETED SUCCESSFULLY!
    echo ===============================================================================
    echo.
    echo 📁 Results saved to: data\exports\evaluation\
    echo.
    echo 📊 Generated files:
    echo   📄 FLAN_T5_Confusion_Matrix_Report.pdf - Comprehensive analysis report
    echo   📊 confusion_matrix.png - Raw confusion matrix heatmap
    echo   📊 confusion_matrix_normalized.png - Normalized confusion matrix
    echo   📈 performance_by_category.png - Precision/Recall/F1 by category
    echo   📋 classification_dashboard.png - 4-panel performance dashboard
    echo   💾 confusion_matrix_results.json - Detailed metrics and raw data
    echo.
    echo 🎯 Key Insights Available:
    echo   • Overall classification accuracy
    echo   • Per-disease category performance (Precision, Recall, F1-Score)
    echo   • Most/least accurate disease classifications
    echo   • Common misclassification patterns
    echo   • Model confusion between similar diseases
    echo   • Recommendations for model improvement
    echo.
    echo 📊 Confusion Matrix Analysis:
    echo   • Diagonal elements = Correct classifications
    echo   • Off-diagonal elements = Misclassifications
    echo   • Normalized version shows classification probabilities
    echo   • Heat map colors indicate classification confidence
    echo.
    
    REM Open results folder
    echo Opening results folder...
    explorer data\exports\evaluation
    echo.
)

echo Press any key to exit...
pause >nul