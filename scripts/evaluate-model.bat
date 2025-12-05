@echo off
REM ===============================================================================
REM 📊 FLAN-T5 Model Evaluation Script
REM 🎯 Comprehensive testing with PNG & PDF reports
REM ===============================================================================

echo.
echo ===============================================================================
echo 📊 FLAN-T5 MODEL EVALUATION
echo ===============================================================================
echo.
echo This script will evaluate the latest FLAN-T5 model and generate:
echo   📄 PDF Report (comprehensive analysis)
echo   📊 PNG Charts (performance visualizations)  
echo   📋 JSON Results (raw data)
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

if not exist "%MODEL_PATH%\pytorch_model.bin" if not exist "%MODEL_PATH%\model.safetensors" (
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
echo [STEP 4/4] Starting evaluation...
echo ===============================================================================
echo 🔧 EVALUATION CONFIGURATION
echo ===============================================================================
echo Model Path:          %MODEL_PATH%
echo Test Categories:     5 (Basic, Chronic, Emergency, Preventive, Mental Health)
echo Questions per Cat:   15 questions each
echo Total Questions:     75 test cases
echo Output Format:       JSON + PDF + PNG
echo Expected Duration:   5-10 minutes
echo GPU Support:         Auto-detected
echo ===============================================================================
echo.

REM Run evaluation
python src\ml_pipeline\evaluate_flan_t5.py --model-path %MODEL_PATH%

if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo ❌ EVALUATION FAILED!
    echo ===============================================================================
    echo.
    echo Check the error messages above and:
    echo   1. Make sure the model is properly trained
    echo   2. Verify all dependencies are installed
    echo   3. Check GPU memory availability
    echo   4. Ensure model files are not corrupted
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ===============================================================================
    echo ✅ EVALUATION COMPLETED SUCCESSFULLY!
    echo ===============================================================================
    echo.
    echo 📁 Results saved to: data\exports\evaluation\
    echo.
    echo 📊 Generated files:
    echo   📄 FLAN_T5_Evaluation_Report.pdf - Comprehensive analysis
    echo   📊 quality_by_category.png - Quality scores by category  
    echo   📊 inference_time_distribution.png - Response time analysis
    echo   📊 performance_dashboard.png - Overall performance metrics
    echo   📋 evaluation_results.json - Raw data for further analysis
    echo.
    echo 🎯 Key Metrics:
    echo   • Model accuracy and quality scores
    echo   • Response time performance
    echo   • Category-specific analysis
    echo   • Emergency detection capability
    echo   • Error analysis and recommendations
    echo.
    
    REM Open results folder
    echo Opening results folder...
    explorer data\exports\evaluation
    echo.
)

echo Press any key to exit...
pause >nul