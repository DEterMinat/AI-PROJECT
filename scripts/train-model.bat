@echo off
REM ============================================================================
REM train-model.bat - FLAN-T5-Base Medical Model Training Script
REM ============================================================================
REM 
REM This script trains FLAN-T5-Base (250M parameters) for medical diagnosis.
REM
REM WHY FLAN-T5-BASE?
REM   - Encoder-Decoder architecture (perfect for Q&A)
REM   - Instruction-tuned (understands Q&A naturally)
REM   - 250M params (4x better than T5-Small 60M)
REM   - Expected accuracy: 60-75% (vs 40% T5-Small)
REM
REM Training Time:
REM   - GPU (6 GB VRAM): 3-4 hours
REM   - CPU (32 GB RAM): 6-8 hours
REM
REM Usage:
REM   train-model.bat           Start training with default settings
REM
REM ============================================================================

title FLAN-T5-Base Medical Training

echo.
echo ===============================================================================
echo    FLAN-T5-BASE MEDICAL DIAGNOSIS TRAINING
echo ===============================================================================
echo.
echo This will train FLAN-T5-Base (250M parameters) with your custom medical data.
echo.
echo WHY FLAN-T5-BASE?
echo   - Encoder-Decoder = Perfect for Q^&A (better than BioGPT Causal LM)
echo   - Instruction-tuned = Understands questions naturally
echo   - 250M params = 4x better than T5-Small
echo   - Expected: 60-75%% accuracy (vs 40%% T5-Small)
echo.
echo Training Configuration (IMPROVED):
echo   - Model: google/flan-t5-base (250M parameters)
echo   - Training Data: 89,079 samples (was 12,335) +622%%!
echo   - Validation Data: 11,135 samples (was 1,542)
echo   - Test Data: 11,135 samples (was 1,542)
echo   - Epochs: 15 (was 5)
echo   - Batch Size: 8 (was 4)
echo   - Learning Rate: 5e-5 (was 1e-4)
echo   - Early Stopping: Yes (patience=3)
echo   - Expected Accuracy: 60-70%% (was 42%%)
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo.
echo [INFO] Checking dependencies...
python -c "import torch, transformers, datasets" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Missing dependencies detected!
    echo.
    echo Installing required packages:
    echo   - torch
    echo   - transformers
    echo   - datasets
    echo   - accelerate
    echo.
    pip install torch transformers datasets accelerate tqdm scikit-learn
    echo.
)

REM Check GPU availability
echo [INFO] Checking GPU availability...
python -c "import torch; print('GPU Available!' if torch.cuda.is_available() else 'CPU Only (slower but works)')"
echo.

REM Check data files (using LARGE dataset)
if not exist "data\model_ready\train_20251007_235239.json" (
    echo [ERROR] Training data not found!
    echo Expected: data\model_ready\train_20251007_235239.json
    echo.
    echo Please run data preparation first:
    echo   python src\ml_pipeline\data_splitting.py
    pause
    exit /b 1
)

echo [INFO] Data files found (LARGE dataset):
echo   - Training: data\model_ready\train_20251007_235239.json (89,079 samples)
echo   - Validation: data\model_ready\val_20251007_235239.json (11,135 samples)
echo   - Test: data\model_ready\test_20251007_235239.json (11,135 samples)
echo   - Total: 111,349 samples (7x more data!)
echo.

REM Estimate training time
echo ===============================================================================
echo    TRAINING TIME ESTIMATE
echo ===============================================================================
echo.
python -c "import torch; gpu = torch.cuda.is_available(); print('  With GPU (RTX 3050 6GB): 3-4 hours' if gpu else '  With CPU (32GB RAM): 6-8 hours'); print('  Model download: ~5 minutes (990 MB)');"
echo.

REM Confirm before training
echo ===============================================================================
set /p confirm="Ready to start training? This will take 6-8 hours on CPU. (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo ===============================================================================
echo    STARTING TRAINING
echo ===============================================================================
echo.
echo Training started at: %date% %time%
echo This will take approximately 6-8 hours on CPU (3-4 hours on GPU).
echo.
echo You can:
echo   - Minimize this window and continue working
echo   - Check progress in: logs\flan_t5_training.log
echo   - Monitor GPU: nvidia-smi (if GPU available)
echo.
echo Press Ctrl+C to stop training (progress will be saved)
echo.
echo ===============================================================================
echo.

REM Run training with LARGE dataset (111,349 samples)
python src\ml_pipeline\train_flan_t5.py ^
    --train data\model_ready\train_20251007_235239.json ^
    --val data\model_ready\val_20251007_235239.json ^
    --test data\model_ready\test_20251007_235239.json ^
    --epochs 15 ^
    --batch-size 8 ^
    --lr 5e-5 ^
    --early-stop-patience 3

if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo [ERROR] Training failed!
    echo ===============================================================================
    echo.
    echo Check the error messages above and:
    echo   1. Make sure all dependencies are installed
    echo   2. Check if you have enough disk space (need ~2 GB)
    echo   3. Review logs\flan_t5_training.log for details
    echo.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo    TRAINING COMPLETED!
echo ===============================================================================
echo.
echo Training finished at: %date% %time%
echo.
echo Model saved to: models\flan_t5_diagnosis_[timestamp]
echo.
echo Next steps:
echo   1. Test the model: python langchain_service\medical_ai.py
echo   2. Start API: start-api.bat
echo   3. Open frontend: web_app\medical_qa_demo\index.html
echo.
echo Expected performance:
echo   - FLAN-T5-Base: 60-75%% accuracy
echo   - T5-Small (old): 40%% accuracy
echo   - Improvement: +20-35%% better!
echo.
echo ===============================================================================

pause
