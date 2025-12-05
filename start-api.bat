@echo off
cd /d "%~dp0"
title T5-SMALL Medical API Server
color 0A

echo.
echo ========================================
echo   T5-SMALL MEDICAL API SERVER
echo ========================================
echo.

REM Check for Conda environment
where conda >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Users\tanak\anaconda3\Scripts\conda.exe" (
        set "PATH=%PATH%;C:\Users\tanak\anaconda3;C:\Users\tanak\anaconda3\Scripts;C:\Users\tanak\anaconda3\Library\bin"
    )
)

call conda activate .\.conda >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Users\tanak\anaconda3\Scripts\activate.bat" (
        call "C:\Users\tanak\anaconda3\Scripts\activate.bat" "%~dp0.conda" >nul 2>&1
    )
)

python -c "import sys; exit(0 if '.conda' in sys.prefix or 'medical-ai' in sys.prefix else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Activated Conda environment '.conda'
    goto :START_SERVER
)

REM Check if Python virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Activating Python virtual environment...
    call .venv\Scripts\activate.bat
    goto :START_SERVER
)

echo [ERROR] No suitable Python environment found!
echo Please run SETUP_CONDA.bat or create a venv.
pause
exit /b 1

:START_SERVER

echo.
echo ========================================
echo   MODEL: T5-SMALL DIAGNOSIS
echo ========================================
echo Architecture: T5ForConditionalGeneration
echo Parameters: 60M
echo Accuracy: 35.67%% (Test Set)
echo Task: Symptom → Disease Diagnosis
echo.

echo [INFO] Starting T5-Small Integrated Medical API...
echo.
echo ========================================
echo   Features:
echo   - FastAPI Gateway (Port 8000)
echo   - T5-Small Disease Diagnosis
echo   - Langchain RAG Engine
echo   - Medical Knowledge Base
echo ========================================
echo.
echo [INFO] Access Points:
echo   - API Docs:  http://localhost:8000/docs
echo   - Health:    http://localhost:8000/health
echo   - Status:    http://localhost:8000/status
echo.
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start the integrated API
python run_api.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Integrated API!
    echo Check that all dependencies are installed:
    echo   pip install -r requirements.txt
    echo.
    pause
)

