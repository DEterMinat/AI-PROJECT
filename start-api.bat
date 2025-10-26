@echo off
title T5-SMALL Medical API Server
color 0A

echo.
echo ========================================
echo   T5-SMALL MEDICAL API SERVER
echo ========================================
echo.

REM Check if Python virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then activate and install dependencies
    pause
    exit /b 1
)

echo [INFO] Activating Python virtual environment...
call .venv\Scripts\activate.bat

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

