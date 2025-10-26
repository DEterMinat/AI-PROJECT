@echo off
title Start Complete Medical AI System
color 0E

echo.
echo ================================================================================
echo                     COMPLETE MEDICAL AI SYSTEM LAUNCHER
echo                       (T5-SMALL Disease Diagnosis System)
echo ================================================================================
echo.
echo This will start all components:
echo   [1] N8N Workflow Server (Docker) - OPTIONAL
echo   [2] Integrated API (FastAPI + Langchain + T5-Small)
echo   [3] Frontend Web Interface - OPTIONAL
echo.
echo ================================================================================
echo.

REM Step 1: Check Docker (Optional)
echo [Step 1/3] Checking Docker (Optional)...
docker version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Docker is not running. N8N will be skipped.
    echo           API can still work without N8N workflow.
    set SKIP_N8N=1
) else (
    echo [OK] Docker is ready
    set SKIP_N8N=0
)
echo.

REM Step 2: Check Python environment
echo [Step 2/3] Checking Python environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found!
    echo Please create it first: python -m venv .venv
    pause
    exit /b 1
)
echo [OK] Python environment ready
echo.

REM Step 3: Check T5 Model
echo [Step 3/3] Checking T5-Small model...
if not exist "models\t5_diagnosis_20251008_111522" (
    echo [ERROR] T5-Small model not found!
    echo Please train the model first:
    echo   python scripts\Main_8_Train_T5_Medical.py
    pause
    exit /b 1
)
echo [OK] T5-Small model ready
echo.

echo ================================================================================
echo.

REM Start N8N if Docker available
if "%SKIP_N8N%"=="0" (
    echo [1/3] Starting N8N Workflow Server...
    echo       This will open in a new window
    timeout /t 2 /nobreak >nul
    start "N8N Workflow Server" cmd /c start-n8n.bat
    echo [OK] N8N starting... (Wait 10 seconds for initialization)
    timeout /t 10 /nobreak
    echo.
) else (
    echo [1/3] Skipping N8N (Docker not available)
    echo.
)

echo [2/3] Starting Integrated Medical API Server...
echo       This will open in a new window
timeout /t 2 /nobreak >nul
start "Integrated Medical API" cmd /c start-api.bat
echo [OK] API starting... (Wait 15 seconds for model loading)
timeout /t 15 /nobreak

echo.
echo [3/3] Opening Frontend Web Interface...
timeout /t 2 /nobreak >nul
if exist "web_app\medical_qa_demo\index.html" (
    start "" "web_app\medical_qa_demo\index.html"
    echo [OK] Frontend opened in browser
) else (
    echo [WARNING] Frontend not found, opening API docs instead
    start "" "http://localhost:8000/docs"
)

echo.
echo ================================================================================
echo                          ALL SERVICES STARTED!
echo ================================================================================
echo.
echo Access Points:
echo   API Docs:    http://localhost:8000/docs
echo   API Status:  http://localhost:8000/status
if "%SKIP_N8N%"=="0" (
    echo   N8N UI:      http://localhost:5678
)
echo.
echo System Architecture:
echo   Frontend → FastAPI → Langchain + T5-Small → Disease Diagnosis
echo.
echo Current Model:
echo   - Architecture: T5-Small (T5ForConditionalGeneration)
echo   - Parameters: 60M
echo   - Task: Symptom → Disease Diagnosis
echo   - Accuracy: 35.67%% (Test Set)
echo   - Training: 15,419 diagnosis-focused samples
echo   - Location: models/t5_diagnosis_20251008_111522/
echo.
echo Usage:
echo   POST /ask
echo   {"question": "fever, cough, difficulty breathing"}
echo.
echo   Response:
echo   {"answer": "Based on symptoms, this may indicate: INFECTION..."}
echo.
echo Notes:
echo   - T5-Small model loading takes ~10-15 seconds
echo   - Frontend connects to http://localhost:8000
echo   - Interactive testing: test-t5.bat
echo   - Deployment check: python scripts\Deploy_T5_API.py
echo.
echo To stop all services:
echo   1. Close the API window (or press Ctrl+C)
if "%SKIP_N8N%"=="0" (
    echo   2. Close the N8N window (or press Ctrl+C)
    echo   3. Or run: docker-compose down
)
echo.
echo ================================================================================
echo.
echo Press any key to exit this launcher window...
pause >nul
