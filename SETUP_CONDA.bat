@echo off
cd /d "%~dp0"
title Setup Medical AI System (Conda)
color 0E

echo.
echo ================================================================================
echo                     MEDICAL AI SYSTEM SETUP (CONDA)
echo ================================================================================
echo.

REM Check if Conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Conda not found in PATH. Checking default locations...
    if exist "C:\Users\tanak\anaconda3\Scripts\conda.exe" (
        echo [INFO] Found Anaconda at C:\Users\tanak\anaconda3
        set "PATH=%PATH%;C:\Users\tanak\anaconda3;C:\Users\tanak\anaconda3\Scripts;C:\Users\tanak\anaconda3\Library\bin"
    ) else (
        echo [ERROR] Conda is not installed or not in PATH.
        echo Please install Anaconda or Miniconda first.
        pause
        exit /b 1
    )
)

echo [1/4] Creating local Conda environment '.conda'...
REM Using nodefaults in environment.yml to avoid ToS issues
call conda env create -f environment.yml -p .conda
if %errorlevel% neq 0 (
    echo [WARNING] Creation failed or environment exists. Attempting update...
    call conda env update -f environment.yml -p .conda --prune
)

echo.
echo [2/4] Activating environment...
call conda activate .\.conda 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Standard activation failed. Trying direct script activation...
    if exist "C:\Users\tanak\anaconda3\Scripts\activate.bat" (
        call "C:\Users\tanak\anaconda3\Scripts\activate.bat" "%~dp0.conda"
    )
)

REM Verify activation
python -c "import sys; exit(0 if '.conda' in sys.prefix or 'medical-ai' in sys.prefix else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment '.conda'!
    echo [ERROR] The environment might not have been created correctly due to errors above.
    pause
    exit /b 1
)

echo.
echo [3/4] Installing project in editable mode...
pip install -e .

echo.
echo [4/4] Setup complete!
echo.
echo To start the system, run: START-ALL.bat
echo.
pause
