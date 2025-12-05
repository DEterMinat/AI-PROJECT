@echo off
title System Cleanup Tool
color 0E

echo.
echo ================================================================================
echo                        SYSTEM CLEANUP TOOL
echo ================================================================================
echo.
echo This tool will help you free up disk space by removing unnecessary files.
echo.
echo Current Disk Usage Analysis:
echo --------------------------------------------------------------------------------
echo [1] .venv (Python Env)      ~5.30 GB  (Required for running, do not delete)
echo [2] data/raw                ~1.34 GB  (Source data, KEEP THIS)
echo [3] data/processed          ~0.68 GB  (Can be regenerated)
echo [4] data/model_ready        ~0.50 GB  (Can be regenerated)
echo [5] models/flan_t5...       ~0.95 GB  (Unused model?)
echo [6] models/t5_extended...   ~0.23 GB  (Unused model?)
echo [7] Logs & Cache            ~0.01 GB  (Safe to delete)
echo.
echo ================================================================================
echo.

:MENU
echo Choose an option:
echo [1] Safe Cleanup (Cache, Logs, Temp files)
echo [2] Deep Cleanup (Data processed/ready) - Saves ~1.2 GB
echo [3] Remove Unused Models (Keep active one) - Saves ~1.2 GB
echo [4] FULL CLEANUP (All of the above) - Saves ~2.4 GB
echo [5] Exit
echo.
set /p CHOICE="Enter your choice (1-5): "

if "%CHOICE%"=="1" goto SAFE_CLEAN
if "%CHOICE%"=="2" goto DATA_CLEAN
if "%CHOICE%"=="3" goto MODEL_CLEAN
if "%CHOICE%"=="4" goto FULL_CLEAN
if "%CHOICE%"=="5" goto END

:SAFE_CLEAN
echo.
echo [INFO] Cleaning cache and logs...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
if exist ".pytest_cache" rd /s /q ".pytest_cache"
del /q logs\*.log
echo [OK] Cache and logs cleaned.
if "%CHOICE%"=="1" goto END
goto DATA_CLEAN

:DATA_CLEAN
echo.
echo [WARNING] This will delete processed data. You may need to re-run data processing scripts.
set /p CONFIRM="Are you sure? (y/n): "
if /i not "%CONFIRM%"=="y" goto END

echo [INFO] Cleaning processed data...
if exist "data\processed" (
    del /q data\processed\*.*
    echo [OK] data/processed cleaned.
)
if exist "data\model_ready" (
    del /q data\model_ready\*.*
    echo [OK] data/model_ready cleaned.
)
if "%CHOICE%"=="2" goto END
goto MODEL_CLEAN

:MODEL_CLEAN
echo.
echo [INFO] The active model is: models\t5_diagnosis_20251008_111522
echo [WARNING] This will delete other models:
echo  - models\flan_t5_diagnosis_20251018_205708
echo  - models\t5_diagnosis_extended
echo.
set /p CONFIRM="Are you sure? (y/n): "
if /i not "%CONFIRM%"=="y" goto END

if exist "models\flan_t5_diagnosis_20251018_205708" (
    rd /s /q "models\flan_t5_diagnosis_20251018_205708"
    echo [OK] Deleted flan_t5_diagnosis_20251018_205708
)
if exist "models\t5_diagnosis_extended" (
    rd /s /q "models\t5_diagnosis_extended"
    echo [OK] Deleted t5_diagnosis_extended
)
if "%CHOICE%"=="3" goto END
goto END

:FULL_CLEAN
REM Logic flows through 1 -> 2 -> 3
goto SAFE_CLEAN

:END
echo.
echo ================================================================================
echo Cleanup Complete!
echo Press any key to exit...
pause >nul
