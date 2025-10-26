@echo off
REM ==============================================================================
REM 🧹 CLEAN SYSTEM - ลบไฟล์ชั่วคราวและ Cache
REM ==============================================================================

echo ========================================
echo    🧹 CLEANING AI-PROJECT SYSTEM
echo ========================================
echo.

REM ลบ Python Cache Files
echo 🗑️ Cleaning Python cache files...
for /r "d:\AI-PROJECT" %%i in (__pycache__) do (
    if exist "%%i" (
        echo Removing: %%i
        rmdir /s /q "%%i" 2>nul
    )
)

REM ลบไฟล์ .pyc
echo 🗑️ Cleaning .pyc files...
del /s /q "d:\AI-PROJECT\*.pyc" 2>nul

REM ลบไฟล์ Log เก่า (เก็บแค่ 5 ไฟล์ล่าสุด)
echo 📝 Cleaning old log files...
pushd "d:\AI-PROJECT\logs"
for /f "skip=5 delims=" %%f in ('dir /b /o-d *.log 2^>nul') do (
    echo Removing old log: %%f
    del /q "%%f" 2>nul
)
popd

REM ลบไฟล์ .tmp และ .bak
echo 🗑️ Cleaning temporary files...
del /s /q "d:\AI-PROJECT\*.tmp" 2>nul
del /s /q "d:\AI-PROJECT\*.bak" 2>nul
del /s /q "d:\AI-PROJECT\*~" 2>nul

REM ล้าง pip cache
echo 🗑️ Cleaning pip cache...
pip cache purge 2>nul

echo.
echo ========================================
echo    ✅ CLEANUP COMPLETED!
echo ========================================
echo.
echo 📊 Disk space freed. System is clean!
echo.
pause
