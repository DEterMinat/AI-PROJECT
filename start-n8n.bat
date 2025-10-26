@echo off
title N8N Workflow Server
color 0B

echo.
echo ========================================
echo   N8N WORKFLOW AUTOMATION SERVER
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)

echo [INFO] Docker is running
echo.
echo [INFO] Starting N8N with Docker Compose...
echo.

REM Start N8N using docker-compose
docker-compose -f docker-compose.yml up n8n

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start N8N!
    echo.
    echo Troubleshooting:
    echo 1. Check Docker Desktop is running
    echo 2. Check port 5678 is not in use
    echo 3. Try: docker-compose -f docker-compose-working.yml down
    echo    Then run this script again
    echo.
    pause
)
