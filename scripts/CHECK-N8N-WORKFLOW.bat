@echo off
REM ==============================================================================
REM 🔍 CHECK N8N WORKFLOW STATUS
REM ==============================================================================

echo ========================================
echo    🔍 CHECKING N8N WORKFLOW STATUS
echo ========================================
echo.

REM ตรวจสอบ N8N container
echo 📦 Checking N8N Docker container...
docker ps --filter "name=n8n" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

REM ตรวจสอบ N8N Health
echo 🏥 Checking N8N health endpoint...
curl -s http://localhost:5678/healthz
echo.
echo.

REM ตรวจสอบ Webhook endpoint
echo 🔗 Testing N8N webhook endpoint...
echo POST http://localhost:5678/webhook/medical-qa
curl -X POST http://localhost:5678/webhook/medical-qa -H "Content-Type: application/json" -d "{\"question\":\"test\"}"
echo.
echo.

REM ตรวจสอบไฟล์ Workflow
echo 📁 Checking workflow file...
set "PROJECT_ROOT=%~dp0.."
if exist "%PROJECT_ROOT%\n8n\medical_qa_workflow_complete.json" (
    echo ✅ Workflow file exists: medical_qa_workflow_complete.json
    powershell -Command "(Get-Content '%PROJECT_ROOT%\n8n\medical_qa_workflow_complete.json' | ConvertFrom-Json).nodes | ForEach-Object { Write-Host '  - Node:' $_.name '(' $_.type ')' }"
) else (
    echo ❌ Workflow file NOT found!
)
echo.

echo ========================================
echo    📋 WORKFLOW STATUS SUMMARY
echo ========================================
echo.
echo ⚠️ If webhook returns error:
echo    1. N8N is running BUT workflow NOT imported
echo    2. Go to http://localhost:5678
echo    3. Import: n8n\medical_qa_workflow_complete.json
echo    4. Activate workflow (toggle ON)
echo.

pause
