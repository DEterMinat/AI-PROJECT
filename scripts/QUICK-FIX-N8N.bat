@echo off
REM ==============================================================================
REM 🚀 QUICK FIX - แก้ปัญหา N8N Workflow
REM ==============================================================================

echo ========================================
echo    🚀 N8N WORKFLOW QUICK FIX
echo ========================================
echo.

echo ⚠️ PROBLEM: N8N workflow not imported = API returns 500 error
echo.
echo ✅ SOLUTION: Manual import workflow into N8N
echo.
echo ========================================
echo    📋 STEP BY STEP INSTRUCTIONS:
echo ========================================
echo.
echo Step 1: Check N8N is running
docker ps --filter "name=n8n" --format "{{.Names}}: {{.Status}}"
echo.

echo Step 2: Open N8N Web Interface
echo Opening browser...
start http://localhost:5678
timeout /t 3 >nul
echo.

echo Step 3: Open workflow file location
echo Opening folder...
start explorer "d:\AI-PROJECT\n8n"
timeout /t 2 >nul
echo.

echo ========================================
echo    🔧 MANUAL STEPS (Do this now):
echo ========================================
echo.
echo 1. In N8N web (http://localhost:5678):
echo    - Click "Workflows" in sidebar
echo    - Click "+" or "Add workflow"
echo    - Click "..." menu -^> "Import from file"
echo.
echo 2. Select file from opened folder:
echo    ^> medical_qa_workflow_complete.json
echo.
echo 3. After import:
echo    - Click "Active" toggle (make it blue/ON)
echo    - Press Ctrl+S to save
echo.
echo ========================================
echo    ✅ AFTER IMPORT - TEST:
echo ========================================
echo.

echo Testing N8N webhook...
powershell -Command "$body = @{question='I have fever'} | ConvertTo-Json; Invoke-WebRequest -Uri 'http://localhost:5678/webhook/medical-qa' -Method POST -Body $body -ContentType 'application/json' -UseBasicParsing | Select-Object -ExpandProperty Content"
echo.

echo ========================================
echo    🎯 EXPECTED RESULT:
echo ========================================
echo If successful, you should see JSON with:
echo - answer: (medical diagnosis)
echo - sources: (document references)
echo - method: "langchain"
echo.
echo If still shows error:
echo - Workflow not activated (toggle not blue)
echo - Webhook path wrong (should be "medical-qa")
echo - Check N8N logs: docker logs medical_n8n --tail 50
echo.

pause
