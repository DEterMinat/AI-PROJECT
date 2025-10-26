# N8N Workflows for Medical AI

## Workflows

### 1. Medical Q&A Workflow (medical_qa_workflow.json)
- รับคำถามผ่าน webhook
- ส่งไปยัง FastAPI
- ส่งคำตอบกลับ

## การ Import Workflow

1. เข้า N8N: http://localhost:5678
2. กด "New workflow" 
3. กด "Import from file"
4. เลือกไฟล์ .json
5. Save และ Activate

## Webhook URLs

- Medical Q&A: `http://localhost:5678/webhook/medical-qa`

## Test Request

```bash
curl -X POST "http://localhost:5678/webhook/medical-qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "อาการของโรคเบาหวานคืออะไร?", "user_id": "test"}'
```