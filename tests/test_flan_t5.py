#!/usr/bin/env python3
"""Quick test of FLAN-T5 model"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.medical_ai import LangchainMedicalService

print("\n" + "="*60)
print("🧪 TESTING FLAN-T5-BASE MODEL")
print("="*60 + "\n")

# Initialize service
service = LangchainMedicalService()

# Test questions
test_cases = [
    "fever, cough, sore throat",
    "severe headache, stiff neck, sensitivity to light",
    "chest pain, shortness of breath, sweating"
]

for i, symptoms in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {symptoms}")
    print("="*60)
    
    result = service.ask_question(symptoms)
    
    print(f"\n✅ Answer: {result.get('answer', 'No answer')[:300]}")
    print(f"\n📊 Stats:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Time: {result.get('processing_time', 0):.2f}s")
    print(f"   Sources: {len(result.get('sources', []))}")

print("\n" + "="*60)
print("✅ Testing complete!")
print("="*60 + "\n")
