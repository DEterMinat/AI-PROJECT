#!/usr/bin/env python3
"""
🏥 T5-SMALL INTERACTIVE DISEASE DIAGNOSIS TESTER
🎯 For: General public users - Input symptoms → Get disease diagnosis
👨‍⚕️ Interactive CLI tool for testing T5-Small model

Usage:
    python scripts/Test_T5_Interactive.py --model models/t5_diagnosis_20251008_111522
    
Features:
    - Input symptoms in natural language
    - Get top-3 disease predictions with scores
    - Show confidence levels
    - Medical specialty classification
    - Easy to use CLI interface
"""

import torch
import argparse
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

class DiagnosisTester:
    """Interactive disease diagnosis tester"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🏥 T5-SMALL DISEASE DIAGNOSIS SYSTEM")
        print("="*70)
        print(f"📁 Loading model from: {self.model_path.name}")
        print(f"🖥️  Device: {self.device}")
        print()
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load disease mapping if available
        self.disease_mapping = {}
        mapping_file = self.model_path / "disease_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.disease_mapping = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"   Total diseases: {len(self.disease_mapping)}")
        print()
    
    def diagnose(self, symptoms, num_predictions=3):
        """
        Diagnose disease from symptoms
        
        Args:
            symptoms: Text description of symptoms
            num_predictions: Number of top predictions to return
        
        Returns:
            List of (disease, score) tuples
        """
        # Format input
        input_text = f"diagnose disease: {symptoms}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with beam search
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=32,
                num_beams=max(10, num_predictions * 2),  # More beams for diversity
                num_return_sequences=num_predictions,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Decode predictions
        predictions = []
        for i in range(min(num_predictions, len(outputs.sequences))):
            disease = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
            
            # Calculate confidence score (rough approximation)
            # Higher beam score = higher confidence
            if hasattr(outputs, 'sequences_scores'):
                score = outputs.sequences_scores[i].item()
                confidence = self._score_to_confidence(score)
            else:
                confidence = 1.0 / (i + 1)  # Fallback: rank-based confidence
            
            predictions.append((disease.strip(), confidence))
        
        return predictions
    
    def _score_to_confidence(self, score):
        """Convert beam score to confidence percentage"""
        # Beam scores are negative log probabilities
        # More negative = less confident
        # Convert to 0-100% range
        import math
        prob = math.exp(score)  # Convert log prob to prob
        confidence = min(100, prob * 100)  # Scale to percentage
        return confidence
    
    def get_specialty(self, disease):
        """Get medical specialty for a disease"""
        disease_lower = disease.lower()
        
        # Specialty mapping
        specialties = {
            'infectious': ['infection', 'pneumonia', 'bronchitis', 'flu', 'covid', 'tuberculosis', 'hepatitis', 'uti'],
            'respiratory': ['asthma', 'copd', 'pneumonia', 'bronchitis', 'emphysema'],
            'cardiovascular': ['hypertension', 'heart attack', 'angina', 'arrhythmia', 'stroke', 'heart disease'],
            'digestive': ['gastritis', 'gerd', 'ulcer', 'ibs', 'crohn', 'colitis', 'appendicitis'],
            'neurological': ['migraine', 'headache', 'alzheimer', 'parkinson', 'epilepsy', 'stroke', 'meningitis'],
            'metabolic': ['diabetes', 'thyroid', 'obesity', 'hypothyroidism', 'hyperthyroidism'],
            'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd', 'ocd'],
            'gynecological': ['pregnancy', 'pcos', 'endometriosis', 'menopause', 'fibroids'],
            'musculoskeletal': ['arthritis', 'osteoporosis', 'fibromyalgia', 'gout', 'fracture', 'tendinitis'],
            'dermatological': ['eczema', 'psoriasis', 'acne', 'dermatitis', 'rash', 'hives'],
            'oncological': ['cancer', 'tumor', 'leukemia', 'lymphoma', 'carcinoma'],
            'renal': ['kidney disease', 'renal failure', 'kidney stones', 'nephritis'],
            'pediatric': ['measles', 'chickenpox', 'mumps', 'rubella', 'whooping cough']
        }
        
        for specialty, keywords in specialties.items():
            if any(keyword in disease_lower for keyword in keywords):
                return specialty.capitalize()
        
        return 'General Medicine'
    
    def format_prediction(self, disease, confidence, rank):
        """Format prediction for display"""
        specialty = self.get_specialty(disease)
        
        # Confidence level
        if confidence >= 70:
            level = "🟢 HIGH"
            color = "\033[92m"  # Green
        elif confidence >= 40:
            level = "🟡 MODERATE"
            color = "\033[93m"  # Yellow
        else:
            level = "🔴 LOW"
            color = "\033[91m"  # Red
        reset = "\033[0m"
        
        return f"{color}   {rank}. {disease.upper():20s} - {specialty:20s} - {level} ({confidence:.1f}%){reset}"
    
    def interactive_mode(self):
        """Run interactive testing mode"""
        print("🎯 INTERACTIVE DIAGNOSIS MODE")
        print("="*70)
        print("Enter symptoms to get disease diagnosis.")
        print("Type 'quit' or 'exit' to stop.")
        print()
        
        # Sample examples
        examples = [
            "I have a fever, cough, and difficulty breathing",
            "Severe headache, nausea, and sensitivity to light",
            "Chest pain, shortness of breath, and palpitations",
            "Abdominal pain, bloating, and diarrhea",
            "Fatigue, weight gain, and cold intolerance"
        ]
        
        print("📝 Example symptoms:")
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
        print()
        
        while True:
            try:
                # Get input
                print("-"*70)
                symptoms = input("🩺 Your symptoms: ").strip()
                
                if not symptoms:
                    continue
                
                if symptoms.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using the diagnosis system!")
                    break
                
                # Check if it's an example number
                if symptoms.isdigit() and 1 <= int(symptoms) <= len(examples):
                    symptoms = examples[int(symptoms) - 1]
                    print(f"   Using example: {symptoms}")
                
                print()
                print("🔍 Analyzing symptoms...")
                print()
                
                # Get predictions
                predictions = self.diagnose(symptoms, num_predictions=3)
                
                if not predictions:
                    print("   ❌ Could not generate diagnosis. Please try different symptoms.")
                    continue
                
                # Display results
                print("📊 DIAGNOSIS RESULTS:")
                print()
                for i, (disease, confidence) in enumerate(predictions, 1):
                    print(self.format_prediction(disease, confidence, i))
                
                print()
                print("⚠️  DISCLAIMER:")
                print("   This is an AI prediction for educational purposes only.")
                print("   Always consult a qualified healthcare professional for medical advice.")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\n   ❌ Error: {e}")
                print("   Please try again.")
    
    def batch_test(self, test_cases):
        """Test multiple cases at once"""
        print("🧪 BATCH TESTING MODE")
        print("="*70)
        print()
        
        for i, symptoms in enumerate(test_cases, 1):
            print(f"Test Case {i}:")
            print(f"   Symptoms: {symptoms}")
            print()
            
            predictions = self.diagnose(symptoms, num_predictions=3)
            
            print("   Predictions:")
            for j, (disease, confidence) in enumerate(predictions, 1):
                print(f"   {self.format_prediction(disease, confidence, j)}")
            
            print()
            print("-"*70)
            print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="T5-Small Interactive Disease Diagnosis Tester")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained T5-Small model")
    parser.add_argument("--batch", nargs='+', type=str,
                       help="Batch test mode: provide symptoms as arguments")
    
    args = parser.parse_args()
    
    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Please check the path and try again.")
        return 1
    
    # Initialize tester
    tester = DiagnosisTester(args.model)
    
    # Run mode
    if args.batch:
        # Batch mode
        tester.batch_test(args.batch)
    else:
        # Interactive mode
        tester.interactive_mode()
    
    return 0


if __name__ == "__main__":
    exit(main())
