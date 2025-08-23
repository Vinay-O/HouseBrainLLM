#!/usr/bin/env python3
"""
Model Evaluation Script for HouseBrain Enhancement
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

def evaluate_housebrain_model():
    """Evaluate enhanced HouseBrain model"""
    
    print("🧪 Evaluating HouseBrain Enhanced Model")
    print("=" * 50)
    
    # Load enhanced model
    model_path = "models/housebrain_deepseek_enhanced"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test prompts
    test_prompts = [
        "Generate a comprehensive architectural design with professional 2D floor plans and 3D models. Requirements: - Total Area: 2000 sqft - Floors: 2 - Style: Modern Contemporary",
        "Create a professional drawing set with 17 sheets including floor plans, electrical, plumbing, HVAC, structural, elevations, sections, and details.",
        "Generate a BIM-quality 3D model with PBR materials, advanced lighting, and export formats for obj, gltf, and ifc."
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🧪 Test {i+1}: {prompt[:50]}...")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result = {
            "prompt": prompt,
            "generated": generated_text,
            "length": len(generated_text),
            "quality_indicators": {
                "mentions_17_sheets": "17" in generated_text and "sheet" in generated_text.lower(),
                "mentions_bim": "bim" in generated_text.lower(),
                "mentions_pbr": "pbr" in generated_text.lower(),
                "mentions_precision": "0.001mm" in generated_text or "precision" in generated_text.lower(),
                "mentions_formats": any(fmt in generated_text.lower() for fmt in ["obj", "gltf", "ifc"])
            }
        }
        
        results.append(result)
        
        # Print summary
        quality_score = sum(result["quality_indicators"].values()) / len(result["quality_indicators"])
        print(f"✅ Generated {result['length']} characters")
        print(f"📊 Quality Score: {quality_score:.1%}")
    
    # Save evaluation results
    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation completed!")
    print("📁 Results saved to: evaluation_results.json")

if __name__ == "__main__":
    evaluate_housebrain_model()\n