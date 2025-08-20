#!/usr/bin/env python3
"""
Smoke test for HouseBrain training pipeline
Tests with minimal settings to ensure everything works
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_data_generation():
    """Test data generation"""
    print("🧪 Testing data generation...")
    
    # Clean up any existing test data
    if os.path.exists("test_smoke_data"):
        subprocess.run(["rm", "-rf", "test_smoke_data"])
    
    # Generate test data
    result = subprocess.run([
        "python", "generate_synthetic_v2.py", 
        "--out_dir", "test_smoke_data", 
        "--n", "50"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Data generation failed: {result.stderr}")
        return False
    
    print("✅ Data generation successful")
    
    # Check files were created
    files = list(Path("test_smoke_data").glob("*.json"))
    if len(files) != 50:
        print(f"❌ Expected 50 files, got {len(files)}")
        return False
    
    print(f"✅ Generated {len(files)} files")
    return True

def test_data_validation():
    """Test data validation"""
    print("🧪 Testing data validation...")
    
    try:
        from src.housebrain.validate_v2 import validate_v2_file
        
        # Test first few files
        files = list(Path("test_smoke_data").glob("*.json"))[:5]
        total_errors = 0
        
        for f in files:
            errors = validate_v2_file(str(f))
            hard_errors = [e for e in errors if not e.startswith('WARN:')]
            total_errors += len(hard_errors)
        
        if total_errors > 0:
            print(f"❌ Found {total_errors} validation errors")
            return False
        
        print("✅ Data validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def test_pipeline():
    """Test v2 pipeline"""
    print("🧪 Testing v2 pipeline...")
    
    try:
        from src.housebrain.pipeline_v2 import run_pipeline
        
        # Test with first file
        test_file = list(Path("test_smoke_data").glob("*.json"))[0]
        run_pipeline(str(test_file), "test_pipeline_output", sheet_modes=["floor"])
        
        # Check outputs
        outputs = list(Path("test_pipeline_output").glob("*"))
        if len(outputs) < 3:  # Should have SVG, DXF, BOQ at minimum
            print(f"❌ Expected at least 3 outputs, got {len(outputs)}")
            return False
        
        print("✅ Pipeline test successful")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_trainer_config():
    """Test trainer configuration without actual training"""
    print("🧪 Testing trainer configuration...")
    
    try:
        # Import trainer to check configuration
        from housebrain_colab_trainer import HouseBrainTrainer
        
        # Test with minimal config
        trainer = HouseBrainTrainer(
            dataset_path="test_smoke_data",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            output_dir="test_output",
            max_samples=10,  # Very small for testing
            batch_size=1,
            grad_accum_steps=1,
            epochs=1,
            max_length=512,  # Smaller for testing
            eval_steps=0,
            save_steps=5,
            save_total_limit=1
        )
        
        print("✅ Trainer configuration successful")
        return True
        
    except Exception as e:
        print(f"❌ Trainer configuration failed: {e}")
        return False

def main():
    """Run all smoke tests"""
    print("🚀 Starting HouseBrain Smoke Tests...")
    print("=" * 50)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Data Validation", test_data_validation),
        ("Pipeline", test_pipeline),
        ("Trainer Config", test_trainer_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for Colab training.")
        print("\n📝 Next steps:")
        print("1. Use the same commands in Colab")
        print("2. Ensure dataset directory exists and has files")
        print("3. Use the optimized settings from docs/OPTIMIZED_3_NOTEBOOK_V2_TRAINING.md")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    subprocess.run(["rm", "-rf", "test_smoke_data", "test_pipeline_output", "test_output"], 
                  capture_output=True)

if __name__ == "__main__":
    main()
