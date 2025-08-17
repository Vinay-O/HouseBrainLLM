#!/usr/bin/env python3
"""
HouseBrain Colab Setup Script
Quick setup for training on Colab Pro+ with 1M dataset
"""

import os
import subprocess
import sys
from pathlib import Path

def check_gpu():
    """Check GPU availability and info"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            return True
        else:
            print("❌ No GPU available. Please use GPU runtime.")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch==2.1.0",
        "transformers==4.36.0", 
        "peft==0.7.0",
        "accelerate==0.25.0",
        "datasets==2.15.0",
        "tqdm",
        "numpy",
        "matplotlib"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ Dependencies installed successfully!")

def extract_dataset():
    """Extract the dataset if tar.gz exists"""
    if Path("housebrain_dataset_r1_super_1M.tar.gz").exists():
        print("📁 Extracting dataset...")
        subprocess.check_call(["tar", "-xzf", "housebrain_dataset_r1_super_1M.tar.gz"])
        print("✅ Dataset extracted successfully!")
        
        # Check dataset structure
        train_files = len(list(Path("housebrain_dataset_r1_super_1M/train").rglob("*.json")))
        val_files = len(list(Path("housebrain_dataset_r1_super_1M/validation").rglob("*.json")))
        print(f"📊 Train files: {train_files:,}")
        print(f"📊 Validation files: {val_files:,}")
    else:
        print("⚠️ Dataset file not found. Please upload housebrain_dataset_r1_super_1M.tar.gz")

def check_dataset_quality():
    """Check dataset quality and distribution"""
    print("🔍 Checking dataset quality...")
    
    try:
        import json
        from collections import Counter
        
        # Check first few samples
        train_dir = Path("housebrain_dataset_r1_super_1M/train")
        problem_types = []
        
        for json_file in list(train_dir.rglob("*.json"))[:100]:  # Check first 100
            with open(json_file, 'r') as f:
                data = json.load(f)
                problem_types.append(data['input']['problem_type'])
        
        # Count problem types
        counter = Counter(problem_types)
        total = sum(counter.values())
        
        print("📈 Problem type distribution (sample):")
        for pt, count in counter.most_common():
            percentage = (count / total) * 100
            print(f"  {pt}: {count} ({percentage:.1f}%)")
        
        # Check geometric construction samples
        geo_count = counter.get('Geometric_Construction', 0)
        geo_percentage = (geo_count / total) * 100
        print(f"\n🎯 Geometric_Construction: {geo_count} ({geo_percentage:.1f}%)")
        
        if geo_percentage > 50:
            print("✅ Excellent geometric focus!")
        elif geo_percentage > 30:
            print("✅ Good geometric focus")
        else:
            print("⚠️ Low geometric focus")
            
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")

def setup_training():
    """Setup training environment"""
    print("🚀 Setting up training environment...")
    
    # Create output directory
    Path("housebrain-r1-super-trained").mkdir(exist_ok=True)
    
    # Check if training script exists
    if not Path("colab_proplus_train_r1_super.py").exists():
        print("❌ Training script not found. Please upload colab_proplus_train_r1_super.py")
        return False
    
    print("✅ Training environment ready!")
    return True

def main():
    """Main setup function"""
    print("🏠 HouseBrain Colab Setup")
    print("=" * 50)
    
    # Check GPU
    if not check_gpu():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Extract dataset
    extract_dataset()
    
    # Check dataset quality
    check_dataset_quality()
    
    # Setup training
    if setup_training():
        print("\n🎉 Setup complete! Ready for training.")
        print("\n📋 Next steps:")
        print("1. Run: python colab_proplus_train_r1_super.py")
        print("2. Monitor: tail -f training_log_r1_super.txt")
        print("3. Check metrics: cat training_metrics_r1_super.json")
    else:
        print("\n❌ Setup incomplete. Please check errors above.")

if __name__ == "__main__":
    main()
