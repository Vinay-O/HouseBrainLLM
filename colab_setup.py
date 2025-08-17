#!/usr/bin/env python3
"""
HouseBrain Colab Setup Script
Quick setup for training on Colab Pro+ with 10K test or 1M full dataset
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

def find_dataset():
    """Find available dataset files"""
    dataset_files = []
    
    # Check for 10K test dataset
    if Path("housebrain_dataset_r1_super_10k_aug.tar.gz").exists():
        dataset_files.append("housebrain_dataset_r1_super_10k_aug.tar.gz")
    
    # Check for 1M full dataset
    if Path("housebrain_dataset_r1_super_1M.tar.gz").exists():
        dataset_files.append("housebrain_dataset_r1_super_1M.tar.gz")
    
    # Check for 1M augmented dataset
    if Path("housebrain_dataset_r1_super_1M_aug_v1_1.tar.gz").exists():
        dataset_files.append("housebrain_dataset_r1_super_1M_aug_v1_1.tar.gz")
    
    return dataset_files

def extract_dataset():
    """Extract the dataset if tar.gz exists"""
    dataset_files = find_dataset()
    
    if not dataset_files:
        print("⚠️ No dataset files found. Please upload one of:")
        print("  - housebrain_dataset_r1_super_10k_aug.tar.gz (32MB - test)")
        print("  - housebrain_dataset_r1_super_1M.tar.gz (2.9GB - full)")
        print("  - housebrain_dataset_r1_super_1M_aug_v1_1.tar.gz (2.9GB - full augmented)")
        return None
    
    # Use the first available dataset
    dataset_file = dataset_files[0]
    print(f"📁 Found dataset: {dataset_file}")
    
    # Determine dataset name from file
    if "10k" in dataset_file:
        dataset_name = "housebrain_dataset_r1_super_10k_aug"
        print("🎯 Using 10K test dataset")
    elif "aug_v1_1" in dataset_file:
        dataset_name = "housebrain_dataset_r1_super_1M_aug_v1_1"
        print("🎯 Using 1M augmented dataset (v1.1)")
    else:
        dataset_name = "housebrain_dataset_r1_super_1M"
        print("🎯 Using 1M full dataset")
    
    print(f"📁 Extracting {dataset_file}...")
    subprocess.check_call(["tar", "-xzf", dataset_file])
    print("✅ Dataset extracted successfully!")
    
    # Check dataset structure
    train_files = len(list(Path(f"{dataset_name}/train").rglob("*.json")))
    val_files = len(list(Path(f"{dataset_name}/validation").rglob("*.json")))
    print(f"📊 Train files: {train_files:,}")
    print(f"📊 Validation files: {val_files:,}")
    
    return dataset_name

def check_dataset_quality():
    """Check dataset quality and distribution"""
    print("🔍 Checking dataset quality...")
    
    try:
        import json
        from collections import Counter
        
        # Find the extracted dataset directory
        possible_dirs = [
            "housebrain_dataset_r1_super_10k_aug",
            "housebrain_dataset_r1_super_1M_aug_v1_1", 
            "housebrain_dataset_r1_super_1M"
        ]
        
        dataset_dir = None
        for dir_name in possible_dirs:
            if Path(dir_name).exists():
                dataset_dir = dir_name
                break
        
        if not dataset_dir:
            print("❌ No dataset directory found")
            return
        
        # Check first few samples
        train_dir = Path(f"{dataset_dir}/train")
        problem_types = []
        
        json_files = list(train_dir.rglob("*.json"))
        if not json_files:
            print("❌ No JSON files found in train directory")
            return
        
        # Check first 100 files or all if less than 100
        check_files = json_files[:min(100, len(json_files))]
        
        for json_file in check_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    problem_types.append(data['input']['problem_type'])
            except Exception as e:
                print(f"⚠️ Error reading {json_file}: {e}")
                continue
        
        if not problem_types:
            print("❌ No valid problem types found")
            return
        
        # Count problem types
        counter = Counter(problem_types)
        total = sum(counter.values())
        
        if total == 0:
            print("❌ No problem types found")
            return
        
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
        
        # Check for augmented features if using augmented dataset
        if "aug" in dataset_dir:
            print("\n🔧 Checking augmented features...")
            sample_file = json_files[0]
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
                output = sample_data.get('output', {})
                
                has_meta = 'metadata_augmented_v1_1' in output
                has_levels = 'levels' in output
                has_2d_dims = 'dimensions_2d' in output
                
                print(f"  Units & Datum: {'✅' if has_meta else '❌'}")
                print(f"  Floor Levels: {'✅' if has_levels else '❌'}")
                print(f"  2D Dimensions: {'✅' if has_2d_dims else '❌'}")
                
                if has_meta and has_levels and has_2d_dims:
                    print("✅ All v1.1 augmentation features present!")
            
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")

def setup_training():
    """Setup training environment"""
    print("🚀 Setting up training environment...")
    
    # Create output directory
    Path("housebrain-r1-super-trained").mkdir(exist_ok=True)
    Path("housebrain-10k-test-trained").mkdir(exist_ok=True)
    
    # Check for training scripts
    has_10k_script = Path("colab_10k_test_train.py").exists()
    has_1m_script = Path("colab_proplus_train_r1_super.py").exists()
    
    if not has_10k_script and not has_1m_script:
        print("❌ No training scripts found. Please upload:")
        print("  - colab_10k_test_train.py (for 10K test)")
        print("  - colab_proplus_train_r1_super.py (for 1M full training)")
        return False
    
    if has_10k_script:
        print("✅ 10K test training script found")
    if has_1m_script:
        print("✅ 1M full training script found")
    
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
    dataset_name = extract_dataset()
    
    # Check dataset quality
    check_dataset_quality()
    
    # Setup training
    if setup_training():
        print("\n🎉 Setup complete! Ready for training.")
        print("\n📋 Next steps:")
        
        has_10k_script = Path("colab_10k_test_train.py").exists()
        has_1m_script = Path("colab_proplus_train_r1_super.py").exists()
        
        if has_10k_script:
            print("1. For 10K test: python colab_10k_test_train.py")
            print("2. Monitor: tail -f training_log_10k_test.txt")
            print("3. Check metrics: cat training_metrics_10k_test.json")
        
        if has_1m_script:
            print("4. For 1M full training: python colab_proplus_train_r1_super.py")
            print("5. Monitor: tail -f training_log_r1_super.txt")
            print("6. Check metrics: cat training_metrics_r1_super.json")
        
        if dataset_name and "10k" in dataset_name:
            print("\n🎯 Using 10K test dataset - perfect for validation!")
            print("   After successful test, upload 1M dataset for full training.")
    else:
        print("\n❌ Setup incomplete. Please check errors above.")

if __name__ == "__main__":
    main()
