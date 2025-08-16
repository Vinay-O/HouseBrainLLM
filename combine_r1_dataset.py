#!/usr/bin/env python3
"""
Combine R1-Enhanced Dataset
Merges existing 575K dataset with new R1 reasoning samples
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

def combine_r1_datasets():
    """Combine existing dataset with R1-enhanced reasoning samples"""
    
    # Source directories
    existing_dataset = Path("housebrain_dataset_v6_1M")
    r1_enhanced_dataset = Path("housebrain_dataset_r1_enhanced")
    
    # Target directory
    target_dir = Path("housebrain_dataset_r1_final")
    target_train = target_dir / "train"
    target_validation = target_dir / "validation"
    
    target_train.mkdir(parents=True, exist_ok=True)
    target_validation.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Combining datasets for R1 optimization...")
    
    # Copy existing dataset samples
    print("📁 Copying existing 575K samples...")
    
    if existing_dataset.exists():
        # Copy training samples
        existing_train = existing_dataset / "train"
        if existing_train.exists():
            existing_train_files = list(existing_train.glob("*.json"))
            for file in tqdm(existing_train_files, desc="Copying existing train files"):
                shutil.copy2(file, target_train / file.name)
        
        # Copy validation samples
        existing_validation = existing_dataset / "validation"
        if existing_validation.exists():
            existing_validation_files = list(existing_validation.glob("*.json"))
            for file in tqdm(existing_validation_files, desc="Copying existing validation files"):
                shutil.copy2(file, target_validation / file.name)
    
    # Copy R1-enhanced reasoning samples
    print("🧠 Copying R1-enhanced reasoning samples...")
    
    if r1_enhanced_dataset.exists():
        # Copy R1 training samples
        r1_train = r1_enhanced_dataset / "train"
        if r1_train.exists():
            r1_train_files = list(r1_train.glob("*.json"))
            for file in tqdm(r1_train_files, desc="Copying R1 train files"):
                shutil.copy2(file, target_train / file.name)
        
        # Copy R1 validation samples
        r1_validation = r1_enhanced_dataset / "validation"
        if r1_validation.exists():
            r1_validation_files = list(r1_validation.glob("*.json"))
            for file in tqdm(r1_validation_files, desc="Copying R1 validation files"):
                shutil.copy2(file, target_validation / file.name)
    
    # Count total samples
    total_train_files = len(list(target_train.glob("*.json")))
    total_validation_files = len(list(target_validation.glob("*.json")))
    
    # Create dataset info
    dataset_info = {
        "name": "housebrain_dataset_r1_final",
        "version": "R1_v1.0",
        "description": "Combined HouseBrain dataset optimized for DeepSeek-R1-Distill-Qwen-7B",
        "total_samples": total_train_files + total_validation_files,
        "train_samples": total_train_files,
        "validation_samples": total_validation_files,
        "composition": {
            "existing_samples": "575K high-quality architectural designs",
            "r1_enhanced_samples": "100K complex reasoning tasks",
            "reasoning_tasks": [
                "Code_Compliance_Analysis",
                "Multi_Constraint_Optimization", 
                "Cost_Optimization_Calculations",
                "Conflict_Resolution_Strategies",
                "Step_By_Step_Design_Analysis"
            ]
        },
        "model_optimized_for": "DeepSeek-R1-Distill-Qwen-7B",
        "training_features": [
            "Chat-style formatting with system prompts",
            "Assistant-only masked loss",
            "Complex reasoning with mathematical calculations",
            "Multi-step architectural problem solving",
            "Code compliance and regulatory analysis"
        ]
    }
    
    with open(target_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n🎉 Dataset combination complete!")
    print(f"📊 Total samples: {dataset_info['total_samples']:,}")
    print(f"📈 Training samples: {total_train_files:,}")
    print(f"📉 Validation samples: {total_validation_files:,}")
    print(f"🧠 R1-optimized for: {dataset_info['model_optimized_for']}")
    print(f"📁 Saved to: {target_dir}")
    
    return target_dir

def update_training_configs():
    """Update training scripts to use the combined R1 dataset"""
    
    print("\n🔧 Updating training configurations...")
    
    # Update simple training script
    simple_script = Path("colab_proplus_train_simple.py")
    if simple_script.exists():
        with open(simple_script, 'r') as f:
            content = f.read()
        
        # Update dataset path
        content = content.replace(
            'DATASET_PATH = "housebrain_dataset_v6_1M"',
            'DATASET_PATH = "housebrain_dataset_r1_final"'
        )
        
        with open(simple_script, 'w') as f:
            f.write(content)
        print("✅ Updated colab_proplus_train_simple.py")
    
    # Update notification training script
    notification_script = Path("colab_proplus_train_with_notifications.py")
    if notification_script.exists():
        with open(notification_script, 'r') as f:
            content = f.read()
        
        # Update dataset path
        content = content.replace(
            'DATASET_PATH = "housebrain_dataset_v6_1M"',
            'DATASET_PATH = "housebrain_dataset_r1_final"'
        )
        
        with open(notification_script, 'w') as f:
            f.write(content)
        print("✅ Updated colab_proplus_train_with_notifications.py")

def main():
    """Main function"""
    print("🚀 HouseBrain R1 Dataset Optimization")
    print("=" * 50)
    
    # Step 1: Generate R1-enhanced dataset
    print("\n🧠 Step 1: Generating R1-enhanced reasoning samples...")
    try:
        from generate_r1_enhanced_dataset import main as generate_r1
        generate_r1()
        print("✅ R1-enhanced dataset generated successfully!")
    except Exception as e:
        print(f"❌ Error generating R1 dataset: {e}")
        print("⚠️ Proceeding with existing dataset only...")
    
    # Step 2: Combine datasets
    print("\n🔄 Step 2: Combining datasets...")
    target_dir = combine_r1_datasets()
    
    # Step 3: Update training configs
    print("\n🔧 Step 3: Updating training configurations...")
    update_training_configs()
    
    print("\n🎉 R1 Dataset Optimization Complete!")
    print("=" * 50)
    print("📊 Your dataset is now optimized for DeepSeek-R1-Distill-Qwen-7B")
    print("🧠 Includes complex reasoning tasks and mathematical calculations")
    print("💡 Ready for training with enhanced reasoning capabilities!")
    print(f"📁 Dataset location: {target_dir}")

if __name__ == "__main__":
    main()
