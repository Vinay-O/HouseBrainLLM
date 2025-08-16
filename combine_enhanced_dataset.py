#!/usr/bin/env python3
"""
Combine Enhanced Dataset Batches
Merges all enhanced dataset batches into one comprehensive dataset
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

def combine_datasets():
    """Combine all enhanced dataset batches"""
    
    # Source directories
    base_dir = Path(".")
    source_dirs = [
        "housebrain_dataset_v6_enhanced_batch_1",
        "housebrain_dataset_v6_enhanced_batch_2", 
        "housebrain_dataset_v6_enhanced_batch_3",
        "housebrain_dataset_v6_enhanced_batch_4",
        "housebrain_dataset_v6_enhanced_batch_5",
        "housebrain_dataset_v6_enhanced_batch_6"
    ]
    
    # Target directory
    target_dir = Path("housebrain_dataset_v6_1M")
    target_train = target_dir / "train"
    target_validation = target_dir / "validation"
    
    # Create target directories
    target_train.mkdir(parents=True, exist_ok=True)
    target_validation.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Combining enhanced dataset batches...")
    
    total_train_files = 0
    total_validation_files = 0
    
    # Combine each batch
    for batch_dir in source_dirs:
        source_path = base_dir / batch_dir
        if not source_path.exists():
            print(f"⚠️ Warning: {batch_dir} not found, skipping...")
            continue
            
        print(f"📁 Processing {batch_dir}...")
        
        # Copy training files
        train_source = source_path / "train"
        if train_source.exists():
            train_files = list(train_source.glob("*.json"))
            for file in tqdm(train_files, desc=f"Copying train files from {batch_dir}"):
                shutil.copy2(file, target_train / file.name)
                total_train_files += 1
        
        # Copy validation files
        validation_source = source_path / "validation"
        if validation_source.exists():
            validation_files = list(validation_source.glob("*.json"))
            for file in tqdm(validation_files, desc=f"Copying validation files from {batch_dir}"):
                shutil.copy2(file, target_validation / file.name)
                total_validation_files += 1
    
    # Create dataset info
    dataset_info = {
        "name": "housebrain_dataset_v6_1M",
        "version": "v6.0",
        "description": "Enhanced HouseBrain dataset with 1M+ high-quality samples",
        "total_samples": total_train_files + total_validation_files,
        "train_samples": total_train_files,
        "validation_samples": total_validation_files,
        "quality_threshold": 0.8,
        "features": [
            "Enhanced architectural designs",
            "Indian market focus (40%)",
            "Quality gates and validation",
            "Regional variations",
            "Climate considerations",
            "NBC 2016 compliance",
            "Vastu compliance options",
            "Green building features"
        ],
        "generated_from": source_dirs
    }
    
    # Save dataset info
    with open(target_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n🎉 Dataset combination complete!")
    print(f"📊 Total samples: {dataset_info['total_samples']:,}")
    print(f"📈 Training samples: {total_train_files:,}")
    print(f"📉 Validation samples: {total_validation_files:,}")
    print(f"📁 Saved to: {target_dir}")
    
    return target_dir

if __name__ == "__main__":
    combine_datasets()
