#!/usr/bin/env python3
"""
HouseBrain Dataset Splitter

Splits the 500K dataset into 6 equal parts for parallel training:
- 3 Colab accounts
- 3 Kaggle accounts
- Each gets ~83K samples
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse

def split_dataset(source_path: str, output_dir: str, num_splits: int = 6):
    """Split dataset into equal parts"""
    print(f"🔪 Splitting dataset: {source_path}")
    print(f"📊 Creating {num_splits} equal parts")
    
    # Load all training files
    train_path = Path(source_path) / "train"
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    
    # Get all JSON files
    json_files = list(train_path.glob("*.json"))
    print(f"📄 Found {len(json_files)} training files")
    
    # Shuffle files for random distribution
    random.shuffle(json_files)
    
    # Calculate samples per split
    total_samples = len(json_files)
    samples_per_split = total_samples // num_splits
    extra_samples = total_samples % num_splits
    
    print(f"📊 {samples_per_split} samples per split")
    print(f"📊 {extra_samples} extra samples to distribute")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split files
    start_idx = 0
    for split_num in range(num_splits):
        # Calculate samples for this split
        if split_num < extra_samples:
            split_size = samples_per_split + 1
        else:
            split_size = samples_per_split
        
        end_idx = start_idx + split_size
        
        # Create split directory
        split_name = f"split_{split_num + 1:02d}"
        split_dir = output_path / split_name
        train_split_dir = split_dir / "train"
        train_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files for this split
        split_files = json_files[start_idx:end_idx]
        print(f"📁 Creating {split_name}: {len(split_files)} samples")
        
        for file_path in split_files:
            dest_path = train_split_dir / file_path.name
            shutil.copy2(file_path, dest_path)
        
        # Create dataset info
        dataset_info = {
            "name": f"HouseBrain Dataset Split {split_num + 1}",
            "split": split_num + 1,
            "total_splits": num_splits,
            "samples": len(split_files),
            "source": source_path,
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        with open(split_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        start_idx = end_idx
    
    print(f"✅ Dataset split complete!")
    print(f"📁 Output directory: {output_path}")
    
    # Create summary
    create_split_summary(output_path, num_splits)

def create_split_summary(output_path: Path, num_splits: int):
    """Create summary of all splits"""
    summary = {
        "total_splits": num_splits,
        "splits": []
    }
    
    for split_num in range(num_splits):
        split_name = f"split_{split_num + 1:02d}"
        split_dir = output_path / split_name
        train_dir = split_dir / "train"
        
        if train_dir.exists():
            sample_files = list(train_dir.glob("*.json"))
            sample_count = len(sample_files)
            # Gather lightweight stats for sanity
            few = sample_files[:10]
            avg_tokens_hint = None
            try:
                lengths = []
                for fp in few:
                    with open(fp, 'r') as f:
                        js = json.load(f)
                        text_len = len(json.dumps(js))
                        lengths.append(text_len)
                if lengths:
                    avg_tokens_hint = int(sum(lengths) / len(lengths) / 4)  # rough bytes->tokens
            except Exception:
                avg_tokens_hint = None
            summary["splits"].append({
                "name": split_name,
                "samples": sample_count,
                "path": str(split_dir),
                "avg_tokens_hint": avg_tokens_hint
            })
    
    with open(output_path / "split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Split summary created: {output_path / 'split_summary.json'}")

def create_training_configs(output_path: Path):
    """Create training configurations for each platform"""
    
    # Colab configurations
    colab_configs = []
    for i in range(3):
        config = {
            "platform": "colab",
            "account": f"colab_{i + 1}",
            "dataset_path": f"split_{i + 1:02d}",
            "max_samples": None,  # Use all samples in split
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_length": 512,
            "num_epochs": 2,
            "learning_rate": 2e-4,
            "lora_r": 8,
            "output_dir": f"models/housebrain-colab-{i + 1}",
            "save_steps": 100,  # Save frequently
            "logging_steps": 10,
            "expected_time": "8-10 hours"
        }
        colab_configs.append(config)
    
    # Kaggle configurations
    kaggle_configs = []
    for i in range(3):
        config = {
            "platform": "kaggle",
            "account": f"kaggle_{i + 1}",
            "dataset_path": f"split_{i + 4:02d}",
            "max_samples": None,  # Use all samples in split
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_length": 512,
            "num_epochs": 2,
            "learning_rate": 2e-4,
            "lora_r": 8,
            "output_dir": f"models/housebrain-kaggle-{i + 1}",
            "save_steps": 100,  # Save frequently
            "logging_steps": 10,
            "expected_time": "8-10 hours"
        }
        kaggle_configs.append(config)
    
    # Save configurations
    configs = {
        "colab": colab_configs,
        "kaggle": kaggle_configs,
        "total_accounts": 6,
        "total_samples": 500000
    }
    
    with open(output_path / "training_configs.json", 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"⚙️ Training configurations created: {output_path / 'training_configs.json'}")

def create_upload_scripts(output_path: Path):
    """Create upload scripts for each platform"""
    
    # Colab upload script
    colab_script = """#!/usr/bin/env python3
# Colab Upload Script
# Upload this to each Colab account

import os
import zipfile
from google.colab import files

# Upload your split dataset
print("📤 Uploading dataset split...")
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted: {filename}")

# Install dependencies
!pip install transformers peft bitsandbytes datasets

# Run training
!python colab_training_fixed.py --dataset split_XX --output models/housebrain-colab-X

print("🚀 Training started! Keep this tab open.")
"""
    
    # Kaggle upload script
    kaggle_script = """#!/usr/bin/env python3
# Kaggle Upload Script
# Upload this to each Kaggle account

import os
import zipfile

# Extract dataset (uploaded as zip)
for filename in os.listdir('.'):
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted: {filename}")

# Install dependencies
!pip install transformers peft bitsandbytes datasets

# Run training
!python colab_training_fixed.py --dataset split_XX --output models/housebrain-kaggle-X

print("🚀 Training started! Check email for completion notification.")
"""
    
    # Save scripts
    with open(output_path / "colab_upload_script.py", 'w') as f:
        f.write(colab_script)
    
    with open(output_path / "kaggle_upload_script.py", 'w') as f:
        f.write(kaggle_script)
    
    print(f"📝 Upload scripts created in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Split HouseBrain dataset for parallel training")
    parser.add_argument("--source", default="housebrain_dataset_v5_350k", help="Source dataset path")
    parser.add_argument("--output", default="housebrain_splits", help="Output directory for splits")
    parser.add_argument("--splits", type=int, default=6, help="Number of splits")
    
    args = parser.parse_args()
    
    print("🔪 HouseBrain Dataset Splitter")
    print("=" * 50)
    
    # Split dataset
    split_dataset(args.source, args.output, args.splits)
    
    # Create training configurations
    create_training_configs(Path(args.output))
    
    # Create upload scripts
    create_upload_scripts(Path(args.output))
    
    print("\n🎉 Dataset splitting complete!")
    print(f"📁 All files ready in: {args.output}")
    print("\n📋 Next steps:")
    print("1. Upload each split to respective Colab/Kaggle accounts")
    print("2. Run training on all 6 accounts simultaneously")
    print("3. Download trained models and merge them")

if __name__ == "__main__":
    main()

