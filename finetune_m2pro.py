#!/usr/bin/env python3
"""
HouseBrain Fine-tuning for MacBook M2 Pro

Optimized for Apple Silicon with MPS backend support.
Supports dataset versions v3 and v4.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from housebrain.finetune import FineTuningConfig, HouseBrainDataset, HouseBrainFineTuner


def check_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available"""
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available!")
        print(f"   - Device: Apple Silicon GPU")
        return True
    else:
        print("⚠️  MPS not available, falling back to CPU")
        return False


def create_train_val_split(dataset_path: str, train_ratio: float = 0.9):
    """Create train/validation split for flat dataset structure (like v4)"""
    import random
    from pathlib import Path
    
    dataset_dir = Path(dataset_path)
    json_files = list(dataset_dir.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path}")
    
    # Shuffle files
    random.shuffle(json_files)
    
    # Split into train/val
    split_idx = int(len(json_files) * train_ratio)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    # Create train/val directories
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Move files
    for file in train_files:
        file.rename(train_dir / file.name)
    
    for file in val_files:
        file.rename(val_dir / file.name)
    
    print(f"📊 Created train/val split:")
    print(f"   - Training: {len(train_files)} files")
    print(f"   - Validation: {len(val_files)} files")
    
    return str(dataset_dir)


def get_optimal_config(dataset_version: str, use_mps: bool = True) -> FineTuningConfig:
    """Get optimal configuration for DeepSeek R1 fine-tuning"""
    
    # Base configuration for DeepSeek R1
    config = FineTuningConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",  # DeepSeek R1 equivalent
        dataset_path=f"housebrain_dataset_{dataset_version}",
        output_dir=f"models/housebrain-{dataset_version}",
        max_length=1024,  # Reduced for memory efficiency
        batch_size=2 if use_mps else 1,  # Optimized for M2 Pro
        num_epochs=3,
        learning_rate=2e-4,
        use_4bit=False,  # MPS doesn't support 4-bit quantization
        lora_r=16,
        lora_alpha=32,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
    )
    
    # Adjust for dataset version
    if dataset_version == "v4_10k":
        config.batch_size = 1 if use_mps else 1  # Even smaller batch for larger dataset
        config.max_length = 2048  # Longer sequences for v4
        config.num_epochs = 2  # Fewer epochs for larger dataset
    
    return config


def main():
    parser = argparse.ArgumentParser(description="HouseBrain Fine-tuning for M2 Pro")
    parser.add_argument("--dataset", choices=["v3", "v4_10k"], default="v3",
                       help="Dataset version to use")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--no-mps", action="store_true", help="Disable MPS (use CPU)")
    parser.add_argument("--split-v4", action="store_true", 
                       help="Create train/val split for v4 dataset")
    
    args = parser.parse_args()
    
    print("🏠 HouseBrain Fine-tuning for MacBook M2 Pro")
    print("=" * 60)
    
    # Check MPS availability
    use_mps = check_mps_availability() and not args.no_mps
    
    # Handle v4 dataset splitting if needed
    dataset_path = f"housebrain_dataset_{args.dataset}"
    if args.dataset == "v4_10k" and args.split_v4:
        print(f"📂 Creating train/val split for {dataset_path}...")
        dataset_path = create_train_val_split(dataset_path)
    
    # Get optimal configuration
    config = get_optimal_config(args.dataset, use_mps)
    config.dataset_path = dataset_path
    
    if args.test:
        print("🧪 Running in test mode...")
        config.num_epochs = 1
        config.batch_size = 1
        config.max_length = 512
        config.logging_steps = 10
        config.save_steps = 100
        config.eval_steps = 100
    
    print(f"\n📊 Configuration:")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Model: {config.model_name}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Max length: {config.max_length}")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Device: {'MPS' if use_mps else 'CPU'}")
    
    # Set device
    if use_mps:
        device = torch.device("mps")
        print(f"🚀 Using MPS device: {device}")
    else:
        device = torch.device("cpu")
        print(f"🐌 Using CPU device: {device}")
    
    try:
        # Initialize fine-tuner
        fine_tuner = HouseBrainFineTuner(config)
        
        # Override device settings for MPS
        if use_mps:
            fine_tuner.device = device
        
        # Start fine-tuning
        print(f"\n🚀 Starting fine-tuning...")
        fine_tuner.train()
        
        print(f"\n✅ Fine-tuning completed!")
        print(f"📁 Model saved to: {config.output_dir}")
        
        # Test the fine-tuned model
        if not args.test:
            print(f"\n🧪 Testing fine-tuned model...")
            test_input = {
                "basicDetails": {
                    "plotSize": 2000,
                    "bedrooms": 3,
                    "floors": 2,
                    "budget": 500000,
                    "style": "modern"
                },
                "plot": {
                    "width": 40,
                    "length": 50
                },
                "roomBreakdown": {
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "kitchen": 1,
                    "livingRoom": 1,
                    "diningRoom": 1
                }
            }
            
            # Test generation
            from housebrain.llm import HouseBrainLLM
            llm = HouseBrainLLM(
                demo_mode=False,
                finetuned_model_path=config.output_dir
            )
            
            from housebrain.schema import HouseInput
            house_input = HouseInput(**test_input)
            result = llm.generate_house_design(house_input)
            
            print(f"✅ Test generation successful!")
            print(f"   - Total area: {result.total_area} sqft")
            print(f"   - Construction cost: ${result.construction_cost:,}")
            print(f"   - Levels: {len(result.levels)}")
        
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
