#!/usr/bin/env python3
"""
HouseBrain LLM Training Script

Optimized training script for HouseBrain LLM with automatic environment detection
and memory optimization. Supports both local (MPS) and cloud (CUDA) training.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import with fallback
try:
    from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner
except ImportError:
    print("❌ Could not import housebrain.finetune. Using fallback.")
    sys.exit(1)

# Disable wandb prompts
os.environ["WANDB_DISABLED"] = "true"


def detect_environment():
    """Detect training environment and optimize settings"""
    print("🔍 Detecting training environment...")
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Optimize for different GPU types
        if "V100" in gpu_name or "A100" in gpu_name:
            # High-end GPU - can handle more
            return {
                "model": "deepseek-ai/deepseek-coder-6.7b-base",
                "batch_size": 2,
                "gradient_accumulation": 8,
                "max_samples": 50000,
                "use_4bit": True
            }
        elif "T4" in gpu_name or "K80" in gpu_name:
            # Colab/Kaggle GPU - conservative settings
            return {
                "model": "deepseek-ai/deepseek-coder-6.7b-base",
                "batch_size": 1,
                "gradient_accumulation": 16,
                "max_samples": 10000,  # 10K as requested
                "use_4bit": True
            }
        else:
            # Generic CUDA GPU
            return {
                "model": "deepseek-ai/deepseek-coder-6.7b-base",
                "batch_size": 1,
                "gradient_accumulation": 16,
                "max_samples": 10000,  # 10K as requested
                "use_4bit": True
            }
    
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon GPU detected (MPS)")
        return {
            "model": "deepseek-ai/deepseek-coder-6.7b-base",
            "batch_size": 1,
            "gradient_accumulation": 8,
            "max_samples": 5000,  # Conservative for MPS
            "use_4bit": False  # MPS doesn't support 4-bit quantization
        }
    
    else:
        device = "cpu"
        print("⚠️  No GPU detected, using CPU (very slow)")
        return {
            "model": "deepseek-ai/deepseek-coder-1.3b-base",  # Smaller model for CPU
            "batch_size": 1,
            "gradient_accumulation": 4,
            "max_samples": 1000,  # Very small for CPU
            "use_4bit": False
        }


def find_dataset():
    """Find available dataset"""
    possible_datasets = [
        "housebrain_dataset_v5_350k",
        "housebrain_dataset_v5_150k_colab",
        "housebrain_dataset_v5_500k_colab", 
        "housebrain_dataset_v4_10k",
        "housebrain_dataset_v3",
        "housebrain_dataset_v1"
    ]
    
    for dataset in possible_datasets:
        if Path(dataset).exists():
            print(f"✅ Found dataset: {dataset}")
            return dataset
    
    print("❌ No dataset found!")
    print("Available datasets:", possible_datasets)
    return None


def main():
    parser = argparse.ArgumentParser(description="Train HouseBrain LLM")
    parser.add_argument("--model", help="Model to use (auto-detected if not specified)")
    parser.add_argument("--dataset", help="Dataset path (auto-detected if not specified)")
    parser.add_argument("--output", default="models/housebrain-trained", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size (auto-detected if not specified)")
    parser.add_argument("--max-samples", type=int, help="Maximum samples (auto-detected if not specified)")
    parser.add_argument("--check-only", action="store_true", help="Only check environment and dataset")
    parser.add_argument("--test", action="store_true", help="Test mode with small dataset")
    
    args = parser.parse_args()
    
    print("🏗️ HouseBrain LLM Training - Optimized")
    print("=" * 50)
    
    # Detect environment
    env_config = detect_environment()
    
    # Find dataset
    dataset_path = args.dataset or find_dataset()
    if not dataset_path:
        print("❌ No dataset found. Please generate a dataset first.")
        return
    
    # Override with command line args if provided
    model_name = args.model or env_config["model"]
    batch_size = args.batch_size or env_config["batch_size"]
    max_samples = args.max_samples or env_config["max_samples"]
    
    # Test mode overrides
    if args.test:
        max_samples = 100
        epochs = 1
        print("🧪 Test mode: Using 100 samples, 1 epoch")
    else:
        epochs = args.epochs
    
    # Configuration
    config = FineTuningConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=args.output,
        num_epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=env_config["gradient_accumulation"],
        use_4bit=env_config["use_4bit"]
    )
    
    # Print configuration
    print(f"\n📋 Training Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Dataset: {config.dataset_path}")
    print(f"   Output: {config.output_dir}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Max Length: {config.max_length}")
    print(f"   LoRA Rank: {config.lora_r}")
    print(f"   Samples: {max_samples:,}")
    print(f"   4-bit Quantization: {config.use_4bit}")
    print(f"   Device: {config.device}")

    # Warn if no validation split present
    val_path = Path(dataset_path) / "validation"
    if not val_path.exists():
        print("ℹ️ No validation split found. Consider generating a validation set for better selection.")
    
    if args.check_only:
        print("\n✅ Environment check completed!")
        return
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = HouseBrainFineTuner(config)
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"⏰ This will take 2-3 hours on Colab GPU")
    print(f"📊 Training on {max_samples:,} samples...")
    print(f" Keep this notebook active and don't close the browser tab!")
    
    success = trainer.train(max_samples)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {config.output_dir}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
