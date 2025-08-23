#!/usr/bin/env python3
"""
HouseBrain Training Script
Main script for training the HouseBrain LLM with fine-tuning.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from housebrain.finetune import HouseBrainFineTuner, FineTuningConfig


def main():
    parser = argparse.ArgumentParser(description="Train HouseBrain LLM")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base", 
                       help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="models/housebrain-trained", 
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, 
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, 
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, 
                       help="LoRA dropout")
    parser.add_argument("--gradient-accumulation", type=int, default=16, 
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100, 
                       help="Warmup steps")

    parser.add_argument("--check-only", action="store_true", 
                       help="Only check configuration without training")
    
    args = parser.parse_args()
    
    # Create configuration
    config = FineTuningConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps
    )
    
    # Create fine-tuner
    fine_tuner = HouseBrainFineTuner(config)
    
    if args.check_only:
        print("Configuration check:")
        print(f"Model: {config.model_name}")
        print(f"Dataset: {config.dataset_path}")
        print(f"Output: {config.output_dir}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max length: {config.max_length}")
        print(f"LoRA r: {config.lora_r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"LoRA dropout: {config.lora_dropout}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Warmup steps: {config.warmup_steps}")
        
        # Check if dataset exists
        if os.path.exists(config.dataset_path):
            print(f"✅ Dataset found: {config.dataset_path}")
        else:
            print(f"❌ Dataset not found: {config.dataset_path}")
            return
        
        # Check dataset structure
        train_dir = os.path.join(config.dataset_path, "train")
        val_dir = os.path.join(config.dataset_path, "validation")
        
        if os.path.exists(train_dir):
            train_files = len([f for f in os.listdir(train_dir) if f.endswith('.json')])
            print(f"✅ Training files: {train_files}")
        else:
            print(f"❌ Training directory not found: {train_dir}")
        
        if os.path.exists(val_dir):
            val_files = len([f for f in os.listdir(val_dir) if f.endswith('.json')])
            print(f"✅ Validation files: {val_files}")
        else:
            print(f"❌ Validation directory not found: {val_dir}")
        
        print("Configuration check complete!")
        return
    
    # Start training
    print("Starting HouseBrain training...")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    
    try:
        fine_tuner.train()
        print("Training completed successfully!")
        print(f"Model saved to: {config.output_dir}")
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()\n