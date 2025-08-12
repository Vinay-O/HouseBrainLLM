#!/usr/bin/env python3
"""
HouseBrain Fine-tuning Script

This script runs the fine-tuning process for the HouseBrain LLM using the synthetic dataset.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from housebrain.finetune import FineTuningConfig, run_finetuning


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HouseBrain LLM")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-base", 
                       help="Base model to fine-tune")
    parser.add_argument("--dataset", default="housebrain_dataset_v1", 
                       help="Path to dataset directory")
    parser.add_argument("--output", default="models/housebrain-finetuned", 
                       help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, 
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--test", action="store_true", 
                       help="Run a quick test with minimal epochs")
    
    args = parser.parse_args()
    
    print("🏠 HouseBrain Fine-tuning Script")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
        return
    
    # Check dataset structure
    train_path = os.path.join(args.dataset, "train")
    val_path = os.path.join(args.dataset, "validation")
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return
    
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found: {val_path}")
        return
    
    # Count training samples
    train_files = [f for f in os.listdir(train_path) if f.endswith('.json')]
    val_files = [f for f in os.listdir(val_path) if f.endswith('.json')]
    
    print(f"📊 Dataset Statistics:")
    print(f"   - Training samples: {len(train_files)}")
    print(f"   - Validation samples: {len(val_files)}")
    print(f"   - Base model: {args.model}")
    print(f"   - Output directory: {args.output}")
    
    # Adjust parameters for test mode
    if args.test:
        print("🧪 Running in test mode with reduced parameters...")
        epochs = 1
        batch_size = 2
        save_steps = 100
        eval_steps = 100
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        save_steps = 500
        eval_steps = 500
    
    # Check CUDA availability and adjust configuration
    import torch
    use_4bit = torch.cuda.is_available()
    
    # Create configuration
    config = FineTuningConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_length=args.max_length,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        num_epochs=epochs,
        save_steps=save_steps,
        eval_steps=eval_steps,
        use_4bit=use_4bit,  # Only use 4-bit if CUDA is available
    )
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Max length: {config.max_length}")
    print(f"   - LoRA rank: {config.lora_r}")
    print(f"   - 4-bit quantization: {config.use_4bit}")
    
    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (this will be very slow)")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start fine-tuning
    print(f"\n🚀 Starting fine-tuning...")
    try:
        trainer = run_finetuning(config)
        print(f"\n✅ Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {args.output}")
        
        # Test the fine-tuned model
        print(f"\n🧪 Testing fine-tuned model...")
        test_finetuned_model(args.output)
        
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()


def test_finetuned_model(model_path: str):
    """Test the fine-tuned model with a sample input"""
    try:
        from housebrain import generate_house_design, HouseInput
        import json
        
        # Load sample input
        with open("data/sample_input.json", "r") as f:
            sample_input = json.load(f)
        
        house_input = HouseInput(**sample_input)
        
        print("🎨 Testing fine-tuned model generation...")
        house_output = generate_house_design(
            house_input, 
            demo_mode=False, 
            finetuned_model_path=model_path
        )
        
        print(f"✅ Fine-tuned model test successful!")
        print(f"   - Total area: {house_output.total_area:.1f} sqft")
        print(f"   - Construction cost: ${house_output.construction_cost:,.0f}")
        print(f"   - Levels: {len(house_output.levels)}")
        
    except Exception as e:
        print(f"❌ Fine-tuned model test failed: {e}")


if __name__ == "__main__":
    main()
