#!/usr/bin/env python3
"""
Google Colab Setup Script for HouseBrain

This script sets up the complete environment for training HouseBrain on Google Colab.
Run this in the first cell of your Colab notebook.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "fastapi==0.111.0",
        "uvicorn[standard]==0.30.1",
        "pydantic>=2.7.0",
        "orjson>=3.10.0",
        "httpx>=0.27.0",
        "numpy>=1.26.4",
        "svgwrite>=1.4.3",
        "trimesh>=4.4.3",
        "python-dotenv>=1.0.1"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      capture_output=True, text=True)

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available, using CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_repository():
    """Setup the HouseBrain repository"""
    print("\n📁 Setting up repository...")
    
    # Check if repository exists
    if os.path.exists("HouseBrain"):
        print("   Repository already exists, updating...")
        os.chdir("HouseBrain")
        subprocess.run(["git", "pull"], capture_output=True, text=True)
    else:
        print("   Cloning repository...")
        # This will be replaced with your actual repo URL
        subprocess.run(["git", "clone", "https://github.com/YOUR_USERNAME/HouseBrain.git"], 
                      capture_output=True, text=True)
        os.chdir("HouseBrain")
    
    print("✅ Repository setup complete")

def verify_dataset():
    """Verify dataset availability"""
    print("\n📊 Checking datasets...")
    
    datasets = []
    for item in os.listdir("."):
        if item.startswith("housebrain_dataset_"):
            datasets.append(item)
    
    if datasets:
        print(f"✅ Found datasets: {datasets}")
        for dataset in datasets:
            train_path = os.path.join(dataset, "train")
            val_path = os.path.join(dataset, "validation")
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                train_files = len([f for f in os.listdir(train_path) if f.endswith('.json')])
                val_files = len([f for f in os.listdir(val_path) if f.endswith('.json')])
                print(f"   {dataset}: {train_files} train, {val_files} validation samples")
            else:
                print(f"   {dataset}: Invalid structure")
    else:
        print("⚠️  No datasets found")
    
    return datasets

def create_training_script():
    """Create optimized training script for Colab"""
    print("\n📝 Creating Colab training script...")
    
    script_content = '''#!/usr/bin/env python3
"""
HouseBrain Training Script for Google Colab
Optimized for Colab's free GPU resources
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner

def train_housebrain():
    """Train HouseBrain model on Colab"""
    
    # Find available dataset
    datasets = [d for d in os.listdir('.') if d.startswith('housebrain_dataset_')]
    if not datasets:
        print("❌ No datasets found!")
        return
    
    dataset_path = datasets[0]  # Use first available dataset
    print(f"📊 Using dataset: {dataset_path}")
    
    # Colab-optimized configuration
    config = FineTuningConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        dataset_path=dataset_path,
        output_dir="models/housebrain-colab-trained",
        max_length=1024,
        batch_size=2,
        num_epochs=3,
        learning_rate=2e-4,
        use_4bit=True,
        lora_r=16,
        lora_alpha=32,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
    )
    
    print("⚙️ Training Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - Dataset: {config.dataset_path}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Max length: {config.max_length}")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - 4-bit quantization: {config.use_4bit}")
    
    # Start training
    print("\\n🚀 Starting training...")
    trainer = HouseBrainFineTuner(config)
    trainer.train()
    
    print("✅ Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")

if __name__ == "__main__":
    train_housebrain()
'''
    
    with open("train_colab.py", "w") as f:
        f.write(script_content)
    
    print("✅ Training script created: train_colab.py")

def main():
    """Main setup function"""
    print("🏠 HouseBrain Colab Setup")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Setup repository
    setup_repository()
    
    # Verify dataset
    datasets = verify_dataset()
    
    # Create training script
    create_training_script()
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("   1. Upload your dataset if not present")
    print("   2. Run: python train_colab.py")
    print("   3. Download the trained model")
    
    if not gpu_available:
        print("⚠️  No GPU detected - training will be very slow!")
    
    if not datasets:
        print("⚠️  No datasets found - upload your dataset first!")

if __name__ == "__main__":
    main()
