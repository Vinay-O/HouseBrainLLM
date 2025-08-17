# 🏗️ HouseBrain 10K Training Guide

**Complete step-by-step guide to train your custom architectural AI**

This guide provides detailed instructions for training HouseBrain LLM on a 10K dataset using Google Colab or local environment.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Environment Setup](#-environment-setup)
3. [Dataset Preparation](#-dataset-preparation)
4. [Model Selection](#-model-selection)
5. [Training Execution](#-training-execution)
6. [Testing & Validation](#-testing--validation)
7. [Troubleshooting](#-troubleshooting)
8. [Next Steps](#-next-steps)

---

## 🚀 Quick Start

### Prerequisites
- Google Colab Pro+ (recommended) or local GPU
- 8GB+ GPU memory
- 30-60 minutes training time

### One-Command Training
```bash
# Clone repository
git clone https://github.com/Vinay-O/HouseBrainLLM.git
cd HouseBrainLLM

# Generate 10K dataset
python generate_advanced_dataset.py --samples 10000 --output housebrain_10k

# Train with DeepSeek R1 (best reasoning)
python housebrain_colab_trainer.py --test --dataset housebrain_10k --output housebrain_r1_model
```

---

## 🔧 Environment Setup

### Step 1: Install Dependencies

```python
# Install required packages
!pip install torch==2.1.0 transformers==4.41.0 accelerate==0.27.0 peft==0.8.0 datasets==2.16.0 bitsandbytes tqdm scikit-learn pandas matplotlib seaborn

# Clone HouseBrain repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# Verify installation
import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Step 2: Verify Environment

```python
# Check system resources
import psutil
import torch

print("🖥️ System Resources:")
print(f"   CPU Cores: {psutil.cpu_count()}")
print(f"   RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")

# Test imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

---

## 📊 Dataset Preparation

### Option A: Generate 10K Dataset

```python
# Generate advanced 10K dataset
print("🏠 Generating 10K HouseBrain dataset...")
!python generate_advanced_dataset.py --samples 10000 --output housebrain_10k_advanced --quality 0.90 --india 0.60

# Verify dataset
import json
import os

dataset_path = "housebrain_10k_advanced"
if os.path.exists(f"{dataset_path}/dataset_info.json"):
    with open(f"{dataset_path}/dataset_info.json", 'r') as f:
        info = json.load(f)
    print(f"✅ Dataset: {info['name']}")
    print(f"✅ Samples: {info['total_samples']}")
    print(f"✅ Quality: {info['quality_threshold']}")
    print(f"✅ India ratio: {info['india_ratio']}")
```

### Option B: Use Existing Dataset

```python
# Download existing dataset (if available)
!wget https://your-dataset-url/housebrain_10k.tar.gz
!tar -xzf housebrain_10k.tar.gz

# Or use your existing 1M dataset subset
!cp -r housebrain_dataset_r1_super_1M_aug_v1_1/train/shard_01 housebrain_10k_subset
```

### Dataset Verification

```python
# Check dataset structure
import json
import os

def verify_dataset(dataset_path):
    """Verify dataset structure and quality"""
    print(f"🔍 Verifying dataset: {dataset_path}")
    
    # Check info file
    info_file = f"{dataset_path}/dataset_info.json"
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info = json.load(f)
        print(f"   📊 Total samples: {info.get('total_samples', 'N/A')}")
        print(f"   🎯 Quality threshold: {info.get('quality_threshold', 'N/A')}")
    
    # Check training files
    train_dir = f"{dataset_path}/train"
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
        print(f"   📁 Training files: {len(train_files)}")
        
        # Check sample structure
        if train_files:
            sample_file = f"{train_dir}/{train_files[0]}"
            with open(sample_file, 'r') as f:
                sample = json.load(f)
            print(f"   📋 Problem type: {sample['input']['problem_type']}")
            print(f"   🧠 Reasoning steps: {len(sample['input']['reasoning_steps'])}")
            print(f"   📤 Output sections: {len(sample['output'])}")
    
    print("✅ Dataset verification complete!")

verify_dataset("housebrain_10k_advanced")
```

---

## 🤖 Model Selection

### Available Models

```python
# Model configurations
MODELS = {
    "deepseek_r1": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "description": "Best reasoning capabilities for architectural tasks",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended": True
    },
    "qwen2_5": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Excellent reasoning with great compatibility",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended": False
    },
    "llama3_1": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "description": "Very stable and reliable",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended": False
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Good reasoning with efficient training",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended": False
    }
}

# Display model options
print("🤖 Available Models for HouseBrain Training:")
print()
for key, config in MODELS.items():
    star = "⭐" if config["recommended"] else "  "
    print(f"{star} {key.upper()}")
    print(f"   Model: {config['name']}")
    print(f"   Description: {config['description']}")
    print(f"   Target Modules: {config['target_modules']}")
    print()
```

### Model Selection

```python
# Choose your model (change this to your preference)
SELECTED_MODEL = "deepseek_r1"  # Options: "deepseek_r1", "qwen2_5", "llama3_1", "mistral"

model_config = MODELS[SELECTED_MODEL]
print(f"🎯 Selected Model: {model_config['name']}")
print(f"📝 Description: {model_config['description']}")
print(f"🔧 Target Modules: {model_config['target_modules']}")

# Training configuration
TRAINING_CONFIG = {
    "model_name": model_config["name"],
    "dataset_path": "housebrain_10k_advanced",
    "output_dir": f"housebrain_{SELECTED_MODEL}_10k_model",
    "max_length": 2048,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": model_config["target_modules"]
}

print(f"\n⚙️ Training Configuration:")
for key, value in TRAINING_CONFIG.items():
    print(f"   {key}: {value}")
```

---

## 🎯 Training Execution

### Method 1: Using Unified Trainer

```python
# Import training modules
import sys
sys.path.append('.')

from housebrain_colab_trainer import HouseBrainTrainer, TrainingConfig
from dataclasses import dataclass

# Create training configuration
@dataclass
class CustomTrainingConfig:
    model_name: str = TRAINING_CONFIG["model_name"]
    dataset_path: str = TRAINING_CONFIG["dataset_path"]
    output_dir: str = TRAINING_CONFIG["output_dir"]
    max_length: int = TRAINING_CONFIG["max_length"]
    batch_size: int = TRAINING_CONFIG["batch_size"]
    gradient_accumulation_steps: int = TRAINING_CONFIG["gradient_accumulation_steps"]
    learning_rate: float = TRAINING_CONFIG["learning_rate"]
    num_train_epochs: int = TRAINING_CONFIG["num_train_epochs"]
    warmup_steps: int = TRAINING_CONFIG["warmup_steps"]
    logging_steps: int = TRAINING_CONFIG["logging_steps"]
    save_steps: int = TRAINING_CONFIG["save_steps"]
    eval_steps: int = TRAINING_CONFIG["eval_steps"]
    lora_r: int = TRAINING_CONFIG["lora_r"]
    lora_alpha: int = TRAINING_CONFIG["lora_alpha"]
    lora_dropout: float = TRAINING_CONFIG["lora_dropout"]
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = TRAINING_CONFIG["target_modules"]

config = CustomTrainingConfig()

# Initialize trainer
print("🚀 Initializing HouseBrain Trainer...")
trainer = HouseBrainTrainer(config)

# Start training
print("\n🎯 Starting HouseBrain Training...")
print(f"📊 Dataset: {config.dataset_path}")
print(f"🤖 Model: {config.model_name}")
print(f"📁 Output: {config.output_dir}")
print(f"⏱️ Expected time: 30-60 minutes")

try:
    trainer.train()
    print("\n✅ Training completed successfully!")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check GPU memory (need ~8GB+)")
    print("2. Reduce batch_size or gradient_accumulation_steps")
    print("3. Try a smaller model (Qwen2.5 or Mistral)")
    print("4. Check dataset format and paths")
```

### Method 2: Command Line Training

```bash
# Direct command line training
python housebrain_colab_trainer.py \
    --test \
    --dataset housebrain_10k_advanced \
    --output housebrain_r1_model \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --learning-rate 2e-4 \
    --epochs 3
```

### Training Monitoring

```python
# Monitor training progress
import time
from pathlib import Path

def monitor_training(output_dir):
    """Monitor training progress"""
    model_dir = Path(output_dir)
    
    while True:
        if model_dir.exists():
            # Check for training logs
            log_file = model_dir / "trainer_state.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                
                print(f"\n📈 Training Progress:")
                print(f"   Global step: {logs.get('global_step', 'N/A')}")
                print(f"   Epoch: {logs.get('epoch', 'N/A'):.2f}")
                
                if 'log_history' in logs and logs['log_history']:
                    latest = logs['log_history'][-1]
                    print(f"   Loss: {latest.get('loss', 'N/A'):.4f}")
                    print(f"   Learning rate: {latest.get('learning_rate', 'N/A'):.6f}")
        
        time.sleep(30)  # Check every 30 seconds

# Start monitoring (run in separate cell)
monitor_training(TRAINING_CONFIG["output_dir"])
```

---

## 🧪 Testing & Validation

### Test Trained Model

```python
# Test the trained model
print("🧪 Testing HouseBrain Model...")

import json
from src.housebrain.llm import HouseBrainLLM

# Load the trained model
model_path = TRAINING_CONFIG["output_dir"]
housebrain = HouseBrainLLM(finetuned_model_path=model_path)

# Test input
test_input = {
    "plot": {"length": 60, "width": 40, "unit": "ft"},
    "setbacks_ft": {"front": 15, "rear": 10, "left": 8, "right": 8},
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 2,
    "budget_inr": 2500000,
    "style": "modern",
    "region": "north"
}

print("\n📋 Test Input:")
print(json.dumps(test_input, indent=2))

# Generate design
print("\n🏗️ Generating House Design...")
try:
    result = housebrain.generate_house_design(test_input)
    print("\n✅ Design Generated Successfully!")
    print(f"📊 Total Cost: ₹{result.total_cost_estimate:,}")
    print(f"⏱️ Timeline: {result.timeline_weeks} weeks")
    print(f"🏠 Floors: {len(result.levels)}")
    print(f"🚪 Rooms: {sum(len(level.rooms) for level in result.levels)}")
    
    # Show optimization notes
    if result.optimization_notes:
        print("\n💡 Optimization Notes:")
        for note in result.optimization_notes:
            print(f"   • {note}")
            
except Exception as e:
    print(f"\n❌ Generation failed: {e}")
    print("\n🔧 This might be normal for a newly trained model.")
    print("   Try training for more epochs or with more data.")
```

### Model Analysis

```python
# Analyze training results
import os
import json
from pathlib import Path

model_dir = Path(TRAINING_CONFIG["output_dir"])

print("📊 Training Analysis:")
print(f"\n📁 Model Directory: {model_dir}")

# Check model files
if model_dir.exists():
    files = list(model_dir.rglob("*"))
    print(f"\n📄 Model Files ({len(files)} total):")
    for file in files[:10]:  # Show first 10 files
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name} ({size_mb:.1f} MB)")
    
    if len(files) > 10:
        print(f"   ... and {len(files) - 10} more files")
else:
    print("❌ Model directory not found")

# Check training logs
log_file = model_dir / "trainer_state.json"
if log_file.exists():
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    print(f"\n📈 Training Logs:")
    print(f"   Total steps: {logs.get('global_step', 'N/A')}")
    print(f"   Epochs completed: {logs.get('epoch', 'N/A')}")
    
    if 'log_history' in logs:
        history = logs['log_history']
        if history:
            latest = history[-1]
            print(f"   Latest loss: {latest.get('loss', 'N/A'):.4f}")
            print(f"   Learning rate: {latest.get('learning_rate', 'N/A'):.6f}")

# Model size analysis
if model_dir.exists():
    total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n💾 Model Size: {total_size_mb:.1f} MB")
    
    if total_size_mb < 100:
        print("✅ Compact model (LoRA weights only)")
    elif total_size_mb < 1000:
        print("📦 Medium model size")
    else:
        print("🔧 Full model (large size)")

print("\n🎉 Training Analysis Complete!")
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Out of Memory (OOM)
```python
# Solution: Reduce memory usage
TRAINING_CONFIG.update({
    "batch_size": 1,  # Reduce from 2 to 1
    "gradient_accumulation_steps": 16,  # Increase from 8 to 16
    "max_length": 1024,  # Reduce from 2048 to 1024
})
```

#### 2. Model Loading Errors
```python
# Solution: Check model access
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_CONFIG["model_name"])
    model = AutoModelForCausalLM.from_pretrained(TRAINING_CONFIG["model_name"])
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("💡 Try different model or check internet connection")
```

#### 3. Dataset Format Issues
```python
# Solution: Verify dataset structure
def fix_dataset_format(dataset_path):
    """Fix common dataset format issues"""
    import json
    import os
    
    # Check and fix sample structure
    train_dir = f"{dataset_path}/train"
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if file.endswith('.json'):
                filepath = os.path.join(train_dir, file)
                with open(filepath, 'r') as f:
                    sample = json.load(f)
                
                # Ensure required fields
                if 'input' not in sample:
                    sample['input'] = sample.get('question', {})
                if 'output' not in sample:
                    sample['output'] = sample.get('answer', {})
                
                # Save fixed sample
                with open(filepath, 'w') as f:
                    json.dump(sample, f, indent=2)
    
    print("✅ Dataset format fixed!")

fix_dataset_format("housebrain_10k_advanced")
```

#### 4. Training Stuck
```python
# Solution: Restart with different parameters
TRAINING_CONFIG.update({
    "learning_rate": 1e-4,  # Reduce learning rate
    "warmup_steps": 200,  # Increase warmup
    "num_train_epochs": 1,  # Start with fewer epochs
})
```

#### 5. Poor Generation Quality
```python
# Solution: Improve training
TRAINING_CONFIG.update({
    "num_train_epochs": 5,  # Train longer
    "learning_rate": 1e-4,  # Lower learning rate
    "lora_r": 32,  # Increase LoRA rank
    "lora_alpha": 64,  # Increase LoRA alpha
})
```

### Performance Optimization

```python
# Optimize for better performance
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Use mixed precision
TRAINING_CONFIG.update({
    "fp16": True,
    "bf16": False,
})

# Optimize data loading
TRAINING_CONFIG.update({
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
})
```

---

## 🚀 Next Steps

### Immediate Actions

```python
print("🚀 Next Steps for HouseBrain:")
print("\n📋 Immediate Actions:")
print("1. ✅ Download your trained model")
print("2. ✅ Test with different inputs")
print("3. ✅ Save model to Google Drive")
print("4. ✅ Share results with team")

# Save to Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

import shutil
model_path = TRAINING_CONFIG["output_dir"]
drive_path = f"/content/drive/MyDrive/HouseBrain/{model_path}"
shutil.copytree(model_path, drive_path, dirs_exist_ok=True)
print(f"✅ Model saved to Google Drive: {drive_path}")
```

### Model Improvements

```python
print("\n🔧 Model Improvements:")
print("1. 🎯 Train on larger dataset (100K, 500K, 1M)")
print("2. 🔄 Fine-tune hyperparameters")
print("3. 🧠 Try different base models")
print("4. 📊 Analyze performance metrics")

# Scale to larger datasets
LARGER_DATASETS = {
    "100K": "python generate_advanced_dataset.py --samples 100000 --output housebrain_100k",
    "500K": "python generate_advanced_dataset.py --samples 500000 --output housebrain_500k",
    "1M": "python generate_advanced_dataset.py --samples 1000000 --output housebrain_1M"
}

for size, command in LARGER_DATASETS.items():
    print(f"   {size}: {command}")
```

### Production Deployment

```python
print("\n🏗️ Production Deployment:")
print("1. 🌐 Deploy as API service")
print("2. 🎨 Integrate with design tools")
print("3. 📱 Create mobile app")
print("4. 🔗 Connect to CAD software")

# API deployment example
API_DEPLOYMENT = """
# FastAPI deployment
from fastapi import FastAPI
from src.housebrain.llm import HouseBrainLLM

app = FastAPI()
housebrain = HouseBrainLLM(finetuned_model_path="housebrain_r1_model")

@app.post("/generate_design")
async def generate_design(input_data: dict):
    result = housebrain.generate_house_design(input_data)
    return result.model_dump()

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

print("\n📝 API Deployment Code:")
print(API_DEPLOYMENT)
```

### Documentation

```python
print("\n📚 Documentation:")
print("• README.md - Project overview")
print("• COLAB_TRAINING_GUIDE.md - Detailed training guide")
print("• MODEL_COMPARISON.md - Model selection guide")
print("• FUTURE_ROADMAP.md - Development plans")

print("\n🎉 Congratulations! Your HouseBrain model is ready! 🏗️⚡")
```

---

## 📞 Support & Resources

### Getting Help
- **Repository**: https://github.com/Vinay-O/HouseBrainLLM
- **Issues**: Check existing issues or create new ones
- **Documentation**: Review all .md files in the repository

### Additional Resources
- **Model Comparison**: See `MODEL_COMPARISON.md` for detailed model analysis
- **Training Guide**: See `COLAB_TRAINING_GUIDE.md` for advanced training options
- **Future Roadmap**: See `FUTURE_ROADMAP.md` for development plans

### Performance Tips
- Use Colab Pro+ for better GPU and longer sessions
- Start with 10K dataset, then scale to larger datasets
- Monitor training logs for loss convergence
- Test model frequently during training
- Save checkpoints regularly

---

**🎯 Ready to build the future of architectural AI! 🏗️⚡**
