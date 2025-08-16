# 🏗️ HouseBrain LLM - Complete Training Guide

## 📋 Overview

This guide will walk you through training HouseBrain, a specialized architectural AI model, on Google Colab Pro+ using our 575K high-quality dataset.

### 🎯 What We're Building
- **Model**: DeepSeek Coder 6.7B base model
- **Dataset**: 575,016 high-quality architectural samples
- **Focus**: Indian market with NBC 2016 compliance
- **Output**: Complete house designs and construction documentation

### 📊 Dataset Details
- **Total Samples**: 575,016
- **Training**: 517,243 samples
- **Validation**: 57,773 samples
- **Quality**: 85%+ acceptance rate with quality gates
- **Features**: 40% India-focused, regional variations, climate considerations

---

## 🚀 Step 1: Google Colab Pro+ Setup

### 1.1 Access Colab Pro+
1. Go to: https://colab.research.google.com/
2. Sign in with your Google account
3. Ensure you have **Colab Pro+** subscription

### 1.2 Create New Notebook
1. Click **"New Notebook"**
2. Rename to: `HouseBrain_1M_Training`
3. Save the notebook

### 1.3 Configure Runtime
1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Set **Runtime shape** to **High-RAM** (if available)
4. Click **Save**

---

## 📁 Step 2: Upload Dataset

### 2.1 Prepare Dataset File
**File to upload**: `housebrain_dataset_v6_1M.tar.gz`
- Contains 575K high-quality samples
- Compressed size: ~55MB
- Extracts to ~2GB

### 2.2 Upload to Colab
```python
# Cell 1: Upload dataset
from google.colab import files
print("📁 Please upload: housebrain_dataset_v6_1M.tar.gz")
uploaded = files.upload()

# Extract the dataset
!tar -xzf housebrain_dataset_v6_1M.tar.gz
!ls -la housebrain_dataset_v6_1M/
```

**Expected Output:**
```
📁 Please upload: housebrain_dataset_v6_1M.tar.gz
Saving housebrain_dataset_v6_1M.tar.gz to housebrain_dataset_v6_1M.tar.gz
housebrain_dataset_v6_1M/
├── dataset_info.json
├── train/          # 517,243 training files
└── validation/     # 57,773 validation files
```

---

## 🔧 Step 3: Install Dependencies

```python
# Cell 2: Install required packages
!pip install -q transformers==4.36.0 peft==0.7.0 accelerate==0.25.0 datasets==2.15.0
!pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q matplotlib tqdm

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

print("✅ Dependencies installed!")
```

**Expected Output:**
```
Collecting transformers==4.36.0
...
Successfully installed transformers-4.36.0 peft-0.7.0 accelerate-0.25.0 datasets-2.15.0
✅ Dependencies installed!
```

---

## 🎯 Step 4: Verify GPU Setup

```python
# Cell 3: Check GPU availability
import torch
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Should show A100 or V100 for Colab Pro+
```

**Expected Output:**
```
✅ CUDA available: True
🎯 GPU: NVIDIA A100-SXM4-40GB
💾 GPU Memory: 40.0 GB
```

---

## 📥 Step 5: Clone Repository

```python
# Cell 4: Clone your repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM
!ls -la
```

**Expected Output:**
```
Cloning into 'HouseBrainLLM'...
remote: Enumerating objects: 100, done.
...
drwxr-xr-x  2 root root 4096 Aug 16 21:15 src
drwxr-xr-x  2 root root 4096 Aug 16 21:15 api
-rw-r--r--  1 root root 1234 Aug 16 21:15 colab_proplus_train_simple.py
-rw-r--r--  1 root root  567 Aug 16 21:15 monitor_training.py
...
```

---

## 🚀 Step 6: Start Training

```python
# Cell 5: Start the training (this will run for 8-12 hours)
print("🚀 Starting HouseBrain training...")
print("📊 This will take 8-12 hours. You can monitor progress in the next cell.")
!python colab_proplus_train_simple.py
```

**Expected Output:**
```
🚀 Starting HouseBrain training...
🔧 Setting up environment...
✅ GPU: NVIDIA A100-SXM4-40GB
💾 GPU Memory: 40.0 GB
📊 Loading dataset...
📈 Training samples: 517,243
📉 Validation samples: 57,773
🤖 Loading model and tokenizer...
[2024-08-16 21:30:15] Starting HouseBrain training with 40.0GB GPU
[2024-08-16 21:30:16] Dataset loaded: 517,243 train, 57,773 validation samples
[2024-08-16 21:30:20] Model and tokenizer loaded successfully
[2024-08-16 21:30:25] 🚀 Starting training...
[2024-08-16 21:30:30] Step 10: Train Loss=2.3456, Eval Loss=2.1234, LR=2.00e-04, GPU=35.67GB
```

---

## 📊 Step 7: Monitor Training Progress

```python
# Cell 6: Monitor training progress (run this periodically)
print("📊 Checking training progress...")
!python monitor_training.py
```

**Expected Output:**
```
📊 Training Monitor
==================================================
📝 Latest log entries:
   [2024-08-16 21:30:30] Step 10: Train Loss=2.3456, Eval Loss=2.1234, LR=2.00e-04, GPU=35.67GB
   [2024-08-16 21:30:45] Step 20: Train Loss=2.1234, Eval Loss=2.0123, LR=2.00e-04, GPU=35.67GB

📈 Metrics Summary:
   Latest Train Loss: 2.1234 (Step 20)
   Latest Eval Loss: 2.0123 (Step 20)
   GPU Memory: 35.67GB (Step 20)

💾 Saved Checkpoints:
   checkpoint-1000
   checkpoint-2000

⏰ Last updated: 2024-08-16 21:30:45
```

---

## 📈 Step 8: Visualize Training Metrics

```python
# Cell 7: Visualize training progress
from monitor_training import plot_metrics
plot_metrics()
```

**Expected Output:**
- 4-panel plot showing:
  - Training Loss (should decrease)
  - Evaluation Loss (should decrease)
  - Learning Rate (should follow scheduler)
  - GPU Memory Usage (should be stable)

---

## ⏱️ Training Timeline

### A100 40GB (Colab Pro+)
| Event | Time | Description |
|-------|------|-------------|
| **Start** | 0h | Training begins |
| **First Checkpoint** | 0.5h | checkpoint-1000 saved |
| **First Evaluation** | 0.25h | First validation loss |
| **Mid Training** | 4-6h | 25,000 steps completed |
| **Final Checkpoint** | 8-10h | checkpoint-50000 saved |
| **Training Complete** | 8-12h | Model ready |

### V100 16GB (Colab Pro)
| Event | Time | Description |
|-------|------|-------------|
| **Start** | 0h | Training begins |
| **First Checkpoint** | 0.75h | checkpoint-1000 saved |
| **First Evaluation** | 0.4h | First validation loss |
| **Mid Training** | 6-8h | 25,000 steps completed |
| **Final Checkpoint** | 12-14h | checkpoint-50000 saved |
| **Training Complete** | 12-16h | Model ready |

---

## 🔍 What to Monitor

### ✅ Good Signs
- **Loss decreasing** over time (both train and eval)
- **GPU utilization** 80-95%
- **Memory usage** stable around 35-38GB
- **Checkpoints saving** every 1000 steps
- **No errors** in training output

### ❌ Warning Signs
- **Loss not decreasing** (learning rate too low)
- **Loss exploding** (learning rate too high)
- **GPU memory errors** (batch size too large)
- **Training stuck** (check for errors)
- **High validation loss** (overfitting)

---

## 💾 What Gets Saved

### During Training
```
housebrain-trained-model/
├── checkpoint-1000/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.bin
├── checkpoint-2000/
├── checkpoint-3000/
├── ...
└── checkpoint-50000/
```

### After Training
```
housebrain-trained-model/
├── final/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── training_log.txt
└── training_metrics.json
```

---

## 🎯 Training Configuration

### Model Settings
- **Base Model**: `deepseek-ai/deepseek-coder-6.7b-base`
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05

### Training Settings
- **Learning Rate**: 2e-4
- **Batch Size**: 4
- **Gradient Accumulation**: 8
- **Sequence Length**: 1024
- **Max Steps**: 50,000
- **Save Steps**: 1,000
- **Eval Steps**: 500

### Optimization
- **Precision**: BF16 (A100) / FP16 (V100)
- **Gradient Checkpointing**: Enabled
- **Optimizer**: AdamW
- **Scheduler**: Cosine
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0

---

## 🚨 Important Notes

### Keep Colab Running
- **Don't close** the browser tab
- **Don't disconnect** from the internet
- **Keep laptop plugged in** if using laptop
- **Monitor periodically** every 30 minutes

### Save Progress
- **Checkpoints are automatic** every 1000 steps
- **Logs are saved** to `training_log.txt`
- **Metrics are saved** to `training_metrics.json`
- **Final model** saved to `housebrain-trained-model/final/`

### Troubleshooting
- **If training stops**: Check for errors in output
- **If memory issues**: Reduce batch size in config
- **If loss not decreasing**: Check learning rate
- **If GPU errors**: Restart runtime and try again

---

## 🎉 After Training

### 1. Download Model
```python
# Download the trained model
from google.colab import files
!zip -r housebrain-trained-model.zip housebrain-trained-model/
files.download('housebrain-trained-model.zip')
```

### 2. Test Model
```python
# Test the trained model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
model = AutoModelForCausalLM.from_pretrained("housebrain-trained-model/final/")
tokenizer = AutoTokenizer.from_pretrained("housebrain-trained-model/final/")

# Test generation
input_text = "Design a 3BHK house in Mumbai with modern style"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. Next Steps
- **2D Floor Plan Generation**: Implement visualization
- **3D Model Generation**: Add 3D rendering
- **API Development**: Create FastAPI server
- **Web Platform**: Build user interface

---

## 📞 Support

### If You Encounter Issues
1. **Check the logs** in `training_log.txt`
2. **Monitor GPU usage** in Colab
3. **Restart runtime** if needed
4. **Check GitHub issues** for common problems

### Expected Performance
- **Training Time**: 8-12 hours (A100) / 12-16 hours (V100)
- **Final Loss**: Should be < 1.0
- **Model Size**: ~13GB (base + LoRA)
- **Inference Speed**: ~2-3 seconds per generation

---

## 🎯 Quick Start Commands

**Copy-paste this complete sequence:**

```python
# 1. Upload dataset
from google.colab import files
uploaded = files.upload()
!tar -xzf housebrain_dataset_v6_1M.tar.gz

# 2. Install dependencies
!pip install -q transformers==4.36.0 peft==0.7.0 accelerate==0.25.0 datasets==2.15.0 torch==2.1.0 matplotlib tqdm

# 3. Clone repo
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# 4. Start training
!python colab_proplus_train_simple.py
```

**That's it! Your HouseBrain model will be training in 8-12 hours! 🚀**
