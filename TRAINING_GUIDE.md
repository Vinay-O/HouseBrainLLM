# 🧠 HouseBrain Training Guide

**Complete guide for training the HouseBrain LLM on various platforms**

## 📋 Overview

This guide covers training the HouseBrain LLM using different platforms and configurations. The training process uses QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning of the DeepSeek model.

## 🎯 Training Options

### 1. **Google Colab (Recommended - Free GPU)**
- **GPU**: T4 (16GB VRAM)
- **Training Time**: 2-4 hours
- **Cost**: Free
- **Best for**: Most users

### 2. **Kaggle Notebooks (Alternative - Free GPU)**
- **GPU**: P100 (16GB VRAM)
- **Training Time**: 3-5 hours
- **Cost**: Free
- **Best for**: Alternative to Colab

### 3. **Local Training (M2 Pro)**
- **GPU**: Apple Silicon MPS
- **Training Time**: 8-12 hours
- **Cost**: Free
- **Best for**: Development and testing

## 🚀 Quick Start - Google Colab

### Step 1: Prepare Dataset

```bash
# Generate 50K samples locally
python generate_dataset.py --samples 50000 --output housebrain_dataset_v5_50k --zip
```

### Step 2: Open Colab

1. Go to: https://colab.research.google.com/
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU

### Step 3: Training Code

```python
# Install dependencies
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic

# Clone repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# Upload dataset
from google.colab import files
uploaded = files.upload()

# Extract dataset
import zipfile
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')

# Start training
import sys
sys.path.append('src')
from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner

config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="housebrain_dataset_v5_50k",
    output_dir="models/housebrain-colab-trained",
    max_length=1024,
    batch_size=2,
    num_epochs=3,
    learning_rate=2e-4,
    use_4bit=True,
    fp16=True,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
)

trainer = HouseBrainFineTuner(config)
trainer.train()

# Save model
trainer.save_model()
```

### Step 4: Download Model

```python
# Create zip archive
import zipfile
import os

model_dir = "models/housebrain-colab-trained"
zip_path = "housebrain-model.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, model_dir)
            zipf.write(file_path, arcname)

# Download
files.download(zip_path)
```

## 🖥️ Local Training (M2 Pro)

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Training Command

```bash
# Basic training
python finetune_housebrain.py --dataset housebrain_dataset_v5_50k --epochs 3

# Advanced options
python finetune_housebrain.py \
    --dataset housebrain_dataset_v5_50k \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --max-length 1024 \
    --output-dir models/housebrain-m2pro-trained
```

### Configuration Options

```python
# M2 Pro optimized config
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="housebrain_dataset_v5_50k",
    output_dir="models/housebrain-m2pro-trained",
    max_length=1024,
    batch_size=1,  # Smaller for M2 Pro
    num_epochs=3,
    learning_rate=2e-4,
    use_4bit=False,  # Disable for MPS
    fp16=False,      # Disable for MPS
    warmup_steps=50,
    logging_steps=25,
    save_steps=250,
)
```

## 📊 Dataset Generation

### Optimal Dataset Size

- **Minimum**: 10K samples
- **Recommended**: 50K samples
- **Optimal**: 100K samples
- **Maximum**: 200K samples (diminishing returns)

### Generation Commands

```bash
# Quick test (1K samples)
python generate_dataset.py --samples 1000 --output housebrain_dataset_test --fast

# Standard training (50K samples)
python generate_dataset.py --samples 50000 --output housebrain_dataset_v5_50k --zip

# Large dataset (100K samples)
python generate_dataset.py --samples 100000 --output housebrain_dataset_v5_100k --fast --zip

# Custom configuration
python generate_dataset.py \
    --samples 75000 \
    --output housebrain_dataset_custom \
    --train-ratio 0.9 \
    --fast \
    --zip
```

### Dataset Features

- **Realistic Parameters**: Plot sizes, room dimensions, budgets
- **Multiple Styles**: 15+ architectural styles
- **Regional Variations**: US, EU, Asia, Australia
- **Climate Zones**: Tropical, Subtropical, Temperate, Cold
- **Material Specifications**: Exterior, roofing, flooring options

## ⚙️ Model Configuration

### Base Models

| Model | Size | VRAM | Quality | Speed |
|-------|------|------|---------|-------|
| `deepseek-ai/deepseek-coder-6.7b-base` | 6.7B | 16GB | High | Medium |
| `deepseek-ai/deepseek-coder-1.3b-base` | 1.3B | 8GB | Medium | Fast |
| `microsoft/DialoGPT-small` | 117M | 4GB | Low | Very Fast |

### Training Parameters

```python
# Optimal configuration
config = FineTuningConfig(
    # Model
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    
    # Dataset
    dataset_path="housebrain_dataset_v5_50k",
    
    # Training
    max_length=1024,
    batch_size=2,  # Adjust based on VRAM
    num_epochs=3,
    learning_rate=2e-4,
    
    # Optimization
    use_4bit=True,  # Enable for CUDA
    fp16=True,      # Enable for CUDA
    gradient_accumulation_steps=4,
    
    # LoRA
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    
    # Monitoring
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
)
```

## 📈 Performance Monitoring

### Training Metrics

```python
# Monitor training progress
trainer = HouseBrainFineTuner(config)

# Training with callbacks
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    warmup_steps=config.warmup_steps,
    weight_decay=0.01,
    logging_dir=f"{config.output_dir}/logs",
    logging_steps=config.logging_steps,
    evaluation_strategy="steps",
    eval_steps=config.eval_steps,
    save_strategy="steps",
    save_steps=config.save_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",  # Optional: Weights & Biases
)

trainer.train()
```

### Expected Results

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Training Loss | < 1.0 | < 0.8 | < 0.6 |
| Validation Loss | < 1.2 | < 1.0 | < 0.8 |
| Compliance Score | > 60% | > 75% | > 85% |
| Generation Speed | < 15s | < 10s | < 5s |

## 🔧 Troubleshooting

### Common Issues

#### 1. **Out of Memory (OOM)**

```python
# Reduce batch size
config.batch_size = 1

# Enable gradient accumulation
config.gradient_accumulation_steps = 8

# Use smaller model
config.model_name = "deepseek-ai/deepseek-coder-1.3b-base"
```

#### 2. **Slow Training**

```python
# Enable mixed precision
config.fp16 = True

# Increase batch size if memory allows
config.batch_size = 4

# Reduce sequence length
config.max_length = 512
```

#### 3. **Poor Model Performance**

```python
# Increase dataset size
# Generate 100K+ samples

# Increase training epochs
config.num_epochs = 5

# Adjust learning rate
config.learning_rate = 1e-4  # Lower for stability
```

#### 4. **MPS Issues (Apple Silicon)**

```python
# Disable 4-bit quantization
config.use_4bit = False

# Disable fp16
config.fp16 = False

# Use smaller batch size
config.batch_size = 1
```

### Platform-Specific Issues

#### Google Colab
- **Disconnect**: Save model frequently
- **GPU Limit**: Use T4, not V100
- **Memory**: Monitor with `!nvidia-smi`

#### Kaggle
- **Session Limit**: 9 hours max
- **GPU**: P100 available
- **Storage**: 20GB limit

#### Local M2 Pro
- **MPS**: Ensure PyTorch 2.1+
- **Memory**: Monitor Activity Monitor
- **Heat**: Ensure proper cooling

## 📦 Model Deployment

### Save and Load

```python
# Save trained model
trainer.save_model("models/housebrain-trained")

# Load for inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/housebrain-trained")
tokenizer = AutoTokenizer.from_pretrained("models/housebrain-trained")
```

### Integration

```python
# Use in HouseBrain LLM
from src.housebrain.llm import HouseBrainLLM

llm = HouseBrainLLM(finetuned_model_path="models/housebrain-trained")
result = llm.generate_design(input_data)
```

## 🎯 Best Practices

### 1. **Dataset Quality**
- Use realistic parameters
- Include diverse styles and regions
- Validate data quality before training

### 2. **Training Strategy**
- Start with smaller datasets for testing
- Use validation split for monitoring
- Save checkpoints frequently

### 3. **Resource Management**
- Monitor GPU memory usage
- Use appropriate batch sizes
- Enable mixed precision when possible

### 4. **Model Evaluation**
- Test on unseen data
- Validate architectural compliance
- Measure generation quality

## 📚 Additional Resources

- **Hugging Face Docs**: https://huggingface.co/docs/transformers/
- **PEFT Documentation**: https://huggingface.co/docs/peft/
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **DeepSeek Models**: https://huggingface.co/deepseek-ai

## 🆘 Support

For training issues:
1. Check the troubleshooting section
2. Review error logs
3. Try with smaller dataset first
4. Create GitHub issue with details

---

**Happy Training! 🚀**
