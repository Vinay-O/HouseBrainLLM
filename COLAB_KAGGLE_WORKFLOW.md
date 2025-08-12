# 🚀 Complete Workflow: Colab → Kaggle

**Generate datasets on Google Colab, train models on Kaggle**

This guide provides a complete workflow for generating large datasets on Colab and training your HouseBrain LLM on Kaggle.

## 📋 Overview

### Why This Approach?
- **Colab**: Great for CPU-intensive dataset generation (free, no time limits)
- **Kaggle**: Better for GPU training (longer sessions, more stable)
- **Cost**: Completely free
- **Performance**: Optimal for each task

### Workflow Steps:
1. **Generate Dataset on Colab** (this guide)
2. **Download Dataset** to your computer
3. **Upload to Kaggle** as a dataset
4. **Train Model on Kaggle** (separate guide)

---

## 🏗️ Step 1: Generate Dataset on Google Colab

### 1.1 Open Google Colab
- Go to: https://colab.research.google.com/
- Create new notebook
- **Runtime**: CPU (no GPU needed for generation)

### 1.2 Setup Environment

```python
# Install dependencies
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic orjson svgwrite trimesh python-dotenv

# Clone repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

print("✅ Environment setup complete!")
```

### 1.3 Configure Dataset Generation

```python
# Import modules
import sys
sys.path.append('.')

from generate_dataset import DatasetConfig, HouseBrainDatasetGenerator

# Configure generation
config = DatasetConfig(
    num_samples=50000,  # Adjust based on your needs
    output_dir="housebrain_dataset_v5_50k_colab",
    train_ratio=0.9,
    min_plot_size=1000,
    max_plot_size=10000,
    min_bedrooms=1,
    max_bedrooms=6,
    min_floors=1,
    max_floors=4,
    min_budget=100000,
    max_budget=2000000,
    fast_mode=True,  # Skip layout solving for speed
)

print(f"📋 Generating {config.num_samples:,} samples...")
```

### 1.4 Generate Dataset

```python
# Initialize generator
generator = HouseBrainDatasetGenerator(config)

# Generate dataset
output_dir = generator.generate_dataset()
print(f"✅ Dataset generated: {output_dir}")
```

### 1.5 Create Zip Archive

```python
import zipfile
import os
from pathlib import Path

output_dir = Path(config.output_dir)
zip_path = f"{config.output_dir}.zip"

# Create zip
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir.parent)
            zipf.write(file_path, arcname)

print(f"📦 Zip created: {zip_path}")
print(f"📁 Size: {os.path.getsize(zip_path) / 1e6:.1f} MB")
```

### 1.6 Download Dataset

```python
from google.colab import files
files.download(zip_path)
print("⬇️ Dataset downloaded to your computer!")
```

---

## 📤 Step 2: Upload to Kaggle

### 2.1 Create Kaggle Dataset
1. Go to: https://www.kaggle.com/
2. Click "Create" → "New Dataset"
3. Fill in details:
   - **Name**: `housebrain-dataset-v5-50k`
   - **Description**: `HouseBrain architectural dataset for LLM training`
   - **License**: MIT
4. Upload the zip file you downloaded from Colab
5. Click "Create"

### 2.2 Dataset URL
Your dataset will be available at:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/housebrain-dataset-v5-50k
```

---

## 🧠 Step 3: Train on Kaggle

### 3.1 Create Kaggle Notebook
1. Go to: https://www.kaggle.com/
2. Click "Create" → "New Notebook"
3. **Accelerator**: GPU (P100)
4. **Language**: Python

### 3.2 Training Code

```python
# Install dependencies
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic

# Clone repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# Add dataset
# Go to "Data" tab → "Add data" → "Search for datasets"
# Search for your dataset and add it

# Import training modules
import sys
sys.path.append('src')

from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner
import torch

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
```

### 3.3 Configure Training

```python
# Training configuration
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="../input/housebrain-dataset-v5-50k-colab",  # Kaggle dataset path
    output_dir="models/housebrain-kaggle-trained",
    max_length=1024,
    batch_size=2,
    num_epochs=3,
    learning_rate=2e-4,
    use_4bit=True,
    fp16=True,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

print(f"📋 Training Configuration:")
print(f"   Model: {config.model_name}")
print(f"   Dataset: {config.dataset_path}")
print(f"   Output: {config.output_dir}")
```

### 3.4 Start Training

```python
# Initialize trainer
trainer = HouseBrainFineTuner(config)

# Start training
print("🎯 Starting training...")
print("⏰ This will take 2-4 hours on Kaggle P100")

trainer.train()
print("🎉 Training completed!")
```

### 3.5 Save and Download Model

```python
# Save model
trainer.save_model()

# Create zip for download
import zipfile
import os

model_dir = config.output_dir
zip_path = "housebrain-model-kaggle.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, model_dir)
            zipf.write(file_path, arcname)

print(f"📦 Model saved: {zip_path}")
print(f"📁 Size: {os.path.getsize(zip_path) / 1e6:.1f} MB")

# Download (Kaggle will show download link)
```

---

## 📊 Dataset Generation Parameters

### Recommended Configurations

#### Small Dataset (10K samples)
```python
config = DatasetConfig(
    num_samples=10000,
    output_dir="housebrain_dataset_v5_10k",
    fast_mode=True,
)
```

#### Medium Dataset (50K samples) - RECOMMENDED
```python
config = DatasetConfig(
    num_samples=50000,
    output_dir="housebrain_dataset_v5_50k",
    fast_mode=True,
)
```

#### Large Dataset (100K samples)
```python
config = DatasetConfig(
    num_samples=100000,
    output_dir="housebrain_dataset_v5_100k",
    fast_mode=True,
)
```

#### Massive Dataset (200K samples)
```python
config = DatasetConfig(
    num_samples=200000,
    output_dir="housebrain_dataset_v5_200k",
    fast_mode=True,
)
```

### Advanced Parameters

```python
config = DatasetConfig(
    num_samples=50000,
    output_dir="housebrain_dataset_v5_50k_advanced",
    train_ratio=0.9,
    min_plot_size=800,
    max_plot_size=15000,
    min_bedrooms=1,
    max_bedrooms=8,
    min_floors=1,
    max_floors=5,
    min_budget=80000,
    max_budget=3000000,
    fast_mode=True,
    # Custom styles
    styles=[
        "Modern", "Contemporary", "Traditional", "Colonial", "Mediterranean",
        "Craftsman", "Victorian", "Minimalist", "Scandinavian", "Industrial",
        "Tropical", "Rustic", "Art Deco", "Mid-Century Modern", "Gothic"
    ],
    # Custom regions
    regions=[
        "US_Northeast", "US_Southeast", "US_Midwest", "US_Southwest", "US_West",
        "EU_UK", "EU_Germany", "EU_France", "EU_Italy", "EU_Spain",
        "Asia_India", "Asia_China", "Asia_Japan", "Asia_Singapore", "Asia_Australia"
    ]
)
```

---

## ⚙️ Training Parameters

### Kaggle-Optimized Configuration

```python
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="../input/housebrain-dataset-v5-50k-colab",
    output_dir="models/housebrain-kaggle-trained",
    max_length=1024,
    batch_size=2,  # P100 can handle this
    num_epochs=3,
    learning_rate=2e-4,
    use_4bit=True,
    fp16=True,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
```

### Alternative Models

#### Smaller Model (Faster Training)
```python
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-1.3b-base",  # Smaller model
    batch_size=4,  # Can use larger batch size
    # ... other parameters
)
```

#### Larger Model (Better Quality)
```python
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",  # Larger model
    batch_size=1,  # Smaller batch size needed
    gradient_accumulation_steps=8,  # Compensate for small batch
    # ... other parameters
)
```

---

## 🕐 Time Estimates

### Dataset Generation (Colab)
- **10K samples**: 10-15 minutes
- **50K samples**: 30-45 minutes
- **100K samples**: 60-90 minutes
- **200K samples**: 2-3 hours

### Model Training (Kaggle)
- **1.3B model**: 1-2 hours
- **6.7B model**: 2-4 hours
- **With 50K samples**: 2-3 hours
- **With 100K samples**: 3-5 hours

---

## 🆘 Troubleshooting

### Colab Issues

#### Out of Memory
```python
# Reduce samples
config.num_samples = 10000

# Use fast mode
config.fast_mode = True
```

#### Slow Generation
```python
# Use fast mode
config.fast_mode = True

# Reduce parameter complexity
config.styles = ["Modern", "Traditional"]  # Fewer styles
```

### Kaggle Issues

#### Out of Memory
```python
# Reduce batch size
config.batch_size = 1

# Use smaller model
config.model_name = "deepseek-ai/deepseek-coder-1.3b-base"
```

#### Slow Training
```python
# Reduce sequence length
config.max_length = 512

# Increase gradient accumulation
config.gradient_accumulation_steps = 8
```

#### Dataset Path Issues
```python
# Check dataset path
import os
print(os.listdir("../input/"))

# Use correct path
config.dataset_path = "../input/YOUR_DATASET_NAME"
```

---

## 📈 Expected Results

### Dataset Quality
- **Realistic Parameters**: 95%+ realistic values
- **Diversity**: 15+ styles, 15+ regions
- **Completeness**: 100% valid JSON structure

### Model Performance
- **Training Loss**: < 1.0 (target), < 0.8 (good), < 0.6 (excellent)
- **Validation Loss**: < 1.2 (target), < 1.0 (good), < 0.8 (excellent)
- **Compliance Score**: > 60% (target), > 75% (good), > 85% (excellent)
- **Generation Speed**: < 15s (target), < 10s (good), < 5s (excellent)

---

## 🎯 Best Practices

### 1. **Dataset Generation**
- Start with 50K samples for testing
- Use `fast_mode=True` for speed
- Include diverse styles and regions
- Validate dataset quality before training

### 2. **Training**
- Use Kaggle P100 GPU for best performance
- Start with 3 epochs, increase if needed
- Monitor training loss and validation
- Save checkpoints frequently

### 3. **Resource Management**
- Colab: CPU for generation (no time limits)
- Kaggle: GPU for training (9-hour sessions)
- Monitor memory usage on both platforms

### 4. **Quality Assurance**
- Test generated datasets locally first
- Validate model outputs after training
- Compare with baseline performance

---

## 🚀 Quick Start Commands

### Complete Colab Workflow
```python
# 1. Setup
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# 2. Generate Dataset
import sys
sys.path.append('.')
from generate_dataset import DatasetConfig, HouseBrainDatasetGenerator

config = DatasetConfig(num_samples=50000, output_dir="housebrain_dataset_v5_50k", fast_mode=True)
generator = HouseBrainDatasetGenerator(config)
output_dir = generator.generate_dataset()

# 3. Create Zip
import zipfile
import os
from pathlib import Path

output_dir = Path(config.output_dir)
zip_path = f"{config.output_dir}.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir.parent)
            zipf.write(file_path, arcname)

# 4. Download
from google.colab import files
files.download(zip_path)
```

### Complete Kaggle Workflow
```python
# 1. Setup
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# 2. Import modules
import sys
sys.path.append('src')
from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner

# 3. Configure training
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="../input/housebrain-dataset-v5-50k",
    output_dir="models/housebrain-kaggle-trained",
    max_length=1024,
    batch_size=2,
    num_epochs=3,
    learning_rate=2e-4,
    use_4bit=True,
    fp16=True,
)

# 4. Train
trainer = HouseBrainFineTuner(config)
trainer.train()

# 5. Save
trainer.save_model()
```

---

**🎉 You now have a complete workflow for generating datasets on Colab and training on Kaggle!**

For more information, visit: https://github.com/Vinay-O/HouseBrainLLM
