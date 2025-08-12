# 🚀 Complete Guide: 150K Enhanced Dataset on Colab

**Generate 150K samples with crucial architectural parameters and train your HouseBrain LLM**

This guide provides step-by-step instructions for generating a massive 150K enhanced dataset on Google Colab and training your custom LLM.

## 📋 Overview

### Why 150K Samples?
- **Optimal for Free Tier**: Maximum value from free platforms
- **Enhanced Quality**: 6+ crucial architectural parameters
- **Training Time**: 90-120 minutes generation, 5-7 hours training
- **Performance**: 85-95% compliance score expected

### Enhanced Features Included:
1. **Plot Shape & Orientation** (Rectangle, L-shape, corner plot, etc.)
2. **Exterior Finishes & Materials** (Brick, stone, stucco, etc.)
3. **Climate & Site Conditions** (Hot, cold, tropical, etc.)
4. **Building Codes & Regulations** (FAR, height limits, parking)
5. **Garage & Parking** (Attached, detached, carport, none)
6. **Utilities & Accessibility** (Water, sewer, solar ready)

---

## 🏗️ Step 1: Generate 150K Dataset on Colab

### 1.1 Open Google Colab
- Go to: https://colab.research.google.com/
- Create new notebook
- **Runtime**: CPU (no GPU needed for generation)

### 1.2 Upload Dataset Generation Notebook
1. **Download**: `colab_dataset_generation.ipynb` from this repository
2. **Upload**: File → Upload notebook → Select the file
3. **Run**: Execute all cells

### 1.3 Generation Process
The notebook will:
1. **Install dependencies** (2-3 minutes)
2. **Clone repository** (1 minute)
3. **Configure 150K samples** with enhanced features
4. **Generate dataset** (90-120 minutes)
5. **Create zip archive** (5-10 minutes)
6. **Download dataset** (2-5 minutes)

### 1.4 Expected Output
- **File**: `housebrain_dataset_v5_150k_colab.zip`
- **Size**: ~1.5GB
- **Samples**: 150,000 (135K train, 15K validation)
- **Features**: 6+ enhanced architectural parameters

---

## 📤 Step 2: Upload to Kaggle

### 2.1 Create Kaggle Dataset
1. Go to: https://www.kaggle.com/
2. Click "Create" → "New Dataset"
3. Fill in details:
   - **Name**: `housebrain-dataset-v5-150k-enhanced`
   - **Description**: `HouseBrain enhanced architectural dataset with 150K samples including plot shape, exterior finishes, climate, and building codes`
   - **License**: MIT
4. Upload the zip file: `housebrain_dataset_v5_150k_colab.zip`
5. Click "Create"

### 2.2 Dataset URL
Your dataset will be available at:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/housebrain-dataset-v5-150k-enhanced
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
# Training configuration for 150K enhanced dataset
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="../input/housebrain-dataset-v5-150k-enhanced",  # Your dataset path
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

print(f"📋 Training Configuration:")
print(f"   Model: {config.model_name}")
print(f"   Dataset: {config.dataset_path}")
print(f"   Output: {config.output_dir}")
print(f"   Samples: 150,000 enhanced")
```

### 3.4 Start Training

```python
# Initialize trainer
trainer = HouseBrainFineTuner(config)

# Start training
print("🎯 Starting training...")
print("⏰ This will take 5-7 hours on Kaggle P100")
print("📊 Training on 150K enhanced samples...")

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
zip_path = "housebrain-model-kaggle-150k.zip"

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

## 📊 Enhanced Dataset Features

### **Plot & Site Parameters**
- **Plot Shapes**: Rectangle, L-shape, irregular, corner plot, square
- **Orientations**: 8 compass directions (N, NE, E, SE, S, SW, W, NW)
- **Slopes**: 0-15 degrees
- **Setbacks**: Front, rear, left, right (corner plot variations)

### **Exterior Finishes & Materials**
- **Exterior Walls**: Brick, stone, stucco, vinyl, wood, concrete, fiber cement
- **Roofing**: Asphalt shingles, metal, tile, slate, flat roof, wood shingles
- **Windows**: Single-hung, double-hung, casement, picture, sliding, bay
- **Doors**: Wood, steel, fiberglass, sliding glass, French
- **Garage**: Attached, detached, carport, none

### **Climate & Site Conditions**
- **Climate Zones**: Hot dry, hot humid, cold, temperate, tropical, Mediterranean
- **Seismic Zones**: Low, medium, high earthquake risk
- **Soil Types**: Clay, sandy, rocky, loamy, silty
- **Utilities**: City/well water, city/septic sewer, solar ready

### **Building Codes & Regulations**
- **Floor Area Ratio**: 0.2-0.8
- **Height Restrictions**: 25-35 feet
- **Parking Requirements**: 1-3 spaces
- **Fire Safety**: Sprinklers, fire exits, fire walls

---

## ⏰ Time Estimates

### **Dataset Generation (Colab CPU)**
- **150K samples**: 90-120 minutes
- **Dependencies**: 2-3 minutes
- **Zip creation**: 5-10 minutes
- **Download**: 2-5 minutes
- **Total**: 2-2.5 hours

### **Model Training (Kaggle P100)**
- **150K samples**: 5-7 hours
- **Model loading**: 2-3 minutes
- **Training**: 5-7 hours
- **Model saving**: 1-2 minutes
- **Total**: 5-7 hours

### **Complete Workflow**
- **Generation**: 2-2.5 hours (Colab)
- **Training**: 5-7 hours (Kaggle)
- **Total**: 7-9.5 hours
- **Cost**: Completely free

---

## 🎯 Expected Results

### **Dataset Quality**
- **Realistic Parameters**: 95%+ realistic values
- **Diversity**: 15+ styles, 15+ regions, 6+ climate zones
- **Completeness**: 100% valid JSON structure
- **Enhanced Features**: 6+ crucial architectural parameters

### **Model Performance**
- **Training Loss**: < 0.8 (target), < 0.6 (excellent)
- **Validation Loss**: < 1.0 (target), < 0.8 (excellent)
- **Compliance Score**: 85-95% (excellent)
- **Generation Speed**: < 10s per design
- **Enhanced Output**: Includes all architectural parameters

---

## 🆘 Troubleshooting

### **Colab Issues**

#### Out of Memory
```python
# Reduce samples
config.num_samples = 100000  # 100K instead of 150K

# Use fast mode (already enabled)
config.fast_mode = True
```

#### Slow Generation
```python
# Use fast mode (already enabled)
config.fast_mode = True

# Reduce parameter complexity
config.styles = ["Modern", "Traditional", "Contemporary"]  # Fewer styles
```

#### Session Timeout
- **Solution**: Keep browser tab active
- **Monitor**: Check progress every 15 minutes
- **Restart**: If needed, restart and continue from checkpoint

### **Kaggle Issues**

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

## 🚀 Quick Start Commands

### **Complete Colab Workflow**
```python
# 1. Setup
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# 2. Generate Enhanced Dataset
import sys
sys.path.append('.')
from generate_dataset import DatasetConfig, HouseBrainDatasetGenerator

config = DatasetConfig(
    num_samples=150000,
    output_dir="housebrain_dataset_v5_150k_colab",
    fast_mode=True
)
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

### **Complete Kaggle Workflow**
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
    dataset_path="../input/housebrain-dataset-v5-150k-enhanced",
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

## 📈 Performance Comparison

| Dataset Size | Generation | Training | Compliance | Quality | Risk |
|--------------|------------|----------|------------|---------|------|
| **50K** | 30-45min | 2-4h | 70-85% | Good | None |
| **100K** | 60-90min | 4-6h | 80-90% | Very Good | Low |
| **150K** | 90-120min | 5-7h | 85-95% | Excellent | Low |
| **200K** | 2-3h | 6-8h | 90-95% | Outstanding | Medium |

---

## 🎯 Best Practices

### 1. **Dataset Generation**
- Use Colab CPU (no GPU needed)
- Keep browser tab active
- Monitor progress every 15 minutes
- Use fast mode for speed

### 2. **Training**
- Use Kaggle P100 GPU for best performance
- Monitor training loss and validation
- Save checkpoints frequently
- Use appropriate batch size

### 3. **Resource Management**
- Colab: CPU for generation (no time limits)
- Kaggle: GPU for training (9-hour sessions)
- Monitor memory usage on both platforms

### 4. **Quality Assurance**
- Verify dataset structure before training
- Check enhanced features are included
- Validate model outputs after training
- Compare with baseline performance

---

**🎉 You now have a complete workflow for generating 150K enhanced samples and training your HouseBrain LLM!**

**This will give you the maximum value from free platforms with outstanding model performance!**

For more information, visit: https://github.com/Vinay-O/HouseBrainLLM
