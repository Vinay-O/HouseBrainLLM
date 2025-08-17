# 🏗️ HouseBrain Colab Training Guide

Complete, unified guide for training HouseBrain in Google Colab with clean, error-free setup.

## 🚀 **Quick Start (Copy-Paste Ready)**

### **Step 1: Setup Environment**
```python
# Clone repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# Check GPU
!nvidia-smi

# Install dependencies (fixed versions)
!python -m pip install --force-reinstall torch==2.1.0 transformers==4.36.0 accelerate==0.25.0 peft==0.7.0 datasets==2.15.0 numpy==1.24.3 tqdm
```

### **Step 2: Generate Dataset (10K Test)**
```python
# Generate 10K test dataset (takes ~5 minutes)
!python colab_generate_10k.py --samples 10000 --output housebrain_10k --fast
```

### **Step 3: Train Model (10K Test)**
```python
# Option A: DeepSeek R1 (Best reasoning - recommended)
!python housebrain_colab_trainer.py --test --dataset housebrain_10k --output housebrain_r1_model

# Option B: Qwen2.5 (Excellent reasoning + compatibility)
!python housebrain_colab_trainer.py --test --dataset housebrain_10k --output housebrain_qwen_model --model "Qwen/Qwen2.5-7B-Instruct"

# Option C: Llama 3.1 (Very stable)
!python housebrain_colab_trainer.py --test --dataset housebrain_10k --output housebrain_llama_model --model "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

### **Step 4: Generate Full Dataset (100K/500K/1M)**
```python
# For 100K samples (~30 minutes)
!python colab_generate_10k.py --samples 100000 --output housebrain_100k --fast

# For 500K samples (~2-3 hours)
!python colab_generate_10k.py --samples 500000 --output housebrain_500k --fast

# For 1M samples (~5-6 hours)
!python colab_generate_10k.py --samples 1000000 --output housebrain_1m --fast
```

### **Step 5: Train Full Model**
```python
# Train on full dataset
!python housebrain_colab_trainer.py --dataset housebrain_100k --output housebrain_production_model --epochs 3
```

## 📊 **Training Configurations**

### **Test Mode (10K samples)**
- **Purpose**: Validate pipeline
- **Time**: 30-60 minutes
- **Memory**: ~8GB GPU
- **Output**: Proof of concept model

### **Production Mode (100K+ samples)**
- **Purpose**: Production-ready model
- **Time**: 3-12 hours (depending on size)
- **Memory**: ~16GB+ GPU
- **Output**: High-quality model

## 🛠️ **Advanced Options**

### **Custom Dataset Size**
```python
# Generate custom size
!python colab_generate_10k.py --samples 50000 --output housebrain_50k

# Train with custom parameters
!python housebrain_colab_trainer.py \
    --dataset housebrain_50k \
    --output my_model \
    --batch-size 2 \
    --epochs 3 \
    --max-samples 25000
```

### **Resume Training**
```python
# Training automatically saves checkpoints
# Resume from checkpoint if interrupted
!python housebrain_colab_trainer.py --dataset housebrain_100k --output housebrain_production_model --epochs 3
```

### **Monitor Progress**
```python
# Check training progress
!ls -la housebrain_production_model/

# View logs
!tail -f housebrain_production_model/trainer_state.json
```

## 🎯 **Expected Results**

### **10K Test Training**
- **Training Loss**: ~1.5 → ~0.8
- **Time**: 30-60 minutes
- **Model Size**: ~13GB
- **Quality**: Basic functionality

### **100K Production Training**
- **Training Loss**: ~1.2 → ~0.4
- **Time**: 3-5 hours
- **Model Size**: ~13GB
- **Quality**: High-quality designs

### **1M Production Training**
- **Training Loss**: ~1.0 → ~0.2
- **Time**: 8-12 hours
- **Model Size**: ~13GB
- **Quality**: Production-ready

## 🚨 **Troubleshooting**

### **Memory Issues**
```python
# Reduce batch size
!python housebrain_colab_trainer.py --dataset housebrain_100k --batch-size 1 --output model
```

### **Runtime Disconnection**
```python
# Check if training was interrupted
!ls -la housebrain_production_model/

# Resume training (automatic checkpoint detection)
!python housebrain_colab_trainer.py --dataset housebrain_100k --output housebrain_production_model
```

### **Dependency Conflicts**
```python
# Fix dependencies manually
!python fix_dependencies.py

# Or use the built-in fix
!python housebrain_colab_trainer.py --dataset housebrain_100k --output model
```

## 📈 **Performance Optimization**

### **Free Colab**
- **Max Samples**: 50K-100K
- **Batch Size**: 1
- **Gradient Accumulation**: 8
- **Expected Time**: 3-6 hours

### **Colab Pro+**
- **Max Samples**: 500K-1M
- **Batch Size**: 2-4
- **Gradient Accumulation**: 4-8
- **Expected Time**: 6-12 hours

## 💾 **Download Trained Model**
```python
# Compress model for download
!tar -czf housebrain_model.tar.gz housebrain_production_model/

# Download via Colab
from google.colab import files
files.download('housebrain_model.tar.gz')
```

## 🎯 **Model Testing**
```python
# Test the trained model
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load trained model
model_path = "housebrain_production_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Test input
test_input = {
    "plot": {"length": 60, "width": 40, "unit": "ft"},
    "bedrooms": 3,
    "bathrooms": 2,
    "budget_inr": 2500000
}

# Generate design
prompt = f"Generate house design: {test_input}"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 📊 **Training Timeline**

### **Complete Workflow**
1. **Setup** (5 minutes): Clone repo, install dependencies
2. **Test** (45 minutes): Generate 10K dataset + train test model
3. **Production** (6-12 hours): Generate full dataset + train production model
4. **Validation** (15 minutes): Test model quality
5. **Download** (10 minutes): Package and download model

### **Total Time**
- **Minimum** (10K test): ~1 hour
- **Recommended** (100K): ~6 hours  
- **Full** (1M): ~18 hours

## ✅ **Success Checklist**

- [ ] GPU detected and working
- [ ] Dependencies installed correctly
- [ ] 10K test dataset generated
- [ ] 10K test model trained successfully
- [ ] Full dataset generated
- [ ] Production model trained
- [ ] Model generates valid JSON
- [ ] Model downloaded successfully

## 🚀 **Next Steps**

After successful training:
1. **Integrate into your app**
2. **Deploy via API**
3. **Scale with multiple models**
4. **Continuous improvement with more data**

---

## 🎯 **One-Click Complete Training**

```python
# Complete pipeline: 10K test + 100K production
!git clone https://github.com/Vinay-O/HouseBrainLLM.git && cd HouseBrainLLM
!python -m pip install --force-reinstall torch==2.1.0 transformers==4.36.0 accelerate==0.25.0 peft==0.7.0 datasets==2.15.0 numpy==1.24.3 tqdm
!python colab_generate_10k.py --samples 10000 --output housebrain_10k --fast
!python housebrain_colab_trainer.py --test --dataset housebrain_10k --output housebrain_10k_model
!python colab_generate_10k.py --samples 100000 --output housebrain_100k --fast  
!python housebrain_colab_trainer.py --dataset housebrain_100k --output housebrain_production_model --epochs 3
```

**This will give you a production-ready HouseBrain model in 6-8 hours!** 🏗️⚡
