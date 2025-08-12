# 🚀 HouseBrain Training on Google Colab & Kaggle

## Quick Start Guide

This guide will help you train the HouseBrain model using **free GPU resources** on Google Colab or Kaggle.

---

## 🎯 **Option 1: Google Colab (Recommended)**

### **Step 1: Create GitHub Repository**

1. **Go to GitHub:**
   - Visit: https://github.com/
   - Create a new **private** repository named `HouseBrain`

2. **Push your code:**
   ```bash
   # Add your GitHub repository as remote
   git remote add origin https://github.com/YOUR_USERNAME/HouseBrain.git
   
   # Push all code
   git push -u origin main
   ```

### **Step 2: Generate Dataset**

```bash
# Generate 10K dataset (for testing)
python generate_dataset.py --samples 10000 --output housebrain_dataset_v5_10k

# Generate 100K dataset (for full training)
python generate_dataset.py --samples 100000 --output housebrain_dataset_v5_100k
```

### **Step 3: Upload to Google Colab**

1. **Go to Google Colab:**
   - Visit: https://colab.research.google.com/
   - Sign in with your Google account

2. **Create New Notebook:**
   - File → New notebook

3. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free) or **V100** (if available)

4. **Copy this code into the first cell:**
   ```python
   # Install dependencies
   !pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic
   
   # Clone your repository (REPLACE YOUR_USERNAME)
   !git clone https://github.com/YOUR_USERNAME/HouseBrain.git
   %cd HouseBrain
   
   # Check GPU
   import torch
   print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

5. **Upload your dataset:**
   ```python
   from google.colab import files
   import zipfile
   
   # Upload your dataset zip file
   uploaded = files.upload()
   
   # Extract dataset
   for filename in uploaded.keys():
       if filename.endswith('.zip'):
           with zipfile.ZipFile(filename, 'r') as zip_ref:
               zip_ref.extractall('.')
   ```

6. **Start training:**
   ```python
   import sys
   sys.path.append('src')
   
   from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner
   
   # Training configuration
   config = FineTuningConfig(
       model_name="deepseek-ai/deepseek-coder-6.7b-base",
       dataset_path="housebrain_dataset_v5_10k",  # or your dataset name
       output_dir="models/housebrain-colab-trained",
       max_length=1024,
       batch_size=2,
       num_epochs=3,
       use_4bit=True,
   )
   
   # Start training
   trainer = HouseBrainFineTuner(config)
   trainer.train()
   ```

7. **Download the model:**
   ```python
   import zipfile
   from google.colab import files
   
   # Compress model
   with zipfile.ZipFile("housebrain-model.zip", 'w') as zipf:
       # Add model files
       # ... compression code
   
   # Download
   files.download("housebrain-model.zip")
   ```

---

## 🎯 **Option 2: Kaggle Notebooks**

### **Advantages:**
- ✅ **Better GPU** (P100, V100)
- ✅ **More RAM** (30GB)
- ✅ **Longer runtime** (9 hours)
- ✅ **More stable** than Colab

### **Steps:**

1. **Go to Kaggle:**
   - Visit: https://www.kaggle.com/
   - Create account (free)

2. **Create New Notebook:**
   - Notebooks → New Notebook
   - Enable GPU: Settings → Accelerator → **GPU**

3. **Upload Dataset:**
   - Add data → Upload dataset
   - Upload your `housebrain_dataset_v5_10k.zip`

4. **Use the same training code as Colab**

---

## 🎯 **Option 3: Paperspace Gradient**

### **Steps:**

1. **Go to Paperspace:**
   - Visit: https://gradient.paperspace.com/
   - Sign up (free)

2. **Create Notebook:**
   - Create → Notebook
   - Choose **Free GPU** instance

3. **Clone and train:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HouseBrain.git
   cd HouseBrain
   # Upload dataset and run training
   ```

---

## 📊 **Dataset Preparation**

### **Generate Different Sizes:**

```bash
# Small (1K) - for testing
python generate_dataset.py --samples 1000 --output housebrain_dataset_v5_1k

# Medium (10K) - for Colab training
python generate_dataset.py --samples 10000 --output housebrain_dataset_v5_10k

# Large (100K) - for full training
python generate_dataset.py --samples 100000 --output housebrain_dataset_v5_100k
```

### **Upload to Colab:**
1. **Zip your dataset:**
   ```bash
   zip -r housebrain_dataset_v5_10k.zip housebrain_dataset_v5_10k/
   ```

2. **Upload in Colab:**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select your zip file
   ```

---

## ⚙️ **Training Configuration**

### **Colab Optimized:**
```python
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="housebrain_dataset_v5_10k",
    output_dir="models/housebrain-colab-trained",
    max_length=1024,      # Reduced for memory
    batch_size=2,         # Small batch for free GPU
    num_epochs=3,         # 3 epochs for good results
    learning_rate=2e-4,   # Standard learning rate
    use_4bit=True,        # Enable quantization
    lora_r=16,           # LoRA rank
    lora_alpha=32,       # LoRA alpha
)
```

### **Memory Issues?**
```python
# Reduce memory usage
config = FineTuningConfig(
    max_length=512,       # Smaller sequences
    batch_size=1,         # Minimal batch
    num_epochs=2,         # Fewer epochs
    use_4bit=True,        # Keep quantization
)
```

---

## 🔄 **Integration Workflow**

### **1. Train on Colab:**
```python
# Complete training in Colab
trainer = HouseBrainFineTuner(config)
trainer.train()
```

### **2. Download Model:**
- Download the compressed model file
- Save to your local machine

### **3. Extract and Test:**
```bash
# Extract model
unzip housebrain-trained-model.zip -d models/

# Test locally
python test_deepseek.py

# Run API
python -m api.main --finetuned-model models/housebrain-colab-trained
```

### **4. Update Repository:**
```bash
# Add trained model
git add models/housebrain-colab-trained
git commit -m "Add Colab-trained model"
git push origin main
```

---

## 📈 **Expected Results**

### **Training Time:**
- **Colab T4:** 2-4 hours (10K samples)
- **Kaggle P100:** 1-2 hours (10K samples)
- **Colab V100:** 1-2 hours (10K samples)

### **Model Performance:**
- **Compliance Score:** 70-85% (vs 50% baseline)
- **Design Quality:** Significantly improved
- **Generation Speed:** 5-10 seconds per design

### **Memory Usage:**
- **Colab:** 8-10GB RAM, 12GB GPU
- **Kaggle:** 15-20GB RAM, 16GB GPU

---

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   ```python
   # Reduce batch size and sequence length
   config.batch_size = 1
   config.max_length = 512
   ```

2. **Colab Disconnects:**
   - Use Kaggle instead
   - Save checkpoints frequently
   - Use smaller dataset

3. **Model Download Fails:**
   - Use Google Drive for large models
   - Split into smaller files

4. **Training Too Slow:**
   - Use smaller dataset for testing
   - Reduce number of epochs

### **Support:**
- Check `test_deepseek.py` for diagnostics
- Review logs in `models/` directory
- Use `--test` flag for quick validation

---

## 🎉 **Success Checklist**

### **After Training:**
- ✅ Model generates valid JSON
- ✅ Designs pass validation
- ✅ Compliance score > 70%
- ✅ Generation time < 10 seconds
- ✅ Memory usage < 8GB

### **Integration Success:**
- ✅ API serves trained model
- ✅ Designs are realistic
- ✅ Cost estimates accurate
- ✅ Multi-floor support works

---

## 📚 **Quick Commands**

### **Local Setup:**
```bash
# Generate dataset
python generate_dataset.py --samples 10000

# Push to GitHub
git add .
git commit -m "Add dataset"
git push origin main
```

### **Colab Training:**
```python
# Setup
!git clone https://github.com/YOUR_USERNAME/HouseBrain.git
%cd HouseBrain

# Upload dataset and train
# (Use the code above)
```

### **Local Integration:**
```bash
# Extract model
unzip housebrain-trained-model.zip -d models/

# Test
python test_deepseek.py

# Run API
python -m api.main --finetuned-model models/housebrain-colab-trained
```

---

**🎯 Ready to train? Follow the steps above and you'll have a trained HouseBrain model in 2-4 hours!**
