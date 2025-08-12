# 🚀 HouseBrain Training Guide

## Free Training Options for HouseBrain LLM

This guide shows you how to train the HouseBrain model using free resources like Google Colab, avoiding memory constraints on your local machine.

## 📋 **Quick Start**

### 1. Generate 100K Dataset
```bash
# Generate 100K synthetic test cases
python generate_dataset.py --samples 100000 --output housebrain_dataset_v5_100k
```

### 2. Create Git Repository
```bash
# Initialize Git (already done)
git remote add origin https://github.com/YOUR_USERNAME/HouseBrain.git
git push -u origin main
```

### 3. Train on Google Colab
- Upload `colab_training.ipynb` to Google Colab
- Follow the notebook instructions
- Download the trained model

---

## 🎯 **Option 1: Google Colab (Recommended)**

### **Advantages:**
- ✅ **Free GPU** (T4, V100, A100)
- ✅ **12GB RAM** available
- ✅ **Easy setup** and download
- ✅ **No installation** required

### **Steps:**

1. **Go to Google Colab:**
   - Visit: https://colab.research.google.com/
   - Sign in with your Google account

2. **Upload the Notebook:**
   - File → Upload notebook
   - Select `colab_training.ipynb`

3. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free) or **V100** (if available)

4. **Clone Your Repository:**
   ```python
   !git clone https://github.com/YOUR_USERNAME/HouseBrain.git
   %cd HouseBrain
   ```

5. **Upload Dataset:**
   - Upload your `housebrain_dataset_v5_100k.zip`
   - Or generate it in Colab

6. **Run Training:**
   - Execute all cells in the notebook
   - Training will take 2-4 hours

7. **Download Model:**
   - Model will be automatically compressed and downloaded
   - Save to your local machine

---

## 🎯 **Option 2: Kaggle Notebooks**

### **Advantages:**
- ✅ **Free GPU** (P100, V100)
- ✅ **30GB RAM** available
- ✅ **Longer runtime** (9 hours)
- ✅ **Better performance** than Colab

### **Steps:**

1. **Go to Kaggle:**
   - Visit: https://www.kaggle.com/
   - Create account (free)

2. **Create New Notebook:**
   - Notebooks → New Notebook
   - Copy content from `colab_training.ipynb`

3. **Upload Dataset:**
   - Add data → Upload dataset
   - Upload your `housebrain_dataset_v5_100k.zip`

4. **Enable GPU:**
   - Settings → Accelerator → **GPU**

5. **Run Training:**
   - Execute all cells
   - Download model when complete

---

## 🎯 **Option 3: Paperspace Gradient**

### **Advantages:**
- ✅ **Free GPU** (RTX 4000)
- ✅ **8GB RAM** available
- ✅ **Jupyter interface**
- ✅ **Git integration**

### **Steps:**

1. **Go to Paperspace:**
   - Visit: https://gradient.paperspace.com/
   - Sign up (free)

2. **Create Notebook:**
   - Create → Notebook
   - Choose **Free GPU** instance

3. **Clone Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HouseBrain.git
   cd HouseBrain
   ```

4. **Upload Dataset:**
   - Upload your dataset files
   - Or generate in notebook

5. **Run Training:**
   - Execute training cells
   - Download results

---

## 🎯 **Option 4: Local Training (Optimized)**

### **For M2 Pro with Memory Optimization:**

```bash
# Generate smaller dataset for testing
python generate_dataset.py --samples 10000 --output housebrain_dataset_v5_10k

# Train with minimal memory usage
python finetune_m2pro.py --dataset housebrain_dataset_v5_10k --no-mps --batch-size 1
```

### **Memory Optimization Tips:**
- Use `--no-mps` to disable GPU (CPU only)
- Set `batch_size=1`
- Reduce `max_length=512`
- Use `use_4bit=False`

---

## 📊 **Dataset Generation**

### **Generate Different Sizes:**

```bash
# Small dataset (1K samples) - for testing
python generate_dataset.py --samples 1000 --output housebrain_dataset_v5_1k

# Medium dataset (10K samples) - for local training
python generate_dataset.py --samples 10000 --output housebrain_dataset_v5_10k

# Large dataset (100K samples) - for Colab training
python generate_dataset.py --samples 100000 --output housebrain_dataset_v5_100k
```

### **Dataset Features:**
- ✅ **Realistic plot dimensions** (1K-10K sqft)
- ✅ **Multiple architectural styles** (15 styles)
- ✅ **Global regions** (15 regions)
- ✅ **Varied room configurations**
- ✅ **Realistic budgets** ($100K-$2M)

---

## 🔄 **Integration Workflow**

### **1. Train on Colab:**
```python
# In Google Colab
!git clone https://github.com/YOUR_USERNAME/HouseBrain.git
%cd HouseBrain
!python generate_dataset.py --samples 100000
!python finetune_m2pro.py --dataset housebrain_dataset_v5_100k
```

### **2. Download Model:**
- Download `housebrain-trained-model.zip`
- Extract to your local machine

### **3. Test Locally:**
```bash
# Extract model
unzip housebrain-trained-model.zip -d models/

# Test the model
python test_deepseek.py

# Run API with trained model
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

## ⚙️ **Training Configuration**

### **Colab Optimized Config:**
```python
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="housebrain_dataset_v5_100k",
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

### **Local Optimized Config:**
```python
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="housebrain_dataset_v5_10k",
    output_dir="models/housebrain-local-trained",
    max_length=512,       # Very small for memory
    batch_size=1,         # Minimal batch size
    num_epochs=2,         # Fewer epochs
    use_4bit=False,       # Disable quantization
    lora_r=8,            # Smaller LoRA
    lora_alpha=16,       # Smaller alpha
)
```

---

## 📈 **Expected Results**

### **Training Time:**
- **Colab T4 GPU:** 2-4 hours (100K samples)
- **Kaggle P100 GPU:** 1-2 hours (100K samples)
- **Local M2 Pro:** 8-12 hours (10K samples)

### **Model Performance:**
- **Compliance Score:** 70-85% (vs 50% baseline)
- **Design Quality:** Significantly improved
- **Generation Speed:** 5-10 seconds per design

### **Memory Usage:**
- **Colab:** 8-10GB RAM, 12GB GPU
- **Local:** 16GB RAM (CPU only)

---

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   ```bash
   # Reduce batch size and sequence length
   python finetune_m2pro.py --batch-size 1 --max-length 512
   ```

2. **Colab Disconnects:**
   - Use Kaggle or Paperspace instead
   - Save checkpoints frequently

3. **Model Download Fails:**
   - Use Google Drive for large models
   - Split into smaller files

4. **Training Too Slow:**
   - Use smaller dataset for testing
   - Reduce number of epochs

### **Support:**
- Check the `test_deepseek.py` for diagnostics
- Review logs in `models/` directory
- Use `--test` flag for quick validation

---

## 🎉 **Success Metrics**

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

## 📚 **Next Steps**

1. **Train on Colab** with 100K dataset
2. **Download and test** the model locally
3. **Integrate with API** for production use
4. **Fine-tune further** based on results
5. **Scale to larger models** when budget allows

**Happy Training! 🚀**
