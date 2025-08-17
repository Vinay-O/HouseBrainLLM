# 🏠 HouseBrain 10K Training Guide for Colab

## 🚀 Quick Start (10K Dataset Training)

### Step 1: Clone Repository in Colab
```python
# Clone the repository
!git clone https://github.com/your-username/housebrain_v1_1.git
%cd housebrain_v1_1
```

### Step 2: Install Dependencies
```python
# Install required packages
!pip install -r requirements.txt
!pip install --upgrade transformers accelerate peft datasets
```

### Step 3: Generate 10K Dataset
```python
# Generate 10K dataset directly in Colab
!python colab_generate_10k.py
```

**Expected Time**: 15-30 minutes
**Output**: `housebrain_dataset_r1_super_10k_aug/` directory

### Step 4: Run Training
```python
# Start 10K test training
!python colab_10k_test_train.py
```

**Expected Time**: 30-60 minutes
**Output**: `housebrain-10k-test-trained/` directory

## 📁 Repository Structure

After cloning, you'll have:
```
housebrain_v1_1/
├── colab_10k_test_train.py          # 10K training script
├── colab_proplus_train_r1_super.py  # 1M training script (for later)
├── colab_setup.py                   # Setup and validation
├── colab_generate_10k.py            # 10K dataset generator
├── generate_1m_super_quality.py     # Main dataset generator
├── augment_dataset_v1_1.py          # Dataset augmentation
├── requirements.txt                  # Dependencies
└── README.md                        # Project documentation
```

## 🔧 Complete Colab Notebook

Here's the complete notebook code:

```python
# Cell 1: Clone Repository
!git clone https://github.com/your-username/housebrain_v1_1.git
%cd housebrain_v1_1

# Cell 2: Install Dependencies
!pip install -r requirements.txt
!pip install --upgrade transformers accelerate peft datasets

# Cell 3: Generate 10K Dataset
!python colab_generate_10k.py

# Cell 4: Verify Dataset
!python colab_setup.py

# Cell 5: Start Training
!python colab_10k_test_train.py
```

## 📊 What the 10K Training Does

### Dataset Generation
- **10,000 samples** with super-quality reasoning
- **9,000 train** + **1,000 validation** split
- **Geometric metadata** for 2D/3D generation
- **India-specific** architectural features
- **NBC 2016 compliance**

### Training Configuration
- **Model**: DeepSeek-R1-Distill-Qwen-7B
- **LoRA Rank**: 32 (optimized for speed)
- **Max Steps**: 1,000 (quick validation)
- **Batch Size**: 1 with gradient accumulation
- **Learning Rate**: 2e-4
- **Expected Time**: 30-60 minutes

## 🎯 Success Criteria

### Dataset Generation Success
- ✅ 10K samples generated
- ✅ Train/validation split created
- ✅ Geometric metadata added
- ✅ Dataset info file created

### Training Success
- ✅ Training starts without errors
- ✅ Loss decreases over time
- ✅ Checkpoints saved every 200 steps
- ✅ Evaluation runs every 100 steps
- ✅ Final evaluation loss < 2.0

## 🚨 Troubleshooting

### If Git Clone Fails
```python
# Alternative: Download from GitHub releases
!wget https://github.com/your-username/housebrain_v1_1/archive/main.zip
!unzip main.zip
%cd housebrain_v1_1-main
```

### If Dependencies Fail
```python
# Manual installation
!pip install torch transformers accelerate peft datasets tqdm numpy
```

### If Dataset Generation Fails
```python
# Check available memory
!free -h
# Reduce workers if needed
!python colab_generate_10k.py --workers 2
```

### If Training Fails
```python
# Check GPU
!nvidia-smi
# Restart runtime and try again
```

## 📈 Monitoring Training

### Check Training Progress
```python
# Monitor logs
!tail -f training_log_10k_test.txt

# Check GPU usage
!nvidia-smi

# Check disk space
!df -h
```

### Training Outputs
- **Logs**: `training_log_10k_test.txt`
- **Metrics**: `training_metrics_10k_test.json`
- **Model**: `housebrain-10k-test-trained/`
- **Checkpoints**: Every 200 steps

## 🔄 Next Steps After 10K Success

1. **Test the model** on sample architectural problems
2. **Generate 1M dataset** for full training
3. **Run full 1M training** with `colab_proplus_train_r1_super.py`
4. **Deploy the model** for 2D/3D generation

## 📞 Support Commands

### Check System Status
```python
# GPU info
!nvidia-smi

# Memory usage
!free -h

# Disk space
!df -h

# Python packages
!pip list | grep -E "(torch|transformers|peft)"
```

### Debug Dataset
```python
# Check dataset structure
!ls -la housebrain_dataset_r1_super_10k_aug/

# Check sample quality
!python colab_setup.py

# View sample
!find housebrain_dataset_r1_super_10k_aug/train -name "*.json" | head -1 | xargs cat | jq '.output' | head -10
```

---

**Ready to start? Run the cells in order and you'll have a trained HouseBrain model in about 1-2 hours!** 🚀
