# 🏠 HouseBrain Training Retry Guide

## Current Status ✅

Your datasets are ready and optimized:
- **10K Test Dataset**: `housebrain_dataset_r1_super_10k_aug.tar.gz` (34MB)
- **1M Full Dataset**: `housebrain_dataset_r1_super_1M_aug_v1_1.tar.gz` (3.37GB)
- **Training Scripts**: Updated with error handling for `KeyError: 'qwen2'`

## 🚀 Quick Start (Recommended)

### Step 1: Upload to Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Upload these files:
   - `housebrain_dataset_r1_super_10k_aug.tar.gz`
   - `colab_setup.py`
   - `colab_10k_test_train.py`
   - `colab_proplus_train_r1_super.py`

### Step 2: Run 10K Test First
```python
# Cell 1: Setup
!python colab_setup.py

# Cell 2: Run 10K test training
!python colab_10k_test_train.py
```

**Expected Time**: 30-60 minutes
**Purpose**: Validate the training pipeline before full 1M training

### Step 3: If 10K Test Succeeds
```python
# Upload the 1M dataset
# housebrain_dataset_r1_super_1M_aug_v1_1.tar.gz

# Run full training
!python colab_proplus_train_r1_super.py
```

**Expected Time**: 6-12 hours
**Purpose**: Full model training on 1M super-quality samples

## 🔧 What's Fixed

### 1. KeyError: 'qwen2' Fix
- Added automatic transformers upgrade
- Added error handling with alternative loading method
- Added `ignore_mismatched_sizes=True` fallback

### 2. Dataset Augmentation
- Both datasets now include precise geometric metadata
- IFC integration for BIM compatibility
- DXF layer specifications
- 2D dimension scaffolds
- Project coordinate systems

### 3. Training Optimizations
- **10K Test**: Fast validation with reduced parameters
- **1M Full**: Production-quality training with full optimizations
- Automatic error recovery and logging

## 📊 Dataset Details

### 10K Test Dataset
- **Size**: 34MB compressed
- **Samples**: 9K train, 1K validation
- **Features**: Full augmentation with geometric metadata
- **Purpose**: Quick validation and debugging

### 1M Full Dataset
- **Size**: 3.37GB compressed
- **Samples**: 900K train, 100K validation
- **Features**: Complete super-quality reasoning dataset
- **Purpose**: Production model training

## 🎯 Training Parameters

### 10K Test Configuration
```python
LORA_R = 32
MAX_STEPS = 1000
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
```

### 1M Full Configuration
```python
LORA_R = 64
MAX_STEPS = 100000
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 1e-4
```

## 🚨 Troubleshooting

### If KeyError: 'qwen2' Still Occurs
1. **Manual transformers upgrade**:
   ```python
   !pip install --upgrade transformers
   !pip install --upgrade accelerate
   ```

2. **Restart runtime** and try again

3. **Alternative model loading** (already included in scripts)

### If Out of Memory
1. **Reduce batch size**: Change `BATCH_SIZE = 1` to `BATCH_SIZE = 1`
2. **Increase gradient accumulation**: Change `GRADIENT_ACCUMULATION = 8` to `GRADIENT_ACCUMULATION = 16`
3. **Enable gradient checkpointing** (already enabled)

### If Training is Too Slow
1. **Use Colab Pro+** for better GPU
2. **Reduce sequence length**: Change `SEQUENCE_LENGTH = 4096` to `SEQUENCE_LENGTH = 2048`
3. **Reduce LoRA rank**: Change `LORA_R = 32` to `LORA_R = 16`

## 📈 Expected Results

### 10K Test Success Criteria
- Training completes without errors
- Loss decreases over time
- Model saves checkpoints successfully
- Evaluation loss is reasonable (< 2.0)

### 1M Full Success Criteria
- Training completes in 6-12 hours
- Final evaluation loss < 1.5
- Model generates coherent architectural responses
- Geometric data is properly formatted

## 🔄 Next Steps After Training

1. **Model Evaluation**: Test on sample architectural problems
2. **2D/3D Generation**: Use model output for floor plan generation
3. **Model Merging**: If using parallel training
4. **Deployment**: Integrate into your application

## 📞 Support

If you encounter issues:
1. Check the training logs in the output directory
2. Verify GPU availability and memory
3. Ensure all files are uploaded correctly
4. Try the 10K test first before full training

---

**Ready to retry? Start with the 10K test to validate everything works!** 🚀
