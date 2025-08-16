# HouseBrain Colab Pro+ Training Guide

## 🚀 Colab Pro+ Setup (A100/V100)

### Step 1: Environment Setup
1. **Go to Google Colab Pro+**
2. **Create new notebook**: `HouseBrain_ProPlus_Training`
3. **Runtime → Change runtime type → Hardware accelerator → GPU → A100 (or V100)**
4. **Runtime → Change runtime type → Runtime shape → High-RAM**

### Step 2: Install Dependencies
```python
# Install optimized dependencies for A100
!pip install -q transformers==4.36.0 peft==0.7.0 accelerate==0.25.0 datasets==2.15.0
!pip install -q wandb bitsandbytes==0.41.1
!pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
```

### Step 3: GPU Verification
```python
import torch
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"🎯 GPU: {torch.cuda.get_device_name()}")
print(f"📊 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"🔧 Compute Capability: {torch.cuda.get_device_capability()}")
```

## 🏗️ Training Configuration

### A100 Optimized Settings
- **Model**: `deepseek-ai/deepseek-coder-6.7b-base`
- **Precision**: BF16 (A100) / FP16 (V100)
- **Sequence Length**: 1024-1536 tokens
- **LoRA Rank**: 8-16 (A100) / 4-8 (V100)
- **Batch Size**: Effective 128 via gradient accumulation
- **Learning Rate**: 1.5e-4 with cosine scheduler
- **Gradient Clipping**: 0.5
- **Gradient Checkpointing**: Enabled
- **Token Packing**: Enabled

### Dataset Expansion Plan
- **Current**: 425K samples
- **Target**: 1.0M high-quality samples
- **Daily Generation**: 20-40K → 10-20K after quality gating
- **Weekly Shards**: 150-200K samples per shard

## 📊 Training Phases

### Phase 1: Baseline Training (Week 1)
- **Dataset**: 425K current samples
- **Goal**: Establish baseline performance
- **Metrics**: JSON parse rate, validation loss, India-specific tasks

### Phase 2: Data Expansion (Week 2-3)
- **Dataset**: 650K → 800K samples
- **Focus**: Hard cases, documentation tasks, multi-turn dialogues
- **Quality Gates**: Schema validation, code compliance, deduplication

### Phase 3: 2D/3D Integration (Week 4)
- **Dataset**: 1.0M samples
- **Add**: 2D floor plan tasks, 3D model metadata
- **Pipeline**: JSON → SVG/PNG → Mesh with metadata

## 🎯 Expected Performance

### A100 Capabilities
- **Training Time**: 4-6 hours for 425K samples
- **Memory Usage**: 30-35GB (comfortable headroom)
- **Throughput**: 2-4x faster than P100
- **Stability**: Long sessions, fewer disconnects

### Quality Improvements
- **JSON Parse Rate**: ≥98%
- **Code Compliance**: ≥95%
- **India-Specific Tasks**: +15-20% improvement
- **Documentation Quality**: BLEU/ROUGE scores

## 📁 Repository Structure
```
housebrain_v1_1/
├── src/housebrain/          # Core modules
├── api/                     # FastAPI server
├── housebrain_dataset_v5_425k/  # Current dataset
├── generate_combined_dataset.py  # Dataset generation
├── generate_india_dataset.py     # India-specific data
├── merge_models.py          # LoRA merging
├── COLAB_PROPLUS_TRAINING.md    # This guide
├── FUTURE_ROADMAP.md        # Long-term goals
└── requirements.txt         # Dependencies
```

## 🚀 Next Steps
1. **Upload dataset to Google Drive**
2. **Create Colab Pro+ notebook**
3. **Start baseline training**
4. **Expand dataset with quality gates**
5. **Implement 2D/3D pipeline**
