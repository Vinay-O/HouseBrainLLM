# HouseBrain Training Optimization Guide

## 🎯 Overview

This guide covers optimization strategies for HouseBrain v2 training, including GPU utilization, training speed, quality considerations, and recommended settings for different scenarios.

## 📊 GPU Utilization Parameters

### Primary Parameters That Increase GPU Utilization:

#### 1. **Batch Size** (Most Important)
```bash
--batch-size 8  # Higher = more GPU utilization
```
- **Higher batch size = more parallel processing**
- **A100 can handle batch_size=8-16 easily**
- **Recommended range: 4-12 for optimal utilization**

#### 2. **Gradient Accumulation Steps**
```bash
--grad-accum-steps 4  # Higher = larger effective batch
```
- **Effective batch = batch_size × grad_accum_steps**
- **Higher effective batch = more GPU work per step**
- **Recommended range: 4-8 for better utilization**

#### 3. **Max Length** (Sequence Length)
```bash
--max-length 1024  # Higher = more GPU memory/compute
```
- **Longer sequences = more GPU memory usage**
- **But too long = slower training**
- **Sweet spot: 768-1024 for v2 data**

## ⚡ Training Speed vs Quality Analysis

### Current Settings Analysis:
```bash
--batch-size 8 --grad-accum-steps 6 --max-length 768
```
**Effective batch size = 48, Steps = 2,084, Time = ~2 hours**

### Quality Impact Assessment:

#### ✅ **Positive Quality Factors:**

1. **Good Effective Batch Size (48)**
   - ✅ **Stable gradients** - larger batches provide more stable gradient estimates
   - ✅ **Better convergence** - reduces training variance
   - ✅ **Consistent learning** - more reliable parameter updates

2. **Appropriate Sequence Length (768)**
   - ✅ **Captures full v2 JSON structure** - sufficient for complete plans
   - ✅ **Balanced memory/compute** - not too short, not too long
   - ✅ **Good for architectural data** - fits typical house plan complexity

3. **Single Epoch Training**
   - ✅ **Prevents overfitting** - especially important for synthetic data
   - ✅ **Maintains generalization** - model learns patterns without memorizing

#### ⚠️ **Potential Quality Concerns:**

1. **Training Time vs Quality Trade-off**
   - ⚠️ **2 hours might be too fast** for 100k samples
   - ⚠️ **Could underfit** - model might not learn complex patterns
   - ⚠️ **Limited parameter updates** - only 2,084 steps

2. **Batch Size Considerations**
   - ⚠️ **Effective batch=48 might be too large** for some patterns
   - ⚠️ **Could miss fine-grained details** in smaller batches

## 🎯 Recommended Training Configurations

### Option A: Quality-Focused (Recommended for Production)
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 6 --grad-accum-steps 6 --epochs 1 --max-length 1024 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```
**Specifications:**
- **Effective batch size = 36**
- **Training steps = 2,778**
- **Expected time = ~3 hours**
- **GPU utilization = 85-95%**
- **Quality level = High**

### Option B: Balanced Quality/Speed
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 8 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```
**Specifications:**
- **Effective batch size = 32**
- **Training steps = 3,125**
- **Expected time = ~2.5 hours**
- **GPU utilization = 70-85%**
- **Quality level = Medium-High**

### Option C: Maximum GPU Utilization
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 12 --grad-accum-steps 4 --epochs 1 --max-length 1024 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```
**Specifications:**
- **Effective batch size = 48**
- **Training steps = 2,084**
- **Expected time = ~2-3 hours**
- **GPU utilization = 95-100%**
- **Quality level = Medium**

### Option D: Fastest Training (Speed Priority)
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 8 --epochs 1 --max-length 512 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```
**Specifications:**
- **Effective batch size = 32**
- **Training steps = 3,125**
- **Expected time = ~2-3 hours**
- **GPU utilization = 70-85%**
- **Quality level = Medium**

## 📈 Speed vs Quality Comparison Table

| Option | Effective Batch | Steps | Time | GPU Util | Quality | Best For |
|--------|----------------|-------|------|----------|---------|----------|
| **A (Quality)** | 36 | 2,778 | 3h | 85-95% | **High** | Production |
| **B (Balanced)** | 32 | 3,125 | 2.5h | 70-85% | **Medium-High** | General use |
| **C (Max GPU)** | 48 | 2,084 | 2-3h | 95-100% | **Medium** | GPU efficiency |
| **D (Speed)** | 32 | 3,125 | 2-3h | 70-85% | **Medium** | Quick iteration |

## 🔍 Quality Validation Strategy

### 1. Monitor Training Loss
```bash
# Check if loss is decreasing steadily
# Should see consistent downward trend
# Look for:
# - Smooth decrease in loss
# - No sudden spikes or plateaus
# - Final loss should be reasonable (< 2.0 for language models)
```

### 2. Test Generated Outputs
```bash
# After training, test with:
python - <<'PY'
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load trained model
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "/content/hb_v2_s01")

# Test generation
prompt = "Generate a HouseBrain Plan v2 for a 2-bedroom modern apartment"
inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=800, temperature=0.7)
result = tok.decode(out[0], skip_special_tokens=True)
print(result)
PY
```

### 3. Validate JSON Schema
```bash
# Check if outputs are valid v2 JSON
python -c "
from src.housebrain.validate_v2 import validate_v2_file
import json

# Test with generated output
sample_output = 'path/to/generated_output.json'
try:
    is_valid = validate_v2_file(sample_output)
    print(f'✅ Schema validation: {is_valid}')
except Exception as e:
    print(f'❌ Validation error: {e}')
"
```

### 4. Pipeline Testing
```bash
# Test full pipeline with generated output
python -c "
from src.housebrain.pipeline_v2 import run_pipeline

# Test rendering
run_pipeline(
    input_path='path/to/generated_output.json',
    out_dir='test_output',
    sheet_modes=['floor', 'rcp', 'power', 'plumbing']
)
print('✅ Pipeline test completed')
"
```

## 🎯 Final Recommendations

### For Production Quality:
**Use Option A (Quality-Focused):**
- **Batch size 6, grad_accum 6** (effective batch 36)
- **Max length 1024** (captures full v2 complexity)
- **~3 hours training time** (good balance)
- **Best for final model deployment**

### For Development/Testing:
**Use Option B (Balanced):**
- **Batch size 4, grad_accum 8** (effective batch 32)
- **Max length 768** (sufficient for most cases)
- **~2.5 hours training time** (reasonable speed)
- **Good for iterative development**

### For Quick Iteration:
**Use Option D (Speed Priority):**
- **Batch size 4, grad_accum 8** (effective batch 32)
- **Max length 512** (faster processing)
- **~2-3 hours training time** (fastest)
- **Best for rapid prototyping**

## 📊 GPU Monitoring Commands

### Monitor GPU Usage:
```bash
# Real-time GPU monitoring
!nvidia-smi -l 1

# Check GPU memory usage
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor during training
!watch -n 1 nvidia-smi
```

### Expected GPU Utilization by Option:
- **Option A**: 85-95% GPU utilization, 30-35GB VRAM
- **Option B**: 70-85% GPU utilization, 25-30GB VRAM
- **Option C**: 95-100% GPU utilization, 35-40GB VRAM
- **Option D**: 70-85% GPU utilization, 20-25GB VRAM

## 🚀 Quick Start Commands

### For 3-Notebook Training Strategy:

**Notebook 1 (Shard 1):**
```bash
# Generate data
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_01 --n 100000

# Train with quality settings
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 6 --grad-accum-steps 6 --epochs 1 --max-length 1024 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```

**Notebook 2 (Shard 2):**
```bash
# Generate data
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_02 --n 100000

# Train with balanced settings
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_02 \
  --max-samples 100000 \
  --output /content/hb_v2_s02 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 8 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```

**Notebook 3 (Shard 3):**
```bash
# Generate data
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_03 --n 100000

# Train with speed settings
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_03 \
  --max-samples 100000 \
  --output /content/hb_v2_s03 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 8 --epochs 1 --max-length 512 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1 \
  --grad-checkpointing
```

## 📝 Notes

- **Always monitor training loss** to ensure quality
- **Test generated outputs** after each training run
- **Validate JSON schema** compliance
- **Use appropriate settings** based on your quality vs speed requirements
- **GPU utilization should be 70-100%** for optimal performance
- **Training time should be 2-4 hours** for 100k samples on A100

---

*This guide is based on testing with NVIDIA A100 40GB GPU in Google Colab Pro+ environment.*
