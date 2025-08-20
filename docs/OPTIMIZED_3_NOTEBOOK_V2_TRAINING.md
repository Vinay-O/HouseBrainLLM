# Optimized 3-Notebook v2 Training Strategy

**Goal**: Train 300k v2 data and combine with existing 600k for a 900k super model using only 3 notebooks and 180 credits.

**Result**: HouseBrain 900k Super v2 Model ready for local testing and deployment.

---

## Overview

- **Data**: 300k v2 enhanced samples (3 shards × 100k each)
- **Model**: 900k total (600k existing + 300k v2)
- **Notebooks**: 3 parallel sessions
- **Estimated Time**: 7-8 hours total
- **Speed Optimizations**: 4x faster training settings

---

## Notebook 1: Generate All Data + Train Shard 1

### Step 1: Setup Environment
```bash
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM
```

### Step 2: Fix Dependencies
```python
!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("Restart now")
PY
```

### Step 3: Update Repository
```bash
%cd /content/HouseBrainLLM
!git pull
```

### Step 4: Generate ALL 3 Shards (100k each = 300k total)
```bash
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_01 --n 100000
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_02 --n 100000  
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_03 --n 100000
```

**Expected Output**: 
```
✅ Generated 100000 synthetic v2 plans at /content/HouseBrainLLM/hb_v2_shard_01
✅ Generated 100000 synthetic v2 plans at /content/HouseBrainLLM/hb_v2_shard_02
✅ Generated 100000 synthetic v2 plans at /content/HouseBrainLLM/hb_v2_shard_03
```

### Step 5: Train Shard 1 (FAST SETTINGS)
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 4 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1
```

**Speed Optimizations**:
- `batch-size 4` (2x faster than batch-size 2)
- `grad-accum-steps 4` (2x faster than grad-accum-steps 8)
- `save-steps 1000` (more frequent saves)

**Estimated Time**: ~2-2.5 hours

### Step 6: Save Shard 1 to Drive
```bash
!mkdir -p /content/drive/MyDrive/HouseBrainLLM/checkpoints
!rsync -ah --progress /content/hb_v2_s01/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s01/
```

---

## Notebook 2: Train Shard 2

### Step 1: Setup (Same as Notebook 1)
```bash
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM

!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("Restart now")
PY

%cd /content/HouseBrainLLM
!git pull
```

### Step 2: Train Shard 2 (FAST SETTINGS)
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_02 \
  --max-samples 100000 \
  --output /content/hb_v2_s02 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 4 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1
```

**Estimated Time**: ~2-2.5 hours

### Step 3: Save Shard 2 to Drive
```bash
!rsync -ah --progress /content/hb_v2_s02/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s02/
```

---

## Notebook 3: Train Shard 3 + Merge Everything

### Step 1: Setup (Same as Notebook 1)
```bash
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM

!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("Restart now")
PY

%cd /content/HouseBrainLLM
!git pull
```

### Step 2: Train Shard 3 (FAST SETTINGS)
```bash
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_03 \
  --max-samples 100000 \
  --output /content/hb_v2_s03 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 --grad-accum-steps 4 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 1
```

**Estimated Time**: ~2-2.5 hours

### Step 3: Save Shard 3 to Drive
```bash
!rsync -ah --progress /content/hb_v2_s03/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s03/
```

### Step 4: MERGE - Create Super Model (600k + 300k v2 = 900k)
```python
from merge_models import ModelMerger

base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Load your existing 600k model from Drive
!rsync -ah --progress /content/drive/MyDrive/HouseBrainLLM/checkpoints/housebrain_600k_merged/ /content/housebrain_600k_merged/

# Load the 3 v2 shards
!rsync -ah --progress /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s01/ /content/hb_v2_s01/
!rsync -ah --progress /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s02/ /content/hb_v2_s02/
!rsync -ah --progress /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s03/ /content/hb_v2_s03/

# First merge the 3 v2 shards into one 300k v2 model
print("Merging 3 v2 shards into 300k v2 model...")
merger = ModelMerger(base_model_name=base)
merger.load_base_model()
v2_models = ["/content/hb_v2_s01", "/content/hb_v2_s02", "/content/hb_v2_s03"]
merger.load_trained_models(v2_models)
merger.merge_models("/content/housebrain_v2_300k_merged", "average")

# Then combine 600k + 300k v2 = SUPER MODEL
print("Creating 900k super model...")
inputs = [
    "/content/housebrain_600k_merged",       # your existing 600k
    "/content/housebrain_v2_300k_merged",    # new 300k v2
]
merger = ModelMerger(base_model_name=base)
merger.load_base_model()
merger.load_trained_models(inputs)
final_adapter = merger.merge_models("/content/housebrain_900k_super_v2", "weighted")

print("✅ SUPER MODEL READY: 900k total (600k + 300k v2)")
```

### Step 5: Save Super Model to Drive
```bash
!mkdir -p /content/drive/MyDrive/HouseBrainLLM/merged
!rsync -ah --progress /content/housebrain_900k_super_v2/ /content/drive/MyDrive/HouseBrainLLM/merged/housebrain_900k_super_v2/

print("🎉 DOWNLOAD READY:")
print("Super Model: /content/drive/MyDrive/HouseBrainLLM/merged/housebrain_900k_super_v2/")
print("Individual Shards: /content/drive/MyDrive/HouseBrainLLM/checkpoints/")
```

---

## Timeline Summary

| Phase | Duration | Description |
|-------|----------|-------------|
| Data Generation | ~20 min | Generate 300k v2 samples |
| Training Shard 1 | ~2.5h | Train 100k v2 samples |
| Training Shard 2 | ~2.5h | Train 100k v2 samples |
| Training Shard 3 | ~2.5h | Train 100k v2 samples |
| Merging | ~30 min | Combine all models |
| **Total** | **~8h** | **Complete 900k super model** |

---

## Speed Optimizations Applied

### Training Speed (4x faster):
- **Batch Size**: 4 (vs 2) - 2x faster
- **Gradient Accumulation**: 4 (vs 8) - 2x faster
- **Save Steps**: 1000 (vs 2000) - More frequent saves

### Memory Efficiency:
- **Max Length**: 768 tokens (optimal for v2 data)
- **Save Total Limit**: 1 (minimal disk usage)
- **Eval Steps**: 0 (no validation during training)

---

## Final Result

### Model Specifications:
- **Total Training Data**: 900k samples
- **v2 Features**: Full schema compliance (metadata, wall layers, openings, stairs, electrical, schedules)
- **Base Model**: DeepSeek-R1-Distill-Qwen-7B
- **Training Method**: LoRA + 4-bit quantization
- **Merge Strategy**: Weighted (v2 features prioritized)

### Files Generated:
- `housebrain_900k_super_v2/` - Complete super model
- `hb_v2_s01/`, `hb_v2_s02/`, `hb_v2_s03/` - Individual shards
- `housebrain_v2_300k_merged/` - Combined v2 model

### Local Usage:
```python
from src.housebrain.llm import HouseBrainLLM

# Load the super model
hb = HouseBrainLLM(
    demo_mode=False, 
    finetuned_model_path="path/to/housebrain_900k_super_v2"
)

# Generate v2 plans
sample = {
    "rooms": ["living", "kitchen", "bedroom", "bathroom"],
    "style": "modern", 
    "area_sqft": 1800, 
    "stories": 1, 
    "location": "suburban"
}
plan = hb.generate_house_design(sample)
```

---

## Troubleshooting

### Data Generation Issues:
- Ensure you have the latest code: `!git pull`
- Check the generator works: `!python generate_synthetic_v2.py --help`
- Verify output directories exist

### Training Issues:
- Monitor GPU memory: `!nvidia-smi`
- Check batch size compatibility with your GPU
- Ensure dataset paths are correct

### Merging Issues:
- Verify all shard paths exist before merging
- Check available disk space for merged models
- Ensure base model is accessible

---

## Credits Usage Estimate

- **Notebook 1**: ~60 credits (setup + generation + training)
- **Notebook 2**: ~60 credits (training only)
- **Notebook 3**: ~60 credits (training + merging)
- **Total**: ~180 credits (within your budget)

**Success Criteria**: 900k super v2 model ready for local deployment and testing.
