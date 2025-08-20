# HouseBrain Colab Sharded Training — Minimal Steps (Local → Save to Drive)

Follow these exact steps per shard. Train locally in `/content`, then copy results to Drive at the end.

---

## 0) Set shard and paths
```python
SHARD_ID = "04"  # set per notebook: 01, 02, 03, ...
DATASET_DIR   = f"/content/HouseBrainLLM/hb_r1_shard_{SHARD_ID}"
OUTPUT_LOCAL  = f"/content/hb_r1_s{SHARD_ID}"
DRIVE_MOUNT   = "/content/gdrive"  # clean mount path
OUTPUT_DRIVE  = f"{DRIVE_MOUNT}/MyDrive/housebrain/shards/hb_r1_s{SHARD_ID}"
```

## 1) Clone repo (fresh runtime)
```python
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM
```

## 2) Fix dependencies (mandatory)
```python
!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("Restart now if Colab asks; otherwise continue.")
PY
```

## 3) Generate dataset for this shard
```python
!python generate_advanced_dataset.py --samples 100000 --output hb_r1_shard_{SHARD_ID} --shard-size 100000
```

## 4) Train locally (writes to /content)
```python
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset "/content/HouseBrainLLM/hb_r1_shard_{SHARD_ID}" \
  --max-samples 100000 \
  --output   "{OUTPUT_LOCAL}" \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 2 --grad-accum-steps 8 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 2000 --save-total-limit 1
```

## 5) Save to Drive (copy finished run)
```python
from google.colab import drive; drive.mount(DRIVE_MOUNT)
import os; os.makedirs(OUTPUT_DRIVE, exist_ok=True)
print("Copying to:", OUTPUT_DRIVE)
!rsync -avh --info=progress2 "{OUTPUT_LOCAL}/" "{OUTPUT_DRIVE}/"

# Verify
from transformers.trainer_utils import get_last_checkpoint
print("Last checkpoint on Drive:", get_last_checkpoint(OUTPUT_DRIVE))
!ls -lah "{OUTPUT_DRIVE}" | head -n 20
```

### Resume later
```python
# In a fresh runtime: re-clone (Step 1), mount Drive, then run training pointing to Drive
from google.colab import drive; drive.mount(DRIVE_MOUNT)
!python /content/HouseBrainLLM/housebrain_colab_trainer.py \
  --dataset "{DATASET_DIR}" \
  --max-samples 100000 \
  --output   "{OUTPUT_DRIVE}" \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 2 --grad-accum-steps 8 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 1000 --save-total-limit 3
```

### Next shard
Open a new Colab, set `SHARD_ID` to the next value, and repeat Steps 0–5. Keep saving each finished shard to Drive.
