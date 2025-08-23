# HouseBrain 8-Hour Optimized Training Plan (200k v2 Data)

## 🎯 Executive Summary

This guide provides an optimized strategy to train **200,000 high-quality v2 samples** with a `max-length` of **2048**, ensuring the total *model training time* is under **8 hours**.

The recommended approach is to use **two separate Colab notebooks to train two 100k sample shards**. This is significantly safer and more robust than a single long-running session.

## 📈 Strategy: Why 2x 100k is Better than 1x 150k

- **Safety & Stability**: A single 8+ hour Colab session has a high risk of disconnection, crashing, or running out of memory. Splitting the work into two ~4-hour training sessions ensures that if one fails, you don't lose all your progress.
- **Resource Management**: Each notebook starts fresh, preventing memory leaks or filesystem clutter from impacting the second half of the training.
- **Flexibility**: You can run the two notebooks in parallel (if you have enough compute units) or sequentially without risking a single point of failure.

## ⏰ Timeline & Time Budget

This plan is designed to fit your 8-hour training window.

- **Total Model Training Time**: **~7.8 hours** (well within your 8-hour limit)
- **Total Project Time (including data generation)**: **~10 hours**

---

### **Breakdown Per Shard (You will do this twice):**

| Task | Estimated Time |
| :--- | :--- |
| 1. Setup & Data Generation (100k samples) | ~1 hour |
| 2. Model Training (100k samples @ `max_length=2048`) | **~3.9 hours** |
| **Total Time Per Shard** | **~5 hours** |

---

## 🚀 The Plan: Step-by-Step Commands

You will run these steps in two separate Colab notebooks.

### Colab Notebook 1: Training Shard 1 (100k Samples)

#### **Step 1: Setup and Data Generation**
```python
# Clone the repository and install dependencies
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM

# This step requires a kernel restart
!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("✅ Dependencies fixed. Please RESTART the kernel now.")
PY

# Pull any recent changes after restart
%cd /content/HouseBrainLLM
!git pull

# Generate the first 100k data shard
# This will take about 1 hour
print("⏳ Generating 100,000 v2 samples for Shard 1...")
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_01 --n 100000
print("✅ Data generation for Shard 1 complete.")
!du -sh /content/HouseBrainLLM/hb_v2_shard_01
```

#### **Step 2: Run Optimized Training for Shard 1**
This command is optimized for an **A100 GPU** to finish in approximately **3.9 hours**.

```python
# Train Shard 1
%env HB_WARUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_01 \
  --max-samples 100000 \
  --output /content/hb_v2_s01 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 8 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --max-length 2048 \
  --eval-steps 0 \
  --save-steps 1000 \
  --save-total-limit 1 \
  --grad-checkpointing
```
**Parameters Explained:**
- `--max-length 2048`: Ensures the full, high-quality v2 JSON is used, preventing data loss.
- `--batch-size 8 --grad-accum-steps 8`: Creates a large **effective batch size of 64**. This maximizes GPU throughput on an A100 and is key to finishing within the time limit.

#### **Step 3: Save Shard 1 to Google Drive**
```python
# Save the trained model adapter to Drive
!mkdir -p /content/drive/MyDrive/HouseBrainLLM/checkpoints
!rsync -ah --progress /content/hb_v2_s01/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s01/
print("✅ Shard 1 saved to Google Drive.")
```

---

### Colab Notebook 2: Training Shard 2 (100k Samples)

Repeat the exact same process in a **new, separate Colab notebook**. Just change the directory names for shard 2.

#### **Step 1: Setup and Data Generation**
```python
# (Same setup steps as above: clone, fix_dependencies, restart, pull)

# Generate the second 100k data shard
print("⏳ Generating 100,000 v2 samples for Shard 2...")
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_02 --n 100000
print("✅ Data generation for Shard 2 complete.")
!du -sh /content/HouseBrainLLM/hb_v2_shard_02
```

#### **Step 2: Run Optimized Training for Shard 2**
```python
# Train Shard 2
%env HB_WARUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_02 \
  --max-samples 100000 \
  --output /content/hb_v2_s02 \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 8 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --max-length 2048 \
  --eval-steps 0 \
  --save-steps 1000 \
  --save-total-limit 1 \
  --grad-checkpointing
```

#### **Step 3: Save Shard 2 to Google Drive**
```python
# Save the trained model adapter to Drive
!mkdir -p /content/drive/MyDrive/HouseBrainLLM/checkpoints
!rsync -ah --progress /content/hb_v2_s02/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s02/
print("✅ Shard 2 saved to Google Drive.")
```

## 🛠️ Safer Alternative (If you get VRAM errors)

The settings above are aggressive to meet the time constraint. If you encounter "Out of Memory" errors, use this slightly slower but safer alternative.

```python
# Safer Training Command (use if the one above fails)
!python housebrain_colab_trainer.py \
  --dataset ... \
  --output ... \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 4 \
  --grad-accum-steps 16 \
  --epochs 1 \
  --max-length 2048 \
  ...
```
This still uses an effective batch size of 64 but requires less VRAM per step. It may increase the training time slightly.

## ✅ Next Steps

After completing this plan, you will have two new high-quality model adapters (`hb_v2_s01` and `hb_v2_s02`). The next step will be to **merge these with your original 600k model** to create the final, enhanced super-model.
