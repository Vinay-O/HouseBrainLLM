## HouseBrain 1M Training (Sharded, A100) — Required Steps Only

Train 1,000,000 samples as 10 parallel shards (100k each). Run one shard per Colab Pro+ A100 notebook/session.

### 0) Naming convention
- Shard dataset folder: `hb_r1_shard_XX` where `XX ∈ {01..10}`
- Output model folder: `/content/hb_r1_sXX`

### 1) New notebook: clone repo
```python
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM
!git pull
```

### 2) Fix dependencies, then restart runtime (mandatory)
```python
!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("Restart now")
PY
```
Restart runtime. Then re-enter the repo:
```python
%cd /content/HouseBrainLLM
!git pull
```

### 3) Generate one 100k dataset shard (unique XX per notebook)
```python
!python generate_advanced_dataset.py --samples 100000 --output hb_r1_shard_XX --shard-size 100000
```

### 4) Train DeepSeek R1 on that shard (A100, bf16 auto, LoRA + 4‑bit)
```python
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_r1_shard_XX \
  --max-samples 100000 \
  --output /content/hb_r1_sXX \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 2 --grad-accum-steps 8 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 2000 --save-total-limit 1
```

Run shards `01..10` in separate notebooks/sessions.

### 5) Merge LoRA adapters after all shards finish
```python
!python merge_models.py \
  --strategy average \
  --inputs /content/hb_r1_s01 /content/hb_r1_s02 /content/hb_r1_s03 /content/hb_r1_s04 /content/hb_r1_s05 \
          /content/hb_r1_s06 /content/hb_r1_s07 /content/hb_r1_s08 /content/hb_r1_s09 /content/hb_r1_s10 \
  --output /content/housebrain_r1_1m_merged
```

### Notes
- Use a new notebook per shard (`XX` differs per notebook) to run in parallel.
- If step time stays >6s after 200 steps, try: `--max-length 640` or `--batch-size 3 --grad-accum-steps 6` (if VRAM allows).
- Global dedupe (optional but recommended if shards are generated over multiple sessions):
```python
# Dry run
!python dedupe_shards.py --roots /content/HouseBrainLLM --glob "hb_r1_shard_*" --dry-run
# Enforce deletion of cross-shard duplicates
!python dedupe_shards.py --roots /content/HouseBrainLLM --glob "hb_r1_shard_*" --action delete
```


