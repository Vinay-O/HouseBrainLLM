# HouseBrain v2: 200k Training (4 shards) + Merge With Existing 600k Model

This guide shows how to:

1) Generate 200k HouseBrain v2 synthetic dataset (4 shards × 50k)
2) Train each v2 shard on Colab A100 and save to Drive
3) Merge the 4 new v2 adapters into a single v2_200k adapter
4) Combine the existing 600k model with the new v2_200k adapter into a final model
5) Smoke‑test with the v2 pipeline (SVG/DXF/glTF/BOQ)

> Base model used throughout: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

---

## Prerequisites

- Colab Pro/Pro+ session with A100 40GB (recommended)
- Google Drive with enough space to store adapters and logs (~1–3 GB/adaptor)
- This repo cloned in Colab under `/content/HouseBrainLLM`

```python
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM
!git pull
```

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 1) Generate 200k v2 Dataset (4 × 50k)

We use the enhanced v2 generator `generate_synthetic_v2.py` (metadata, wall layers, openings, optional stairs, basic electrical, schedules).

```python
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_01 --n 50000
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_02 --n 50000
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_03 --n 50000
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_04 --n 50000
```

Optional: copy shards to Drive for safekeeping

```bash
mkdir -p /content/drive/MyDrive/HouseBrainLLM/v2_shards
rsync -ah --progress /content/HouseBrainLLM/hb_v2_shard_01/ /content/drive/MyDrive/HouseBrainLLM/v2_shards/hb_v2_shard_01/
rsync -ah --progress /content/HouseBrainLLM/hb_v2_shard_02/ /content/drive/MyDrive/HouseBrainLLM/v2_shards/hb_v2_shard_02/
rsync -ah --progress /content/HouseBrainLLM/hb_v2_shard_03/ /content/drive/MyDrive/HouseBrainLLM/v2_shards/hb_v2_shard_03/
rsync -ah --progress /content/HouseBrainLLM/hb_v2_shard_04/ /content/drive/MyDrive/HouseBrainLLM/v2_shards/hb_v2_shard_04/
```

---

## 2) Train Each v2 Shard (A100, LoRA+4bit)

Estimated time per 50k shard: ~4.25 hours on A100 40GB (based on ~8.5h for 100k). Four notebooks in parallel: ~4.5–5.5h wall‑clock.

```python
%env HB_WARMUP=1
!python housebrain_colab_trainer.py \
  --dataset /content/HouseBrainLLM/hb_v2_shard_0X \
  --max-samples 50000 \
  --output /content/hb_v2_s0X \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --batch-size 2 --grad-accum-steps 8 --epochs 1 --max-length 768 \
  --eval-steps 0 --save-steps 2000 --save-total-limit 1
```

Replace `0X` with `01..04` and run each in its own notebook/session.

When a shard finishes, save adapter to Drive:

```bash
mkdir -p /content/drive/MyDrive/HouseBrainLLM/checkpoints
rsync -ah --progress /content/hb_v2_s0X/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s0X/
```

---

## 3) Merge the 4 v2 Adapters → `housebrain_v2_200k_merged`

Do this in a fresh Colab cell after all four shards finish:

```python
from merge_models import ModelMerger

base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
v2_models = [
    "/content/hb_v2_s01",
    "/content/hb_v2_s02",
    "/content/hb_v2_s03",
    "/content/hb_v2_s04",
]

merger = ModelMerger(base_model_name=base)
merger.load_base_model()
merger.load_trained_models(v2_models)
merger.merge_models("/content/housebrain_v2_200k_merged", "average")
```

Save to Drive:

```bash
rsync -ah --progress /content/housebrain_v2_200k_merged/ /content/drive/MyDrive/HouseBrainLLM/merged/housebrain_v2_200k_merged/
```

---

## 4) Combine Existing 600k Model with v2_200k

We weight v2 slightly higher to bias toward v2 features.

```python
from merge_models import ModelMerger

base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
inputs = [
    "/content/housebrain_600k_merged",       # your existing 600k adapter
    "/content/housebrain_v2_200k_merged",    # new v2 200k adapter
]

merger = ModelMerger(base_model_name=base)
merger.load_base_model()
merger.load_trained_models(inputs)
final_adapter = merger.merge_models("/content/housebrain_600k_v2_200k_final", "weighted")
```

Optional: bake LoRA into full weights (if you need a single folder without PEFT):

```python
try:
    full = final_adapter.merge_and_unload()
    full.save_pretrained("/content/housebrain_600k_v2_200k_full")
    merger.tokenizer.save_pretrained("/content/housebrain_600k_v2_200k_full")
except Exception as e:
    print("Skip bake (adapter-only is fine):", e)
```

Save outputs to Drive:

```bash
mkdir -p /content/drive/MyDrive/HouseBrainLLM/merged
rsync -ah --progress /content/housebrain_600k_v2_200k_final/ /content/drive/MyDrive/HouseBrainLLM/merged/housebrain_600k_v2_200k_final/
[ -d /content/housebrain_600k_v2_200k_full ] && rsync -ah --progress /content/housebrain_600k_v2_200k_full/ /content/drive/MyDrive/HouseBrainLLM/merged/housebrain_600k_v2_200k_full/ || true
```

---

## 5) Smoke‑Test Inference + v2 Pipeline

If you keep the merged adapter:

```python
from src.housebrain.llm import HouseBrainLLM

hb = HouseBrainLLM(demo_mode=False, finetuned_model_path="/content/housebrain_600k_v2_200k_final")
sample = {
  "rooms": ["living", "kitchen", "bedroom", "bathroom"],
  "style": "modern",
  "area_sqft": 1800,
  "stories": 1,
  "location": "suburban"
}
plan_json = hb.generate_house_design(sample)
```

Render with v2 pipeline:

```python
import json, pathlib
from src.housebrain.pipeline_v2 import run_pipeline

out_dir = pathlib.Path("/content/out_v2_test"); out_dir.mkdir(parents=True, exist_ok=True)
tmp = out_dir / "plan.json"
with tmp.open("w") as f: json.dump(plan_json, f, indent=2)

run_pipeline(str(tmp), str(out_dir), sheet_modes=["floor","rcp","power","plumbing"])
```

This writes SVG, DXF, glTF, and a BOQ JSON along with an index.html.

---

## Training Time and Resource Estimates

- 100k shard (baseline): ~8.5 hours on A100 40GB
- 50k v2 shard (this guide): ~4.25 hours each
- 4 shards in parallel → ~4.5–5.5 hours wall‑clock, depending on Colab load

Tips:

- If step time > 8s after 200 steps, reduce `--max-length 640` or try `--batch-size 3 --grad-accum-steps 6` if VRAM allows.
- Keep `WANDB_MODE=offline`; sync later with `wandb sync` if needed.

---

## Recommended Layout and Naming

- Datasets
  - `/content/HouseBrainLLM/hb_v2_shard_01..04`
- Checkpoints
  - `/content/hb_v2_s01..04`, `/content/housebrain_v2_200k_merged`
  - `/content/housebrain_600k_merged` (existing)
  - `/content/housebrain_600k_v2_200k_final` (new)
- Drive backups under `MyDrive/HouseBrainLLM/{v2_shards,checkpoints,merged}`

---

## Troubleshooting

- Out‑of‑memory: lower `--max-length` or `--batch-size` and increase `--grad-accum-steps` to keep effective batch similar.
- Tokenizer/adapter mismatch: ensure all adapters and merges use the same base `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.
- JSON validity: use `src/housebrain/validate_v2.py` or the pipeline’s validation step.

---

## Appendix: Saving/Restoring From Drive

```python
from google.colab import drive
drive.mount('/content/drive')

!rsync -ah --progress /content/housebrain_600k_v2_200k_final/ \
  /content/drive/MyDrive/HouseBrainLLM/merged/housebrain_600k_v2_200k_final/
```


