## HouseBrain Parallel Training Guide (350K Dataset)

This guide explains how to split the 350K dataset into 6–7 parts and train them in parallel using multiple free Kaggle and Colab accounts, then merge the adapters into one production model.

### Summary
- **Dataset**: `housebrain_dataset_v5_350k/` (315K train, 35K val)
- **Split Tool**: `split_dataset.py`
- **Training Script**: `colab_training_fixed.py`
- **Merger**: `merge_models.py` (LoRA adapter averaging)

---

### 1) Split the Dataset into 6–7 Parts
Run locally:
```bash
python split_dataset.py --source housebrain_dataset_v5_350k --output housebrain_splits --splits 6
```
Output structure:
- `housebrain_splits/split_01/train/*.json`
- ...
- `housebrain_splits/split_06/train/*.json`
- `housebrain_splits/split_summary.json`
- `housebrain_splits/training_configs.json`

Zip each split for upload:
```bash
cd housebrain_splits
for d in split_*; do zip -qr "$d.zip" "$d"; done
```

---

### 2) Train Each Split on Free Kaggle/Colab Accounts

Recommended mapping (6-way):
- Kaggle: `split_04`, `split_05`, `split_06`
- Colab: `split_01`, `split_02`, `split_03`

Baseline settings (per account):
- `batch_size=1`, `gradient_accumulation_steps=16`, `max_length=512`, `epochs=2`, `lora_r=8`

Kaggle steps:
```bash
!pip -q install transformers==4.43.3 peft==0.11.1 bitsandbytes==0.43.1 datasets==2.20.0 accelerate==0.33.0 safetensors==0.4.3 json-repair==0.21.0
!unzip -q split_04.zip && ls -la split_04/train | head -10
```
```python
from colab_training_fixed import FixedFineTuningConfig, FixedHouseBrainFineTuner
config = FixedFineTuningConfig(dataset_path="split_04", output_dir="models/hb-kaggle-04")
FixedHouseBrainFineTuner(config).train()
```

Colab steps:
```bash
!pip -q install transformers==4.43.3 peft==0.11.1 bitsandbytes==0.43.1 datasets==2.20.0 accelerate==0.33.0 safetensors==0.4.3 json-repair==0.21.0
!unzip -q split_01.zip -d /content
```
```python
from colab_training_fixed import FixedFineTuningConfig, FixedHouseBrainFineTuner
config = FixedFineTuningConfig(dataset_path="/content/split_01", output_dir="/content/models/hb-colab-01")
FixedHouseBrainFineTuner(config).train()
```

Tips:
- Keep each account running a single split.
- Save often (`save_steps=200`) to mitigate disconnect risk.

---

### 3) Collect Trained Adapters
From each account, download its model folder, e.g.:
- Kaggle: `models/hb-kaggle-04/`
- Colab: `/content/models/hb-colab-01/`

Place all 6–7 adapters on your local machine or a server.

---

### 4) Merge Adapters into One Production Model

Example (6 models):
```bash
python merge_models.py \
  --models \
    models/hb-colab-01 \
    models/hb-colab-02 \
    models/hb-colab-03 \
    models/hb-kaggle-04 \
    models/hb-kaggle-05 \
    models/hb-kaggle-06 \
  --output models/housebrain-merged \
  --strategy average \
  --validate
```

Notes:
- Ensure all adapters share identical LoRA config (rank, alpha, target modules). The script checks and warns.
- Strategies: `average` (default), `weighted`, or `best`.

---

### 5) What It Looks Like After Training Completes
Each trained adapter folder typically contains:
- `adapter_config.json`, `adapter_model.bin`
- `trainer_state.json`, `training_args.bin`
- tokenizer files

After merging, `models/housebrain-merged/` contains the merged adapter and tokenizer.

---

### 6) Next Steps After Training

1. Quick sanity eval:
```python
from src.housebrain.llm import HouseBrainLLM
llm = HouseBrainLLM(model_path="models/housebrain-merged")
print(llm.generate_design({"basicDetails": {"totalArea": 1200, "unit": "sqft", "bedrooms": 2, "bathrooms": 1, "floors": 1, "budget": 900000, "style": "Modern"}, "plot": {"length": 40, "width": 30, "unit": "ft", "orientation": "N", "setbacks_ft": {"front": 5, "rear": 5, "left": 3, "right": 3}}, "roomBreakdown": []}))
```

2. Evaluate metrics (invalid-JSON rate, parse rate, compliance):
- Use validation split and `strict_json_parse` in `src/housebrain/llm.py`.

3. Optional fine-tuning round:
- Train a short second round on harder examples or India-focused subset for specialization.

4. Deployment:
- Package `models/housebrain-merged/` with base model in inference stack.
- Optionally export to Hugging Face Hub or use Ollama integration if desired.

---

### 7) FAQ / Tips
- Prefer Kaggle for long free sessions; Colab for setup/quick tests.
- If you have 7 accounts, set `--splits 7` and repeat the same steps.
- If a run crashes, just restart using the latest checkpoint in the output dir.


