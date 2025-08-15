## HouseBrain India Training Guide (75K)

This guide covers training the India-focused 75K dataset on Kaggle and Google Colab. It includes a free-tier recommendation, step-by-step instructions, and next steps after training.

### Summary
- **Dataset**: `housebrain_dataset_india_75k/` (67,500 train, 7,499 validation)
- **Model**: `deepseek-ai/deepseek-coder-6.7b-base` + QLoRA
- **Script**: `colab_training_fixed.py` (masked SFT, eval, stable args)
- **Output**: LoRA adapter folder (can be merged later)

### Recommendation (Free Tier)
- **Train on Kaggle**: Longer uninterrupted GPU sessions (T4/P100), persistent outputs, reliable for multi-hour runs.
- **Use Colab for debugging/quick iterations**: Faster to iterate, but more likely to disconnect/limit.

---

### Train on Kaggle (Recommended)

1) Start a new Kaggle Notebook with GPU:
- Settings → Accelerator: GPU (T4/P100)

2) Upload or mount the dataset directory:
- Zip locally (optional):
```bash
zip -r housebrain_dataset_india_75k.zip housebrain_dataset_india_75k/
```
- In Kaggle, upload `housebrain_dataset_india_75k.zip` to the notebook and extract:
```bash
!unzip -q housebrain_dataset_india_75k.zip
!ls -la housebrain_dataset_india_75k | head -20
```

3) Install dependencies:
```bash
!pip -q install transformers==4.43.3 peft==0.11.1 bitsandbytes==0.43.1 datasets==2.20.0 accelerate==0.33.0 safetensors==0.4.3 json-repair==0.21.0
```

4) Upload project files or git clone your repo:
```bash
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM
```

5) Run training with India dataset path:
```python
from colab_training_fixed import FixedFineTuningConfig, FixedHouseBrainFineTuner

config = FixedFineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="../housebrain_dataset_india_75k",  # adjust if path differs
    output_dir="models/housebrain-india-75k",
    num_epochs=2,
    batch_size=1,
    gradient_accumulation_steps=16,
    max_length=512,
    learning_rate=2e-4,
    lora_r=8,
    lora_alpha=16,
)

trainer = FixedHouseBrainFineTuner(config)
success = trainer.train()
```

6) Download the model artifacts (if needed):
```bash
!ls -la models/housebrain-india-75k
```

---

### Train on Google Colab

1) Launch a GPU runtime (Runtime → Change runtime type → GPU).

2) Upload and extract the dataset (Drive or direct upload):
```bash
!unzip -q housebrain_dataset_india_75k.zip -d /content
```

3) Install dependencies:
```bash
!pip -q install transformers==4.43.3 peft==0.11.1 bitsandbytes==0.43.1 datasets==2.20.0 accelerate==0.33.0 safetensors==0.4.3 json-repair==0.21.0
```

4) Upload `colab_training_fixed.py` and run training:
```python
from colab_training_fixed import FixedFineTuningConfig, FixedHouseBrainFineTuner

config = FixedFineTuningConfig(
    dataset_path="/content/housebrain_dataset_india_75k",
    output_dir="/content/models/housebrain-india-75k",
)

trainer = FixedHouseBrainFineTuner(config)
trainer.train()
```

5) Save/download model artifacts from `/content/models/housebrain-india-75k`.

---

### What You Get After Training
- A folder like `models/housebrain-india-75k/` containing LoRA adapter files:
  - `adapter_config.json`, `adapter_model.bin`
  - `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`
  - `training_args.bin`, `trainer_state.json`

---

### Next Steps (Post-Training)
- **Quick eval**: Run a small validation set through `src/housebrain/llm.py` to check JSON validity and domain fit.
- **Optional more runs**: Train 2–3 variants (different seeds, slightly different LR) to ensemble/merge later.
- **Merge (if multiple runs)**: Use `merge_models.py` to average LoRA adapters (see parallel guide for details).
- **Deploy**: Use the merged adapter with base model in `src/housebrain/llm.py` for inference, or export to Hub.

---

### Tips
- Prefer Kaggle for full runs; use Colab for smoke tests.
- Keep `max_length<=512` and batch_size=1 with gradient accumulation for free GPUs.
- Save frequently (`save_steps` ≈ 200–500) to avoid losing progress on disconnects.


