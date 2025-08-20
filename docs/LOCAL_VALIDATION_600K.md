# Local Validation with the 600k Merged Model (v2 JSON + SVG/DXF/glTF/BOQ)

This guide shows how to use your merged 600k LoRA adapter locally to:
- Generate a HouseBrain v2 plan JSON from requirements
- Validate it against the v2 schema
- Render professional outputs (SVG, DXF, glTF) and a BOQ

Works on Linux/macOS with Python 3.10+ and a recent GPU. CPU also works but will be slower.

---

## 1) Setup environment

```bash
cd /Users/vinay/Desktop/housebrain_v1_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Place your merged adapter and set path

Your merged adapter directory must contain `adapter_model.safetensors`, `adapter_config.json`, and tokenizer files.

```bash
export MODEL_DIR="/absolute/path/to/housebrain_600k_merged"
```

If you only have it on Drive, copy it locally first for speed.

---

## 3) Generate a v2 plan JSON using the adapter (PEFT)

```python
python - <<'PY'
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

MODEL_DIR = os.environ.get("MODEL_DIR")
BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("Loading base + adapter…")
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, MODEL_DIR)
model.eval()

requirements = {
  "rooms": ["living","kitchen","bedroom","bathroom"],
  "style": "modern",
  "area_sqft": 1800,
  "stories": 1,
  "location": "suburban"
}

sys_prompt = "You are HouseBrain. Output strictly valid HouseBrain Plan v2 JSON only, no prose."
user_prompt = f"Generate a HouseBrain Plan v2 that satisfies:\n{json.dumps(requirements, indent=2)}"
prompt = f"<|im_start|>system\n{sys_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"

inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=1600, temperature=0.7, do_sample=True,
        pad_token_id=tok.eos_token_id
    )
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Extract JSON block
start = text.find("{"); end = text.rfind("}")
if start == -1 or end == -1:
    raise SystemExit("No JSON block found in model output")
plan_json = json.loads(text[start:end+1])

with open("tmp_plan_v2.json","w") as f: json.dump(plan_json, f, indent=2)
print("✅ Wrote tmp_plan_v2.json")
PY
```

---

## 4) Validate against the v2 schema

```python
python - <<'PY'
from src.housebrain.validate_v2 import validate_v2_file
errs = validate_v2_file("tmp_plan_v2.json")
if errs:
    print("Validation issues:")
    for e in errs: print("-", e)
    raise SystemExit(1)
print("✅ OK: v2 JSON is valid")
PY
```

---

## 5) Full check with v2 pipeline (SVG/DXF/glTF/BOQ)

This validates again, renders sheets, writes DXF/glTF, a BOQ JSON, and an `index.html` viewer.

```bash
python src/housebrain/pipeline_v2.py \
  --input tmp_plan_v2.json \
  --out_dir out_v2/check_600k \
  --modes floor rcp power plumbing
```

Outputs are written under `out_v2/check_600k/`.

---

## 6) Optional: using a baked full model instead of an adapter

If you merged and baked the adapter into full weights:

```python
python - <<'PY'
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/absolute/path/to/housebrain_600k_full"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto").eval()

requirements = {"rooms":["living","kitchen","bedroom","bathroom"],"style":"modern","area_sqft":1800,"stories":1}
sys_prompt = "You are HouseBrain. Output strictly valid HouseBrain Plan v2 JSON only, no prose."
user_prompt = f"Generate a HouseBrain Plan v2 that satisfies:\n{json.dumps(requirements, indent=2)}"
prompt = f"<|im_start|>system\n{sys_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"

inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1600, temperature=0.7, do_sample=True, pad_token_id=tok.eos_token_id)
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
start = text.find("{"); end = text.rfind("}")
plan_json = json.loads(text[start:end+1])
open("tmp_plan_v2.json","w").write(json.dumps(plan_json, indent=2))
print("✅ Wrote tmp_plan_v2.json (full model)")
PY
```

Then repeat steps 4–5 for validation and rendering.

---

## Troubleshooting

- Slow generation on CPU: ensure CUDA is available; otherwise reduce `max_new_tokens` to ~800–1000 for faster draft checks.
- Tokenizer mismatch: if loading the adapter path fails, verify the base is `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and the adapter dir contains the tokenizer files; if not, use the base tokenizer.
- Validation failures: open `tmp_plan_v2.json` and fix fields flagged by `validate_v2_file` (levels, walls/openings required fields, etc.).

---

## Where things live

- Adapter path: `MODEL_DIR` (e.g., `/models/housebrain_600k_merged`)
- Generated JSON: `tmp_plan_v2.json`
- Renders/exports: `out_v2/check_600k/`
