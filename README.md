# HouseBrain

Architect-grade 2D/3D pipeline with Indian residential expertise.

## End-to-end flow
1. Frontend collects: plot (L×W), PIN code, style, plot shape, floors.
2. Backend sends prompt to DeepSeek-R1 with Indian architectural system prompt.
3. LLM returns JSON. If incomplete, `LayoutSolver` fills gaps.
4. Pipeline v2 validates JSON (schema + geometry) and renders:
   - SVG floor/RCP/power/plumbing
   - DXF export
   - glTF shell for 3D
5. Outputs are written to `professional_outputs/` or the provided folder.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip ruff pytest
pytest -q
```

Run the v2 pipeline manually:
```bash
python -m src.housebrain.pipeline_v2 --input /path/to/plan.json --out_dir outputs
```

## Key modules
- `src/housebrain/llm.py`: Indian-architecture-focused LLM interface
- `src/housebrain/layout.py`: rule-based layout solver fallback
- `src/housebrain/plan_renderer.py`: CAD-grade SVG export
- `src/housebrain/revolutionary_3d_generator.py`: 3D generation core
- `src/housebrain/validate_v2.py`: v2 JSON schema + geometric validation
- `src/housebrain/pipeline_v2.py`: end-to-end orchestrator

## Fine-Tuning the Model

The model can be fine-tuned on custom, high-quality examples to improve its architectural reasoning.

### 1. Prepare Your Data
Add your own `prompt-reasoning-output` JSON files to the `data/training/indian_residential/` directory. Follow the format of the existing examples.

### 2. Run the Fine-Tuning Script
This script uses LoRA for memory-efficient training. For a model like `deepseek-coder-6.7b`, you will need a GPU with at least 24GB of VRAM (e.g., NVIDIA RTX 3090/4090).

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Run the fine-tuning script (enable --use_4bit for memory savings)
python scripts/run_finetuning.py \
    --model_id "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --dataset_path "data/training/indian_residential" \
    --output_dir "models/housebrain-deepseek-v1-finetuned" \
    --use_4bit
```

### 3. Evaluate the Fine-Tuned Model
After training, you can objectively evaluate your model's performance on a test prompt. This script will generate a design and run it through the entire HouseBrain pipeline, reporting on validation and 3D model quality.

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the evaluation script
python scripts/evaluate_model.py \
    --base_model_id "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --finetuned_model_path "models/housebrain-deepseek-v1-finetuned/final_model" \
    --prompt_text "Design a 3BHK house for a 30x40 east-facing plot with Vastu compliance."
```