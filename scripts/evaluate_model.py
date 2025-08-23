#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned HouseBrain model.")
    parser.add_argument("--base_model_id", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct", help="The base model ID.")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to the fine-tuned LoRA model weights.")
    parser.add_argument("--prompt_text", type=str, required=True, help="The design prompt to evaluate.")
    args = parser.parse_args()

    # --- 1. Load Fine-Tuned Model ---
    print("\n--- Step 1: Loading Fine-Tuned Model and Tokenizer ---")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(base_model, args.finetuned_model_path)
    model = model.merge_and_unload()
    print("Successfully loaded and merged fine-tuned model.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Generate JSON from Prompt ---
    print(f"\n--- Step 2: Generating design for prompt: '{args.prompt_text}' ---")
    
    # Format the prompt exactly as the model was trained
    formatted_prompt = f"""### Instruction:
{args.prompt_text}

### Response:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False, temperature=0.0, top_p=1.0, eos_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the JSON part of the response
    try:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        generated_plan = json.loads(json_str)
        print("Successfully extracted and parsed JSON output from model.")
    except (IndexError, json.JSONDecodeError) as e:
        print("❌ ERROR: Failed to extract or parse JSON from the model's response.")
        print(f"   Error: {e}")
        print(f"   Full Response:\n{response_text}")
        sys.exit(1)

    # --- 3. Run Pipeline and Evaluate Output ---
    print("\n--- Step 3: Running HouseBrain pipeline on generated JSON ---")
    
    # Add src to path to import pipeline
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_dir))
    from housebrain.pipeline_v2 import run_pipeline
    from housebrain.validate_v2 import validate_v2_file

    # Create temporary files for evaluation
    eval_dir = Path("evaluation_output")
    eval_dir.mkdir(exist_ok=True)
    plan_path = eval_dir / "generated_plan.json"
    
    # The training data format has an outer 'output' key which we need to unwrap for the pipeline
    if 'output' in generated_plan:
        plan_for_pipeline = generated_plan['output']
    else:
        plan_for_pipeline = generated_plan
        
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_for_pipeline, f, indent=2)

    try:
        run_pipeline(str(plan_path), str(eval_dir))
        print("Pipeline executed successfully.")
    except SystemExit as e:
        print(f"❌ ERROR: Pipeline failed with exit code {e.code}.")
        sys.exit(1)
        
    # --- 4. Report Quality Metrics ---
    print("\n--- Step 4: Reporting Quality Metrics ---")
    
    validation_errors = validate_v2_file(str(plan_path))
    gltf_path = eval_dir / "generated_plan_3d.gltf"
    total_triangles = 0

    if gltf_path.exists():
        with open(gltf_path, "r") as f:
            gltf_data = json.load(f)
        if "meshes" in gltf_data and "accessors" in gltf_data:
            accessors = gltf_data["accessors"]
            for mesh in gltf_data["meshes"]:
                for primitive in mesh.get("primitives", []):
                    if "indices" in primitive:
                        indices_accessor = accessors[primitive["indices"]]
                        total_triangles += indices_accessor.get("count", 0) // 3

    # Calculate score
    score = 100
    if validation_errors:
        score -= len(validation_errors) * 10
    if total_triangles > 150000:
        score -= (total_triangles - 150000) // 10000 # Penalize for excessive triangles

    print("\n========================================")
    print("         EVALUATION REPORT")
    print("========================================")
    print(f"  Schema & Geometric Validation: {'✅ PASS' if not validation_errors else f'❌ FAIL ({len(validation_errors)} errors)'}")
    for err in validation_errors:
        print(f"    - {err}")
    print(f"  3D Model Triangle Count: {total_triangles}")
    print(f"  Production Readiness Score: {max(0, score)}/100")
    print("========================================")
    print(f"\n✅ Evaluation complete. All outputs are in the '{eval_dir.resolve()}' directory.")

if __name__ == "__main__":
    main()
