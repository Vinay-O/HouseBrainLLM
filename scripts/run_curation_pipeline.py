import os
import sys
import shutil
import subprocess
import argparse
from tqdm import tqdm

# Add parent and curation_scripts directories to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../curation_scripts')))

from curation_scripts.normalize_schema import normalize_plan
# from curation_scripts.validate_and_filter import validate_plan # Temporarily disabling for this test run
from curation_scripts.enrich_plans import enrich_plan
from curation_scripts.semantic_validator import get_original_prompt, analyze_plan, validate_with_llm

def run_pipeline(input_file_path: str, dirs: dict, all_prompts: list):
    """
    Runs the full 3-stage curation pipeline on a single raw data file.
    """
    if not os.path.exists(input_file_path):
        tqdm.write(f"❌ SKIPPING: Input file not found at '{input_file_path}'")
        return

    base_filename = os.path.basename(input_file_path) # Use the actual filename
    
    # --- Setup Directories ---
    RAW_DIR = dirs["raw"]
    STAGE0_NORMALIZED_DIR = dirs["normalized"]
    STAGE2_ENRICHED_DIR = dirs["enriched"]
    STAGE3_PROD_DIR = dirs["production"]
    QUARANTINE_DIR = dirs["quarantine"]
    
    # Copy the raw file to the pipeline's starting directory
    current_file_path = os.path.join(RAW_DIR, base_filename)
    shutil.copy(input_file_path, current_file_path)
    
    # --- Stage 0: Schema Normalization ---
    normalized_file_path = normalize_plan(current_file_path, STAGE0_NORMALIZED_DIR, QUARANTINE_DIR)

    if not normalized_file_path:
        return
    current_file_path = normalized_file_path

    # --- Stage 2: Auto-Architect ---
    enriched_file_path = enrich_plan(current_file_path, STAGE2_ENRICHED_DIR)
    
    if not enriched_file_path:
        return
    current_file_path = enriched_file_path

    # --- Stage 3: AI Sanity Check ---
    try:
        # NOTE: We need to import json here as it's used in this block
        import json
        with open(current_file_path, 'r') as f:
            plan_data = json.load(f)

        # NOTE: This is a simplification. The real script matches prompt to file.
        # Here we just grab the first prompt as a stand-in for the demo.
        try:
            # Logic for 'prompt_XXXX.json'
            prompt_index_str = base_filename.split('.')[0].split('_')[1]
            prompt_index = int(prompt_index_str) - 1 # Adjust for 0-based index
            if 0 <= prompt_index < len(all_prompts):
                original_prompt = all_prompts[prompt_index]
            else:
                raise IndexError("Prompt index out of range.")
        except (IndexError, ValueError):
            original_prompt = "Default prompt: Design a cozy 2BHK apartment."
            tqdm.write(f"⚠️ Could not parse prompt index from '{base_filename}'. Using default.")

        plan_analysis = analyze_plan(plan_data)
        
        JUDGE_MODEL = "llama3:instruct"
        is_valid = validate_with_llm(original_prompt, plan_analysis, JUDGE_MODEL)

        if is_valid:
            # Inject the original prompt into the final JSON
            if 'input' not in plan_data: plan_data['input'] = {}
            if 'basicDetails' not in plan_data['input']: plan_data['input']['basicDetails'] = {}
            plan_data['input']['basicDetails']['prompt'] = original_prompt
            
            with open(current_file_path, 'w') as f:
                json.dump(plan_data, f, indent=4)

            final_path = os.path.join(STAGE3_PROD_DIR, base_filename)
            shutil.move(current_file_path, final_path)
        else:
            quarantine_path = os.path.join(QUARANTINE_DIR, "semantic_errors")
            shutil.move(current_file_path, os.path.join(quarantine_path, base_filename))
            
    except Exception as e:
        tqdm.write(f"❌ ERROR: An error occurred during semantic validation for {base_filename}: {e}")
        quarantine_path = os.path.join(QUARANTINE_DIR, "semantic_errors")
        shutil.move(current_file_path, os.path.join(quarantine_path, base_filename))


def print_tree(startpath):
    print(f"\n📂 Final Directory Structure:")
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full curation pipeline on a directory of raw JSON files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the raw JSON files.")
    parser.add_argument("--output-dir", type=str, default="curation_pipeline_run", help="Directory to store the pipeline output.")
    args = parser.parse_args()

    # The semantic validator needs the ollama library, which isn't a direct dependency.
    # We will import it here to avoid making it a hard requirement for the other scripts.
    try:
        import json
        import ollama
    except ImportError:
        print("❌ FATAL: 'ollama' and 'json' libraries are required for Stage 3.")
        print("Please install it: pip install ollama")
        sys.exit(1)

    # --- Load all prompts ---
    try:
        with open("platinum_prompts.txt", 'r') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("❌ FATAL: 'platinum_prompts.txt' not found. This file is required to associate plans with their original prompts.")
        sys.exit(1)

    # --- Setup Directories ---
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir) # Clean up previous runs
    
    PIPELINE_DIR = args.output_dir
    pipeline_dirs = {
        "raw": os.path.join(PIPELINE_DIR, "0_raw"),
        "normalized": os.path.join(PIPELINE_DIR, "1_normalized"),
        "enriched": os.path.join(PIPELINE_DIR, "2_enriched"),
        "production": os.path.join(PIPELINE_DIR, "3_production"),
        "quarantine": os.path.join(PIPELINE_DIR, "quarantine")
    }

    for d in pipeline_dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(pipeline_dirs["quarantine"], "semantic_errors"), exist_ok=True)


    # --- Find all .json files ---
    raw_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".json"):
                raw_files.append(os.path.join(root, file))

    if not raw_files:
        print(f"❌ No '.json' files found in '{args.input_dir}'")
        sys.exit(1)

    print("="*80)
    print(f"🚀 Starting Curation Pipeline for {len(raw_files)} files...")
    print(f"   Input Dir: {args.input_dir}")
    print(f"   Output Dir: {args.output_dir}")
    print("="*80)
    
    for file_path in tqdm(raw_files, desc="Curating Files"):
        run_pipeline(file_path, pipeline_dirs, all_prompts)

    print("\n" + "="*80)
    print("✅ Pipeline Finished.")
    print("="*80)
    print_tree(PIPELINE_DIR)
