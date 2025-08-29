import os
import json
import argparse
import random
import sys
import tempfile
import shutil
from tqdm import tqdm
import re

# Add parent and curation_scripts directories to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curation_scripts.normalize_schema import normalize_plan

def normalize_plan(plan_data: dict) -> dict:
    """
    Normalizes a single JSON house plan to ensure it adheres to the official schema.
    This function is kept for compatibility but is not directly used in the new process_file.
    """
    # This function is now primarily for schema validation and potential future use.
    # The new process_file aggressively extracts JSON and associates prompts.
    return plan_data


def find_and_parse_json(raw_content: str) -> dict:
    """
    Aggressively finds a JSON object within a string and parses it.
    """
    # Find the first '{' and the last '}'
    start_index = raw_content.find('{')
    end_index = raw_content.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise ValueError("Could not find a valid JSON structure in the content.")

    # Extract the potential JSON string
    json_str = raw_content[start_index : end_index + 1]
    
    # Let the json library parse it, which is good at handling unicode
    return json.loads(json_str)


def process_file(filepath: str, prompts: list) -> dict:
    tqdm.write("-" * 50)
    tqdm.write(f"--- 0. Processing File: {os.path.basename(filepath)} ---")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Step 1: Aggressively find and parse the JSON content
        tqdm.write("--- 1. Extracting JSON from raw text ---")
        healed_plan = find_and_parse_json(raw_content)

        # Step 2: Associate with a prompt
        filename = os.path.basename(filepath)
        try:
            # Assumes filename is like 'plan_0001.json'
            prompt_index_str = re.sub(r'\D', '', filename)
            prompt_index = int(prompt_index_str)
            # --> Add a check to see if we have enough prompts <--
            if 0 <= prompt_index < len(prompts):
                user_prompt = prompts[prompt_index]
                # Ensure the prompt is embedded in the structure for the training format
                if 'input' not in healed_plan:
                    healed_plan['input'] = {}
                if 'basicDetails' not in healed_plan['input']:
                    healed_plan['input']['basicDetails'] = {}
                healed_plan['input']['basicDetails']['prompt'] = user_prompt
                tqdm.write(f"--- 2. Associated with prompt #{prompt_index} ---")
            else:
                # If we run out of unique prompts, gracefully reuse the last available one.
                # This is useful for large generation runs with a smaller prompt set.
                user_prompt = prompts[-1]
                tqdm.write(f"⚠️ WARN: Prompt index {prompt_index} out of range. Reusing last available prompt.")
                if 'input' not in healed_plan:
                    healed_plan['input'] = {}
                if 'basicDetails' not in healed_plan['input']:
                    healed_plan['input']['basicDetails'] = {}
                healed_plan['input']['basicDetails']['prompt'] = user_prompt


        except (ValueError, IndexError):
            tqdm.write(f"⚠️ WARN: Could not extract a valid prompt index from filename '{filename}'. Skipping.")
            return None

        # Step 3: Normalize and validate the schema (optional, but good practice)
        # healed_plan = normalize_plan(healed_plan)
        # For now, we trust the extracted JSON and focus on formatting.

        return healed_plan

    except json.JSONDecodeError as e:
        tqdm.write(f"❌ FAILED: JSON Decode Error. Details: {e}")
        return None
    except ValueError as e:
        tqdm.write(f"❌ FAILED: Value Error (likely during extraction). Details: {e}")
        return None
    except Exception as e:
        tqdm.write(f"🚨 FAILED: An unexpected error occurred. Details: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Prepare a JSONL dataset for fine-tuning from curated house plan JSONs.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the raw/curated JSON files.")
    parser.add_argument("--output-file", type=str, default="finetune_dataset.jsonl", help="Base name of the output JSONL file.")
    # Add an argument to append to the file instead of overwriting
    parser.add_argument("--append", action="store_true", help="Append to the output file instead of overwriting it.")
    parser.add_argument("--prompts-file", type=str, default="platinum_prompts.txt", help="Path to the file containing prompts.")
    args = parser.parse_args()

    all_prompts = []
    try:
        with open(args.prompts_file, 'r') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(all_prompts)} prompts from {args.prompts_file}")
    except FileNotFoundError:
        print(f"🚨 FATAL: {args.prompts_file} not found. Cannot associate prompts.")
        return

    files_to_process = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.json')]
    
    if not files_to_process:
        print(f"❌ No JSON files found in '{args.input_dir}'")
        return

    # Determine write mode based on the new --append argument
    write_mode = 'a' if args.append else 'w'
    successful_conversions = 0
    with open(args.output_file, write_mode) as outfile:
        for filepath in tqdm(files_to_process, desc="Processing and Converting Files"):
            processed_plan = process_file(filepath, all_prompts)

            if processed_plan:
                try:
                    # Final check: get the prompt for the output format
                    user_prompt = processed_plan.get("input", {}).get("basicDetails", {}).get("prompt", "")
                    if not user_prompt:
                        tqdm.write(f"⚠️ WARN: Skipping file {os.path.basename(filepath)} because final prompt is missing.")
                        continue
                    
                    # Create the final training format
                    output_record = {
                        "messages": [
                            {"role": "system", "content": "You are an expert architect. Generate a JSON house plan based on the user's request."},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": json.dumps(processed_plan, indent=2)}
                        ]
                    }
                    outfile.write(json.dumps(output_record) + '\n')
                    successful_conversions += 1
                except Exception as e:
                    tqdm.write(f"🚨 FAILED to write record for {os.path.basename(filepath)}: {e}")

    print("\n" + "="*50)
    if successful_conversions > 0:
        print(f"✅ Success! Created dataset with {successful_conversions} samples.")
        print(f"   Dataset saved to: '{args.output_file}'")
    else:
        print(f"❌ Error: No files were successfully converted. No dataset created.")
    print("="*50)

if __name__ == "__main__":
    main()
