import os
import json
import argparse
import re
from tqdm import tqdm

def find_and_parse_json(raw_content: str) -> dict:
    """
    Aggressively finds a JSON object within a string that might have leading/trailing text.
    """
    start_index = raw_content.find('{')
    end_index = raw_content.rfind('}')
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise ValueError("Could not find a valid JSON structure in the content.")
    json_str = raw_content[start_index : end_index + 1]
    return json.loads(json_str)

def main():
    parser = argparse.ArgumentParser(description="Combine and prepare a JSONL dataset for Junior Architect fine-tuning.")
    parser.add_argument("--input-dirs", nargs='+', required=True, help="A list of directories containing the raw JSON plan files.")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to the master file containing all prompts, one per line.")
    parser.add_argument("--output-file", type=str, default="finetune_dataset_junior_architect_v1.jsonl", help="Name of the output JSONL file.")
    args = parser.parse_args()

    # Step 1: Load all prompts into memory
    try:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
        tqdm.write(f"✅ Loaded {len(all_prompts)} prompts from {args.prompts_file}")
    except FileNotFoundError:
        tqdm.write(f"🚨 FATAL: Prompts file not found at {args.prompts_file}. Aborting.")
        return

    # Step 2: Gather all file paths from input directories
    files_to_process = []
    for directory in args.input_dirs:
        if not os.path.isdir(directory):
            tqdm.write(f"⚠️ WARN: Directory not found, skipping: {directory}")
            continue
        # We only care about files in the root of these directories for prompt association
        for file in os.listdir(directory):
            if file.endswith('.json'):
                files_to_process.append(os.path.join(directory, file))

    if not files_to_process:
        print("❌ No JSON files found in the specified directories.")
        return

    # Step 3: Process files and create the dataset
    successful_conversions = 0
    with open(args.output_file, 'w') as outfile:
        for filepath in tqdm(files_to_process, desc="Processing and Combining Datasets"):
            try:
                # Get prompt by extracting index from filename
                filename = os.path.basename(filepath)
                prompt_index_match = re.search(r'\d+', filename)
                if not prompt_index_match:
                    tqdm.write(f"⚠️ WARN: Skipping {filename} - Could not find a number in the filename.")
                    continue
                
                prompt_index = int(prompt_index_match.group(0))
                if prompt_index >= len(all_prompts):
                    tqdm.write(f"⚠️ WARN: Skipping {filename} - Index {prompt_index} is out of range for prompts list (size {len(all_prompts)}).")
                    continue
                
                user_prompt = all_prompts[prompt_index]

                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                plan_json = find_and_parse_json(raw_content)
                assistant_response = json.dumps(plan_json, indent=2)

                # Create the final training format
                output_record = {
                    "messages": [
                        {"role": "system", "content": "You are an expert architect. Generate a complete and detailed house plan in a single JSON object based on the user's request. Ensure the JSON is well-formed and complete."},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                outfile.write(json.dumps(output_record) + '\n')
                successful_conversions += 1

            except (ValueError, json.JSONDecodeError) as e:
                tqdm.write(f"⚠️ WARN: Skipping {os.path.basename(filepath)} due to parsing error: {e}")
            except Exception as e:
                tqdm.write(f"🚨 ERROR: An unexpected error occurred with file {os.path.basename(filepath)}: {e}")

    print("\n" + "="*50)
    if successful_conversions > 0:
        print(f"✅ Success! Created Junior Architect dataset with {successful_conversions} samples.")
        print(f"   Dataset saved to: '{args.output_file}'")
    else:
        print(f"❌ Error: No files were successfully converted. No dataset created.")
    print("="*50)

if __name__ == "__main__":
    main()
