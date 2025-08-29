import os
import argparse
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import re

# +++ PASTE YOUR API KEY HERE +++
# Replace the placeholder text with your actual DeepSeek API key.
# Example: API_KEY = "ds_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
API_KEY = "sk-a651252b530d44b396e069a8b492d166"
# +++++++++++++++++++++++++++++++


# This prompt is refined to be extra insistent on JSON-only output for models
# that don't have a dedicated JSON mode.
V4_MASTER_PROMPT_TEMPLATE = """
You are a world-class AI architect specializing in generating structured data. Your single and only task is to generate a complete JSON object representing a house plan based on the user's request. You must adhere strictly to the provided schema and instructions.

### USER REQUEST:
**{user_prompt}**

### CRITICAL INSTRUCTIONS:
1.  **JSON ONLY:** Your entire response MUST be a single, raw, valid JSON object. Do not wrap it in markdown (```json), do not add any introductory text, explanations, or any characters outside of the main JSON structure starting with `{{` and ending with `}}`.
2.  **STRICT SCHEMA:** The generated JSON MUST perfectly conform to the structure in the `GOLDEN_EXAMPLE`. Use the exact key names and data types.
3.  **GEOMETRIC CONSISTENCY:** Room `bounds` (`x`, `y`, `width`, `height`) must NOT overlap with other rooms on the same level.
4.  **POPULATE ALL FIELDS:** Ensure all required fields like `total_area` are present and calculated correctly.

### GOLDEN_EXAMPLE (Schema Compliant):
{{
  "input": {{ "basicDetails": {{"bedrooms": 2, "floors": 1, "totalArea": 840, "style": "Modern"}}, "plot": {{}}, "roomBreakdown": [] }},
  "total_area": 840.0,
  "construction_cost": 150000.0,
  "levels": [
    {{
      "level_number": 0,
      "rooms": [
        {{ "id": "living_room", "type": "living_room", "bounds": {{"x": 0, "y": 0, "width": 15, "height": 20}}, "doors": [], "windows": [] }},
        {{ "id": "kitchen", "type": "kitchen", "bounds": {{"x": 15, "y": 0, "width": 10, "height": 12}}, "doors": [], "windows": [] }}
      ]
    }}
  ]
}}

Now, generate the JSON object for the user request.
"""

def get_deepseek_client(api_key):
    """Initializes the OpenAI client to connect to DeepSeek's API."""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

def find_and_parse_json(raw_content: str) -> dict:
    """Aggressively finds and parses a JSON object within a raw string."""
    start_index = raw_content.find('{')
    end_index = raw_content.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_str = raw_content[start_index : end_index + 1]
        return json.loads(json_str)
    raise ValueError("Could not find a valid JSON object in the response.")

def call_deepseek_api(client, user_prompt, model_name):
    """Makes a single API call to DeepSeek to generate a house plan."""
    full_prompt = V4_MASTER_PROMPT_TEMPLATE.format(user_prompt=user_prompt)
    messages = [{"role": "user", "content": full_prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
        )
        # We use our robust parser here since JSON mode isn't guaranteed
        return find_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        return {"error": "API call or JSON parsing failed", "details": str(e)}

def main(args):
    # Safety check for the API key
    if not API_KEY or API_KEY == "PASTE_YOUR_DEEPSEEK_API_KEY_HERE":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: API Key Not Found                                !!!")
        print("!!!                                                         !!!")
        print("!!! Please open the 'scripts/generate_with_deepseek.py' file  !!!")
        print("!!! and paste your DeepSeek API key into the API_KEY variable !!!")
        print("!!! at the top of the script.                               !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    client = get_deepseek_client(API_KEY)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        with open(args.prompts_file, 'r') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ FATAL: Prompt file not found at '{args.prompts_file}'")
        return
        
    print(f"✅ Found {len(all_prompts)} total prompts.")
    
    tasks = []
    for i, prompt_text in enumerate(all_prompts[:args.num_plans]):
        output_path = os.path.join(args.output_dir, f"plan_{i:04d}.json")
        if not os.path.exists(output_path) or args.overwrite:
            tasks.append((prompt_text, output_path))
            
    if not tasks:
        print("✅ All requested plans already exist.")
        return
        
    print(f"🚀 Starting generation for {len(tasks)} new plans using '{args.model}'...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {executor.submit(call_deepseek_api, client, task[0], args.model): task[1] for task in tasks}

        pbar = tqdm(as_completed(future_to_path), total=len(tasks), desc="Generating Plans")
        for future in pbar:
            output_path = future_to_path[future]
            try:
                result_json = future.result()
                with open(output_path, 'w') as f:
                    json.dump(result_json, f, indent=2)
            except Exception as exc:
                tqdm.write(f"⚠️ An error occurred for {os.path.basename(output_path)}: {exc}")

    print("\n" + "="*50)
    print("✅ Data generation complete!")
    print(f"   Generated {len(tasks)} plans in '{args.output_dir}'")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate house plan data using the DeepSeek API.")
    # The --api-key argument is no longer needed from the command line
    # parser.add_argument("--api-key", type=str, required=True, help="Your DeepSeek API key.")
    parser.add_argument("--num-plans", type=int, default=1000, help="The total number of plans to generate.")
    parser.add_argument("--output-dir", type=str, default="deepseek_raw_data", help="Directory to save the generated JSON files.")
    parser.add_argument("--prompts-file", type=str, default="platinum_prompts.txt", help="Path to the file containing prompts.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="The DeepSeek model to use for generation.")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel API requests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    args = parser.parse_args()
    main(args)
