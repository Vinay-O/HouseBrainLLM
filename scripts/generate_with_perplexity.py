import os
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# This is our most advanced prompt template, designed to get the best possible JSON output.
V3_MASTER_PROMPT_TEMPLATE = """
You are a world-class AI architect with a deep understanding of spatial design, residential building codes, and architectural aesthetics. Your task is to generate a single, complete, and meticulously detailed JSON object representing a house plan based on the user's request. You must adhere strictly to the provided schema and instructions.

### USER REQUEST:
**{user_prompt}**

### CRITICAL INSTRUCTIONS:
1.  **Output ONLY JSON:** Your entire response must be a single JSON object. Do not include any introductory text, conversation, or markdown (```json) formatting.
2.  **Strict Schema Adherence:** The generated JSON MUST perfectly conform to the structure shown in the `GOLDEN_EXAMPLE` below. Use the exact key names and data types as defined in the `HouseOutput` Pydantic schema.
3.  **Geometric Consistency & Realism:** Rooms must have realistic dimensions and be placed logically. The `bounds` (`x`, `y`, `width`, `height`) for each room must NOT overlap with other rooms on the same level.
4.  **Populate All Required Fields:** Ensure top-level fields like `input`, `total_area`, and `construction_cost` are present and filled with reasonable, calculated values based on the generated plan. Use `RoomType` enums for room types.

### GOLDEN_EXAMPLE (Schema Compliant):
{{
  "input": {{ "basicDetails": {{"bedrooms": 2, "floors": 1, "totalArea": 840, "style": "Modern"}}, "plot": {{}}, "roomBreakdown": [] }},
  "total_area": 840.0,
  "construction_cost": 150000.0,
  "levels": [
    {{
      "level_number": 0,
      "rooms": [
        {{
          "id": "living_room",
          "type": "living_room",
          "bounds": {{"x": 0, "y": 0, "width": 15, "height": 20}},
          "doors": [],
          "windows": []
        }},
        {{
          "id": "kitchen",
          "type": "kitchen",
          "bounds": {{"x": 15, "y": 0, "width": 10, "height": 12}},
          "doors": [],
          "windows": []
        }}
      ]
    }}
  ]
}}

Now, generate the complete, valid, and architecturally sound JSON for the user request.
"""

def get_perplexity_client(api_key):
    """Initializes the OpenAI client to connect to Perplexity's API."""
    return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

def call_perplexity_api(client, user_prompt, model_name):
    """Makes a single API call to Perplexity to generate a house plan."""
    full_prompt = V3_MASTER_PROMPT_TEMPLATE.format(user_prompt=user_prompt)
    messages = [{"role": "user", "content": full_prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7, # A bit of creativity
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "API call failed", "details": "{str(e)}"}}'


def main(args):
    # --- 1. Setup ---
    client = get_perplexity_client(args.api_key)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 2. Load Prompts ---
    try:
        with open(args.prompts_file, 'r') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ FATAL: Prompt file not found at '{args.prompts_file}'")
        return
        
    print(f"✅ Found {len(all_prompts)} total prompts.")
    
    # --- 3. Prepare Batch ---
    prompts_to_process = all_prompts[:args.num_plans]
    
    tasks = []
    for i, prompt_text in enumerate(prompts_to_process):
        output_filename = f"plan_{i:04d}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        if not os.path.exists(output_path) or args.overwrite:
            tasks.append((prompt_text, output_path))
            
    if not tasks:
        print("✅ All requested plans already exist. Nothing to do.")
        print(f"   (Use --overwrite to regenerate them)")
        return
        
    print(f"🚀 Starting generation for {len(tasks)} new plans...")

    # --- 4. Run Generation in Parallel ---
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a dictionary to map futures to their output paths
        future_to_path = {executor.submit(call_perplexity_api, client, task[0], args.model): task[1] for task in tasks}

        # Use tqdm to create a progress bar
        pbar = tqdm(as_completed(future_to_path), total=len(tasks), desc="Generating Plans")
        for future in pbar:
            output_path = future_to_path[future]
            try:
                result = future.result()
                with open(output_path, 'w') as f:
                    f.write(result)
            except Exception as exc:
                tqdm.write(f"⚠️ An error occurred for {os.path.basename(output_path)}: {exc}")


    print("\n" + "="*50)
    print("✅ Data generation complete!")
    print(f"   Generated {len(tasks)} plans in '{args.output_dir}'")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate house plan data using the Perplexity API.")
    parser.add_argument(
        "--api-key", 
        type=str, 
        required=True, 
        help="Your Perplexity Pro API key."
    )
    parser.add_argument(
        "--num-plans", 
        type=int, 
        default=1000, 
        help="The total number of plans to generate."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="perplexity_raw_data", 
        help="Directory to save the generated JSON files."
    )
    parser.add_argument(
        "--prompts-file", 
        type=str, 
        default="platinum_prompts.txt", 
        help="Path to the file containing line-separated prompts."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama-3-sonar-large-32k-online", 
        help="The Perplexity model to use for generation."
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=10, 
        help="Number of parallel API requests to make."
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing files instead of skipping them."
    )

    args = parser.parse_args()
    main(args)
