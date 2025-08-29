import ollama
import os
import hashlib
import time
import argparse
import json
import random

def load_prompts_from_file(filepath):
    """Loads prompts from a text file, one prompt per line."""
    print(f"Loading prompts from {filepath}...")
    with open(filepath, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts.")
    return prompts

def generate_plan_mockable(prompt, model_name, tier_dir, is_mock=True):
    """
    Generates a house plan. If is_mock is True, it only simulates the generation.
    """
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:10]
    timestamp = int(time.time())
    unique_id = f"prompt_{prompt_hash}_{timestamp}"
    
    run_output_dir = os.path.join(tier_dir, model_name, unique_id)
    os.makedirs(run_output_dir, exist_ok=True)
    
    output_filename = os.path.join(run_output_dir, "raw_output.json")

    if is_mock:
        # MOCK BEHAVIOR: Just create a placeholder file and return
        mock_content = {"status": "mock_run", "prompt": prompt}
        with open(output_filename, 'w') as f:
            json.dump(mock_content, f, indent=4)
        return True, output_filename

    # --- REAL GENERATION LOGIC ---
    try:
        structured_prompt = f"""
        Please act as an expert architect specializing in Indian residential and commercial design. Your task is to generate a detailed JSON representation of a floor plan based on the following request, keeping local building norms and Vastu principles in mind where appropriate.

        **Architectural Request:** "{prompt}"

        **Instructions:**
        1.  **Think Step-by-Step:** Reason about the layout, room adjacencies, and flow.
        2.  **Generate JSON:** Provide ONLY the JSON output in the specified schema.

        **JSON Schema:**
        {{
          "levels": [
            {{
              "level_id": "ground_floor",
              "rooms": [
                {{"id": "living_room", "bounds": [x1, y1, x2, y2], "label": "Living Room"}}
              ],
              "openings": [
                {{"id": "front_door", "bounds": [x1, y1, x2, y2], "type": "door"}}
              ]
            }}
          ]
        }}
        
        Now, begin.
        """
        
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': structured_prompt}],
            format='json'
        )
        raw_output = response['message']['content']
        
        with open(output_filename, 'w') as f:
            f.write(raw_output)
            
        return True, output_filename

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        with open(os.path.join(run_output_dir, "error.log"), 'w') as f:
            f.write(error_message)
        return False, None

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_TO_TEST = "phi4-reasoning:latest"
    DATASET_TIER = "gold_tier"
    BASE_OUTPUT_DIR = "raw_generated_data"
    NUM_PLANS_TO_GENERATE = 15000
    PROMPT_FILE = "platinum_prompts.txt"
    
    TIER_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, DATASET_TIER)
    os.makedirs(TIER_OUTPUT_DIR, exist_ok=True)

    all_prompts = load_prompts_from_file(PROMPT_FILE)

    if not all_prompts:
        print("No prompts found. Exiting.")
    else:
        print("\n--- Starting Mock Generation Test ---")
        print(f"Simulating {NUM_PLANS_TO_GENERATE} generations...")
        
        # --- Perform ONE REAL generation first ---
        print("Performing REAL generation for the first prompt...")
        real_prompt = all_prompts[0]
        success, real_filepath = generate_plan_mockable(
            prompt=real_prompt,
            model_name=MODEL_TO_TEST,
            tier_dir=TIER_OUTPUT_DIR,
            is_mock=False # Force a real run
        )

        if success:
            print(f"✅ Real generation successful. File saved.")
            successful_generations = 1
        else:
            print(f"❌ Real generation failed. Check logs.")
            successful_generations = 0

        # --- Now, simulate the rest ---
        # We simulate one less than the total, as we've already done one real run.
        num_to_simulate = NUM_PLANS_TO_GENERATE - 1
        for i in range(num_to_simulate):
            # Use random prompts for the mock runs
            mock_prompt = random.choice(all_prompts[1:]) # Exclude the first prompt
            
            # This part remains the same: create placeholder files
            mock_success, _ = generate_plan_mockable(
                prompt=mock_prompt,
                model_name=MODEL_TO_TEST,
                tier_dir=TIER_OUTPUT_DIR,
                is_mock=True
            )
            if mock_success:
                successful_generations += 1
            print(f"\rMocking generation {i+1}/{num_to_simulate}...", end="")


        print("\n" + "="*50)
        print("Mock Test Complete!")
        print(f"Successfully simulated {successful_generations} generations.")
        print(f"A single REAL output was saved to '{TIER_OUTPUT_DIR}/{MODEL_TO_TEST}'")
        print("="*50)
