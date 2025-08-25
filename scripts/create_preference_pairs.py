# HouseBrain Iridium Tier: Preference Pair Generation (Blueprint)
# This script outlines the future workflow for creating the Iridium dataset.
# It is a blueprint and cannot be run until we have a fine-tuned "Master Architect" model.

import argparse
import json
import os
from pathlib import Path

# --- Placeholder for our Future Fine-Tuned Model ---

def call_finetuned_housebrain_model(prompt: str, temperature: float) -> dict:
    """
    This is a placeholder function.
    In the future, this function will call our fine-tuned "Master Architect" model
    to generate a house plan. The 'temperature' parameter will be crucial for
    generating different variations for the same prompt.
    """
    print(f"--- Calling Master Architect Model (temp={temperature}) ---")
    print(f"PROMPT: {prompt}")
    
    # In a real scenario, this would return a valid HouseOutput dictionary.
    # For this blueprint, we will return a mock object.
    mock_plan = {
        "input": {"basicDetails": {"prompt": prompt}},
        "levels": [{"level_number": 0, "rooms": [{"id": f"room_{temperature}", "type": "living_room", "bounds": {"x": 0, "y": 0, "width": 10, "height": 10}}]}],
        "total_area": 100.0,
        "__model_metadata__": {"temperature": temperature} # Add metadata for clarity
    }
    print("--- Model returned a plan. ---\n")
    return mock_plan

# --- The Core Workflow ---

def generate_preference_pair(prompt: str, output_dir: Path):
    """
    Generates two distinct plans for the same prompt and saves them for comparison.
    """
    
    # Generate Plan A (more conservative)
    plan_a = call_finetuned_housebrain_model(prompt, temperature=0.4)
    
    # Generate Plan B (more creative)
    plan_b = call_finetuned_housebrain_model(prompt, temperature=0.8)
    
    # Save both plans to a dedicated folder for this prompt
    prompt_hash = abs(hash(prompt)) % (10**8)
    pair_dir = output_dir / f"prompt_{prompt_hash}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    with open(pair_dir / "plan_A.json", 'w') as f:
        json.dump(plan_a, f, indent=2)
        
    with open(pair_dir / "plan_B.json", 'w') as f:
        json.dump(plan_b, f, indent=2)
        
    print(f"Successfully generated comparison pair in: {pair_dir}")

def main():
    parser = argparse.ArgumentParser(description="Blueprint for generating preference pairs for the Iridium dataset.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to a file containing complex prompts (e.g., diamond_prompts.txt)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/iridium_pairs",
        help="Directory to save the generated preference pairs."
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=500,
        help="The number of preference pairs to generate."
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # Ensure we don't ask for more pairs than we have prompts
    num_to_generate = min(args.num_pairs, len(prompts))
    
    print(f"--- Starting Iridium Preference Pair Generation ---")
    print(f"Generating {num_to_generate} pairs from '{args.prompt_file}'.\n")

    for i in range(num_to_generate):
        print(f"--- Processing Prompt {i+1}/{num_to_generate} ---")
        generate_preference_pair(prompts[i], output_path)
        print("-" * 50)
        
    print(f"✅ Blueprint execution complete. {num_to_generate} pairs are ready for human review.")
    print("Next step: A human would review each pair in '{output_path}' and create the final preference dataset for DPO/RLHF training.")

if __name__ == "__main__":
    main()
