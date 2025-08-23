#!/usr/bin/env python3
"""
HouseBrain Silver Standard Data Generator
Generates a larger, high-quality dataset by prompting an LLM for architecturally
sound designs and validating them against the HouseBrain schema.
"""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import requests
import re
from typing import Dict, Optional, List

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.housebrain.llm import HouseBrainLLM
from src.housebrain.schema import validate_house_design, HouseOutput
from scripts.generate_gold_standard_data import load_few_shot_examples, ARCHITECTURAL_SCENARIOS

# --- Constants ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-coder:6.7b-instruct"
GENERATION_TIMEOUT = 400

def call_ollama(llm: HouseBrainLLM, prompt: str) -> Optional[str]:
    """Sends a prompt to the Ollama API and returns the raw response string."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "system": llm.system_prompt,
                "prompt": prompt,
                "stream": False,
            },
            timeout=GENERATION_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Ollama API request failed: {e}")
        return None

def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extracts a JSON object from a markdown code block."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to decode JSON: {e}")
            return None
    print("  ❌ Could not find a JSON block in the response.")
    return None

def generate_and_validate_example(llm: HouseBrainLLM, scenario: str, few_shot_examples: str) -> Optional[Dict]:
    """
    Generates a single design and validates it. If valid, returns the example.
    This simplified approach is more robust than the critique/refine loop.
    """
    print(f"\n-> Generating design for: '{scenario}'")
    prompt = f"{few_shot_examples}\n\n---\n\n### Instruction:\n{scenario}\n\n### Reasoning:"
    response_text = call_ollama(llm, prompt)
    if not response_text:
        return None

    # --- Extract and Validate ---
    generated_json = extract_json_from_response(response_text)
    if not generated_json:
        return None

    try:
        house_output = HouseOutput.model_validate(generated_json)
        validation_result = validate_house_design(house_output)
        if validation_result.is_valid:
            print("  ✅ Design is valid! Saving.")
            scratchpad_match = re.search(r"<scratchpad>(.*?)</scratchpad>", response_text, re.DOTALL)
            scratchpad = scratchpad_match.group(1).strip() if scratchpad_match else "Reasoning not captured."
            return {"prompt": scenario, "scratchpad": scratchpad, "output": generated_json}
        else:
            print(f"  ❌ Design failed validation: {validation_result.errors}")
            return None
    except Exception as e:
        print(f"  ❌ Design raised Pydantic exception: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate SILVER standard data for HouseBrain using a direct generate-and-validate approach.")
    parser.add_argument("--output-dir", type=str, default="data/training/silver_standard", help="Directory to save the generated files.")
    parser.add_argument("--num-examples", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("--examples-dir", type=str, default="data/training/gold_standard", help="Directory of gold examples for few-shot prompting.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated files to: {output_path.resolve()}")

    llm = HouseBrainLLM(model_name=MODEL_NAME)
    few_shot_prompt_string = load_few_shot_examples(args.examples_dir)

    existing_files = list(output_path.glob("*.json"))
    start_index = len(existing_files)
    # Cycle through the scenarios to generate the desired number of examples
    scenarios_to_run = [s for s in ARCHITECTURAL_SCENARIOS * (args.num_examples // len(ARCHITECTURAL_SCENARIOS) + 1)]
    
    generated_count = 0
    with tqdm(total=args.num_examples, desc="Generating Silver Standard Data") as pbar:
        while generated_count < args.num_examples:
            # This loop ensures we keep trying scenarios until we have enough successful examples
            if not scenarios_to_run:
                 # Refill scenarios if we run out, to keep trying
                scenarios_to_run = ARCHITECTURAL_SCENARIOS * (args.num_examples // len(ARCHITECTURAL_SCENARIOS) + 1)
            
            scenario = scenarios_to_run.pop(0)
            
            example = generate_and_validate_example(llm, scenario, few_shot_prompt_string)
            if example:
                file_index = start_index + generated_count + 1
                scenario_slug = scenario.lower().replace(" ", "_").replace(",", "")[:50]
                file_path = output_path / f"{file_index:04d}_{scenario_slug}.json"

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(example, f, indent=2)

                generated_count += 1
                pbar.update(1)
                pbar.set_postfix({"saved": generated_count})

    print(f"\n🎉 Generation complete. Saved {generated_count} new silver-standard examples.")

if __name__ == "__main__":
    main()
