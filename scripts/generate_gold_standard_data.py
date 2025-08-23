#!/usr/bin/env python3
"""
HouseBrain Gold Standard Data Generator
Generates and validates architect-grade training examples for the HouseBrain LLM.
"""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess
from typing import Optional

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.housebrain.llm import HouseBrainLLM
from src.housebrain.schema import validate_house_design, HouseOutput

def load_few_shot_examples(directory: str) -> str:
    """Loads all finished gold standard examples from a directory into a single string."""
    examples = []
    path = Path(directory)
    print(f"Loading few-shot examples from: {path.resolve()}")
    
    # Sort files to ensure consistent ordering
    files = sorted(path.glob("*.json"))
    
    if not files:
        print("Warning: No few-shot examples found. Using a basic one-shot prompt.")
        return "" # Fallback or a default single example could be returned here

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            prompt = data["prompt"]
            scratchpad = data["scratchpad"]
            # The output JSON needs to be dumped back into a string for the prompt
            output_json = json.dumps(data["output"], indent=2)
            
            example = (
                f"### Instruction:\n{prompt}\n\n"
                f"### Reasoning:\n<scratchpad>\n{scratchpad}\n</scratchpad>\n\n"
                f"### Response:\n```json\n{output_json}\n```"
            )
            examples.append(example)
    
    print(f"Loaded {len(examples)} few-shot examples.")
    return "\n\n---\n\n".join(examples)


# A diverse list of architectural scenarios to ensure broad training data coverage
ARCHITECTURAL_SCENARIOS = [
    # Compact & Vastu
    "A Vastu-compliant 2BHK for a small 25x40 feet east-facing plot in Pune, focusing on maximum carpet area.",
    "A G+1 3BHK design for a narrow 20x50 feet north-facing plot in Chennai, designed to mitigate heat.",
    # Mid-Size Family Homes
    "A 3BHK duplex for a 30x50 feet west-facing corner plot in Bangalore, with a home office and excellent natural light.",
    "A modern, single-floor 3BHK for a 40x60 square plot in Chandigarh, with an open-plan living and dining area.",
    "A G+1 4BHK for a south-facing 35x55 plot in Hyderabad, with a large kitchen and utility area.",
    # Luxury & Villa
    "A sprawling 5BHK luxury villa for a 60x80 feet plot in Delhi, featuring a double-height living room and an internal courtyard.",
    "A modern farmhouse design with 4 bedrooms on a 1-acre plot near Mumbai, with large verandahs and indoor-outdoor flow.",
    "A G+2 6-bedroom joint family home for a 50x70 feet plot, with separate kitchenettes on each floor.",
    # Unique Constraints
    "A 3BHK design for an irregularly shaped (trapezoidal) plot in Kolkata, maximizing usable space.",
    "A 'linked home' design for two brothers on a 60x60 plot, with two distinct 3BHK units connected by a common area.",
    "A G+1 home on a hill slope with a split-level design to adapt to the terrain.",
    "A home for a professional artist, with a 2BHK living space and a large, north-lit art studio with high ceilings.",
    # Apartment & Urban
    "A luxury 4BHK apartment layout for a 2500 sq ft floor plate in a high-rise, focusing on views and privacy.",
    "A compact but highly functional 2BHK apartment design for a 900 sq ft carpet area in Mumbai.",
    "A penthouse design with a private terrace garden and splash pool for a 3000 sq ft space."
]


def call_ollama(llm: HouseBrainLLM, prompt: str, few_shot_examples: str) -> str:
    """Sends a prompt to the Ollama API and returns the response."""
    import requests
    
    # Construct the final prompt with the few-shot examples
    final_prompt = f"{few_shot_examples}\n\n---\n\n### Instruction:\n{prompt}\n\n### Reasoning:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-coder:6.7b-instruct", # Use the confirmed available model
                "system": llm.system_prompt,
                "prompt": final_prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Ollama API request failed: {e}")
        print("     Ensure the Ollama application is running.")
        return None


def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parses the full <scratchpad> and ```json response from the LLM."""
    try:
        scratchpad_start = response_text.find("<scratchpad>")
        scratchpad_end = response_text.find("</scratchpad>")
        if scratchpad_start == -1 or scratchpad_end == -1:
            # If the model didn't use the scratchpad tags, we'll have to infer it
            json_start_for_scratchpad = response_text.find("```json")
            if json_start_for_scratchpad != -1:
                scratchpad = response_text[:json_start_for_scratchpad].strip()
            else:
                raise ValueError("Could not find ```json block to infer scratchpad.")
        else:
             scratchpad = response_text[scratchpad_start + len("<scratchpad>"):scratchpad_end].strip()

        json_start = response_text.find("```json")
        json_end = response_text.rfind("```")
        if json_start == -1 or json_end == -1:
            raise ValueError("Could not find JSON code block.")
            
        json_str = response_text[json_start + len("```json"):json_end].strip()

        return {
            "prompt": None, # Will be filled in by the calling function
            "scratchpad": scratchpad,
            "output": json_str
        }

    except Exception as e:
        print(f"  ❌ Failed to parse LLM response: {e}")
        print(f"     Full response: {response_text[:500]}...") # Log snippet
        return None


def generate_raw_example(llm: HouseBrainLLM, scenario: str, few_shot_examples: str) -> Optional[dict]:
    """
    Generates a single raw, unvalidated training example for a given scenario.

    Args:
        llm: The HouseBrainLLM instance.
        scenario: The high-level architectural prompt.

    Returns:
        A dictionary containing the prompt and the raw LLM response.
    """
    print(f"-> Generating raw example for: '{scenario}'")
    full_response_text = call_ollama(llm, scenario, few_shot_examples)
    if not full_response_text:
        return None

    # For this raw generation, we just save the prompt and the full response
    return {
        "prompt": scenario,
        "raw_response": full_response_text
    }


def main():
    """Main function to generate the dataset."""
    parser = argparse.ArgumentParser(description="Generate RAW, UNVALIDATED drafts for the HouseBrain Gold Standard dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training/gold_standard_raw",
        help="Directory to save the generated raw draft files."
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=15,
        help="Number of examples to generate from the scenarios list."
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="data/training/gold_standard",
        help="Directory containing existing gold standard examples for few-shot prompting."
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated files to: {output_path.resolve()}")

    llm = HouseBrainLLM(model_name="deepseek-coder:6.7b-instruct")
    
    # Load all existing gold standard examples for the few-shot prompt
    few_shot_prompt_string = load_few_shot_examples(args.examples_dir)

    # Find the starting index for file naming to avoid overwriting and to know which scenarios to run
    existing_files = list(output_path.glob("*.json"))
    start_index = len(existing_files)
    print(f"Found {start_index} existing raw files. New files will start from index {start_index + 1}.")
    
    # We only want to generate scenarios that we don't already have examples for.
    scenarios_to_run = ARCHITECTURAL_SCENARIOS[start_index:start_index + args.num_examples]
    num_to_generate = len(scenarios_to_run)

    generated_count = 0
    with tqdm(total=num_to_generate, desc="Generating Raw Drafts") as pbar:
        for i, scenario in enumerate(scenarios_to_run):
            example = generate_raw_example(llm, scenario, few_shot_prompt_string)
            if example:
                file_index = start_index + generated_count + 1
                # Create a filename-friendly version of the scenario
                scenario_slug = scenario.lower().replace(" ", "_").replace(",", "")[:50]
                file_path = output_path / f"{file_index:04d}_{scenario_slug}.json"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(example, f, indent=2)
                
                generated_count += 1
            
            pbar.update(1)
            pbar.set_postfix({"drafts_saved": generated_count})

    print(f"\n🎉 Generation complete. Successfully saved {generated_count} new raw draft examples.")
    print("Next step: Manually review and correct the files in the output directory to create the final Gold Standard dataset.")


if __name__ == "__main__":
    main()
