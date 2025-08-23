#!/usr/bin/env python3
"""
HouseBrain Silver Standard Data Generator
Generates a larger, high-quality dataset using an automated generate-and-refine loop.
The base LLM generates a draft, critiques it, and then creates a final, validated version.
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

# --- Prompt Templates ---
CRITIQUE_PROMPT_TEMPLATE = """
You are an expert architectural design reviewer. Your task is to analyze the provided JSON output for a house design.
The design was generated based on the following instruction:
---
INSTRUCTION:
{instruction}
---

Here is the generated JSON to review:
---
GENERATED JSON:
{generated_json}
---

Critically evaluate the design based on the following criteria:
1.  **Adherence to Instruction:** Does the design meet all explicit requirements (e.g., plot size, number of rooms, Vastu compliance, facing)?
2.  **Architectural Soundness:** Are the room placements logical? Is circulation efficient? Are wall placements realistic? Are dimensions practical for an Indian context?
3.  **Vastu Compliance:** If requested, is Vastu followed correctly? (e.g., Kitchen in SE, Master Bedroom in SW, Pooja in NE).
4.  **Technical Correctness:** Is the JSON schema likely valid? Are there any obvious errors like overlapping walls or impossible dimensions?

Provide your critique as a concise list of flaws and suggestions for improvement. Be specific. If the design is good, state that and suggest minor refinements.

CRITIQUE:
"""

REFINE_PROMPT_TEMPLATE = """
You are a master architect tasked with refining a house design based on a critique.

Original Instruction:
---
{instruction}
---

Original (flawed) JSON design:
---
{generated_json}
---

Critique and Suggestions for Improvement:
---
{critique}
---

Your task is to generate a new, corrected, and improved JSON output that addresses all points in the critique.
Do not repeat the original mistakes. Produce a complete, valid JSON object that represents the final, superior design.
The final output must be only the JSON object enclosed in ```json ... ```.

### Reasoning for Refinement:
<scratchpad>
I will now incorporate the feedback from the critique. The main points to address are: {critique_summary}. I will adjust the wall coordinates, change room boundaries, and ensure all constraints from the original instruction are met in this new version.
</scratchpad>

### Refined Response:
```json
"""

def call_ollama(llm: HouseBrainLLM, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """Sends a prompt to the Ollama API and returns the raw response string."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "system": system_prompt if system_prompt else llm.system_prompt,
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

def generate_and_refine_example(llm: HouseBrainLLM, scenario: str, few_shot_examples: str) -> Optional[Dict]:
    """Runs the full generate -> critique -> refine -> validate loop."""
    print(f"\n-> Generating initial draft for: '{scenario}'")
    initial_prompt = f"{few_shot_examples}\n\n---\n\n### Instruction:\n{scenario}\n\n### Reasoning:"
    initial_response = call_ollama(llm, initial_prompt)
    if not initial_response:
        return None

    initial_json_str = json.dumps(extract_json_from_response(initial_response))
    if not initial_json_str or initial_json_str == "null":
        print("  ❌ Could not extract initial JSON. Skipping.")
        return None

    print("  -> Critiquing initial draft...")
    critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(instruction=scenario, generated_json=initial_json_str)
    critique = call_ollama(llm, critique_prompt, system_prompt="You are a world-class architectural design reviewer.")
    if not critique:
        print("  ❌ Critique generation failed. Skipping.")
        return None
    print(f"  Critique received: {critique[:150]}...")

    print("  -> Refining design based on critique...")
    critique_summary = critique.split('\n')[0] # Use the first line of the critique for the scratchpad
    refine_prompt = REFINE_PROMPT_TEMPLATE.format(instruction=scenario, generated_json=initial_json_str, critique=critique, critique_summary=critique_summary)
    refined_response = call_ollama(llm, refine_prompt)
    if not refined_response:
        print("  ❌ Refinement generation failed. Skipping.")
        return None

    refined_json = extract_json_from_response(refined_response)
    if not refined_json:
        print("  ❌ Could not extract refined JSON. Skipping.")
        return None

    print("  -> Validating final refined design...")
    try:
        house_output = HouseOutput.model_validate(refined_json)
        validation_result = validate_house_design(house_output)
        if not validation_result.is_valid:
            print(f"  ❌ Final design failed validation: {validation_result.errors}")
            # Optionally save the failed attempt for debugging
            return None
    except Exception as e:
        print(f"  ❌ Pydantic validation failed with an exception: {e}")
        return None

    print("  ✅ Design generated, refined, and validated successfully!")
    # We need to extract the scratchpad from the refinement response
    scratchpad_match = re.search(r"<scratchpad>(.*?)</scratchpad>", refined_response, re.DOTALL)
    scratchpad = scratchpad_match.group(1).strip() if scratchpad_match else "Refinement reasoning not captured."

    return {
        "prompt": scenario,
        "scratchpad": scratchpad,
        "output": refined_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate SILVER standard data for HouseBrain using a generate-and-refine loop.")
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
    scenarios_to_run = [s for s in ARCHITECTURAL_SCENARIOS * (args.num_examples // len(ARCHITECTURAL_SCENARIOS) + 1)][start_index:args.num_examples]

    generated_count = 0
    pbar = tqdm(scenarios_to_run, desc="Generating Silver Standard Data")
    for scenario in pbar:
        example = generate_and_refine_example(llm, scenario, few_shot_prompt_string)
        if example:
            file_index = start_index + generated_count + 1
            scenario_slug = scenario.lower().replace(" ", "_").replace(",", "")[:50]
            file_path = output_path / f"{file_index:04d}_{scenario_slug}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(example, f, indent=2)

            generated_count += 1
            pbar.set_postfix({"saved": generated_count})

    print(f"\n🎉 Generation complete. Saved {generated_count} new silver-standard examples.")

if __name__ == "__main__":
    main()
