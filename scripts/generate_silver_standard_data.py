#!/usr/bin/env python3
"""
HouseBrain Silver Standard Data Generator
Generates a larger, high-quality dataset by prompting an LLM for architecturally
sound designs and validating them against the HouseBrain schema.
"""
import argparse
import json
import logging
import os
import random
import re
import requests
import sys
from pathlib import Path
from typing import Optional

# --- Pre-computation Setup ---
# Add src directory to the Python path before other imports
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from housebrain.schema import validate_house_design, HouseOutput

# --- Logging Configuration ---
# Configure logger to output simple, clean messages for Colab monitoring
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Few-Shot Example ---
# A high-quality example to guide the LLM's output structure.
FEW_SHOT_EXAMPLE = {
    "prompt": "Design a Vastu-compliant 2BHK for a small 25x40 feet east-facing plot in Pune, focusing on maximum carpet area.",
    "output": {
        "input": {
            "plot_dimensions": {"length": 40.0, "width": 25.0},
            "facing_direction": "east",
            "number_of_stories": 1,
        },
        "levels": [
            {
                "level_id": "ground_floor",
                "level_name": "Ground Floor",
                "floor_height": 10.0,
                "spaces": [
                    {"id": "living_room_1", "name": "Living Room", "space_type": "living_room", "boundary": [[1, 23], [1, 39], [13, 39], [13, 23]]},
                    {"id": "kitchen_1", "name": "Kitchen", "space_type": "kitchen", "boundary": [[13, 31], [13, 39], [24, 39], [24, 31]]},
                    {"id": "bedroom_1", "name": "Master Bedroom", "space_type": "bedroom", "boundary": [[1, 1], [1, 12], [12, 12], [12, 1]]},
                    {"id": "bedroom_2", "name": "Second Bedroom", "space_type": "bedroom", "boundary": [[12, 1], [12, 12], [24, 12], [24, 1]]},
                    {"id": "bathroom_1", "name": "Common Bathroom", "space_type": "bathroom", "boundary": [[1, 12], [1, 17], [6, 17], [6, 12]]},
                ],
                "walls": [], "openings": [], "stairs": [],
            }
        ],
        "total_area": 1000.0,
        "construction_cost": 1800000.0,
    },
}

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a specialized AI architect for Indian residences. Your task is to generate a complete and valid HouseBrain JSON object based on the user's request.

**IMPORTANT: The output must be a single JSON object that strictly follows the structure and schema of the example provided below.**

---
**EXAMPLE**

**User Request:**
"{example_prompt}"

**Correctly Formatted JSON Output:**
```json
{example_output}
```
---

**ACTUAL TASK**

**User Request:**
"{prompt}"

**Your Output (JSON only):**
"""


# --- Dynamic Scenario Generation ---
def generate_dynamic_scenario() -> str:
    """Generates a unique, complex architectural scenario."""
    plot_size = f"{random.randint(25, 40)}x{random.randint(50, 70)} feet"
    facing = random.choice(["north", "south", "east", "west"])
    city = random.choice(["Pune (moderate)", "Hyderabad (hot, semi-arid)", "Kolkata (hot, humid)"])
    client_need = random.choice([
        "a dedicated study room for remote work.",
        "a large balcony connected to the master bedroom.",
        "a separate dining area that can seat eight people.",
        "an open-air courtyard in the center of the house.",
        "a two-car garage with internal access.",
    ])
    bhk = random.choice(["2BHK", "3BHK", "4BHK"])
    levels = random.choice(["single-story", "G+1 duplex"])
    return f"Design a Vastu-compliant {bhk} {levels} house for a {plot_size} {facing}-facing plot in {city}. The design must include {client_need}"


# --- Core LLM and Validation Functions ---
def call_ollama(prompt: str) -> Optional[str]:
    """Sends a request to the Ollama API and returns the response string."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-coder:6.7b-instruct", "stream": False, "format": "json",
        "prompt": prompt,
        "system": "You are an AI architect that only outputs valid JSON conforming to a specific schema.",
    }
    try:
        response = requests.post(url, json=payload, timeout=240) # 4-minute timeout
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.Timeout:
        logger.error("TIMEOUT: Ollama API request timed out after 4 minutes.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"REQUEST_ERROR: Ollama API request failed: {e}")
        return None

def extract_json_from_response(response: str) -> Optional[dict]:
    """Extracts a JSON object from a string, supporting markdown code blocks."""
    try:
        # First, try to load directly, as format="json" should handle it.
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback for cases where the model might still wrap in markdown
        match = re.search(r"```json\n({.*?})\n```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("DECODE_ERROR: Could not decode JSON from markdown block.")
                return None
    logger.warning("DECODE_ERROR: Response was not a valid JSON object or markdown block.")
    return None


def generate_and_validate_example(scenario: str, output_dir: Path, file_name: str) -> bool:
    """Generates a single house plan, validates it, and saves if it's perfect."""
    logger.info(f"ATTEMPTING: {scenario[:90]}...")
    
    # Create the few-shot prompt
    prompt = PROMPT_TEMPLATE.format(
        example_prompt=FEW_SHOT_EXAMPLE["prompt"],
        example_output=json.dumps(FEW_SHOT_EXAMPLE["output"], indent=4),
        prompt=scenario,
    )

    raw_response = call_ollama(prompt)
    if not raw_response:
        return False

    json_output = extract_json_from_response(raw_response)
    if not json_output:
        return False
        
    # Perform strict Pydantic validation
    try:
        HouseOutput.model_validate(json_output)
        output_path = output_dir / file_name
        with open(output_path, "w") as f:
            # Save the raw JSON output from the model
            json.dump({"prompt": scenario, "output": json_output}, f, indent=2)
        logger.info(f"SUCCESS: Saved valid plan to {file_name}")
        return True
    except Exception as e:
        logger.warning(f"VALIDATION_ERROR: The generated JSON failed schema validation. Discarding. Errors: {e}")
        return False


def main(args):
    """Main function to run the data generation for a specific worker."""
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_scenarios = [generate_dynamic_scenario() for _ in range(args.num_examples)]
    
    # Distribute scenarios among workers
    scenarios_for_worker = all_scenarios[args.worker_id::args.num_workers]
    
    logger.info(f"Worker {args.worker_id}/{args.num_workers} starting. Assigned {len(scenarios_for_worker)} scenarios.")
    
    for i, scenario in enumerate(scenarios_for_worker):
        # Create a unique file name for each attempt
        file_name = f"silver_standard_{args.worker_id}_{i}.json"
        generate_and_validate_example(scenario, output_dir, file_name)

    logger.info(f"Worker {args.worker_id} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Silver Standard HouseBrain training data in parallel.")
    parser.add_argument("--num-examples", type=int, default=100, help="Total number of examples to generate across all workers.")
    parser.add_argument("--output-dir", type=str, default="data/training/silver_standard", help="Directory to save the generated data.")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers.")
    parser.add_argument("--worker-id", type=int, default=0, help="ID of this worker (e.g., 0, 1, 2...).")
    args = parser.parse_args()
    main(args)
