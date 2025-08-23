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
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Prompt Template ---
# A simple, direct prompt. We are not using few-shot here to maximize speed and variety.
# The goal is to get a raw JSON structure, even if it's not perfectly schema-compliant.
PROMPT_TEMPLATE = """You are a specialized AI architect for Indian residences. Your task is to generate a HouseBrain JSON object based on the user's request. The output must be only the JSON object.

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
        "a dedicated study room for remote work.", "a large balcony connected to the master bedroom.",
        "a separate dining area that can seat eight people.", "an open-air courtyard in the center of the house.",
        "a two-car garage with internal access.",
    ])
    bhk = random.choice(["2BHK", "3BHK", "4BHK"])
    levels = random.choice(["single-story", "G+1 duplex"])
    return f"Design a Vastu-compliant {bhk} {levels} house for a {plot_size} {facing}-facing plot in {city}. The design must include {client_need}"


# --- Core LLM Function ---
def call_ollama(prompt: str) -> Optional[str]:
    """Sends a request to the Ollama API and returns the response string."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-coder:6.7b-instruct", "stream": False, "format": "json",
        "prompt": prompt,
        "system": "You are an AI architect that only outputs JSON.",
    }
    try:
        response = requests.post(url, json=payload, timeout=240) # 4-minute timeout
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.Timeout:
        logger.error(f"TIMEOUT: Ollama request timed out for prompt: {prompt[:80]}...")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"REQUEST_ERROR: Ollama request failed: {e}")
        return None


def generate_and_save_raw_draft(scenario: str, output_dir: Path, file_name: str):
    """Generates a single raw draft and saves it without validation."""
    logger.info(f"GENERATING: {scenario[:90]}...")
    
    prompt = PROMPT_TEMPLATE.format(prompt=scenario)
    raw_response = call_ollama(prompt)
    
    if raw_response:
        output_path = output_dir / file_name
        # Save the prompt and the raw, unvalidated JSON response
        with open(output_path, "w") as f:
            json.dump({"prompt": scenario, "raw_response": raw_response}, f, indent=2)
        logger.info(f"SUCCESS: Saved raw draft to {file_name}")
    else:
        logger.warning(f"FAILURE: No response from Ollama for scenario: {scenario[:90]}...")


def main(args):
    """Main function to run the raw draft generation for a specific worker."""
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_scenarios = [generate_dynamic_scenario() for _ in range(args.num_examples)]
    
    scenarios_for_worker = all_scenarios[args.worker_id::args.num_workers]
    
    logger.info(f"Worker {args.worker_id}/{args.num_workers} starting. Assigned {len(scenarios_for_worker)} scenarios.")
    
    for i, scenario in enumerate(scenarios_for_worker):
        file_name = f"raw_draft_{args.worker_id}_{i}.json"
        generate_and_save_raw_draft(scenario, output_dir, file_name)

    logger.info(f"Worker {args.worker_id} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw, unvalidated HouseBrain drafts.")
    parser.add_argument("--num-examples", type=int, default=200, help="Total number of raw drafts to generate.")
    parser.add_argument("--output-dir", type=str, default="data/training/silver_standard_raw", help="Directory to save the raw drafts.")
    parser.add_argument("--num-workers", type=int, default=8, help="Total number of parallel workers.")
    parser.add_argument("--worker-id", type=int, default=0, help="ID of this worker.")
    args = parser.parse_args()
    main(args)
