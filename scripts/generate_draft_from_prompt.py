import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# --- Add project root to sys.path ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.housebrain.schema import HouseOutput


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def call_ollama(prompt: str, model: str) -> Optional[str]:
    """Sends a request to the Ollama API."""
    import requests  # Import locally for Colab compatibility
    url = "http://localhost:11434/api/generate"
    # Hint to Ollama/models to return JSON only when possible.
    payload = { "model": model, "stream": False, "prompt": prompt, "format": "json" }
    try:
        response = requests.post(url, json=payload, timeout=1800) # 30 min timeout
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {e}")
        return None

def extract_json_from_response(raw_text: str) -> str:
    """
    Extracts a JSON object from a potentially messy string.
    Tries to find a JSON object within ```json ... ```,
    then falls back to the largest balanced '{...}' block.
    """
    # 1. Prioritize JSON within markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. Fallback: Find the largest valid JSON object in the string
    best_json = ""
    max_len = 0
    stack = []
    start_index = -1

    for i, char in enumerate(raw_text):
        if char == '{':
            if not stack:
                start_index = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_index != -1:
                    # Found a complete, balanced JSON object
                    candidate = raw_text[start_index : i + 1]
                    if len(candidate) > max_len:
                        try:
                            json.loads(candidate) # Check if it's valid
                            best_json = candidate
                            max_len = len(candidate)
                        except json.JSONDecodeError:
                            continue # Not a valid JSON, keep searching
    
    if best_json:
        return best_json

    # 3. If no JSON is found, return the raw text for debugging
    return raw_text


def generate_and_save_draft(prompt: str, model: str, output_file: Path):
    """Generates a draft using Ollama and saves it."""
    logger.info(f"Sending request to Ollama with model '{model}'...")
    raw_response = call_ollama(prompt, model)

    if not raw_response:
        logger.error("Failed to get a response from Ollama.")
        return

    json_str = extract_json_from_response(raw_response)

    try:
        # Validate that the extracted string is valid JSON
        parsed_json = json.loads(json_str)
        
        # Save the clean, valid JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(parsed_json, f, indent=2)
        logger.info(f"✅ Successfully generated and saved valid JSON to {output_file}")

    except json.JSONDecodeError:
        logger.error("Response was not valid JSON. Saving raw response for debugging.")
        error_file = output_file.with_suffix(f"{output_file.suffix}.raw_error.txt")
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, "w") as f:
            f.write(raw_response)
        logger.info(f"Raw response saved to {error_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate a draft house plan using a local LLM.")
    parser.add_argument("--model", type=str, required=True, help="Ollama model to use (e.g., 'llama3').")
    
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--scenario", type=str, help="A short text description of the house plan to generate.")
    prompt_group.add_argument("--prompt-file", type=str, help="Path to a text file containing the full prompt.")

    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output JSON file.")

    args = parser.parse_args()

    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).read_text()
        except FileNotFoundError:
            logger.error(f"Error: Prompt file not found at {args.prompt_file}")
            sys.exit(1)
    else:
        prompt = args.scenario

    output_path = Path(args.output_file)
    
    logger.info(f"Generating a draft for scenario: '{args.scenario or 'from file'}'")
    generate_and_save_draft(prompt, args.model, output_path)

if __name__ == "__main__":
    main()
