import argparse
import json
import logging
import os
import random
import requests
import sys
import time
from pathlib import Path
from typing import Optional, Dict
import re # Added for markdown block parsing

# --- Pre-computation Setup ---
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from housebrain.schema import HouseOutput, HouseInput

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Schema and Prompt Loading ---
def load_schema_string() -> str:
    """Loads the Pydantic schema file as a string to be injected into the prompt."""
    try:
        with open(SRC_PATH / "housebrain" / "schema.py", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Could not find schema.py file. Please ensure it exists at src/housebrain/schema.py")
        sys.exit(1)

SCHEMA_PY_STRING = load_schema_string()

PROMPT_TEMPLATE = """You are an expert Indian architect AI. Your task is to generate a complete, valid, and architecturally sound house design in JSON format based on a user request.

**User Request:**
Design a {style} for a {width}x{length} feet {orientation} plot in {city}, India. Ensure the design is Vastu compliant.

**Your Task:**
Create a JSON object that represents the house plan. This JSON object must be **100% compliant** with the following Pydantic schema definition. You must invent all necessary details, including coordinates, areas, and costs, to create a complete and valid design.

**TARGET PYDANTIC SCHEMA DEFINITION:**
```python
{schema_definition}
```

**Instructions:**
1.  Adhere strictly to all data types, required fields, and `Enum` values specified in the schema.
2.  Ensure all coordinates and dimensions are plausible and geometrically consistent.
3.  Your output must be **ONLY the valid JSON object**, enclosed in a markdown block.
"""

CITIES = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad", "Pune"]
STYLES = ["2BHK single-story house", "3BHK single-story house", "4BHK single-story house", "3BHK G+1 duplex house", "4BHK G+1 duplex house", "5BHK G+1 duplex house"]
ORIENTATIONS = ["north-facing", "south-facing", "east-facing", "west-facing"]

def generate_dynamic_prompt() -> str:
    """Creates a random, plausible scenario and injects it into the full prompt template."""
    return PROMPT_TEMPLATE.format(
        style=random.choice(STYLES),
        width=random.randint(25, 40),
        length=random.randint(50, 70),
        orientation=random.choice(ORIENTATIONS),
        city=random.choice(CITIES),
        schema_definition=SCHEMA_PY_STRING,
    )

def call_ollama(prompt: str) -> Optional[str]:
    """Sends a request to the Ollama API."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "stream": False,
        "format": "json",
        "prompt": prompt,
    }
    try:
        response = requests.post(url, json=payload, timeout=180) # 3-minute timeout
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.Timeout:
        logger.error("TIMEOUT: Ollama request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"REQUEST_ERROR: Ollama request failed: {e}")
        return None

def generate_and_validate_example(max_retries: int = 3) -> Optional[Dict]:
    """Generates a design and validates it against the Pydantic schema, with retries."""
    for attempt in range(max_retries):
        prompt = generate_dynamic_prompt()
        logger.info(f"Attempt {attempt + 1}/{max_retries}: Generating a design...")
        
        raw_response = call_ollama(prompt)
        if not raw_response:
            logger.warning("Generation failed: No response from Ollama.")
            continue

        try:
            # The prompt asks for a markdown block, but the 'format: json' parameter
            # often returns raw JSON. We need to handle both cases.
            try:
                llm_output = json.loads(raw_response)
            except json.JSONDecodeError:
                # Fallback for markdown block
                match = re.search(r"```json\n({.*?})\n```", raw_response, re.DOTALL)
                if match:
                    llm_output = json.loads(match.group(1))
                else:
                    raise

            # Perform Pydantic validation
            HouseOutput.model_validate(llm_output)
            logger.info("✅ SUCCESS: Generated design passed validation.")
            # We no longer have a simple "scenario", the whole prompt is the scenario
            return {"prompt": "See generated output", "output": llm_output}
        except json.JSONDecodeError:
            logger.warning("Validation failed: Response was not valid JSON or a markdown block.")
        except Exception as e:
            logger.warning(f"Validation failed: Pydantic schema validation error: {e}")
        
        logger.info("Retrying with a new scenario...")
        time.sleep(1) # Brief pause before retrying

    logger.error(f"Failed to generate a valid example after {max_retries} retries.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate a validated Silver Standard dataset for HouseBrain.")
    parser.add_argument("--num-examples", type=int, default=100, help="Number of valid examples to generate.")
    parser.add_argument("--output-dir", type=str, default="data/training/silver_standard_validated", help="Directory to save the validated data.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting generation of {args.num_examples} validated examples...")
    
    valid_examples_count = 0
    while valid_examples_count < args.num_examples:
        validated_example = generate_and_validate_example()
        if validated_example:
            valid_examples_count += 1
            file_name = f"silver_standard_{valid_examples_count:04d}.json"
            output_path = output_dir / file_name
            with open(output_path, "w") as f:
                json.dump(validated_example, f, indent=2)
            logger.info(f"Saved validated example {valid_examples_count}/{args.num_examples} to {output_path.name}")

    logger.info(f"--- Successfully generated {args.num_examples} validated examples. ---")

if __name__ == "__main__":
    main()
