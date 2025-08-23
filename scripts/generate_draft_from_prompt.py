import argparse
import json
import logging
import requests
import sys
import re
from pathlib import Path
from typing import Optional

# --- Pre-computation Setup ---
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_schema_string() -> str:
    """Loads the Pydantic schema file as a string."""
    try:
        # Assuming the schema is now in a more standard location
        with open(SRC_PATH / "housebrain" / "schemas" / "house_schema.py", "r") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to the old location for compatibility
        try:
            with open(SRC_PATH / "housebrain" / "schema.py", "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.error("Could not find the schema definition file.")
            sys.exit(1)

SCHEMA_PY_STRING = load_schema_string()

PROMPT_TEMPLATE = """You are an expert Indian architect AI. Your task is to generate a complete, valid, and architecturally sound house design in JSON format based on a user request.

**User Request:**
{scenario}

**Your Task:**
Create a JSON object that represents the house plan. This JSON object must be **100% compliant** with the following Pydantic schema definition. You must invent all necessary details, including coordinates, areas, and costs, to create a complete and valid design.

**TARGET PYDANTIC SCHEMA DEFINITION:**
```python
{schema_definition}
```

**Instructions:**
1.  Adhere strictly to all data types, required fields, and `Enum` values specified in the schema.
2.  Ensure all coordinates and dimensions are plausible and geometrically consistent.
3.  Your output must be **ONLY the valid JSON object**, enclosed in a markdown block. Do not include any other text, explanations, or apologies.
"""

def call_ollama(prompt: str, model: str) -> Optional[str]:
    """Sends a request to the Ollama API."""
    url = "http://localhost:11434/api/generate"
    payload = { "model": model, "stream": False, "prompt": prompt }
    logger.info(f"Sending request to Ollama with model '{model}'...")
    try:
        response = requests.post(url, json=payload, timeout=1800) # 30-minute timeout for complex generation
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        return None

def extract_json_from_response(raw_text: str) -> str:
    """Extracts a JSON object from a markdown block or raw string."""
    match = re.search(r"```(json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(2)
    # Fallback for raw JSON output
    return raw_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate a single 'best effort' draft for a training example.")
    parser.add_argument("--scenario", type=str, required=True, help="The design scenario prompt.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the generated draft.")
    parser.add_argument("--model", type=str, default="deepseek-r1:8b", help="The Ollama model to use.")
    args = parser.parse_args()

    logger.info(f"Generating a draft for scenario: '{args.scenario}'")

    full_prompt = PROMPT_TEMPLATE.format(
        scenario=args.scenario,
        schema_definition=SCHEMA_PY_STRING
    )

    raw_response = call_ollama(full_prompt, args.model)

    if raw_response:
        try:
            json_content = extract_json_from_response(raw_response)
            # Validate if it's valid JSON before saving
            json.loads(json_content)

            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                # Save the extracted JSON content directly
                f.write(json_content)
            logger.info(f"✅ Successfully extracted and saved valid JSON draft to {args.output_file}")
        except json.JSONDecodeError:
            logger.error("Response was not valid JSON. Saving raw response for debugging.")
            # Save raw response if JSON parsing fails
            output_path = Path(args.output_file + ".raw_error.txt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(raw_response)
            logger.info(f"Raw response saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save the draft: {e}")
    else:
        logger.error("Failed to get a response from the model.")

if __name__ == "__main__":
    main()
