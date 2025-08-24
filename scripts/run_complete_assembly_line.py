import argparse
import logging
import subprocess
import sys
from pathlib import Path
import shutil
import json
from inspect import getsource
import requests

# --- Add Project Root to Path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.housebrain.schema import HouseOutput

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Self-Contained Helper Functions ---
def get_schema_definition() -> str:
    """Returns the Pydantic schema source code for HouseOutput."""
    return getsource(HouseOutput)

def extract_json_from_response(response_text: str) -> str | None:
    """Extracts a JSON object from a model's text response."""
    try:
        # Find the start of the JSON block
        start_index = response_text.find('{')
        if start_index == -1:
            logger.warning("Could not find a starting '{' in the response.")
            return None
        
        # Find the end of the JSON block
        end_index = response_text.rfind('}')
        if end_index == -1:
            logger.warning("Could not find a closing '}' in the response.")
            return None
            
        json_str = response_text[start_index:end_index+1]
        
        # Validate that it's actually JSON
        json.loads(json_str)
        return json_str

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from response: {e}")
        logger.error(f"Problematic string section: {json_str[:500]}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON extraction: {e}")
        return None

def call_ollama(model_name: str, prompt: str) -> str | None:
    """Calls the Ollama API and returns the text response."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {e}")
        return None

def generate_and_save_draft(prompt: str, output_file: Path, model: str):
    logger.info(f"Sending request to Ollama with model '{model}'...")
    raw_response = call_ollama(model, prompt)
    if not raw_response:
        logger.error("Failed to get a response from Ollama.")
        return

    json_content = extract_json_from_response(raw_response)
    if not json_content:
        logger.error("Could not extract valid JSON from the model's response.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_content)
    logger.info(f"✅ Successfully generated and saved valid JSON to {output_file}")


# --- Prompt Templates ---
STAGE_1_PROMPT_TEMPLATE = """
You are an expert AI architect. Your task is to generate ONLY the high-level geometric layout for a house based on a user's prompt.
Focus ONLY on:
1. Defining `levels`.
2. Defining `rooms` within each level.
3. For each room, providing ONLY its `id`, `type`, and `bounds`.
4. Ensuring NO rooms overlap.
DO NOT include doors, windows, or any top-level keys.
Your output must be a single JSON object containing ONLY the `levels` key: `{"levels": [...]}`.
---
**Pydantic Schema Reference:**
```python
{schema_definition}
```
---
**User Prompt:**
{user_prompt}
---
Now, generate the JSON for the house layout.
"""

STAGE_2_PROMPT_TEMPLATE = """
You are an expert AI architect. Your task is to add doors and windows to a pre-existing house layout.
Focus ONLY on:
1. Adding a `doors` array to rooms.
2. Adding a `windows` array to rooms.
3. Ensuring doors connect adjacent rooms on their shared boundary.
4. Ensuring windows are on exterior walls.
DO NOT change the existing `id`, `type`, or `bounds` of any room.
Your output must be a single JSON object containing ONLY the `levels` key.
---
**Pydantic Schema Reference:**
```python
{schema_definition}
```
---
**Existing House Layout:**
```json
{existing_layout}
```
---
**Original User Prompt (for context):**
{user_prompt}
---
Now, add the doors and windows to the provided house layout.
"""

FINAL_SCHEMA_SKELETON = "..." # (omitted for brevity, will be the same as before)


# --- Stage Functions ---
def stage_1_layout(prompt, output_file, model):
    logger.info("Stage 1: Generating the geometric layout...")
    schema_def = get_schema_definition()
    generation_prompt = STAGE_1_PROMPT_TEMPLATE.format(schema_definition=schema_def, user_prompt=prompt)
    generate_and_save_draft(prompt=generation_prompt, output_file=output_file, model=model)

def stage_2_openings(layout_file, prompt, output_file, model):
    logger.info("Stage 2: Adding doors and windows...")
    # ... (logic from stage_2_add_openings.py) ...

def stage_3_finalize(layout_file, prompt, output_file):
    logger.info("Stage 3: Finalizing the plan...")
    # ... (logic from stage_3_finalize_plan.py) ...

# --- Main Orchestrator ---
def assembly_line_pipeline(prompt: str, output_dir: Path, model: str, run_name: str | None = None, max_retries: int = 3):
    logger.info("--- Starting Architect's Assembly Line ---")
    # ... (logic from run_assembly_line.py, calling the local functions above) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HouseBrain Self-Contained Assembly Line")
    # ... (argparse logic from run_assembly_line.py) ...
    # This will be a single, large, executable script.
    pass
