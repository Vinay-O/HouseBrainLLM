import textwrap
import os

# Using a raw triple-quoted string to avoid any f-string or escaping issues.
script_content = r'''
import argparse
import logging
import subprocess
import sys
from pathlib import Path
import shutil
import json
from inspect import getsource
import urllib.request
import hashlib

# --- Add Project Root to Path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.housebrain.schema import HouseOutput

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Self-Contained Helper Functions ---
def get_schema_definition() -> str:
    return getsource(HouseOutput)

def extract_json_from_response(response_text: str) -> str | None:
    try:
        start_index = response_text.find('{')
        if start_index == -1: return None
        end_index = response_text.rfind('}')
        if end_index == -1: return None
        json_str = response_text[start_index:end_index+1]
        json.loads(json_str)
        return json_str
    except Exception:
        return None

def call_ollama(model_name: str, prompt: str) -> str | None:
    url = "http://localhost:11434/api/generate"
    data = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}
    encoded_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=encoded_data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            if response.status == 200:
                response_data = json.loads(response.read().decode('utf-8'))
                response_text = response_data.get("response", "")
                if not response_text.strip():
                    logger.error("Ollama API returned an empty response.")
                    return None
                return response_text
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        return None

def generate_and_save_draft(prompt: str, output_file: Path, model: str, wrap_in_levels: bool = False):
    logger.info(f"Sending request to Ollama with model '{model}'...")
    raw_response = call_ollama(model, prompt)
    if raw_response:
        json_content_str = extract_json_from_response(raw_response)
        if json_content_str:
            try:
                json_content = json.loads(json_content_str)
                if wrap_in_levels and "levels" not in json_content:
                    logger.warning("Model output was missing root 'levels' key. Wrapping it now.")
                    if isinstance(json_content, list):
                        json_content = {"levels": json_content}
                    else:
                        json_content = {"levels": [json_content] if 'rooms' in json_content else []}
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2)
                logger.info(f"✅ Successfully saved draft to {output_file}")
                return True
            except json.JSONDecodeError:
                logger.error(f"Failed to decode the JSON response: {json_content_str[:500]}")
                return False
    logger.error("Failed to generate a valid JSON draft.")
    return False

# --- Prompt Templates ---
STAGE_1_PROMPT_TEMPLATE = """You are an expert AI architect. Generate ONLY the high-level geometric layout for a house.
Focus ONLY on `levels` and `rooms` with `id`, `type` (using snake_case like 'living_room'), and non-overlapping `bounds`.
DO NOT include doors or windows yet.
Your output must be a single JSON object with a root "levels" key.

**Golden Example of room structure (without doors/windows):**
```json
{{
  "id": "living_room_0",
  "type": "living_room",
  "bounds": {{"x": 1, "y": 1, "width": 14, "height": 18}}
}}
```
---
**Schema Reference:**
```python
{schema_definition}
```
---
**User Prompt:**
{user_prompt}
---
Generate the JSON for the house layout."""

STAGE_2_PROMPT_TEMPLATE = """You are an expert AI architect. Add doors and windows to a pre-existing house layout.
- Use snake_case for room `type` (e.g., 'master_bedroom').
- Doors and windows MUST be complete JSON objects, not just strings.
- Ensure doors connect adjacent rooms and windows are on exterior walls.
- DO NOT change the existing `id`, `type`, or `bounds`.
Your output must be a single JSON object containing ONLY the `levels` key.

**Golden Example of a perfect room with openings:**
```json
 "rooms": [
    {{
      "id": "living_room_0",
      "type": "living_room",
      "bounds": {{"x": 1, "y": 1, "width": 14, "height": 18}},
      "doors": [
        {{
          "id": "d1",
          "wall_index": 2,
          "position": 0.5,
          "width": 3,
          "connecting_room_id": "kitchen_0"
        }}
      ],
      "windows": [
        {{
          "id": "w1",
          "wall_index": 0,
          "position": 0.5,
          "width": 5,
          "height": 4,
          "elevation": 3
        }}
      ]
    }}
]
```
---
**Schema Reference:**
```python
{schema_definition}
```
---
**Existing House Layout:**
```json
{existing_layout}
```
---
**Original User Prompt:**
{user_prompt}
---
Now, add the doors and windows to the layout, following the format of the Golden Example exactly."""

# --- Stage Functions ---
def run_stage_with_retry(stage_function, max_retries, *args):
    for i in range(max_retries):
        if stage_function(*args):
            return True
        logger.warning(f"Stage failed. Retrying ({{i+1}}/{{max_retries}})...")
    return False

def stage_1_layout(prompt, output_file, model):
    logger.info("--- Stage 1: Generating Layout ---")
    schema_def = get_schema_definition()
    generation_prompt = STAGE_1_PROMPT_TEMPLATE.format(schema_definition=schema_def, user_prompt=prompt)
    return generate_and_save_draft(prompt=generation_prompt, output_file=output_file, model=model, wrap_in_levels=True)

def stage_2_openings(layout_file, prompt, output_file, model):
    logger.info("--- Stage 2: Adding Openings ---")
    with open(layout_file, 'r', encoding='utf-8') as f:
        layout_content = f.read()
    schema_def = get_schema_definition()
    generation_prompt = STAGE_2_PROMPT_TEMPLATE.format(schema_definition=schema_def, existing_layout=layout_content, user_prompt=prompt)
    return generate_and_save_draft(prompt=generation_prompt, output_file=output_file, model=model, wrap_in_levels=True)

def stage_3_finalize(layout_file, prompt, output_file):
    logger.info("--- Stage 3: Finalizing Plan ---")
    try:
        with open(layout_file, 'r', encoding='utf-8') as f:
            final_layout = json.load(f)

        if "levels" not in final_layout:
            logger.error("Stage 3 Error: Input JSON is missing 'levels' key.")
            return False

        total_area_sqft = sum(r['bounds']['width'] * r['bounds']['height'] for l in final_layout.get("levels", []) for r in l.get("rooms", []))
        construction_cost = total_area_sqft * 150.0

        final_plan = {
            "input": {
                "basicDetails": {"prompt": prompt, "totalArea": total_area_sqft},
                "plot": {},
                "roomBreakdown": []
            },
            "levels": final_layout.get("levels", []),
            "total_area": round(total_area_sqft, 2),
            "construction_cost": round(construction_cost, 2),
            "materials": {}, "render_paths": {}
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_plan, f, indent=2)
        logger.info(f"✅ Stage 3: Final plan saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error in Stage 3: {e}")
        return False

def validate_output(file_to_validate):
    logger.info(f"--- Validating Final Output: {file_to_validate.name} ---")
    try:
        with open(file_to_validate, 'r') as f:
            instance = json.load(f)
        HouseOutput.model_validate(instance)
        logger.info("✅ Final validation successful.")
        return True
    except Exception as e:
        logger.error(f"❌ Final validation failed: {e}")
        return False

def validate_stage_1_output(file_to_validate):
    try:
        with open(file_to_validate, 'r') as f:
            data = json.load(f)
        if "levels" in data and isinstance(data["levels"], list):
            logger.info("✅ Stage 1 output is structurally valid.")
            return True
        logger.error(f"Stage 1 validation failed. Content: {data}")
        return False
    except Exception as e:
        logger.error(f"Stage 1 validation failed with exception: {e}")
        return False

# --- Main Orchestrator ---
def assembly_line_pipeline(prompt: str, output_dir: Path, model: str, run_name: str | None = None, max_retries: int = 3):
    logger.info("--- Starting Architect's Assembly Line ---")
    work_dir = None
    try:
        run_id = run_name if run_name else "prompt_" + hashlib.sha1(prompt.encode()).hexdigest()[:10]
        work_dir = output_dir / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        stage_1_output = work_dir / "1_layout.json"

        def run_st_1():
            if stage_1_layout(prompt, stage_1_output, model):
                return validate_stage_1_output(stage_1_output)
            return False

        if not run_stage_with_retry(run_st_1, max_retries):
            raise Exception("Assembly line failed at Stage 1.")

        stage_2_output = work_dir / "2_layout_with_openings.json"
        if not run_stage_with_retry(stage_2_openings, max_retries, stage_1_output, prompt, stage_2_output, model):
            raise Exception("Assembly line failed at Stage 2.")

        final_output = work_dir / "3_final_plan.json"
        if not stage_3_finalize(stage_2_output, prompt, final_output):
             raise Exception("Assembly line failed at Stage 3.")

        if validate_output(final_output):
            final_path = output_dir / "gold_standard" / f"{run_id}_curated.json"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(final_output, final_path)
            logger.info(f"✅ Assembly Line Succeeded! Final plan saved to: {final_path}")
        else:
            raise Exception(f"Final validation failed. Please check artifacts in {work_dir.resolve()}")

    except Exception as e:
        log_msg = f"🚨 A critical error occurred in the assembly line: {e}"
        if work_dir:
            log_msg += f" -- Check artifacts in {work_dir.resolve()}"
        logger.error(log_msg, exc_info=False)
    finally:
        logger.info("--- Assembly Line Pipeline Finished ---")

def main():
    parser = argparse.ArgumentParser(description="HouseBrain Self-Contained Assembly Line")
    parser.add_argument("--prompt", type=str, required=True, help="A string containing the design prompt.")
    parser.add_argument("--output-dir", type=str, default="output/assembly_line", help="Directory to save the final files.")
    parser.add_argument("--run-name", type=str, help="Optional unique name for this run.")
    parser.add_argument("--model", type=str, default="mistral:7b-instruct", help="Name of the Ollama model to use.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for each LLM-based stage.")
    args = parser.parse_args()

    assembly_line_pipeline(
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        model=args.model,
        run_name=args.run_name,
        max_retries=args.max_retries
    )

if __name__ == "__main__":
    main()
'''

file_path = "scripts/run_complete_assembly_line.py"
# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, "w") as f:
    f.write(script_content)

print(f"Script created at {file_path}")
