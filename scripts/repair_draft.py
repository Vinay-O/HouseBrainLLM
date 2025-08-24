import argparse
import json
import logging
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.generate_draft_from_prompt import generate_and_save_draft
from src.housebrain.schema import HouseOutput

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPAIR_PROMPT_TEMPLATE = """
You are an expert AI architect specializing in fixing flawed building plan data.
The following JSON draft was generated to satisfy a user's request, but it failed validation against the required Pydantic schema.

Your task is to meticulously correct the JSON to fix *only the specific errors listed below*.
You must adhere to the original design intent described in the user's prompt as closely as possible.
Do not add, remove, or significantly change architectural elements unless it's necessary to resolve a validation error.
The final output must be a single, complete, and valid JSON object that strictly follows the provided schema.
{expert_hints}
---
**Golden Example of a Perfect JSON Structure:**
(Use this as a reference for the correct format and level of detail)
```json
{golden_example}
```
---
**Original User Prompt:**
{original_prompt}
---
**Flawed JSON Draft:**
```json
{flawed_json}
```
---
**Validation Errors to Fix:**
```json
{validation_errors}
```
---
**Pydantic Schema for Reference:**
```python
{schema_definition}
```
---

Now, provide the corrected and complete JSON object.
"""

# A dictionary of expert hints to inject based on error messages
REPAIR_HINTS = {
    "Field required": {
        "trigger_text": "Field required",
        "hint": "\n**Expert Hint:** The draft is missing one or more fundamental keys required by the schema. Ensure the final JSON object includes all necessary fields like `input`, `levels`, `total_area`, etc., at the correct hierarchy."
    },
    "total_area": {
        "trigger_text": "total_area",
        "hint": "\n**Expert Hint:** The `total_area` value seems incorrect. Recalculate the sum of all room areas across all levels and update the `total_area` field to match. Also ensure the `input.basicDetails.totalArea` is consistent."
    },
    "overlap": {
        "trigger_text": "overlap",
        "hint": "\n**Expert Hint:** Two or more rooms are overlapping. Review the `bounds` (x, y, width, height) of the mentioned rooms and adjust them so they are adjacent but do not intersect."
    },
    "Door": {
         "trigger_text": "Door",
         "hint": "\n**Expert Hint:** There is a problem with a Door's placement. A door must connect two rooms that share a wall. Check the `room1` and `room2` fields and the `bounds` of those rooms to ensure they are adjacent. Adjust the door's `position` to be on the shared boundary."
    }
}


def get_schema_definition():
    """Extracts the HouseOutput schema definition as a string."""
    from inspect import getsource
    return getsource(HouseOutput)

def generate_expert_hints(error_json: str) -> str:
    """Analyzes errors and generates contextual hints."""
    hints_to_add = set()
    errors_data = json.loads(error_json)
    error_messages = " ".join(errors_data.get("errors", []))

    for key, hint_data in REPAIR_HINTS.items():
        if hint_data["trigger_text"] in error_messages:
            hints_to_add.add(hint_data["hint"])
    
    if not hints_to_add:
        return ""
        
    header = "\n\n--- Expert Hints to Guide Your Repair ---"
    return header + "".join(hints_to_add)


def repair_draft(
    draft_file: Path,
    error_file: Path,
    prompt_file: Path,
    output_file: Path,
    model: str
):
    """
    Reads a flawed draft, validation errors, and the original prompt,
    then calls an LLM to repair the draft.
    """
    logger.info(f"Starting repair process for {draft_file.name}")
    schema_content = get_schema_definition()

    # --- Load all necessary content from files ---
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read()

    with open(draft_file, 'r', encoding='utf-8') as f:
        draft_content = f.read()

    with open(error_file, 'r', encoding='utf-8') as f:
        errors = f.read()
        
    # --- Load the Golden Example ---
    # This example guides the model on the expected structure and quality.
    golden_example_path = Path("data/gold_standard/gold_standard_22_curated_llama_3.json")
    with open(golden_example_path, 'r', encoding='utf-8') as f:
        golden_example_content = f.read()


    # --- Generate contextual hints based on the errors ---
    expert_hints = generate_expert_hints(errors)

    # --- Construct the full prompt for the repair model ---
    repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
        original_prompt=prompt_content,
        flawed_json=draft_content,
        validation_errors=errors,
        schema_definition=schema_content,
        expert_hints=expert_hints,
        golden_example=golden_example_content
    )

    prompt_log_file = output_file.parent / f"{output_file.stem}.repair_prompt.txt"
    with open(prompt_log_file, 'w', encoding='utf-8') as f:
        f.write(repair_prompt)
    logger.info(f"Full repair prompt saved to {prompt_log_file}")

    # --- 3. Call the LLM to generate the fix ---
    logger.info(f"Calling model '{model}' to attempt the repair...")
    generate_and_save_draft(
        prompt=repair_prompt,
        output_file=output_file,
        model=model
    )
    logger.info(f"Repair attempt saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="HouseBrain AI-Powered Plan Repairer")
    parser.add_argument("--draft-file", type=str, required=True, help="Path to the flawed JSON draft.")
    parser.add_argument("--error-file", type=str, required=True, help="Path to the JSON file containing validation errors.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the file containing the original user prompt.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the repaired JSON file.")
    parser.add_argument("--model", type=str, default="llama3", help="Name of the Ollama model to use for the repair.")
    
    args = parser.parse_args()

    repair_draft(
        draft_file=Path(args.draft_file),
        error_file=Path(args.error_file),
        prompt_file=Path(args.prompt_file),
        output_file=Path(args.output_file),
        model=args.model
    )

if __name__ == "__main__":
    main()
