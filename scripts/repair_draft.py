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

REPAIR_HINTS = {
    "total_area": (
        "REPAIR HINT: The `total_area` field must be the sum of the areas of all rooms "
        "across all `levels`. Recalculate this value precisely and update the field. "
        "Also ensure the `input.basicDetails.totalArea` field matches this new value if it exists."
    ),
    "overlap": (
        "REPAIR HINT: The error indicates that two rooms are overlapping. "
        "Carefully review the `bounds` (x, y, width, height) of the specified rooms "
        "and adjust them so they no longer intersect. Do not change the overall plot dimensions."
    ),
    "input": (
        "REPAIR HINT: The error indicates a missing or malformed `input` object. "
        "Ensure the `input` object and its nested fields (`basicDetails`, `plot`, `roomBreakdown`) "
        "are present and correctly structured according to the schema."
    )
}


def get_schema_definition():
    """Extracts the HouseOutput schema definition as a string."""
    from inspect import getsource
    return getsource(HouseOutput)

def repair_draft(
    draft_path: Path,
    error_path: Path,
    prompt_path: Path,
    output_path: Path,
    model: str
):
    """
    Reads a flawed draft, validation errors, and the original prompt,
    then calls an LLM to repair the draft.
    """
    logger.info(f"Starting repair process for {draft_path.name}")

    # --- 1. Read all inputs ---
    try:
        with open(draft_path, 'r') as f:
            flawed_json_str = f.read()
        
        with open(error_path, 'r') as f:
            errors_str = f.read()
            # Also load as JSON to inspect errors for hints
            errors_json = json.loads(errors_str)

        with open(prompt_path, 'r') as f:
            original_prompt = f.read()

    except FileNotFoundError as e:
        logger.error(f"Error reading input files: {e}")
        sys.exit(1)

    # --- 2. Construct the repair prompt ---
    schema_def = get_schema_definition()
    
    # Add contextual hints for common, difficult errors
    final_prompt = REPAIR_PROMPT_TEMPLATE
    injected_hints = ""
    error_text_for_prompt = json.dumps(errors_json, indent=2)

    if "total_area" in error_text_for_prompt.lower():
        injected_hints += f"\n{REPAIR_HINTS['total_area']}\n"
    if "overlap" in error_text_for_prompt.lower():
        injected_hints += f"\n{REPAIR_HINTS['overlap']}\n"
    if "'input'" in error_text_for_prompt.lower(): # Check for the key 'input'
        injected_hints += f"\n{REPAIR_HINTS['input']}\n"

    if injected_hints:
        final_prompt = final_prompt.replace(
            "---",
            f"---\n**Specific Instructions:**\n{injected_hints}---",
            1  # Replace only the first occurrence
        )

    repair_prompt = final_prompt.format(
        original_prompt=original_prompt,
        flawed_json=flawed_json_str,
        validation_errors=error_text_for_prompt,
        schema_definition=schema_def
    )

    # Save the full repair prompt for debugging
    debug_prompt_path = output_path.parent / f"{output_path.stem}.repair_prompt.txt"
    with open(debug_prompt_path, 'w') as f:
        f.write(repair_prompt)
    logger.info(f"Full repair prompt saved to {debug_prompt_path}")

    # --- 3. Call the LLM to generate the fix ---
    logger.info(f"Calling model '{model}' to attempt the repair...")
    generate_and_save_draft(
        prompt=repair_prompt,
        output_file=output_path,
        model=model
    )
    logger.info(f"Repair attempt saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="HouseBrain AI-Powered Plan Repairer")
    parser.add_argument("--draft-file", type=str, required=True, help="Path to the flawed JSON draft.")
    parser.add_argument("--error-file", type=str, required=True, help="Path to the JSON file containing validation errors.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the file containing the original user prompt.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the repaired JSON file.")
    parser.add_argument("--model", type=str, default="llama3", help="Name of the Ollama model to use for repair.")
    
    args = parser.parse_args()

    repair_draft(
        draft_path=Path(args.draft_file),
        error_path=Path(args.error_file),
        prompt_path=Path(args.prompt_file),
        output_path=Path(args.output_file),
        model=args.model
    )

if __name__ == "__main__":
    main()
