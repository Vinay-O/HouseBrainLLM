import argparse
import logging
import sys
from pathlib import Path

from housebrain.generation.prompts import (
    get_schema_definition,
    get_generation_prompt
)
from housebrain.generation.ollama_generator import generate_and_save_draft

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STAGE_1_PROMPT_TEMPLATE = """
You are an expert AI architect. Your task is to generate ONLY the high-level geometric layout for a house based on a user's prompt.

Focus ONLY on the following tasks:
1.  Define the `levels` of the house.
2.  Within each level, define all the `rooms`.
3.  For each room, provide ONLY its `id`, `type`, and `bounds` (x, y, width, height).
4.  Ensure that NO rooms overlap. The geometric layout must be valid.

DO NOT include:
- Doors or windows.
- The top-level `input`, `total_area`, or `construction_cost` keys.
- Any fields not explicitly listed above.

Your output must be a single JSON object containing ONLY the `levels` key, structured like this: `{"levels": [...]}`.

---
**Pydantic Schema for Reference (only generate the parts mentioned above):**
```python
{schema_definition}
```
---
**User Prompt:**
{user_prompt}
---

Now, generate the JSON for the house layout.
"""

def generate_layout(prompt: str, output_file: Path, model: str):
    """Generates the initial geometric layout of the house."""
    logger.info("Stage 1: Generating the geometric layout...")
    
    schema_def = get_schema_definition()
    
    generation_prompt = STAGE_1_PROMPT_TEMPLATE.format(
        schema_definition=schema_def,
        user_prompt=prompt
    )
    
    generate_and_save_draft(
        prompt=generation_prompt,
        output_file=output_file,
        model=model
    )
    logger.info(f"✅ Stage 1: Layout saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generate House Layout")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the file containing the user prompt.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the generated layout JSON file.")
    parser.add_argument("--model", type=str, default="qwen3:30b", help="Name of the Ollama model to use.")
    args = parser.parse_args()

    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()

    generate_layout(
        prompt=prompt,
        output_file=Path(args.output_file),
        model=args.model
    )

if __name__ == "__main__":
    main()
