import argparse
import logging
import json
from pathlib import Path

from housebrain.generation.prompts import get_schema_definition
from housebrain.generation.ollama_generator import generate_and_save_draft

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STAGE_2_PROMPT_TEMPLATE = """
You are an expert AI architect. Your task is to add doors and windows to a pre-existing house layout.

Focus ONLY on the following tasks:
1.  Analyze the provided JSON layout of rooms.
2.  Add a `doors` array to each room where appropriate.
3.  Add a `windows` array to each room where appropriate.
4.  Ensure every door connects two adjacent rooms (`room1`, `room2`) and its `position` is on their shared boundary.
5.  Ensure every window is placed on an exterior wall of a room.
6.  Ensure every bedroom has at least one window.
7.  Ensure there is a clear path from the entrance to all main rooms.

DO NOT change the existing `id`, `type`, or `bounds` of any room. Your job is only to add the openings.

Your output must be a single JSON object containing ONLY the `levels` key, with the rooms now including `doors` and `windows`.

---
**Pydantic Schema for Reference (only add doors and windows):**
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

def add_openings(layout_file: Path, prompt_file: Path, output_file: Path, model: str):
    """Adds doors and windows to the geometric layout."""
    logger.info("Stage 2: Adding doors and windows...")
    
    schema_def = get_schema_definition()
    
    with open(layout_file, 'r', encoding='utf-8') as f:
        layout_content = f.read()
        
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
        
    generation_prompt = STAGE_2_PROMPT_TEMPLATE.format(
        schema_definition=schema_def,
        existing_layout=layout_content,
        user_prompt=prompt_content
    )
    
    generate_and_save_draft(
        prompt=generation_prompt,
        output_file=output_file,
        model=model
    )
    logger.info(f"✅ Stage 2: Layout with openings saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Add Doors and Windows")
    parser.add_argument("--layout-file", type=str, required=True, help="Path to the Stage 1 layout JSON file.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the file containing the original user prompt.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the enhanced layout with openings.")
    parser.add_argument("--model", type=str, default="qwen3:30b", help="Name of the Ollama model to use.")
    args = parser.parse_args()

    add_openings(
        layout_file=Path(args.layout_file),
        prompt_file=Path(args.prompt_file),
        output_file=Path(args.output_file),
        model=args.model
    )

if __name__ == "__main__":
    main()
