import argparse
import logging
import json
from pathlib import Path
import jsonschema

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# A simplified representation of the final schema for assembly.
FINAL_SCHEMA_SKELETON = {
    "input": {},
    "levels": [],
    "total_area": 0.0,
    "construction_cost": 0.0,
    "materials": {},
    "render_paths": {}
}

def finalize_plan(layout_with_openings_file: Path, prompt_file: Path, output_file: Path):
    """
    Assembles the final, complete HouseOutput JSON from the intermediate parts.
    This stage is deterministic and does not use an LLM.
    """
    logger.info("Stage 3: Finalizing the plan...")

    with open(layout_with_openings_file, 'r', encoding='utf-8') as f:
        final_layout = json.load(f)
        
    # --- Perform Calculations ---
    total_area_sqft = 0.0
    for level in final_layout.get("levels", []):
        for room in level.get("rooms", []):
            bounds = room.get("bounds", {})
            width = bounds.get("width", 0)
            height = bounds.get("height", 0)
            total_area_sqft += (width * height)
    
    # Simple cost estimation: $150 per sqft
    construction_cost = total_area_sqft * 150.0

    # --- Assemble the Final Object ---
    # NOTE: In a real system, the 'input' would be derived more intelligently.
    # For now, we'll use a placeholder based on the prompt.
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
        
    final_plan = FINAL_SCHEMA_SKELETON.copy()
    final_plan["input"] = {
        "basicDetails": {"prompt": prompt_content},
        "plot": {"prompt": "Details inferred from prompt."},
        "roomBreakdown": [] # Could be generated in a future step
    }
    final_plan["levels"] = final_layout.get("levels", [])
    final_plan["total_area"] = round(total_area_sqft, 2)
    final_plan["construction_cost"] = round(construction_cost, 2)
    
    # --- Save the Final Plan ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_plan, f, indent=2)
        
    logger.info(f"✅ Stage 3: Final, complete plan saved to {output_file}")
    logger.info(f"Calculated Total Area: {total_area_sqft:.2f} sqft")
    logger.info(f"Estimated Construction Cost: ${construction_cost:,.2f}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Finalize House Plan")
    parser.add_argument("--layout-file", type=str, required=True, help="Path to the Stage 2 layout with openings JSON file.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the file containing the original user prompt.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the final, complete HouseOutput JSON file.")
    args = parser.parse_args()

    finalize_plan(
        layout_with_openings_file=Path(args.layout_file),
        prompt_file=Path(args.prompt_file),
        output_file=Path(args.output_file)
    )

if __name__ == "__main__":
    main()
