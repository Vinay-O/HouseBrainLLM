import sys
import os
from pathlib import Path
import json
import subprocess
import shutil

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.housebrain.schema import HouseOutput
    from src.housebrain.rendering.core_2d import render_2d_plan
except ImportError as e:
    print(f"❌ Failed to import necessary modules. Please ensure your project structure is correct.")
    print(f"Error: {e}")
    sys.exit(1)

def main():
    """
    Loads a curated plan, parses it, and runs the 2D rendering pipeline.
    """
    # --- Configuration ---
    # We now point directly to our hand-crafted, "Big Boss" approved plan.
    INPUT_JSON_PATH = Path("test_data/big_boss_plan.json")
    OUTPUT_DIR = Path("output/2d_renders_big_boss_plan")
    
    print("="*80)
    print("🚀 Starting 2D Rendering Test on BIG BOSS Plan 🚀")
    print("="*80)

    # 1. Create output directory, ensuring it's clean
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Clean output directory ensured at: {OUTPUT_DIR.resolve()}")

    # 2. Check if input file exists
    if not INPUT_JSON_PATH.exists():
        print(f"❌ ERROR: Input JSON file not found at '{INPUT_JSON_PATH}'.")
        sys.exit(1)
        
    print(f"📄 Loading plan from: {INPUT_JSON_PATH}")

    # 3. Load and parse the JSON file into a HouseOutput object
    try:
        with open(INPUT_JSON_PATH, 'r') as f:
            json_data = json.load(f)
        
        house_plan = HouseOutput.model_validate(json_data)
        print("✅ Successfully parsed and VALIDATED JSON into HouseOutput schema.")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse the input JSON file.")
        print(f"Pydantic validation might have failed. Error: {e}")
        sys.exit(1)

    # 4. Run the rendering function
    try:
        base_filename = "big_boss_render"
        render_2d_plan(house_plan, OUTPUT_DIR, base_filename)
        print("\n🎉 Rendering process completed.")
        print(f"Check the directory '{OUTPUT_DIR.resolve()}' for the SVG output file(s).")
    except Exception as e:
        print(f"❌ ERROR: The rendering process failed.")
        print(f"Error: {e}")
        sys.exit(1)
        
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
