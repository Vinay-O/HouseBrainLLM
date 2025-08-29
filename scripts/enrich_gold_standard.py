import sys
import os
import json
import shutil
from pathlib import Path

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curation_scripts.enrich_plans import enrich_plan

def main():
    """
    Takes the gold standard plan, copies it, and runs the enricher script on it
    to automatically add doors.
    """
    INPUT_JSON_PATH = Path("test_data/gold_standard_plan.json")
    WORKING_DIR = Path("test_data/temp_enrich_test")
    OUTPUT_DIR = Path("test_data/enriched_gold_standard")

    print("="*80)
    print("🚀 Enriching Gold Standard Plan with Doors 🚀")
    print("="*80)

    # 1. Setup directories
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        
    WORKING_DIR.mkdir()
    OUTPUT_DIR.mkdir()

    # 2. Copy the gold standard file to the working directory
    working_file = WORKING_DIR / INPUT_JSON_PATH.name
    shutil.copy(INPUT_JSON_PATH, working_file)
    print(f"📄 Copied '{INPUT_JSON_PATH}' to working directory.")

    # 3. Run the enrich_plan function
    enriched_file_path = enrich_plan(str(working_file), str(OUTPUT_DIR))

    if enriched_file_path:
        print(f"\n✅ Successfully enriched plan and saved to '{enriched_file_path}'")
        
        # Optional: Print the number of doors added
        with open(enriched_file_path, 'r') as f:
            data = json.load(f)
        
        door_count = 0
        for level in data.get('levels', []):
            for room in level.get('rooms', []):
                door_count += len(room.get('doors', []))
        print(f"🚪 Found {door_count} auto-generated doors in the new plan.")

    else:
        print("❌ ERROR: Enrichment process failed.")

    # 4. Clean up working directory
    shutil.rmtree(WORKING_DIR)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
