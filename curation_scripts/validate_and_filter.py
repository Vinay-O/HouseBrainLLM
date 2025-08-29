import json
import os
import sys
from pydantic import BaseModel, ValidationError, RootModel
from typing import List, Dict, Any

# --- Minimal Schema for Validation ---
# We only check for the absolute essentials needed for rendering.
# This makes the validation fast and robust.
class MinimalBounds(RootModel[List[float]]):
    pass

class MinimalRoom(BaseModel):
    id: str
    bounds: MinimalBounds

class MinimalLevel(BaseModel):
    level_id: str
    rooms: List[MinimalRoom]

class MinimalPlan(BaseModel):
    levels: List[MinimalLevel]

def validate_plan(file_path: str, validated_dir: str, quarantine_dir: str):
    """
    Validates a single JSON file for basic structure and schema.
    """
    base_filename = os.path.basename(file_path)
    print(f"--- 1. Validating: {base_filename} ---")

    # 1. Check if the file is valid JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ FAILED: Invalid JSON.")
        quarantine_path = os.path.join(quarantine_dir, "json_errors")
        os.makedirs(quarantine_path, exist_ok=True)
        os.rename(file_path, os.path.join(quarantine_path, base_filename))
        return None
    except Exception as e:
        print(f"❌ FAILED: Could not read file. Error: {e}")
        return None
        
    # 2. Check if the JSON adheres to our minimal schema
    try:
        MinimalPlan.parse_obj(data)
        print("✅ PASSED: JSON is valid and adheres to the minimal schema.")
        
        # Move the validated file to the next stage
        os.makedirs(validated_dir, exist_ok=True)
        new_path = os.path.join(validated_dir, base_filename)
        os.rename(file_path, new_path)
        return new_path

    except ValidationError as e:
        print(f"❌ FAILED: Schema validation error.")
        # print(e) # Uncomment for detailed error
        quarantine_path = os.path.join(quarantine_dir, "schema_errors")
        os.makedirs(quarantine_path, exist_ok=True)
        os.rename(file_path, os.path.join(quarantine_path, base_filename))
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_and_filter.py <path_to_raw_json_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # Define output directories
    VALIDATED_DIR = "curated/gold_tier_validated"
    QUARANTINE_DIR = "quarantine"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    # Make a copy for the test run, so we don't destroy the original
    temp_dir = "temp_curation_test"
    os.makedirs(temp_dir, exist_ok=True)
    test_file_path = os.path.join(temp_dir, os.path.basename(input_file))
    import shutil
    shutil.copy(input_file, test_file_path)

    validated_file_path = validate_plan(test_file_path, VALIDATED_DIR, QUARANTINE_DIR)
    
    # Clean up the temp directory
    shutil.rmtree(temp_dir)

    if validated_file_path:
        print(f"\nFile successfully validated and moved to:\n{validated_file_path}")
    else:
        print(f"\nFile failed validation and was moved to a quarantine directory.")
