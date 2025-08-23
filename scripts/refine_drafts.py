import json
import os
import sys
from pathlib import Path
import subprocess

# --- Setup ---
# Add src directory to the Python path
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from housebrain.schema import HouseOutput, validate_house_design

# --- Constants ---
RAW_DIR = Path("data/training/silver_standard_raw")
REFINED_DIR = Path("data/training/silver_standard")
EDITOR = os.environ.get("EDITOR", "vim")  # Use vim as a fallback editor

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def open_in_editor(filepath):
    """Opens a file in the default or specified terminal editor."""
    subprocess.run([EDITOR, str(filepath)])

def main():
    """Main loop to refine raw drafts."""
    os.makedirs(REFINED_DIR, exist_ok=True)
    raw_files = sorted(list(RAW_DIR.glob("*.json")))

    if not raw_files:
        print(f"No raw drafts found in {RAW_DIR}. Run the generation script first.")
        return

    print(f"Found {len(raw_files)} raw drafts to refine.")
    
    refined_count = len(list(REFINED_DIR.glob("*.json")))

    for i, raw_file_path in enumerate(raw_files):
        clear_screen()
        
        with open(raw_file_path, "r") as f:
            data = json.load(f)
        
        prompt = data["prompt"]
        raw_response = data["raw_response"] # This is a string

        # Create a temporary file to edit
        temp_file = RAW_DIR / f"{raw_file_path.stem}_editable.json"
        
        # Try to format the raw response as JSON for easier editing
        try:
            # First, see if the raw response is already valid JSON
            editable_json = json.loads(raw_response)
        except json.JSONDecodeError:
            # If not, it might be in a markdown block or just malformed
            # For simplicity, we just dump the raw string for the user to fix
            print("--- WARNING: Raw response is not valid JSON. You will need to fix it manually. ---")
            editable_json = {"raw_response_needs_fixing": raw_response}

        with open(temp_file, "w") as f:
            json.dump(editable_json, f, indent=2)

        while True:
            print("--- HouseBrain Data Refiner ---")
            print(f"Progress: Refining {i+1}/{len(raw_files)} (Already refined: {refined_count})")
            print(f"File: {raw_file_path.name}")
            print("\n--- Original Prompt ---")
            print(prompt)
            print("\n--- Instructions ---")
            print("The JSON for this prompt will be opened in your terminal editor.")
            print("Your task is to edit the file until it is a PERFECT, schema-compliant HouseOutput.")
            print("Save and close the editor when you are done.")

            action = input("\n[E]dit file, [S]kip this file, [Q]uit refinement -> ").lower()

            if action == 'e':
                open_in_editor(temp_file)
                
                # After editing, validate the file
                try:
                    with open(temp_file, "r") as f:
                        edited_json = json.load(f)
                    
                    # 1. Pydantic Model Validation
                    house_obj = HouseOutput.model_validate(edited_json)
                    print("\n✅ Pydantic Schema: OK")

                    # 2. Advanced Logical Validation
                    validation_result = validate_house_design(house_obj)
                    if not validation_result.is_valid:
                        print(f"❌ Logical Validation Failed: {validation_result.errors}")
                        input("Press Enter to re-edit the file...")
                        continue
                    
                    print("✅ Advanced Validation: OK")
                    
                    # If all validations pass, save to the refined directory
                    refined_file_path = REFINED_DIR / f"silver_standard_{raw_file_path.stem}.json"
                    final_data = {"prompt": prompt, "output": edited_json}
                    with open(refined_file_path, "w") as f:
                        json.dump(final_data, f, indent=2)
                    
                    print(f"\n🎉 Success! Saved perfected data to {refined_file_path.name}")
                    refined_count += 1
                    os.remove(temp_file) # Clean up temp file
                    input("Press Enter to continue to the next file...")
                    break # Move to the next file

                except json.JSONDecodeError:
                    print("\n❌ Error: The edited file is not valid JSON.")
                    input("Press Enter to re-edit the file...")
                except Exception as e:
                    print(f"\n❌ Pydantic Validation Error: {e}")
                    input("Press Enter to re-edit the file...")

            elif action == 's':
                print("Skipping file...")
                os.remove(temp_file)
                break
            elif action == 'q':
                print("Quitting refinement process.")
                if temp_file.exists():
                    os.remove(temp_file)
                return

if __name__ == "__main__":
    main()
