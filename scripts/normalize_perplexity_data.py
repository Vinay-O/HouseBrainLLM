import os
import json
import re
from tqdm import tqdm

def normalize_perplexity_plan(perplexity_plan: dict) -> dict:
    """Converts a single plan from Perplexity's schema to our internal schema."""
    
    # Start with the basic structure
    normalized_plan = {
        "input": perplexity_plan.get("input", {}),
        "total_area": perplexity_plan.get("total_area", 0.0),
        "construction_cost": None,  # Perplexity doesn't provide this
        "levels": []
    }

    for p_level in perplexity_plan.get("levels", []):
        new_level = {
            "level_number": p_level.get("level_number", 0),
            "rooms": []
        }
        
        for p_room in p_level.get("rooms", []):
            room_name = p_room.get("name", "unknown_room").lower().replace(" ", "_")
            
            # Convert bounds array [x, y, w, h] to object {x, y, width, height}
            p_bounds = p_room.get("bounds", [0, 0, 0, 0])
            new_bounds = {
                "x": p_bounds[0],
                "y": p_bounds[1],
                "width": p_bounds[2],
                "height": p_bounds[3]
            }

            # Simple logic to infer room type from its name
            room_type = "room" # default
            if "living" in room_name: room_type = "living_room"
            elif "kitchen" in room_name: room_type = "kitchen"
            elif "bedroom" in room_name: room_type = "bedroom"
            elif "bathroom" in room_name: room_type = "bathroom"
            elif "garage" in room_name: room_type = "garage"
            elif "balcony" in room_name: room_type = "balcony"
            elif "storage" in room_name: room_type = "storage"
            elif "courtyard" in room_name: room_type = "courtyard"
            elif "garden" in room_name: room_type = "garden"


            new_room = {
                "id": room_name,
                "type": room_type,
                "bounds": new_bounds,
                "doors": [], # Add empty lists for compatibility
                "windows": []
            }
            new_level["rooms"].append(new_room)
            
        normalized_plan["levels"].append(new_level)
        
    return normalized_plan


def main():
    input_file = "proplexity_output_json.txt"
    output_dir = "perplexity_normalized_data"
    
    print(f"--- Starting Normalization of {input_file} ---")
    
    # 1. Read the raw file content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{input_file}'")
        return

    # 2. Clean up the content to make it a single valid JSON array
    # This replaces `] [` with `, ` to merge the two lists.
    cleaned_content = re.sub(r']\s*\[', ',', raw_content, flags=re.DOTALL)
    
    # 3. Parse the cleaned JSON
    try:
        all_plans = json.loads(cleaned_content)
        if not isinstance(all_plans, list):
            raise TypeError("Top-level JSON is not a list.")
        print(f"✅ Successfully parsed {len(all_plans)} plans from the file.")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ ERROR: Could not parse the cleaned JSON. Error: {e}")
        return
        
    # 4. Normalize and save each plan
    print(f"🚀 Normalizing plans and saving to '{output_dir}'...")
    for i, p_plan in enumerate(tqdm(all_plans, desc="Normalizing Plans")):
        try:
            normalized_plan = normalize_perplexity_plan(p_plan)
            
            output_filename = f"plan_perplexity_{i:04d}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_plan, f, indent=2)
        
        except Exception as e:
            tqdm.write(f"⚠️ WARN: Could not process plan #{i}. Error: {e}")

    print("\n" + "="*50)
    print("✅ Success! Normalization complete.")
    print(f"   Check the '{output_dir}' directory for the processed files.")
    print("="*50)


if __name__ == "__main__":
    main()
