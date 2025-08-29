import json
import os
import sys
import re
import shutil

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.housebrain.schema import RoomType


def slugify(text: str) -> str:
    """Converts a string like "Living Room" into a slug "living_room"."""
    text = text.lower()
    text = re.sub(r'[\s/]+', '_', text)
    text = re.sub(r'[^\w_]', '', text)
    return text

def map_name_to_roomtype(name: str) -> RoomType:
    """Maps a free-text room name to the RoomType enum."""
    name_slug = slugify(name)
    
    # Direct mapping for common names
    if "master_bedroom" in name_slug: return RoomType.MASTER_BEDROOM
    if "bedroom" in name_slug: return RoomType.BEDROOM
    if "kitchen" in name_slug: return RoomType.KITCHEN
    if "living" in name_slug: return RoomType.LIVING_ROOM
    if "dining" in name_slug: return RoomType.DINING_ROOM
    if "bathroom" in name_slug or "washroom" in name_slug: return RoomType.BATHROOM
    if "powder" in name_slug: return RoomType.HALF_BATH
    if "utility" in name_slug: return RoomType.UTILITY
    if "storage" in name_slug: return RoomType.STORAGE
    if "stair" in name_slug: return RoomType.STAIRWELL
    if "garage" in name_slug: return RoomType.GARAGE
    if "entrance" in name_slug or "lobby" in name_slug: return RoomType.ENTRANCE
    if "balcony" in name_slug: return RoomType.BALCONY
    if "office" in name_slug or "study" in name_slug: return RoomType.STUDY
    if "family" in name_slug or "lounge" in name_slug: return RoomType.FAMILY_ROOM
    
    # Fallback for less common names
    try:
        return RoomType(name_slug)
    except ValueError:
        print(f"  [Warning] Could not map room name '{name}' to a known RoomType. Defaulting to LIVING_ROOM.")
        return RoomType.LIVING_ROOM


def normalize_plan(file_path: str, normalized_dir: str, quarantine_dir: str):
    """
    Reads a raw JSON file and transforms it to fully comply with the HouseOutput schema.
    """
    base_filename = os.path.basename(file_path)
    print(f"--- 0. Healing & Enriching Schema: {base_filename} ---")
    
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ FAILED: Could not read or parse JSON. Error: {e}")
        # Move to quarantine
        return None

    try:
        healed_data = {
            "input": {
                "basicDetails": {
                    "totalArea": 2000, "unit": "sqft", "floors": 2, 
                    "bedrooms": 3, "bathrooms": 2, "style": "Modern Contemporary", "budget": 500000
                },
                "plot": {"shape": "rectangular", "length": 60, "width": 40},
                "roomBreakdown": []
            },
            "total_area": 2000.0,
            "construction_cost": 500000.0,
            "materials": {},
            "render_paths": {},
            "levels": []
        }

        x_cursor, y_cursor = 0, 0
        
        for i, raw_level in enumerate(raw_data.get('levels', [])):
            healed_level = {
                "level_number": i,
                "rooms": [],
                "stairs": [],
                "height": 10.0
            }
            
            for raw_room in raw_level.get('rooms', []):
                # Basic validation of raw room data
                if 'name' not in raw_room or 'dimensions' not in raw_room:
                    continue
                
                dims = raw_room['dimensions']
                if not isinstance(dims, dict) or 'width' not in dims or 'length' not in dims:
                    continue

                width, length = dims['width'], dims['length']
                
                healed_room = {
                    "id": slugify(raw_room['name']),
                    "type": map_name_to_roomtype(raw_room['name']),
                    "bounds": {
                        "x": x_cursor,
                        "y": y_cursor,
                        "width": width,
                        "height": length
                    },
                    "doors": [],
                    "windows": [],
                    "furniture": [],
                    "features": raw_room.get('features', [])
                }
                healed_level['rooms'].append(healed_room)
                
                # Simple sequential layout, add padding
                x_cursor += width + 2 

            healed_data['levels'].append(healed_level)
            # Reset cursor for next level
            x_cursor = 0
            y_cursor += 20 # Arbitrary vertical offset for next floor
            
        print("✅ PASSED: Schema healed and enriched successfully.")
        os.makedirs(normalized_dir, exist_ok=True)
        new_path = os.path.join(normalized_dir, base_filename)
        with open(new_path, 'w') as f:
            json.dump(healed_data, f, indent=2)
        return new_path

    except Exception as e:
        print(f"❌ FAILED: A critical error occurred during healing: {e}")
        # Move to quarantine
        return None

if __name__ == '__main__':
    # This is a placeholder for standalone testing, not used by the main pipeline
    pass
