import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from pydantic import ValidationError
from src.housebrain.schema import HouseOutput, HouseInput, Level, Room, Rectangle, Point2D, Door, Window, RoomType

# --- Constants ---
MM_TO_FEET = 0.00328084

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def mm_to_ft(mm_val: float) -> float:
    return round(mm_val * MM_TO_FEET, 2)

def convert_point(pt: List[float]) -> Point2D:
    return Point2D(x=mm_to_ft(pt[0]), y=mm_to_ft(pt[1]))

def create_bounds_from_boundary(boundary: List[List[float]]) -> Rectangle:
    x_coords = [p[0] for p in boundary]
    y_coords = [p[1] for p in boundary]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return Rectangle(
        x=mm_to_ft(min_x),
        y=mm_to_ft(min_y),
        width=mm_to_ft(max_x - min_x),
        height=mm_to_ft(max_y - min_y)
    )

def migrate_data(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms a single legacy house plan object to the new HouseOutput schema.
    """
    # Handle both direct legacy format and nested training format
    if 'output' in legacy_data:
        legacy_output = legacy_data['output']
        # The 'output' can sometimes be a JSON string
        if isinstance(legacy_output, str):
            try:
                legacy_output = json.loads(legacy_output)
            except json.JSONDecodeError:
                logger.error("Failed to decode the 'output' JSON string.")
                raise
    else:
        # This case is for files that are *just* the old data structure
        legacy_output = legacy_data

    # --- Create placeholder HouseInput from prompt if available ---
    prompt_text = legacy_data.get('prompt', 'Input prompt not available.')
    
    # *** ADDED CHECK ***
    # Ensure there is something to migrate
    if not legacy_output or 'levels' not in legacy_output:
        raise ValueError("The provided data does not contain a valid 'levels' key to migrate.")

    # --- Create placeholder HouseInput ---
    house_input = HouseInput(
        basicDetails={
            "totalArea": 0, "unit": "feet", "floors": 1, "bedrooms": 2, 
            "bathrooms": 1, "style": "Modern Contemporary", "budget": 5000000
        },
        plot={"area": 1000, "shape": "rectangular", "prompt": prompt_text},
        roomBreakdown=[]
    )
    
    # --- Migrate Levels and Rooms ---
    new_levels = []
    legacy_levels = legacy_output.get('levels', [])
    legacy_spaces = legacy_output.get('spaces', [])
    
    level_map = {l['id']: l for l in legacy_levels}
    rooms_by_level: Dict[str, List[Room]] = {l['id']: [] for l in legacy_levels}

    for space in legacy_spaces:
        level_id = space.get('level_id')
        if level_id in rooms_by_level:
            try:
                room = Room(
                    id=space['id'],
                    type=RoomType(space['type']),
                    bounds=create_bounds_from_boundary(space['boundary']),
                    doors=[],
                    windows=[]
                )
                rooms_by_level[level_id].append(room)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid space '{space.get('id')}': {e}")
                continue

    for i, l_id in enumerate(level_map):
        new_levels.append(
            Level(
                level_number=i,
                rooms=rooms_by_level[l_id],
                stairs=[], # Stairs migration not implemented from legacy format
                height=mm_to_ft(level_map[l_id].get('elevation_mm', 3000)) # Placeholder
            )
        )
    
    # --- Calculate Total Area ---
    total_area = sum(room.bounds.area for level in new_levels for room in level.rooms)

    # --- Construct Final HouseOutput ---
    house_output = HouseOutput(
        input=house_input,
        levels=new_levels,
        total_area=round(total_area, 2),
        construction_cost=round(total_area * 2000, 2), # Placeholder cost
        materials={},
        render_paths={}
    )
    
    return house_output.model_dump(mode='json')

def process_file(input_path: Path, output_dir: Path):
    """
    Reads a legacy file, migrates it, and saves the new version.
    """
    try:
        with open(input_path, 'r') as f:
            legacy_data = json.load(f)
        
        migrated_dict = migrate_data(legacy_data)
        
        # Validate the migrated data one last time
        HouseOutput.model_validate(migrated_dict)

        output_path = output_dir / f"migrated_{input_path.name}"
        with open(output_path, 'w') as f:
            json.dump(migrated_dict, f, indent=2)
        
        logger.info(f"✅ Successfully migrated {input_path.name} -> {output_path.name}")

    except json.JSONDecodeError:
        logger.error(f"❌ Failed to parse JSON from {input_path.name}")
    except (ValueError, KeyError) as e:
        logger.warning(f"⚠️ Skipping file {input_path.name} due to missing or invalid data: {e}")
    except ValidationError as e:
        logger.error(f"❌ Validation failed for migrated {input_path.name}: {e}")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred for {input_path.name}: {e}", exc_info=True)


def main():
    source_dir = Path("data/training/gold_standard")
    output_dir = Path("data/migrated_gold_standard")
    
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return
        
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting migration from '{source_dir}' to '{output_dir}'...")
    
    json_files = list(source_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in the source directory.")
        return
        
    for file_path in json_files:
        process_file(file_path, output_dir)
        
    logger.info("Migration process complete.")

if __name__ == "__main__":
    main()
