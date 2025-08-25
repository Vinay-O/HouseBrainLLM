import json
from pathlib import Path
import argparse
import shutil

# This mapping helps correct common, minor deviations from the official RoomType enum.
ROOM_TYPE_CORRECTIONS = {
    "entryway": "entrance",
    "servants_quarter": "storage", # Or map to a more generic type if servant_room doesn't exist
    "servants_quarters_2_0": "storage", # Accommodate IDs used as types
    "laundry_room": "utility",
    "open_plan_kitchen_living": "living_room", # Simplify complex types
    "bedroom_1": "bedroom", # Normalize numbered types
    "bathroom_1": "bathroom",
    "bathroom_2": "bathroom",
    "north_entry": "entrance",
    "home_office": "study",
    # This list can be expanded as we find more variations.
}

def find_room_by_id(levels, room_id):
    """Utility to find a room object by its ID across all levels."""
    for level in levels:
        for room in level['rooms']:
            if room['id'] == room_id:
                return room
    return None

def upgrade_plan(plan_data: dict) -> dict:
    """
    Upgrades a single house plan JSON to the latest schema version.
    """
    
    # --- Task 1: Correct Room Types ---
    for level in plan_data['levels']:
        for room in level['rooms']:
            original_type = room['type']
            if original_type in ROOM_TYPE_CORRECTIONS:
                room['type'] = ROOM_TYPE_CORRECTIONS[original_type]

    # --- Task 2: Rebuild Door Connectivity ---
    # This task is complex as it requires cross-room awareness.
    all_doors = []
    # First, gather all doors and their host rooms
    for level in plan_data['levels']:
        for room in level['rooms']:
            for door_data in room.get('doors', []):
                # The door belongs to the room it's defined in.
                room1_id = room['id']
                room2_id = door_data.get('connecting_room_id')
                
                if not room2_id:
                    continue # Skip malformed doors

                # To avoid duplicates, we use a sorted tuple as a key
                door_key = tuple(sorted((room1_id, room2_id)))
                
                # We need to find the center point of the shared wall. This is a simplification.
                # A full implementation would require geometric analysis.
                # For now, we assume the provided position is good enough.
                
                new_door = {
                    "position": {"x": -1, "y": -1}, # Placeholder - a real implementation needs geometry
                    "width": door_data.get('width', 3.0),
                    "type": "interior",
                    "room1": room1_id,
                    "room2": room2_id
                }
                all_doors.append({'key': door_key, 'door': new_door})

    # Clear all old doors
    for level in plan_data['levels']:
        for room in level['rooms']:
            room['doors'] = []
            
    # Add the processed, unique doors back to the first room in the pair
    processed_keys = set()
    for door_info in all_doors:
        if door_info['key'] not in processed_keys:
            room1 = find_room_by_id(plan_data['levels'], door_info['door']['room1'])
            if room1:
                # We are skipping position calculation for now as it's very complex.
                # The key is to get the room1/room2 linkage right.
                del door_info['door']['position'] 
                room1['doors'].append(door_info['door'])
                processed_keys.add(door_info['key'])

    return plan_data


def process_directory(input_dir: Path, output_dir: Path):
    """
    Processes all '3_final_plan.json' files in a directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plan_files = list(input_dir.rglob("3_final_plan.json"))
    
    if not plan_files:
        print(f"Warning: No '3_final_plan.json' files found in {input_dir}")
        return

    print(f"Found {len(plan_files)} plans to upgrade...")

    for i, plan_path in enumerate(plan_files):
        print(f"Processing [{i+1}/{len(plan_files)}]: {plan_path.relative_to(input_dir)}")
        
        with open(plan_path, 'r') as f:
            data = json.load(f)
        
        # Perform the upgrade
        upgraded_data = upgrade_plan(data)
        
        # Save to the new directory, preserving the structure
        relative_path = plan_path.relative_to(input_dir)
        new_path = output_dir / relative_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(new_path, 'w') as f:
            json.dump(upgraded_data, f, indent=2)

    print(f"\n✅ Upgrade complete. {len(plan_files)} plans have been processed and saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Upgrade HouseBrain dataset from v1 (Mixtral output) to v2 (strict schema).")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing the raw generated dataset (e.g., 'platinum_batch_1')."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where the cleaned dataset will be saved."
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found at '{input_path}'")
        return

    process_directory(input_path, output_path)

if __name__ == "__main__":
    main()
