import json
import os
import sys
import uuid
from typing import List, Dict, Any

def find_adjacency(room1: Dict, room2: Dict, tolerance=0.1):
    """
    Checks if two rooms are adjacent and finds the shared wall segment.
    Returns the axis of adjacency ('x' or 'y') and the shared interval.
    """
    r1_bounds = room1['bounds']
    r2_bounds = room2['bounds']
    
    # Correctly access coordinates from the Rectangle object
    r1_x1, r1_y1 = r1_bounds['x'], r1_bounds['y']
    r1_x2, r1_y2 = r1_bounds['x'] + r1_bounds['width'], r1_bounds['y'] + r1_bounds['height']

    r2_x1, r2_y1 = r2_bounds['x'], r2_bounds['y']
    r2_x2, r2_y2 = r2_bounds['x'] + r2_bounds['width'], r2_bounds['y'] + r2_bounds['height']


    # Check for adjacency along the Y-axis (shared vertical wall)
    if abs(r1_x2 - r2_x1) < tolerance or abs(r1_x1 - r2_x2) < tolerance:
        # Find the overlapping interval on the Y-axis
        overlap_y_start = max(r1_y1, r2_y1)
        overlap_y_end = min(r1_y2, r2_y2)
        if overlap_y_end > overlap_y_start:
            # They share a vertical wall segment
            shared_x = r1_x2 if abs(r1_x2 - r2_x1) < tolerance else r1_x1
            return 'y', shared_x, overlap_y_start, overlap_y_end

    # Check for adjacency along the X-axis (shared horizontal wall)
    if abs(r1_y2 - r2_y1) < tolerance or abs(r1_y1 - r2_y2) < tolerance:
        # Find the overlapping interval on the X-axis
        overlap_x_start = max(r1_x1, r2_x1)
        overlap_x_end = min(r1_x2, r2_x2)
        if overlap_x_end > overlap_x_start:
            # They share a horizontal wall segment
            shared_y = r1_y2 if abs(r1_y2 - r2_y1) < tolerance else r1_y1
            return 'x', shared_y, overlap_x_start, overlap_x_end
            
    return None, None, None, None

def enrich_plan(file_path: str, enriched_dir: str):
    """
    Enriches a single validated plan by adding missing doors between adjacent rooms.
    """
    base_filename = os.path.basename(file_path)
    print(f"--- 2. Enriching: {base_filename} ---")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    new_openings_added = 0
    door_width = 3.0  # Standard door width in feet, consistent with schema

    for level in data.get('levels', []):
        rooms = level.get('rooms', [])
        
        # The new schema doesn't have a top-level 'openings'. Doors/windows are in rooms.
        # This script's purpose is to add missing doors between rooms.

        # Compare every pair of rooms on the same level
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                room1 = rooms[i]
                room2 = rooms[j]
                
                axis, shared_coord, start, end = find_adjacency(room1, room2)
                
                if axis:
                    # Check if a door already exists between these two rooms
                    door_exists = False
                    # Check both rooms for a door connecting to the other
                    for door in room1.get('doors', []):
                        if door.get('room2') == room2['id']:
                            door_exists = True
                            break
                    if not door_exists:
                        for door in room2.get('doors', []):
                            if door.get('room2') == room1['id']:
                                door_exists = True
                                break
                    
                    if door_exists:
                        continue

                    print(f"  > Found adjacency between '{room1['id']}' and '{room2['id']}'. Adding door.")
                    
                    # Calculate door position at the midpoint of the shared wall
                    midpoint = (start + end) / 2
                    
                    if axis == 'y': # Vertical wall
                        pos_x, pos_y = shared_coord, midpoint
                    else: # Horizontal wall
                        pos_x, pos_y = midpoint, shared_coord

                    new_door = {
                        "position": {"x": pos_x, "y": pos_y},
                        "width": door_width,
                        "type": "interior",
                        "room1": room1['id'],
                        "room2": room2['id']
                    }
                    
                    # Add the door to the first room's door list
                    if 'doors' not in room1:
                        room1['doors'] = []
                    room1['doors'].append(new_door)
                    new_openings_added += 1

    print(f"✅ PASSED: Added {new_openings_added} new door(s) to the plan.")

    # Save the enriched file
    os.makedirs(enriched_dir, exist_ok=True)
    new_path = os.path.join(enriched_dir, base_filename)
    with open(new_path, 'w') as f:
        json.dump(data, f, indent=4)
    return new_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python enrich_plans.py <path_to_validated_json_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    ENRICHED_DIR = "curated/gold_tier_enriched"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    
    enriched_file_path = enrich_plan(input_file, ENRICHED_DIR)
    
    if enriched_file_path:
        print(f"\\nFile successfully enriched and saved to:\\n{enriched_file_path}")
