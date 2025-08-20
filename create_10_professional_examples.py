#!/usr/bin/env python3
"""
Generate 10 professional architectural examples in the housebrain_plan.schema.json format.
These examples will showcase different house types and layouts for testing 2D/3D generation.
"""

import json
import os
from pathlib import Path

def create_example_1_single_story_ranch():
    """3BR/2BA Single Story Ranch - 1,200 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Single Story Ranch",
            "total_area_sqft": 1200,
            "floors": 1,
            "bedrooms": 3,
            "bathrooms": 2
        },
        "walls": [
            # Exterior walls
            {"id": "ext_1", "start": [0, 0], "end": [12000, 0], "type": "exterior", "thickness": 230},
            {"id": "ext_2", "start": [12000, 0], "end": [12000, 8000], "type": "exterior", "thickness": 230},
            {"id": "ext_3", "start": [12000, 8000], "end": [0, 8000], "type": "exterior", "thickness": 230},
            {"id": "ext_4", "start": [0, 8000], "end": [0, 0], "type": "exterior", "thickness": 230},
            
            # Interior walls
            {"id": "int_1", "start": [3000, 0], "end": [3000, 5000], "type": "interior", "thickness": 115},
            {"id": "int_2", "start": [3000, 5000], "end": [9000, 5000], "type": "interior", "thickness": 115},
            {"id": "int_3", "start": [9000, 0], "end": [9000, 8000], "type": "interior", "thickness": 115},
            {"id": "int_4", "start": [6000, 5000], "end": [6000, 8000], "type": "interior", "thickness": 115},
        ],
        "openings": [
            # Front entrance
            {"id": "door_1", "wall_id": "ext_1", "type": "door", "position": 0.4, "width": 915, "metadata": {"swing": "inward"}},
            
            # Windows
            {"id": "win_1", "wall_id": "ext_1", "type": "window", "position": 0.15, "width": 1200, "metadata": {"style": "double_hung"}},
            {"id": "win_2", "wall_id": "ext_1", "type": "window", "position": 0.65, "width": 1200, "metadata": {"style": "double_hung"}},
            {"id": "win_3", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 1000, "metadata": {"style": "casement"}},
            {"id": "win_4", "wall_id": "ext_2", "type": "window", "position": 0.7, "width": 1000, "metadata": {"style": "casement"}},
            {"id": "win_5", "wall_id": "ext_3", "type": "window", "position": 0.2, "width": 1500, "metadata": {"style": "picture"}},
            {"id": "win_6", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 1000, "metadata": {"style": "double_hung"}},
            
            # Interior doors
            {"id": "door_2", "wall_id": "int_1", "type": "door", "position": 0.6, "width": 815, "metadata": {"swing": "inward"}},
            {"id": "door_3", "wall_id": "int_2", "type": "door", "position": 0.3, "width": 815, "metadata": {"swing": "outward"}},
            {"id": "door_4", "wall_id": "int_2", "type": "door", "position": 0.7, "width": 815, "metadata": {"swing": "inward"}},
            {"id": "door_5", "wall_id": "int_3", "type": "door", "position": 0.4, "width": 815, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [9000, 0], [9000, 5000], [3000, 5000], [3000, 3000], [0, 3000]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[0, 3000], [3000, 3000], [3000, 8000], [0, 8000]]},
            {"id": "dining", "name": "Dining Room", "type": "dining", "boundary": [[3000, 5000], [6000, 5000], [6000, 8000], [3000, 8000]]},
            {"id": "master_bed", "name": "Master Bedroom", "type": "bedroom", "boundary": [[9000, 0], [12000, 0], [12000, 5000], [9000, 5000]]},
            {"id": "bed_2", "name": "Bedroom 2", "type": "bedroom", "boundary": [[6000, 5000], [9000, 5000], [9000, 8000], [6000, 8000]]},
            {"id": "bed_3", "name": "Bedroom 3", "type": "bedroom", "boundary": [[9000, 5000], [12000, 5000], [12000, 8000], [9000, 8000]]},
            {"id": "bath_1", "name": "Master Bath", "type": "bathroom", "boundary": [[9000, 3000], [12000, 3000], [12000, 5000], [9000, 5000]]},
            {"id": "bath_2", "name": "Guest Bath", "type": "bathroom", "boundary": [[7500, 5000], [9000, 5000], [9000, 6500], [7500, 6500]]},
        ]
    }

def create_example_2_two_story_colonial():
    """4BR/3BA Two Story Colonial - 2,400 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Two Story Colonial",
            "total_area_sqft": 2400,
            "floors": 2,
            "bedrooms": 4,
            "bathrooms": 3
        },
        "walls": [
            # Exterior perimeter
            {"id": "ext_1", "start": [0, 0], "end": [14000, 0], "type": "exterior", "thickness": 230},
            {"id": "ext_2", "start": [14000, 0], "end": [14000, 10000], "type": "exterior", "thickness": 230},
            {"id": "ext_3", "start": [14000, 10000], "end": [0, 10000], "type": "exterior", "thickness": 230},
            {"id": "ext_4", "start": [0, 10000], "end": [0, 0], "type": "exterior", "thickness": 230},
            
            # First floor interior walls
            {"id": "int_1", "start": [4000, 0], "end": [4000, 6000], "type": "interior", "thickness": 115},
            {"id": "int_2", "start": [4000, 6000], "end": [10000, 6000], "type": "interior", "thickness": 115},
            {"id": "int_3", "start": [10000, 0], "end": [10000, 10000], "type": "interior", "thickness": 115},
            {"id": "int_4", "start": [0, 6000], "end": [4000, 6000], "type": "interior", "thickness": 115},
            {"id": "int_5", "start": [7000, 6000], "end": [7000, 10000], "type": "interior", "thickness": 115},
        ],
        "openings": [
            # Main entrance with sidelights
            {"id": "door_main", "wall_id": "ext_1", "type": "door", "position": 0.5, "width": 1220, "metadata": {"style": "entry_with_sidelights"}},
            
            # Large windows
            {"id": "win_living_1", "wall_id": "ext_1", "type": "window", "position": 0.2, "width": 1800, "metadata": {"style": "bay_window"}},
            {"id": "win_living_2", "wall_id": "ext_1", "type": "window", "position": 0.8, "width": 1200, "metadata": {"style": "double_hung"}},
            {"id": "win_dining", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 1500, "metadata": {"style": "casement"}},
            {"id": "win_kitchen", "wall_id": "ext_3", "type": "window", "position": 0.2, "width": 1200, "metadata": {"style": "double_hung"}},
            {"id": "win_family", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 1800, "metadata": {"style": "sliding"}},
            {"id": "patio_door", "wall_id": "ext_4", "type": "door", "position": 0.7, "width": 1830, "metadata": {"style": "sliding_patio"}},
            
            # Interior doors
            {"id": "door_powder", "wall_id": "int_1", "type": "door", "position": 0.8, "width": 610, "metadata": {"swing": "outward"}},
            {"id": "door_kitchen", "wall_id": "int_2", "type": "door", "position": 0.3, "width": 915, "metadata": {"style": "no_door_opening"}},
            {"id": "door_family", "wall_id": "int_5", "type": "door", "position": 0.5, "width": 915, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "foyer", "name": "Foyer", "type": "entrance", "boundary": [[4000, 0], [6000, 0], [6000, 2000], [4000, 2000]]},
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [4000, 0], [4000, 6000], [0, 6000]]},
            {"id": "dining", "name": "Dining Room", "type": "dining", "boundary": [[10000, 0], [14000, 0], [14000, 6000], [10000, 6000]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[4000, 6000], [7000, 6000], [7000, 10000], [4000, 10000]]},
            {"id": "family", "name": "Family Room", "type": "family", "boundary": [[7000, 6000], [14000, 6000], [14000, 10000], [7000, 10000]]},
            {"id": "powder", "name": "Powder Room", "type": "bathroom", "boundary": [[6000, 0], [10000, 0], [10000, 3000], [6000, 3000]]},
            {"id": "utility", "name": "Utility Room", "type": "utility", "boundary": [[0, 6000], [4000, 6000], [4000, 8000], [0, 8000]]},
            {"id": "pantry", "name": "Pantry", "type": "storage", "boundary": [[0, 8000], [4000, 8000], [4000, 10000], [0, 10000]]},
        ]
    }

def create_example_3_modern_loft():
    """2BR/2BA Modern Loft - 1,800 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Modern Urban Loft",
            "total_area_sqft": 1800,
            "floors": 1,
            "bedrooms": 2,
            "bathrooms": 2
        },
        "walls": [
            # Exterior - rectangular footprint
            {"id": "ext_1", "start": [0, 0], "end": [15000, 0], "type": "exterior", "thickness": 300},
            {"id": "ext_2", "start": [15000, 0], "end": [15000, 8000], "type": "exterior", "thickness": 300},
            {"id": "ext_3", "start": [15000, 8000], "end": [0, 8000], "type": "exterior", "thickness": 300},
            {"id": "ext_4", "start": [0, 8000], "end": [0, 0], "type": "exterior", "thickness": 300},
            
            # Minimal interior walls for open concept
            {"id": "int_1", "start": [10000, 0], "end": [10000, 5000], "type": "interior", "thickness": 90},
            {"id": "int_2", "start": [10000, 5000], "end": [15000, 5000], "type": "interior", "thickness": 90},
            {"id": "int_3", "start": [12500, 5000], "end": [12500, 8000], "type": "interior", "thickness": 90},
        ],
        "openings": [
            # Large glass entrance
            {"id": "entry", "wall_id": "ext_1", "type": "door", "position": 0.1, "width": 1220, "metadata": {"style": "glass_entry"}},
            
            # Floor-to-ceiling windows
            {"id": "win_living_1", "wall_id": "ext_1", "type": "window", "position": 0.4, "width": 3000, "metadata": {"style": "floor_to_ceiling"}},
            {"id": "win_living_2", "wall_id": "ext_1", "type": "window", "position": 0.8, "width": 2000, "metadata": {"style": "floor_to_ceiling"}},
            {"id": "win_dining", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 2500, "metadata": {"style": "floor_to_ceiling"}},
            {"id": "win_bed1", "wall_id": "ext_2", "type": "window", "position": 0.8, "width": 1500, "metadata": {"style": "casement"}},
            {"id": "win_bed2", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 1500, "metadata": {"style": "casement"}},
            {"id": "win_kitchen", "wall_id": "ext_3", "type": "window", "position": 0.2, "width": 2000, "metadata": {"style": "awning"}},
            
            # Minimal interior doors
            {"id": "door_bed1", "wall_id": "int_1", "type": "door", "position": 0.6, "width": 915, "metadata": {"swing": "inward"}},
            {"id": "door_bed2", "wall_id": "int_2", "type": "door", "position": 0.7, "width": 915, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "great_room", "name": "Great Room", "type": "living", "boundary": [[0, 0], [10000, 0], [10000, 5000], [6000, 5000], [6000, 8000], [0, 8000]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[6000, 5000], [10000, 5000], [10000, 8000], [6000, 8000]]},
            {"id": "master_bed", "name": "Master Suite", "type": "bedroom", "boundary": [[10000, 0], [15000, 0], [15000, 5000], [10000, 5000]]},
            {"id": "bedroom_2", "name": "Bedroom 2", "type": "bedroom", "boundary": [[10000, 5000], [12500, 5000], [12500, 8000], [10000, 8000]]},
            {"id": "master_bath", "name": "Master Bath", "type": "bathroom", "boundary": [[13000, 0], [15000, 0], [15000, 3000], [13000, 3000]]},
            {"id": "guest_bath", "name": "Guest Bath", "type": "bathroom", "boundary": [[12500, 5000], [15000, 5000], [15000, 8000], [12500, 8000]]},
        ]
    }

def create_example_4_cottage_bungalow():
    """2BR/1BA Cottage Bungalow - 950 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Cottage Bungalow",
            "total_area_sqft": 950,
            "floors": 1,
            "bedrooms": 2,
            "bathrooms": 1
        },
        "walls": [
            # Compact exterior
            {"id": "ext_1", "start": [0, 0], "end": [10000, 0], "type": "exterior", "thickness": 200},
            {"id": "ext_2", "start": [10000, 0], "end": [10000, 7000], "type": "exterior", "thickness": 200},
            {"id": "ext_3", "start": [10000, 7000], "end": [0, 7000], "type": "exterior", "thickness": 200},
            {"id": "ext_4", "start": [0, 7000], "end": [0, 0], "type": "exterior", "thickness": 200},
            
            # Simple interior layout
            {"id": "int_1", "start": [3500, 0], "end": [3500, 4500], "type": "interior", "thickness": 100},
            {"id": "int_2", "start": [3500, 4500], "end": [7000, 4500], "type": "interior", "thickness": 100},
            {"id": "int_3", "start": [7000, 0], "end": [7000, 7000], "type": "interior", "thickness": 100},
        ],
        "openings": [
            # Cozy front door
            {"id": "front_door", "wall_id": "ext_1", "type": "door", "position": 0.5, "width": 815, "metadata": {"style": "cottage_entry"}},
            
            # Traditional windows
            {"id": "win_living", "wall_id": "ext_1", "type": "window", "position": 0.2, "width": 1000, "metadata": {"style": "cottage_window"}},
            {"id": "win_dining", "wall_id": "ext_1", "type": "window", "position": 0.8, "width": 800, "metadata": {"style": "cottage_window"}},
            {"id": "win_kitchen", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 900, "metadata": {"style": "casement"}},
            {"id": "win_bed1", "wall_id": "ext_2", "type": "window", "position": 0.8, "width": 1000, "metadata": {"style": "double_hung"}},
            {"id": "win_bed2", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 1000, "metadata": {"style": "double_hung"}},
            {"id": "back_door", "wall_id": "ext_4", "type": "door", "position": 0.7, "width": 815, "metadata": {"style": "back_door"}},
            
            # Interior doors
            {"id": "door_bed1", "wall_id": "int_1", "type": "door", "position": 0.7, "width": 710, "metadata": {"swing": "inward"}},
            {"id": "door_bed2", "wall_id": "int_2", "type": "door", "position": 0.6, "width": 710, "metadata": {"swing": "inward"}},
            {"id": "door_bath", "wall_id": "int_3", "type": "door", "position": 0.3, "width": 610, "metadata": {"swing": "outward"}},
        ],
        "spaces": [
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [3500, 0], [3500, 4500], [0, 4500]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[0, 4500], [3500, 4500], [3500, 7000], [0, 7000]]},
            {"id": "dining", "name": "Dining Nook", "type": "dining", "boundary": [[3500, 4500], [5000, 4500], [5000, 7000], [3500, 7000]]},
            {"id": "bedroom_1", "name": "Bedroom 1", "type": "bedroom", "boundary": [[3500, 0], [7000, 0], [7000, 4500], [3500, 4500]]},
            {"id": "bedroom_2", "name": "Bedroom 2", "type": "bedroom", "boundary": [[7000, 0], [10000, 0], [10000, 4500], [7000, 4500]]},
            {"id": "bathroom", "name": "Bathroom", "type": "bathroom", "boundary": [[5000, 4500], [7000, 4500], [7000, 7000], [5000, 7000]]},
            {"id": "utility", "name": "Utility", "type": "utility", "boundary": [[7000, 4500], [10000, 4500], [10000, 7000], [7000, 7000]]},
        ]
    }

def create_example_5_luxury_villa():
    """5BR/4BA Luxury Villa - 3,500 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Luxury Villa",
            "total_area_sqft": 3500,
            "floors": 1,
            "bedrooms": 5,
            "bathrooms": 4
        },
        "walls": [
            # Large exterior perimeter
            {"id": "ext_1", "start": [0, 0], "end": [18000, 0], "type": "exterior", "thickness": 250},
            {"id": "ext_2", "start": [18000, 0], "end": [18000, 12000], "type": "exterior", "thickness": 250},
            {"id": "ext_3", "start": [18000, 12000], "end": [0, 12000], "type": "exterior", "thickness": 250},
            {"id": "ext_4", "start": [0, 12000], "end": [0, 0], "type": "exterior", "thickness": 250},
            
            # Complex interior layout
            {"id": "int_1", "start": [6000, 0], "end": [6000, 8000], "type": "interior", "thickness": 150},
            {"id": "int_2", "start": [6000, 8000], "end": [12000, 8000], "type": "interior", "thickness": 150},
            {"id": "int_3", "start": [12000, 0], "end": [12000, 12000], "type": "interior", "thickness": 150},
            {"id": "int_4", "start": [0, 8000], "end": [6000, 8000], "type": "interior", "thickness": 150},
            {"id": "int_5", "start": [9000, 8000], "end": [9000, 12000], "type": "interior", "thickness": 150},
            {"id": "int_6", "start": [15000, 8000], "end": [15000, 12000], "type": "interior", "thickness": 150},
            {"id": "int_7", "start": [3000, 8000], "end": [3000, 12000], "type": "interior", "thickness": 150},
        ],
        "openings": [
            # Grand entrance
            {"id": "grand_entry", "wall_id": "ext_1", "type": "door", "position": 0.5, "width": 1830, "metadata": {"style": "double_door_entry"}},
            
            # Large windows throughout
            {"id": "win_living_1", "wall_id": "ext_1", "type": "window", "position": 0.2, "width": 2500, "metadata": {"style": "picture_window"}},
            {"id": "win_living_2", "wall_id": "ext_1", "type": "window", "position": 0.8, "width": 2000, "metadata": {"style": "bay_window"}},
            {"id": "win_master", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 2500, "metadata": {"style": "floor_to_ceiling"}},
            {"id": "win_dining", "wall_id": "ext_2", "type": "window", "position": 0.7, "width": 2000, "metadata": {"style": "casement"}},
            {"id": "win_kitchen", "wall_id": "ext_3", "type": "window", "position": 0.2, "width": 1800, "metadata": {"style": "garden_window"}},
            {"id": "win_family", "wall_id": "ext_3", "type": "window", "position": 0.5, "width": 3000, "metadata": {"style": "sliding"}},
            {"id": "win_study", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 1500, "metadata": {"style": "double_hung"}},
            {"id": "patio_doors", "wall_id": "ext_4", "type": "door", "position": 0.4, "width": 2440, "metadata": {"style": "french_doors"}},
            
            # Interior doors
            {"id": "door_study", "wall_id": "int_1", "type": "door", "position": 0.8, "width": 915, "metadata": {"swing": "inward"}},
            {"id": "door_powder", "wall_id": "int_2", "type": "door", "position": 0.2, "width": 710, "metadata": {"swing": "outward"}},
            {"id": "door_master", "wall_id": "int_3", "type": "door", "position": 0.4, "width": 915, "metadata": {"swing": "inward"}},
            {"id": "door_guest1", "wall_id": "int_5", "type": "door", "position": 0.6, "width": 815, "metadata": {"swing": "inward"}},
            {"id": "door_guest2", "wall_id": "int_6", "type": "door", "position": 0.6, "width": 815, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "foyer", "name": "Grand Foyer", "type": "entrance", "boundary": [[6000, 0], [9000, 0], [9000, 3000], [6000, 3000]]},
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [6000, 0], [6000, 8000], [0, 8000]]},
            {"id": "study", "name": "Study", "type": "office", "boundary": [[9000, 0], [12000, 0], [12000, 4000], [9000, 4000]]},
            {"id": "dining", "name": "Formal Dining", "type": "dining", "boundary": [[12000, 0], [18000, 0], [18000, 8000], [12000, 8000]]},
            {"id": "kitchen", "name": "Gourmet Kitchen", "type": "kitchen", "boundary": [[0, 8000], [3000, 8000], [3000, 12000], [0, 12000]]},
            {"id": "family", "name": "Family Room", "type": "family", "boundary": [[3000, 8000], [9000, 8000], [9000, 12000], [3000, 12000]]},
            {"id": "master_suite", "name": "Master Suite", "type": "bedroom", "boundary": [[12000, 8000], [18000, 8000], [18000, 12000], [12000, 12000]]},
            {"id": "guest_bed_1", "name": "Guest Bedroom 1", "type": "bedroom", "boundary": [[9000, 8000], [12000, 8000], [12000, 12000], [9000, 12000]]},
            {"id": "guest_bed_2", "name": "Guest Bedroom 2", "type": "bedroom", "boundary": [[15000, 8000], [18000, 8000], [18000, 12000], [15000, 12000]]},
        ]
    }

def create_example_6_townhouse():
    """3BR/2.5BA Townhouse - 1,850 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm", 
            "project_name": "Urban Townhouse",
            "total_area_sqft": 1850,
            "floors": 2,
            "bedrooms": 3,
            "bathrooms": 2.5
        },
        "walls": [
            # Narrow townhouse footprint
            {"id": "ext_1", "start": [0, 0], "end": [8000, 0], "type": "exterior", "thickness": 230},
            {"id": "ext_2", "start": [8000, 0], "end": [8000, 12000], "type": "exterior", "thickness": 230},
            {"id": "ext_3", "start": [8000, 12000], "end": [0, 12000], "type": "exterior", "thickness": 230},
            {"id": "ext_4", "start": [0, 12000], "end": [0, 0], "type": "exterior", "thickness": 230},
            
            # Efficient interior layout
            {"id": "int_1", "start": [0, 4000], "end": [6000, 4000], "type": "interior", "thickness": 115},
            {"id": "int_2", "start": [6000, 0], "end": [6000, 8000], "type": "interior", "thickness": 115},
            {"id": "int_3", "start": [0, 8000], "end": [8000, 8000], "type": "interior", "thickness": 115},
            {"id": "int_4", "start": [4000, 8000], "end": [4000, 12000], "type": "interior", "thickness": 115},
        ],
        "openings": [
            # Street entrance
            {"id": "front_door", "wall_id": "ext_1", "type": "door", "position": 0.3, "width": 915, "metadata": {"style": "townhouse_entry"}},
            
            # Street-facing windows
            {"id": "win_living", "wall_id": "ext_1", "type": "window", "position": 0.7, "width": 1500, "metadata": {"style": "bay_window"}},
            {"id": "win_kitchen", "wall_id": "ext_2", "type": "window", "position": 0.2, "width": 1000, "metadata": {"style": "casement"}},
            {"id": "win_dining", "wall_id": "ext_2", "type": "window", "position": 0.6, "width": 1200, "metadata": {"style": "double_hung"}},
            {"id": "win_family", "wall_id": "ext_3", "type": "window", "position": 0.5, "width": 1800, "metadata": {"style": "sliding"}},
            {"id": "patio_access", "wall_id": "ext_4", "type": "door", "position": 0.8, "width": 1525, "metadata": {"style": "sliding_door"}},
            
            # Interior circulation
            {"id": "door_powder", "wall_id": "int_1", "type": "door", "position": 0.8, "width": 610, "metadata": {"swing": "outward"}},
            {"id": "door_kitchen", "wall_id": "int_2", "type": "door", "position": 0.6, "width": 915, "metadata": {"style": "opening"}},
            {"id": "stairs_up", "wall_id": "int_3", "type": "door", "position": 0.2, "width": 915, "metadata": {"style": "stair_opening"}},
        ],
        "spaces": [
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [6000, 0], [6000, 4000], [0, 4000]]},
            {"id": "dining", "name": "Dining Area", "type": "dining", "boundary": [[6000, 0], [8000, 0], [8000, 8000], [6000, 8000]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[0, 4000], [4000, 4000], [4000, 8000], [0, 8000]]},
            {"id": "family", "name": "Family Room", "type": "family", "boundary": [[0, 8000], [4000, 8000], [4000, 12000], [0, 12000]]},
            {"id": "breakfast", "name": "Breakfast Nook", "type": "dining", "boundary": [[4000, 8000], [8000, 8000], [8000, 12000], [4000, 12000]]},
            {"id": "powder", "name": "Powder Room", "type": "bathroom", "boundary": [[4000, 4000], [6000, 4000], [6000, 6000], [4000, 6000]]},
            {"id": "stairs", "name": "Stairwell", "type": "circulation", "boundary": [[4000, 6000], [6000, 6000], [6000, 8000], [4000, 8000]]},
        ]
    }

def create_example_7_studio_apartment():
    """Studio Apartment - 650 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Studio Apartment",
            "total_area_sqft": 650,
            "floors": 1,
            "bedrooms": 0,
            "bathrooms": 1
        },
        "walls": [
            # Compact rectangular layout
            {"id": "ext_1", "start": [0, 0], "end": [8500, 0], "type": "exterior", "thickness": 200},
            {"id": "ext_2", "start": [8500, 0], "end": [8500, 7000], "type": "exterior", "thickness": 200},
            {"id": "ext_3", "start": [8500, 7000], "end": [0, 7000], "type": "exterior", "thickness": 200},
            {"id": "ext_4", "start": [0, 7000], "end": [0, 0], "type": "exterior", "thickness": 200},
            
            # Minimal interior walls
            {"id": "int_1", "start": [6000, 0], "end": [6000, 3000], "type": "interior", "thickness": 100},
            {"id": "int_2", "start": [6000, 3000], "end": [8500, 3000], "type": "interior", "thickness": 100},
        ],
        "openings": [
            # Main entrance
            {"id": "entry", "wall_id": "ext_1", "type": "door", "position": 0.2, "width": 815, "metadata": {"style": "apartment_entry"}},
            
            # Large windows for light
            {"id": "win_main", "wall_id": "ext_1", "type": "window", "position": 0.7, "width": 2500, "metadata": {"style": "floor_to_ceiling"}},
            {"id": "win_kitchen", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 1200, "metadata": {"style": "casement"}},
            {"id": "win_living", "wall_id": "ext_3", "type": "window", "position": 0.6, "width": 1800, "metadata": {"style": "sliding"}},
            
            # Interior access
            {"id": "door_bath", "wall_id": "int_1", "type": "door", "position": 0.8, "width": 710, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "main_area", "name": "Living/Sleeping Area", "type": "living", "boundary": [[0, 0], [6000, 0], [6000, 7000], [0, 7000]]},
            {"id": "kitchen", "name": "Kitchenette", "type": "kitchen", "boundary": [[6000, 0], [8500, 0], [8500, 3000], [6000, 3000]]},
            {"id": "bathroom", "name": "Bathroom", "type": "bathroom", "boundary": [[6000, 3000], [8500, 3000], [8500, 7000], [6000, 7000]]},
        ]
    }

def create_example_8_split_level():
    """4BR/3BA Split Level - 2,200 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Split Level Home",
            "total_area_sqft": 2200,
            "floors": 1.5,
            "bedrooms": 4,
            "bathrooms": 3
        },
        "walls": [
            # Split-level exterior
            {"id": "ext_1", "start": [0, 0], "end": [13000, 0], "type": "exterior", "thickness": 230},
            {"id": "ext_2", "start": [13000, 0], "end": [13000, 10000], "type": "exterior", "thickness": 230},
            {"id": "ext_3", "start": [13000, 10000], "end": [0, 10000], "type": "exterior", "thickness": 230},
            {"id": "ext_4", "start": [0, 10000], "end": [0, 0], "type": "exterior", "thickness": 230},
            
            # Split-level interior divisions
            {"id": "int_1", "start": [4000, 0], "end": [4000, 6000], "type": "interior", "thickness": 150},
            {"id": "int_2", "start": [4000, 6000], "end": [9000, 6000], "type": "interior", "thickness": 150},
            {"id": "int_3", "start": [9000, 0], "end": [9000, 10000], "type": "interior", "thickness": 150},
            {"id": "int_4", "start": [0, 6000], "end": [4000, 6000], "type": "interior", "thickness": 150},
            {"id": "int_5", "start": [6500, 6000], "end": [6500, 10000], "type": "interior", "thickness": 150},
        ],
        "openings": [
            # Multiple entries due to split levels
            {"id": "main_entry", "wall_id": "ext_1", "type": "door", "position": 0.4, "width": 915, "metadata": {"style": "main_entry"}},
            {"id": "lower_entry", "wall_id": "ext_1", "type": "door", "position": 0.7, "width": 815, "metadata": {"style": "side_entry"}},
            
            # Windows on all levels
            {"id": "win_living", "wall_id": "ext_1", "type": "window", "position": 0.15, "width": 1500, "metadata": {"style": "picture_window"}},
            {"id": "win_rec", "wall_id": "ext_1", "type": "window", "position": 0.85, "width": 1200, "metadata": {"style": "basement_window"}},
            {"id": "win_master", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 1800, "metadata": {"style": "double_hung"}},
            {"id": "win_bed2", "wall_id": "ext_2", "type": "window", "position": 0.7, "width": 1200, "metadata": {"style": "casement"}},
            {"id": "win_kitchen", "wall_id": "ext_3", "type": "window", "position": 0.2, "width": 1200, "metadata": {"style": "garden_window"}},
            {"id": "win_dining", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 1500, "metadata": {"style": "bay_window"}},
            
            # Interior connections
            {"id": "door_master", "wall_id": "int_3", "type": "door", "position": 0.3, "width": 915, "metadata": {"swing": "inward"}},
            {"id": "door_bed2", "wall_id": "int_2", "type": "door", "position": 0.3, "width": 815, "metadata": {"swing": "inward"}},
            {"id": "door_bed3", "wall_id": "int_5", "type": "door", "position": 0.4, "width": 815, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [4000, 0], [4000, 6000], [0, 6000]]},
            {"id": "rec_room", "name": "Recreation Room", "type": "family", "boundary": [[0, 6000], [4000, 6000], [4000, 10000], [0, 10000]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[4000, 6000], [6500, 6000], [6500, 10000], [4000, 10000]]},
            {"id": "dining", "name": "Dining Room", "type": "dining", "boundary": [[6500, 6000], [9000, 6000], [9000, 10000], [6500, 10000]]},
            {"id": "foyer", "name": "Foyer", "type": "entrance", "boundary": [[4000, 0], [6000, 0], [6000, 3000], [4000, 3000]]},
            {"id": "master_bed", "name": "Master Bedroom", "type": "bedroom", "boundary": [[9000, 0], [13000, 0], [13000, 6000], [9000, 6000]]},
            {"id": "bedroom_2", "name": "Bedroom 2", "type": "bedroom", "boundary": [[4000, 3000], [9000, 3000], [9000, 6000], [4000, 6000]]},
            {"id": "bedroom_3", "name": "Bedroom 3", "type": "bedroom", "boundary": [[9000, 6000], [13000, 6000], [13000, 10000], [9000, 10000]]},
        ]
    }

def create_example_9_mediterranean_villa():
    """4BR/3.5BA Mediterranean Villa - 2,800 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Mediterranean Villa",
            "total_area_sqft": 2800,
            "floors": 1,
            "bedrooms": 4,
            "bathrooms": 3.5
        },
        "walls": [
            # Villa with courtyard design
            {"id": "ext_1", "start": [0, 0], "end": [16000, 0], "type": "exterior", "thickness": 300},
            {"id": "ext_2", "start": [16000, 0], "end": [16000, 14000], "type": "exterior", "thickness": 300},
            {"id": "ext_3", "start": [16000, 14000], "end": [0, 14000], "type": "exterior", "thickness": 300},
            {"id": "ext_4", "start": [0, 14000], "end": [0, 0], "type": "exterior", "thickness": 300},
            
            # Courtyard walls (interior courtyard)
            {"id": "court_1", "start": [6000, 4000], "end": [10000, 4000], "type": "exterior", "thickness": 200},
            {"id": "court_2", "start": [10000, 4000], "end": [10000, 8000], "type": "exterior", "thickness": 200},
            {"id": "court_3", "start": [10000, 8000], "end": [6000, 8000], "type": "exterior", "thickness": 200},
            {"id": "court_4", "start": [6000, 8000], "end": [6000, 4000], "type": "exterior", "thickness": 200},
            
            # Interior walls
            {"id": "int_1", "start": [4000, 0], "end": [4000, 14000], "type": "interior", "thickness": 150},
            {"id": "int_2", "start": [12000, 0], "end": [12000, 14000], "type": "interior", "thickness": 150},
            {"id": "int_3", "start": [0, 4000], "end": [6000, 4000], "type": "interior", "thickness": 150},
            {"id": "int_4", "start": [10000, 4000], "end": [16000, 4000], "type": "interior", "thickness": 150},
            {"id": "int_5", "start": [0, 8000], "end": [6000, 8000], "type": "interior", "thickness": 150},
            {"id": "int_6", "start": [10000, 8000], "end": [16000, 8000], "type": "interior", "thickness": 150},
            {"id": "int_7", "start": [2000, 8000], "end": [2000, 14000], "type": "interior", "thickness": 150},
            {"id": "int_8", "start": [14000, 8000], "end": [14000, 14000], "type": "interior", "thickness": 150},
        ],
        "openings": [
            # Grand entrance
            {"id": "main_entry", "wall_id": "ext_1", "type": "door", "position": 0.5, "width": 1830, "metadata": {"style": "arched_double_doors"}},
            
            # Courtyard access
            {"id": "court_entry_1", "wall_id": "court_1", "type": "door", "position": 0.3, "width": 1220, "metadata": {"style": "french_doors"}},
            {"id": "court_entry_2", "wall_id": "court_3", "type": "door", "position": 0.7, "width": 1220, "metadata": {"style": "french_doors"}},
            
            # Large arched windows
            {"id": "win_living_1", "wall_id": "ext_1", "type": "window", "position": 0.2, "width": 2000, "metadata": {"style": "arched_window"}},
            {"id": "win_living_2", "wall_id": "ext_1", "type": "window", "position": 0.8, "width": 2000, "metadata": {"style": "arched_window"}},
            {"id": "win_master", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 2500, "metadata": {"style": "floor_to_ceiling"}},
            {"id": "win_guest1", "wall_id": "ext_2", "type": "window", "position": 0.7, "width": 1500, "metadata": {"style": "casement"}},
            {"id": "win_kitchen", "wall_id": "ext_3", "type": "window", "position": 0.2, "width": 1800, "metadata": {"style": "arched_window"}},
            {"id": "win_family", "wall_id": "ext_3", "type": "window", "position": 0.8, "width": 2200, "metadata": {"style": "sliding"}},
            {"id": "patio_doors", "wall_id": "ext_4", "type": "door", "position": 0.6, "width": 2440, "metadata": {"style": "folding_glass_doors"}},
            
            # Interior doors
            {"id": "door_study", "wall_id": "int_1", "type": "door", "position": 0.2, "width": 915, "metadata": {"swing": "inward"}},
            {"id": "door_master", "wall_id": "int_2", "type": "door", "position": 0.4, "width": 1070, "metadata": {"swing": "inward"}},
            {"id": "door_guest1", "wall_id": "int_4", "type": "door", "position": 0.3, "width": 815, "metadata": {"swing": "inward"}},
            {"id": "door_guest2", "wall_id": "int_6", "type": "door", "position": 0.7, "width": 815, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "foyer", "name": "Grand Foyer", "type": "entrance", "boundary": [[4000, 0], [8000, 0], [8000, 4000], [4000, 4000]]},
            {"id": "living", "name": "Living Room", "type": "living", "boundary": [[0, 0], [4000, 0], [4000, 4000], [0, 4000]]},
            {"id": "dining", "name": "Formal Dining", "type": "dining", "boundary": [[8000, 0], [12000, 0], [12000, 4000], [8000, 4000]]},
            {"id": "study", "name": "Study", "type": "office", "boundary": [[12000, 0], [16000, 0], [16000, 4000], [12000, 4000]]},
            {"id": "courtyard", "name": "Courtyard", "type": "outdoor", "boundary": [[6000, 4000], [10000, 4000], [10000, 8000], [6000, 8000]]},
            {"id": "kitchen", "name": "Gourmet Kitchen", "type": "kitchen", "boundary": [[0, 4000], [4000, 4000], [4000, 8000], [0, 8000]]},
            {"id": "family", "name": "Family Room", "type": "family", "boundary": [[4000, 4000], [6000, 4000], [6000, 8000], [4000, 8000]]},
            {"id": "master_suite", "name": "Master Suite", "type": "bedroom", "boundary": [[12000, 4000], [16000, 4000], [16000, 8000], [12000, 8000]]},
            {"id": "guest_bed_1", "name": "Guest Bedroom 1", "type": "bedroom", "boundary": [[10000, 4000], [12000, 4000], [12000, 8000], [10000, 8000]]},
            {"id": "guest_bed_2", "name": "Guest Bedroom 2", "type": "bedroom", "boundary": [[0, 8000], [2000, 8000], [2000, 14000], [0, 14000]]},
            {"id": "guest_bed_3", "name": "Guest Bedroom 3", "type": "bedroom", "boundary": [[14000, 8000], [16000, 8000], [16000, 14000], [14000, 14000]]},
            {"id": "breakfast", "name": "Breakfast Nook", "type": "dining", "boundary": [[2000, 8000], [4000, 8000], [4000, 14000], [2000, 14000]]},
            {"id": "media", "name": "Media Room", "type": "entertainment", "boundary": [[4000, 8000], [12000, 8000], [12000, 14000], [4000, 14000]]},
            {"id": "flex", "name": "Flex Room", "type": "multipurpose", "boundary": [[12000, 8000], [14000, 8000], [14000, 14000], [12000, 14000]]},
        ]
    }

def create_example_10_farmhouse():
    """3BR/2.5BA Modern Farmhouse - 2,100 sq ft"""
    return {
        "metadata": {
            "scale": 100,
            "units": "mm",
            "project_name": "Modern Farmhouse",
            "total_area_sqft": 2100,
            "floors": 1,
            "bedrooms": 3,
            "bathrooms": 2.5
        },
        "walls": [
            # Farmhouse rectangular footprint
            {"id": "ext_1", "start": [0, 0], "end": [15000, 0], "type": "exterior", "thickness": 250},
            {"id": "ext_2", "start": [15000, 0], "end": [15000, 9000], "type": "exterior", "thickness": 250},
            {"id": "ext_3", "start": [15000, 9000], "end": [0, 9000], "type": "exterior", "thickness": 250},
            {"id": "ext_4", "start": [0, 9000], "end": [0, 0], "type": "exterior", "thickness": 250},
            
            # Open concept with strategic walls
            {"id": "int_1", "start": [5000, 0], "end": [5000, 5500], "type": "interior", "thickness": 120},
            {"id": "int_2", "start": [5000, 5500], "end": [10000, 5500], "type": "interior", "thickness": 120},
            {"id": "int_3", "start": [10000, 0], "end": [10000, 9000], "type": "interior", "thickness": 120},
            {"id": "int_4", "start": [7500, 5500], "end": [7500, 9000], "type": "interior", "thickness": 120},
            {"id": "int_5", "start": [12500, 5500], "end": [12500, 9000], "type": "interior", "thickness": 120},
        ],
        "openings": [
            # Farmhouse front porch entry
            {"id": "front_entry", "wall_id": "ext_1", "type": "door", "position": 0.6, "width": 1070, "metadata": {"style": "farmhouse_entry"}},
            
            # Large farmhouse windows
            {"id": "win_living_1", "wall_id": "ext_1", "type": "window", "position": 0.2, "width": 1800, "metadata": {"style": "farmhouse_double_hung"}},
            {"id": "win_living_2", "wall_id": "ext_1", "type": "window", "position": 0.8, "width": 1500, "metadata": {"style": "farmhouse_double_hung"}},
            {"id": "win_master", "wall_id": "ext_2", "type": "window", "position": 0.3, "width": 2000, "metadata": {"style": "farmhouse_casement"}},
            {"id": "win_bed2", "wall_id": "ext_2", "type": "window", "position": 0.8, "width": 1200, "metadata": {"style": "farmhouse_double_hung"}},
            {"id": "win_kitchen", "wall_id": "ext_3", "type": "window", "position": 0.15, "width": 1500, "metadata": {"style": "farmhouse_casement"}},
            {"id": "win_dining", "wall_id": "ext_3", "type": "window", "position": 0.5, "width": 2200, "metadata": {"style": "farmhouse_picture"}},
            {"id": "win_bed3", "wall_id": "ext_3", "type": "window", "position": 0.85, "width": 1200, "metadata": {"style": "farmhouse_double_hung"}},
            {"id": "back_porch", "wall_id": "ext_4", "type": "door", "position": 0.4, "width": 1830, "metadata": {"style": "sliding_patio"}},
            
            # Interior openings
            {"id": "door_powder", "wall_id": "int_1", "type": "door", "position": 0.8, "width": 610, "metadata": {"swing": "outward"}},
            {"id": "door_master", "wall_id": "int_3", "type": "door", "position": 0.3, "width": 915, "metadata": {"swing": "inward"}},
            {"id": "door_bed2", "wall_id": "int_2", "type": "door", "position": 0.3, "width": 815, "metadata": {"swing": "inward"}},
            {"id": "door_bed3", "wall_id": "int_4", "type": "door", "position": 0.6, "width": 815, "metadata": {"swing": "inward"}},
        ],
        "spaces": [
            {"id": "great_room", "name": "Great Room", "type": "living", "boundary": [[0, 0], [5000, 0], [5000, 5500], [0, 5500]]},
            {"id": "kitchen", "name": "Kitchen", "type": "kitchen", "boundary": [[0, 5500], [5000, 5500], [5000, 9000], [0, 9000]]},
            {"id": "dining", "name": "Dining Room", "type": "dining", "boundary": [[5000, 5500], [7500, 5500], [7500, 9000], [5000, 9000]]},
            {"id": "pantry", "name": "Pantry", "type": "storage", "boundary": [[7500, 5500], [10000, 5500], [10000, 7000], [7500, 7000]]},
            {"id": "mudroom", "name": "Mudroom", "type": "utility", "boundary": [[7500, 7000], [10000, 7000], [10000, 9000], [7500, 9000]]},
            {"id": "foyer", "name": "Foyer", "type": "entrance", "boundary": [[5000, 0], [7500, 0], [7500, 2500], [5000, 2500]]},
            {"id": "powder", "name": "Powder Room", "type": "bathroom", "boundary": [[5000, 2500], [7500, 2500], [7500, 5500], [5000, 5500]]},
            {"id": "master_suite", "name": "Master Suite", "type": "bedroom", "boundary": [[10000, 0], [15000, 0], [15000, 5500], [10000, 5500]]},
            {"id": "bedroom_2", "name": "Bedroom 2", "type": "bedroom", "boundary": [[7500, 0], [10000, 0], [10000, 5500], [7500, 5500]]},
            {"id": "bedroom_3", "name": "Bedroom 3", "type": "bedroom", "boundary": [[12500, 5500], [15000, 5500], [15000, 9000], [12500, 9000]]},
            {"id": "master_bath", "name": "Master Bath", "type": "bathroom", "boundary": [[10000, 5500], [12500, 5500], [12500, 9000], [10000, 9000]]},
        ]
    }

def main():
    """Generate all 10 professional examples."""
    examples = [
        ("example_01_single_story_ranch.json", create_example_1_single_story_ranch()),
        ("example_02_two_story_colonial.json", create_example_2_two_story_colonial()),
        ("example_03_modern_loft.json", create_example_3_modern_loft()),
        ("example_04_cottage_bungalow.json", create_example_4_cottage_bungalow()),
        ("example_05_luxury_villa.json", create_example_5_luxury_villa()),
        ("example_06_townhouse.json", create_example_6_townhouse()),
        ("example_07_studio_apartment.json", create_example_7_studio_apartment()),
        ("example_08_split_level.json", create_example_8_split_level()),
        ("example_09_mediterranean_villa.json", create_example_9_mediterranean_villa()),
        ("example_10_farmhouse.json", create_example_10_farmhouse()),
    ]
    
    # Create output directory
    output_dir = Path("professional_examples")
    output_dir.mkdir(exist_ok=True)
    
    print("🏗️  Generating 10 Professional Architectural Examples...")
    print("=" * 60)
    
    for filename, data in examples:
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        metadata = data["metadata"]
        print(f"✅ {filename}")
        print(f"   📐 {metadata['project_name']}")
        print(f"   🏠 {metadata['total_area_sqft']:,} sq ft | {metadata['bedrooms']}BR/{metadata['bathrooms']}BA")
        print(f"   📏 {len(data['walls'])} walls | {len(data['openings'])} openings | {len(data['spaces'])} spaces")
        print()
    
    print("🎯 All examples generated successfully!")
    print(f"📁 Files saved in: {output_dir.absolute()}")
    print()
    print("🔧 Usage:")
    print("   # Generate 2D SVG:")
    print("   python src/housebrain/plan_renderer.py --input professional_examples/example_01_single_story_ranch.json --output test_2d.svg")
    print()
    print("   # Generate 3D OBJ:")
    print("   python src/housebrain/export_plan_dxf.py --input professional_examples/example_01_single_story_ranch.json --output test_3d.obj")

if __name__ == "__main__":
    main()
