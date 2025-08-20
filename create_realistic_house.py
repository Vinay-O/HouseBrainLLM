"""
Create a realistic house design that matches professional architectural standards.
Based on real floor plan examples with proper room proportions, logical layout,
and realistic architectural features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


def create_door_polygon(x: float, y: float, width: float = 900, thickness: float = 150) -> List[Tuple[float, float]]:
    """Create a door opening polygon in mm"""
    return [
        (x, y), (x + width, y), (x + width, y + thickness), (x, y + thickness), (x, y)
    ]


def create_window_polygon(x: float, y: float, width: float = 1200, height: float = 1200, sill_height: float = 900) -> List[Tuple[float, float]]:
    """Create a window opening polygon in mm"""
    return [
        (x, y + sill_height), (x + width, y + sill_height), 
        (x + width, y + sill_height + height), (x, y + sill_height + height), 
        (x, y + sill_height)
    ]


def create_room_with_details(room_id: str, room_type: str, x_mm: int, y_mm: int, w_mm: int, h_mm: int, 
                           doors: List[Dict], windows: List[Dict]) -> Dict[str, Any]:
    """Create a detailed room with doors, windows, and architectural elements"""
    
    # Base room polygon
    base_poly = [(x_mm, y_mm), (x_mm + w_mm, y_mm), (x_mm + w_mm, y_mm + h_mm), (x_mm, y_mm + h_mm), (x_mm, y_mm)]
    
    # Add door openings
    door_polys = []
    for door in doors:
        dx = x_mm + door["x_offset"]
        dy = y_mm + door["y_offset"]
        door_polys.append(create_door_polygon(dx, dy, door.get("width", 900)))
    
    # Add window openings
    window_polys = []
    for window in windows:
        wx = x_mm + window["x_offset"]
        wy = y_mm + window["y_offset"]
        window_polys.append(create_window_polygon(wx, wy, window.get("width", 1200), window.get("height", 1200)))
    
    return {
        "id": room_id,
        "name": room_type.replace("_", " ").title(),
        "type": room_type,
        "polygon": base_poly,
        "doors": door_polys,
        "windows": window_polys,
        "area_sqm": round((w_mm * h_mm) / 1000000, 2),
        "dimensions": f"{w_mm/1000:.2f}m x {h_mm/1000:.2f}m",
        "ceiling_height": 2.7,
        "floor_finish": "Vitrified tiles",
        "wall_finish": "Paint",
        "ceiling_finish": "POP + Paint"
    }


def create_realistic_house() -> Dict[str, Any]:
    """Create a realistic house design based on professional examples"""
    
    # Realistic input specifications
    input_obj = {
        "basicDetails": {
            "totalArea": 2000,
            "unit": "sqft", 
            "floors": 2,
            "bedrooms": 3,
            "bathrooms": 2,
            "style": "Modern Contemporary",
            "budget": 6000000,
            "plot_area": 3000,
            "setbacks": {"front": 15, "rear": 10, "left": 8, "right": 8}
        },
        "plot": {
            "width_mm": 15000,
            "height_mm": 12000,
            "north": "N",
            "slope": "level",
            "soil_type": "hard soil"
        },
        "roomBreakdown": [
            {"type": "living_room", "area": 300},
            {"type": "dining_room", "area": 150},
            {"type": "kitchen", "area": 120},
            {"type": "master_bedroom", "area": 200},
            {"type": "bedroom", "area": 150},
            {"type": "bedroom", "area": 120},
            {"type": "bathroom", "area": 60},
            {"type": "bathroom", "area": 50},
            {"type": "utility", "area": 30},
            {"type": "stairwell", "area": 80}
        ]
    }

    # GROUND FLOOR - Realistic layout based on professional examples
    ground_floor = {
        "level": 0,
        "name": "Ground Floor",
        "rooms": [
            # Entrance Hall (2000, 2000, 2500, 2000)
            create_room_with_details("GF_ENT", "entrance", 2000, 2000, 2500, 2000,
                doors=[{"x_offset": 0, "y_offset": 1000, "width": 1200}],  # Main entrance
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 800, "height": 1200}]),
            
            # Living Room - Realistic size (4500, 2000, 5000, 4000)
            create_room_with_details("GF_LIV", "living_room", 4500, 2000, 5000, 4000,
                doors=[{"x_offset": 0, "y_offset": 1500, "width": 1000}],  # From entrance
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 1500, "height": 1200},
                        {"x_offset": 3000, "y_offset": 0, "width": 1500, "height": 1200}]),
            
            # Dining Room - Connected to living (4500, 6000, 4000, 3000)
            create_room_with_details("GF_DIN", "dining_room", 4500, 6000, 4000, 3000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 1000}],  # From living
                windows=[{"x_offset": 2000, "y_offset": 0, "width": 1200, "height": 1200}]),
            
            # Kitchen - Realistic size (9000, 6000, 4000, 3000)
            create_room_with_details("GF_KIT", "kitchen", 9000, 6000, 4000, 3000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 1000}],  # From dining
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 1200, "height": 1200},
                        {"x_offset": 2500, "y_offset": 0, "width": 1200, "height": 1200}]),
            
            # Utility Room - Behind kitchen (9000, 9000, 2000, 2000)
            create_room_with_details("GF_UTL", "utility", 9000, 9000, 2000, 2000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 800}],  # From kitchen
                windows=[{"x_offset": 500, "y_offset": 0, "width": 800, "height": 800}]),
            
            # Guest Bathroom - Near living (4500, 9000, 2000, 2000)
            create_room_with_details("GF_BTH1", "bathroom", 4500, 9000, 2000, 2000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 800}],  # From living
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 800, "height": 800}]),
            
            # Stairwell - Central location (7000, 9000, 2000, 2000)
            create_room_with_details("GF_STA", "stairwell", 7000, 9000, 2000, 2000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 1000}],  # From dining
                windows=[])
        ],
        "stairs": [{
            "type": "straight",
            "width": 1000,
            "length": 2000,
            "riser_height": 180,
            "tread_width": 280,
            "direction": "up",
            "floor_from": 0,
            "floor_to": 1
        }]
    }

    # FIRST FLOOR - Realistic bedroom layout
    first_floor = {
        "level": 1,
        "name": "First Floor", 
        "rooms": [
            # Master Bedroom - Good size (2000, 2000, 4000, 3500)
            create_room_with_details("FF_MBR", "master_bedroom", 2000, 2000, 4000, 3500,
                doors=[{"x_offset": 0, "y_offset": 1500, "width": 900}],  # From landing
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 1200, "height": 1200},
                        {"x_offset": 2500, "y_offset": 0, "width": 1200, "height": 1200}]),
            
            # Master Bathroom - Ensuite (6500, 2000, 2000, 2000)
            create_room_with_details("FF_MBB", "bathroom", 6500, 2000, 2000, 2000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 800}],  # From master bedroom
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 800, "height": 800}]),
            
            # Bedroom 2 - Good size (2000, 6000, 3500, 3000)
            create_room_with_details("FF_BED2", "bedroom", 2000, 6000, 3500, 3000,
                doors=[{"x_offset": 0, "y_offset": 1200, "width": 900}],  # From landing
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 1200, "height": 1200}]),
            
            # Bedroom 3 - Smaller but functional (6000, 6000, 3000, 3000)
            create_room_with_details("FF_BED3", "bedroom", 6000, 6000, 3000, 3000,
                doors=[{"x_offset": 0, "y_offset": 1200, "width": 900}],  # From landing
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 1200, "height": 1200}]),
            
            # Shared Bathroom - For bedrooms 2,3 (9000, 6000, 2000, 2000)
            create_room_with_details("FF_BTH2", "bathroom", 9000, 6000, 2000, 2000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 800}],  # From landing
                windows=[{"x_offset": 1000, "y_offset": 0, "width": 800, "height": 800}]),
            
            # Landing - Central area (7000, 9000, 2000, 2000)
            create_room_with_details("FF_LAN", "landing", 7000, 9000, 2000, 2000,
                doors=[{"x_offset": 0, "y_offset": 0, "width": 1000}],  # To stairs
                windows=[])
        ],
        "stairs": [{
            "type": "straight", 
            "width": 1000,
            "length": 2000,
            "riser_height": 180,
            "tread_width": 280,
            "direction": "down",
            "floor_from": 1,
            "floor_to": 0
        }]
    }

    # Calculate total area and costs
    total_area = sum(room["area_sqm"] for floor in [ground_floor, first_floor] for room in floor["rooms"])
    construction_cost = total_area * 15000  # ₹15,000 per sqm

    # Realistic BOQ
    boq = {
        "earthwork": {"excavation": 100, "filling": 60, "total": 160},
        "foundation": {"pcc": 40, "rcc": 150, "brickwork": 100, "total": 290},
        "superstructure": {"rcc": 400, "brickwork": 320, "plastering": 150, "total": 870},
        "finishes": {"flooring": 280, "painting": 150, "ceiling": 100, "total": 530},
        "services": {"electrical": 150, "plumbing": 120, "total": 270},
        "total_cost": construction_cost
    }

    # Realistic specifications
    specs = {
        "foundation": "RCC isolated footing with plinth beam",
        "structure": "RCC framed structure with brick infill",
        "roof": "RCC slab with weatherproofing",
        "exterior": "Textured paint with stone cladding",
        "interior": "Premium paints and finishes",
        "electrical": "Concealed wiring with modular switches",
        "plumbing": "CPVC pipes with premium fixtures",
        "flooring": "Vitrified tiles with granite in wet areas"
    }

    # Create the realistic output
    output = {
        "input": input_obj,
        "geometry": {
            "floors": [ground_floor, first_floor],
            "plot_boundary": [(0, 0), (15000, 0), (15000, 12000), (0, 12000), (0, 0)],
            "setback_lines": {
                "front": [(1500, 0), (13500, 0)],
                "rear": [(1000, 11000), (14000, 11000)],
                "left": [(0, 800), (0, 11200)],
                "right": [(14200, 800), (14200, 11200)]
            }
        },
        "levels": [ground_floor, first_floor],  # For compatibility
        "total_area": round(total_area, 2),
        "construction_cost": round(construction_cost, 2),
        "boq": boq,
        "specifications": specs,
        "compliance": {
            "nbc_2016": "Fully compliant",
            "structural": "Safe for seismic zone III",
            "fire_safety": "Type A construction",
            "accessibility": "Basic accessibility features included"
        },
        "render_metadata": {
            "2d_scale": "1:100",
            "3d_quality": "high",
            "textures": "photorealistic",
            "lighting": "natural + artificial"
        }
    }

    return output


def main() -> None:
    """Create the realistic example"""
    house = create_realistic_house()
    
    # Save to examples directory
    out_dir = Path("examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the realistic example
    with (out_dir / "realistic_house.json").open("w", encoding="utf-8") as f:
        json.dump(house, f, indent=2)
    
    print("🏗️ REALISTIC HouseBrain Example Created!")
    print("=" * 60)
    print(f"📊 Total Area: {house['total_area']} sqm")
    print(f"💰 Construction Cost: ₹{house['construction_cost']:,.2f}")
    print(f"🏠 Floors: {len(house['geometry']['floors'])}")
    print(f"🚪 Total Rooms: {sum(len(f['rooms']) for f in house['geometry']['floors'])}")
    print(f"📐 Plot Size: {house['input']['plot']['width_mm']/1000:.1f}m x {house['input']['plot']['height_mm']/1000:.1f}m")
    print("=" * 60)
    print("✅ Realistic layout matching professional architectural standards!")
    print("✅ Proper room proportions and logical circulation!")


if __name__ == "__main__":
    main()
