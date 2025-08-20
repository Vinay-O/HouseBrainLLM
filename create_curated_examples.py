"""
Create 20 curated HouseBrain example outputs to unblock pipeline work.

Outputs go to examples/sXX_*.json with both:
- geometry.floors[*].rooms[*].polygon in mm (for exporters)
- levels[] with rectangle bounds in feet (for validators)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


def rect_poly_mm(x_mm: int, y_mm: int, w_mm: int, h_mm: int) -> List[Tuple[int, int]]:
    return [(x_mm, y_mm), (x_mm + w_mm, y_mm), (x_mm + w_mm, y_mm + h_mm), (x_mm, y_mm + h_mm), (x_mm, y_mm)]


def ft(v_mm: int) -> float:
    return round(v_mm / 304.8, 2)


def add_room(level: Dict[str, Any], rid: str, rtype: str, x_mm: int, y_mm: int, w_mm: int, h_mm: int) -> None:
    # polygon (mm)
    level.setdefault("rooms", []).append({
        "name": rtype,
        "polygon": rect_poly_mm(x_mm, y_mm, w_mm, h_mm)
    })


def add_room_rect(level_rect: Dict[str, Any], rid: str, rtype: str, x_mm: int, y_mm: int, w_mm: int, h_mm: int) -> None:
    level_rect.setdefault("rooms", []).append({
        "id": rid,
        "type": rtype,
        "bounds": {"x": ft(x_mm), "y": ft(y_mm), "width": ft(w_mm), "height": ft(h_mm)},
        "doors": [],
        "windows": []
    })


def make_example(idx: int, floors: int = 1, plot_w_mm: int = 12000, plot_h_mm: int = 9000) -> Dict[str, Any]:
    # Basic input scaffold
    input_obj = {
        "basicDetails": {
            "totalArea": 1200, "unit": "sqft", "floors": floors, "bedrooms": 3, "bathrooms": 2,
            "style": "Modern Contemporary", "budget": 3500000
        },
        "plot": {"width_mm": plot_w_mm, "height_mm": plot_h_mm, "north": "N"},
        "roomBreakdown": [
            {"type": "living_room"}, {"type": "dining_room"}, {"type": "kitchen"},
            {"type": "bedroom"}, {"type": "bedroom"}, {"type": "bathroom"}
        ]
    }

    # Geometry (mm)
    floors_geo: List[Dict[str, Any]] = []
    # Rect levels (ft)
    levels_rect: List[Dict[str, Any]] = []

    for f in range(floors):
        lvl = {"level": f, "rooms": []}
        lvrect = {"level_number": f, "rooms": [], "stairs": [], "height": 10.0}

        # Simple layout variants across examples
        base_x = 1000
        base_y = 1000
        grid = 4000
        # living
        add_room(lvl, f"F{f}_LIV", "living_room", base_x, base_y, grid, grid)
        add_room_rect(lvrect, f"F{f}_LIV", "living_room", base_x, base_y, grid, grid)
        # dining
        add_room(lvl, f"F{f}_DIN", "dining_room", base_x + grid, base_y, grid, grid)
        add_room_rect(lvrect, f"F{f}_DIN", "dining_room", base_x + grid, base_y, grid, grid)
        # kitchen smaller
        add_room(lvl, f"F{f}_KIT", "kitchen", base_x + 2 * grid, base_y, int(grid * 0.8), int(grid * 0.8))
        add_room_rect(lvrect, f"F{f}_KIT", "kitchen", base_x + 2 * grid, base_y, int(grid * 0.8), int(grid * 0.8))
        # bedroom
        add_room(lvl, f"F{f}_BED1", "bedroom", base_x, base_y + grid, grid, grid)
        add_room_rect(lvrect, f"F{f}_BED1", "bedroom", base_x, base_y + grid, grid, grid)
        # bedroom 2
        add_room(lvl, f"F{f}_BED2", "bedroom", base_x + grid, base_y + grid, grid, grid)
        add_room_rect(lvrect, f"F{f}_BED2", "bedroom", base_x + grid, base_y + grid, grid, grid)
        # bath
        add_room(lvl, f"F{f}_BTH", "bathroom", base_x + 2 * grid, base_y + grid, int(grid * 0.6), int(grid * 0.6))
        add_room_rect(lvrect, f"F{f}_BTH", "bathroom", base_x + 2 * grid, base_y + grid, int(grid * 0.6), int(grid * 0.6))

        floors_geo.append(lvl)
        levels_rect.append(lvrect)

    total_area = round(sum(r["bounds"]["width"] * r["bounds"]["height"] for L in levels_rect for r in L["rooms"]), 1)

    output = {
        "geometry": {"floors": floors_geo}
    }

    house = {
        "input": input_obj,
        "levels": levels_rect,
        "total_area": total_area,
        "construction_cost": round(total_area * 2200, 2),
        "materials": {"brick_cuft": round(total_area * 1.2, 1)},
        "output": output,
    }
    return house


def main() -> None:
    out_dir = Path("examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (1, 12000, 9000), (1, 14000, 9000), (2, 12000, 9000), (2, 16000, 10000),
        (1, 10000, 8000), (1, 12000, 7000), (2, 12000, 12000), (1, 15000, 10000),
        (2, 14000, 9000), (1, 11000, 9000), (2, 15000, 11000), (1, 13000, 8500),
        (1, 9000, 9000), (2, 9000, 12000), (1, 16000, 12000), (2, 16000, 14000),
        (1, 12000, 12000), (2, 18000, 12000), (1, 10000, 10000), (2, 14000, 14000),
    ]
    for i, (floors, pw, ph) in enumerate(specs, start=1):
        ex = make_example(i, floors=floors, plot_w_mm=pw, plot_h_mm=ph)
        path = out_dir / f"s{i:02d}_{floors}f.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(ex, f, indent=2)
    print(f"✅ Wrote {len(specs)} curated examples to {out_dir}")


if __name__ == "__main__":
    main()


