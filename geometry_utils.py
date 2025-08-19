"""
Lightweight geometry helpers to parse HouseBrain output JSON and provide
normalized 2D polygons for export.

Assumptions:
- Coordinates are in millimeters (mm). We expose a scale factor to convert.
- We look for polygons in several common locations/keys to be tolerant of
  model variations: geometry.floors[*].rooms[*].polygon | outline | boundary

Returned structure:
{
    "floors": [
        {
            "level": 0,
            "rooms": [
                {"name": str, "polygon": List[Tuple[float, float]]}
            ]
        },
        ...
    ]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _get_first_polygon(room_like: Dict[str, Any]) -> List[Tuple[float, float]] | None:
    """Extract a polygon from a room-like dict by checking common keys."""
    for key in ("polygon", "outline", "boundary", "points"):
        value = room_like.get(key)
        if isinstance(value, list) and len(value) >= 3:
            try:
                poly = [(float(x), float(y)) for x, y in value]
                return poly
            except Exception:
                continue
    return None


def _normalize_polygon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ensure last point closes the polygon; remove duplicates at the end."""
    if not poly:
        return poly
    if poly[0] != poly[-1]:
        poly = poly + [poly[0]]
    return poly


def _coerce_floors(output_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find floors list in several possible locations."""
    # Common: output["geometry"]["floors"] or output["floors"]
    floors = []
    if isinstance(output_obj, dict):
        if isinstance(output_obj.get("geometry"), dict):
            g_floors = output_obj["geometry"].get("floors")
            if isinstance(g_floors, list):
                floors = g_floors
        if not floors and isinstance(output_obj.get("floors"), list):
            floors = output_obj["floors"]
        # Single-floor fallback with top-level rooms
        if not floors and isinstance(output_obj.get("rooms"), list):
            floors = [{"level": 0, "rooms": output_obj["rooms"]}]
    return floors if isinstance(floors, list) else []


def load_housebrain_output(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file and extract floor/room polygons in a normalized shape."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Some datasets wrap the payload under {"output": {...}}
    output_obj = raw.get("output", raw)

    floors_src = _coerce_floors(output_obj)
    result: Dict[str, Any] = {"floors": []}

    for floor in floors_src:
        level = floor.get("level") if isinstance(floor, dict) else None
        level = 0 if level is None else level
        rooms_list = []
        for room in (floor.get("rooms") or []):
            if not isinstance(room, dict):
                continue
            name = str(room.get("name") or room.get("type") or "Room")
            poly = _get_first_polygon(room)
            if poly and len(poly) >= 3:
                rooms_list.append({"name": name, "polygon": _normalize_polygon(poly)})
        if rooms_list:
            result["floors"].append({"level": level, "rooms": rooms_list})

    return result


def scale_polygon(poly: List[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    """Scale polygon coordinates by a factor (e.g., 0.001 to convert mm->m)."""
    return [(x * scale, y * scale) for x, y in poly]


def polygons_bbox(polys: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    """Compute bounding box (minx, miny, maxx, maxy) for a list of polygons."""
    xs: List[float] = []
    ys: List[float] = []
    for poly in polys:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else (0.0, 0.0, 0.0, 0.0)


