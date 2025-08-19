"""
Very simple OBJ exporter that turns room polygons into extruded slabs (height=3m).
This is a visualization aid; not a full BIM export.

Usage:
python export_obj.py --input sample_output.json --output plan.obj --scale 0.001 --height 3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from geometry_utils import load_housebrain_output, scale_polygon


def _write_obj(
    path: str,
    faces: List[List[int]],
    vertices: List[Tuple[float, float, float]],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for face in faces:
            idxs = " ".join(str(i) for i in face)
            f.write(f"f {idxs}\n")


def export_to_obj(input_path: str, output_path: str, scale: float = 1.0, height: float = 3.0) -> None:
    data = load_housebrain_output(input_path)

    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []

    def add_prism(poly2d: List[Tuple[float, float]]):
        nonlocal vertices, faces
        base = poly2d[:-1] if poly2d and poly2d[0] == poly2d[-1] else poly2d
        n = len(base)
        if n < 3:
            return
        start_idx = len(vertices) + 1
        # bottom
        for x, y in base:
            vertices.append((x, y, 0.0))
        # top
        for x, y in base:
            vertices.append((x, y, height))
        # sides
        for i in range(n):
            a = start_idx + i
            b = start_idx + (i + 1) % n
            c = start_idx + (i + 1) % n + n
            d = start_idx + i + n
            faces.append([a, b, c, d])
        # caps (fan)
        cap = [start_idx + i for i in range(n)]
        faces.append(cap[::-1])  # bottom
        cap_top = [start_idx + n + i for i in range(n)]
        faces.append(cap_top)

    for floor in data.get("floors", []):
        for room in floor.get("rooms", []):
            poly = room.get("polygon") or []
            if scale != 1.0:
                poly = scale_polygon(poly, scale)
            add_prism(poly)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _write_obj(output_path, faces, vertices)
    print(f"✅ OBJ written to {output_path} (vertices={len(vertices)}, faces={len(faces)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export HouseBrain JSON to simple OBJ prisms")
    ap.add_argument("--input", required=True, help="Path to HouseBrain output JSON")
    ap.add_argument("--output", required=True, help="OBJ file path")
    ap.add_argument("--scale", type=float, default=1.0, help="Coordinate scale (e.g., 0.001 for mm->m)")
    ap.add_argument("--height", type=float, default=3.0, help="Extrusion height (meters)")
    args = ap.parse_args()

    export_to_obj(args.input, args.output, args.scale, args.height)


if __name__ == "__main__":
    main()


