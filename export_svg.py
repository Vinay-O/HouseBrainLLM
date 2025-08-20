"""
Create an inline SVG from HouseBrain output polygons for direct frontend display.

Usage:
python export_svg.py --input sample.json --output out/plan.svg --width 1200 --height 800 --scale 0.001

You can also import get_svg_string(...) to embed directly in API responses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from geometry_utils import load_housebrain_output, scale_polygon, polygons_bbox


def _transform(polys: List[List[Tuple[float, float]]], width: int, height: int, margin: int = 24):
    minx, miny, maxx, maxy = polygons_bbox(polys)
    if maxx - minx < 1e-6 or maxy - miny < 1e-6:
        sx = sy = 1.0
    else:
        sx = (width - 2 * margin) / (maxx - minx)
        sy = (height - 2 * margin) / (maxy - miny)
    s = min(sx, sy)
    tx = margin - minx * s
    ty = margin + maxy * s  # flip Y

    transformed: List[List[Tuple[float, float]]] = []
    for poly in polys:
        pts = []
        for x, y in poly:
            X = x * s + tx
            Y = -y * s + ty
            pts.append((X, Y))
        transformed.append(pts)
    return transformed


def get_svg_string(input_path: str, width: int = 1200, height: int = 800, scale: float = 1.0) -> str:
    data = load_housebrain_output(input_path)
    polys: List[List[Tuple[float, float]]] = []
    for floor in data.get("floors", []):
        for room in floor.get("rooms", []):
            poly = room.get("polygon") or []
            if scale != 1.0:
                poly = scale_polygon(poly, scale)
            if len(poly) >= 4:
                polys.append(poly)
    tpolys = _transform(polys, width, height)

    # Build SVG
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
             "<g fill='none' stroke='#111' stroke-width='2'>"]
    for poly in tpolys:
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in poly) + " Z"
        parts.append(f"<path d='{d}' />")
    parts.append("</g></svg>")
    return "".join(parts)


def export_to_svg_file(input_path: str, output_path: str, width: int = 1200, height: int = 800, scale: float = 1.0) -> None:
    svg = get_svg_string(input_path, width, height, scale)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"✅ SVG written to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export inline SVG from HouseBrain output JSON")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--scale", type=float, default=1.0, help="Coordinate scale (e.g., 0.001 for mm->m)")
    args = ap.parse_args()
    export_to_svg_file(args.input, args.output, args.width, args.height, args.scale)


if __name__ == "__main__":
    main()


