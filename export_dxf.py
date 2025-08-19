"""
Export HouseBrain geometry JSON to a 2D DXF file with basic layers.

Usage:
python export_dxf.py --input sample_output.json --output plan.dxf --scale 0.001

Layers:
- HB_WALLS, HB_ROOMS (polylines)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ezdxf

from geometry_utils import load_housebrain_output, scale_polygon


def export_to_dxf(input_path: str, output_path: str, scale: float = 1.0) -> None:
    data = load_housebrain_output(input_path)

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # Create layers
    if "HB_ROOMS" not in doc.layers:
        doc.layers.add("HB_ROOMS", color=3)  # green
    if "HB_WALLS" not in doc.layers:
        doc.layers.add("HB_WALLS", color=1)  # red (placeholder)

    for floor in data.get("floors", []):
        for room in floor.get("rooms", []):
            poly = room.get("polygon") or []
            if scale != 1.0:
                poly = scale_polygon(poly, scale)
            # Add as closed LWPolyline
            if len(poly) >= 4:  # includes closed point
                msp.add_lwpolyline(poly, dxfattribs={"layer": "HB_ROOMS", "closed": True})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export HouseBrain JSON to DXF")
    ap.add_argument("--input", required=True, help="Path to HouseBrain output JSON")
    ap.add_argument("--output", required=True, help="DXF file path")
    ap.add_argument("--scale", type=float, default=1.0, help="Coordinate scale (e.g., 0.001 for mm->m)")
    args = ap.parse_args()

    export_to_dxf(args.input, args.output, args.scale)
    print(f"✅ DXF written to {args.output}")


if __name__ == "__main__":
    main()


