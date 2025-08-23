"""
HouseBrain pipeline runner:
- Accepts a simplified input JSON, calls mock inference (or real later)
- Validates against schema/validators
- Exports DXF and OBJ for quick visualization

Usage:
python hb_pipeline.py --scenario s03 --input data/sample_input.json --out out/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mock_inference import generate_housebrain_output
from export_dxf import export_to_dxf
from export_obj import export_to_obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Run HouseBrain pipeline with curated examples")
    ap.add_argument("--scenario", default="s01", help="Scenario id (s01..s20)")
    ap.add_argument("--input", default="data/sample_input.json", help="Input payload JSON")
    ap.add_argument("--out", default="out/demo", help="Output directory")
    ap.add_argument("--scale", type=float, default=0.001, help="mm->m scale for exports")
    ap.add_argument("--height", type=float, default=3.0, help="OBJ extrusion height (m)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = generate_housebrain_output(payload, scenario_id=args.scenario)

    # Save the combined output
    json_path = out_dir / f"{args.scenario}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Export 2D/3D
    dxf_path = out_dir / f"{args.scenario}.dxf"
    obj_path = out_dir / f"{args.scenario}.obj"
    export_to_dxf(str(json_path), str(dxf_path), scale=args.scale)
    export_to_obj(str(json_path), str(obj_path), scale=args.scale, height=args.height)

    print(f"✅ Pipeline complete: {json_path}, {dxf_path}, {obj_path}")


if __name__ == "__main__":
    main()\n