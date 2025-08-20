from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .validate_v2 import validate_v2_file
from .export_gltf import export_gltf
from .export_plan_dxf import export_plan_to_dxf
from .plan_renderer import export_plan_svg
from .reports import write_boq, write_index_html


def run_pipeline(input_path: str, out_dir: str, width: int = 1800, height: int = 1200, sheet_modes: list[str] | None = None) -> None:
    errs = validate_v2_file(input_path)
    # Separate warnings (prefixed) from errors
    warnings = [e for e in errs if isinstance(e, str) and e.startswith("WARN:")]
    hard_errors = [e for e in errs if not (isinstance(e, str) and e.startswith("WARN:"))]
    if warnings:
        print("⚠️  Validation warnings:")
        for w in warnings:
            print(" -", w)
    if hard_errors:
        print("❌ Validation failed:")
        for e in hard_errors:
            print(" -", e)
        raise SystemExit(1)

    name = Path(input_path).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    modes = sheet_modes or ["floor"]
    for mode in modes:
        # 2D SVG per sheet mode (renderer reads mode via constructor in a future step)
        svg_out = str(out / f"{name}_{mode}.svg")
        export_plan_svg(input_path, svg_out, width, height, mode)
        # DXF (single for now)
        dxf_out = str(out / f"{name}.dxf")
        export_plan_to_dxf(input_path, dxf_out, units="mm")
        # glTF per sheet mode (placeholder)
        gltf_out = str(out / f"{name}_{mode}.gltf")
        export_gltf(input_path, gltf_out, mode)

    # BoQ and index
    boq_out = str(out / f"{name}_boq.json")
    write_boq(input_path, boq_out)
    write_index_html(str(out), name, modes)

    print(f"✅ Pipeline complete → {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="HouseBrain v2 pipeline (validate → SVG → DXF → glTF)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--modes", nargs="*", default=["floor", "rcp", "power", "plumbing"])
    args = ap.parse_args()
    run_pipeline(args.input, args.out_dir, args.width, args.height, args.modes)


