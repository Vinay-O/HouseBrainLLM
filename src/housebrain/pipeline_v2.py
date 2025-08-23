from __future__ import annotations
import json
from pathlib import Path
from .validate_v2 import validate_v2_file
from .export_plan_dxf import export_plan_to_dxf
from .plan_renderer import export_plan_svg
from .reports import write_boq, write_index_html
from .geometry_builder import build_geometry_from_plan
from .revolutionary_3d_generator import generate_gltf_from_scene


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

    # Load plan data once
    with open(input_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    modes = sheet_modes or ["floor"]
    for mode in modes:
        # 2D SVG per sheet mode
        svg_out = str(out / f"{name}_{mode}.svg")
        export_plan_svg(input_path, svg_out, width, height, mode)

    # Generate 3D Geometry and Export to glTF
    print("🏗️ Building 3D geometry from plan...")
    scene = build_geometry_from_plan(plan_data)
    print(f"   - Generated {len(scene.meshes)} meshes")
    
    print("🎨 Serializing 3D scene to glTF...")
    gltf_data = generate_gltf_from_scene(scene)
    gltf_out = out / f"{name}_3d.gltf"
    with open(gltf_out, "w", encoding="utf-8") as f:
        json.dump(gltf_data, f, indent=2)
    print(f"   - ✅ Saved glTF to {gltf_out}")

    # DXF (single for now)
    dxf_out = str(out / f"{name}.dxf")
    export_plan_to_dxf(input_path, dxf_out, units="mm")

    # BoQ and index
    boq_out = str(out / f"{name}_boq.json")
    write_boq(input_path, boq_out)
    write_index_html(str(out), name, modes, has_3d=True)

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