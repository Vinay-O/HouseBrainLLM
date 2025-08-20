from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from jsonschema import validate as jsonschema_validate, Draft7Validator


def load_schema() -> Dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "housebrain_plan_v2.schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _poly_area(points: List[Tuple[float, float]]) -> float:
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def validate_geometric(plan: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    # Space polygon sanity
    for s in plan.get("spaces", []):
        bnd = s.get("boundary", [])
        if len(bnd) < 3:
            errors.append(f"Space {s.get('id')} has <3 vertices")
            continue
        try:
            area = _poly_area([tuple(p) for p in bnd])
        except Exception:
            errors.append(f"Space {s.get('id')} boundary invalid numbers")
            continue
        if area < 1.0:
            errors.append(f"Space {s.get('id')} area too small: {area:.2f} mm^2")

    # Door width/code sanity
    for o in plan.get("openings", []):
        if o.get("type") == "door":
            w = float(o.get("width", 0))
            # Allow 610mm interior minimum; warn below this threshold
            if w < 610.0:
                errors.append(f"Door {o.get('id')} width {w}mm < 610mm")
            # For main entry door (heuristic), recommend >= 910mm
            if str(o.get("id", "")).lower().startswith("door_main") and w < 910.0:
                errors.append(f"WARN: Door {o.get('id')} main entry recommended >= 910mm")
        if o.get("type") == "window":
            # Basic vertical placement sanity: sill >= 450mm for bedrooms (egress), head <= level height
            sill = float(o.get("sill_height", 0))
            head = float(o.get("head_height", sill + o.get("height", 0)))
            if sill < 300.0:
                errors.append(f"Window {o.get('id')} sill too low: {sill}mm < 300mm")
            if head - sill <= 0:
                errors.append(f"Window {o.get('id')} height invalid: head <= sill")

    # Wall length sanity
    for w in plan.get("walls", []):
        (x1, y1) = w.get("start", [0, 0])
        (x2, y2) = w.get("end", [0, 0])
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 < 1.0:
            errors.append(f"Wall {w.get('id')} has ~zero length")

    # Space min width sanity (avoid too-narrow spaces < 700mm)
    for s in plan.get("spaces", []):
        bnd = s.get("boundary", [])
        if len(bnd) >= 3:
            xs = [p[0] for p in bnd]
            ys = [p[1] for p in bnd]
            w_mm = max(xs) - min(xs)
            h_mm = max(ys) - min(ys)
            if min(w_mm, h_mm) < 700:
                errors.append(f"Space {s.get('id')} too narrow: min dimension < 700mm")

    return errors


def validate_v2_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    schema = load_schema()

    # JSON schema validation
    validator = Draft7Validator(schema)
    json_errors = [f"Schema: {e.message} at {list(e.path)}" for e in validator.iter_errors(plan)]

    # Geometric/code sanity
    geo_errors = validate_geometric(plan)

    return json_errors + geo_errors


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate HouseBrain v2 plan JSON")
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    errs = validate_v2_file(args.input)
    if errs:
        print("❌ Validation failed:")
        for e in errs:
            print(" -", e)
    else:
        print("✅ Validation passed")


