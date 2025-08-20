from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def poly_area(points: List[List[float]]) -> float:
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def mm2_to_sqft(mm2: float) -> float:
    return mm2 / 92903.04


def boq_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    # Simple BoQ: wall lengths by type, total floor area, counts of doors/windows by width class
    wall_lengths = {"exterior": 0.0, "interior": 0.0}
    for w in plan.get("walls", []):
        (x1, y1) = w.get("start", [0, 0])
        (x2, y2) = w.get("end", [0, 0])
        L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        wall_lengths[w.get("type", "interior")] = wall_lengths.get(w.get("type", "interior"), 0.0) + L

    total_area_sqft = 0.0
    for s in plan.get("spaces", []):
        bnd = s.get("boundary", [])
        if len(bnd) >= 3:
            total_area_sqft += mm2_to_sqft(poly_area(bnd))

    door_counts = {"<700": 0, "700-900": 0, ">=900": 0}
    window_counts = {"<1200": 0, "1200-1800": 0, ">=1800": 0}
    for o in plan.get("openings", []):
        w = float(o.get("width", 0))
        if o.get("type") == "door":
            if w < 700:
                door_counts["<700"] += 1
            elif w < 900:
                door_counts["700-900"] += 1
            else:
                door_counts[">=900"] += 1
        elif o.get("type") == "window":
            if w < 1200:
                window_counts["<1200"] += 1
            elif w < 1800:
                window_counts["1200-1800"] += 1
            else:
                window_counts[">=1800"] += 1

    return {
        "wall_lengths_mm": wall_lengths,
        "total_area_sqft": round(total_area_sqft, 1),
        "door_counts": door_counts,
        "window_counts": window_counts,
    }


def write_boq(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    report = boq_from_plan(plan)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ BoQ report written to {output_path}")


def write_index_html(out_dir: str, base_name: str, modes: List[str]) -> None:
    # Generate a simple index HTML linking to SVGs and glTF files per mode
    out = Path(out_dir)
    lines: List[str] = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'><title>HouseBrain Outputs</title>")
    lines.append("<style>body{font-family:Arial,sans-serif;padding:20px} .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px} iframe{width:100%;height:600px;border:1px solid #ccc}</style>")
    lines.append("</head><body>")
    lines.append(f"<h1>{base_name}</h1>")
    for m in modes:
        svg = f"{base_name}_{m}.svg"
        gltf = f"{base_name}_{m}.gltf"
        lines.append(f"<h2>{m.upper()}</h2>")
        lines.append("<div class='grid'>")
        lines.append(f"<div><h3>SVG</h3><iframe src='{svg}'></iframe></div>")
        lines.append(f"<div><h3>glTF</h3><p><a href='{gltf}'>{gltf}</a></p></div>")
        lines.append("</div>")
    lines.append("</body></html>")
    (out / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ HTML index written to {out/'index.html'}")


