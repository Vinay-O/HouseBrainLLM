#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def convert_v1_to_v2(v1_path: str, v2_path: str, project_name: str = "Converted v2") -> None:
    with open(v1_path, "r", encoding="utf-8") as f:
        v1 = json.load(f)

    # Minimal, best-effort mapping to v2 (level-1 only)
    v2: Dict[str, Any] = {
        "metadata": {"project_name": project_name, "units": v1.get("metadata", {}).get("units", "mm")},
        "levels": [{"id": "L1", "name": "Level 1", "elevation_mm": 0}],
        "walls": [],
        "openings": [],
        "spaces": [],
        "stairs": [],
        "electrical": {},
        "schedules": {}
    }

    for w in v1.get("walls", []):
        v2["walls"].append({
            "id": w["id"],
            "level_id": "L1",
            "start": w["start"],
            "end": w["end"],
            "type": w.get("type", "interior"),
            "thickness": w.get("thickness", 115 if w.get("type") == "interior" else 230),
            "height": 2700 if w.get("type", "interior") == "interior" else 3000,
            "load_bearing": True if w.get("type") == "exterior" else False
        })

    def map_swing(val: str | None) -> str | None:
        if not val:
            return None
        v = val.lower()
        if v in ("in", "inward"): return "in"
        if v in ("out", "outward"): return "out"
        return None

    def default_window_op(style: str | None) -> str:
        if not style: return "fixed"
        s = style.lower()
        if "casement" in s:
            # pick a default opening side deterministically
            return "casement_left_out"
        if "awning" in s:
            return "awning_top_out"
        if "slider" in s:
            return "slider_left"
        if "double_hung" in s:
            return "double_hung"
        return "fixed"

    for o in v1.get("openings", []):
        meta = o.get("metadata", {})
        base = {
            "id": o["id"],
            "level_id": "L1",
            "wall_id": o["wall_id"],
            "type": o["type"],
            "position": o.get("position", 0.5),
            "width": o.get("width", 900),
            "height": meta.get("height", 2100),
            "sill_height": meta.get("sill_height", 0 if o["type"] == "door" else 900),
            "head_height": meta.get("head_height", 2100),
            "frame_depth": meta.get("frame_depth", 100),
            "frame_width": meta.get("frame_width", 50),
            "mullion_pattern": meta.get("mullion_pattern", "none"),
        }
        if o["type"] == "door":
            base["handing"] = meta.get("handing", "RHR")
            base["swing"] = map_swing(meta.get("swing", "in")) or "in"
        elif o["type"] == "window":
            base["window_operation"] = default_window_op(meta.get("style"))
        v2["openings"].append(base)

    for s in v1.get("spaces", []):
        v2["spaces"].append({
            "id": s["id"],
            "name": s.get("name", s["id"]),
            "type": s.get("type", "room"),
            "level_id": "L1",
            "boundary": s.get("boundary", []),
            "ceiling_height": 2700,
            "finish_schedule": {"floor": "tile" if "bath" in s.get("name", "").lower() else "wood", "wall": "paint", "ceiling": "gyp"},
            "occupancy_load": 2,
            "egress_required": True if "bed" in s.get("name", "").lower() else False
        })

    Path(v2_path).parent.mkdir(parents=True, exist_ok=True)
    with open(v2_path, "w", encoding="utf-8") as f:
        json.dump(v2, f, indent=2)
    print(f"✅ Converted v1 → v2: {v2_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert HouseBrain v1 plan to v2 schema")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", default="Converted v2")
    args = ap.parse_args()
    convert_v1_to_v2(args.input, args.output, args.name)


