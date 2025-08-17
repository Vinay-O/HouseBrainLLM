#!/usr/bin/env python3
"""
HouseBrain Dataset Augmentation v1.1
- Adds units/datum/origin/north
- Adds per-floor levels (FFL/TOS)
- Enriches doors/windows with sill/head/swing/frame/lintel
- Adds layered wall assemblies and U-values
- Adds stair parameters
- Adds roof metadata (simple)
- Adds IFC class hints and DXF layer mapping
- Adds deterministic GUIDs for elements
- Preserves original structure; writes to a new output dataset directory

Usage:
  python augment_dataset_v1_1.py --input housebrain_dataset_r1_super_1M \
    --output housebrain_dataset_r1_super_1M_aug_v1_1 --workers 8 [--only-shard shard_01]
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULTS = {
    "units": "mm",
    "north_angle_deg": 0.0,
    "elevation": {
        "ground_ffl_mm": 0,
        "floor_to_floor_mm": 3000,
        "slab_thickness_mm": 150,
        "parapet_height_mm": 1000,
        "plinth_height_mm": 450,
    },
    "doors": {
        "sill_height_mm": 0,
        "head_height_mm": 2100,
        "frame_thickness_mm": 100,
        "swing_direction": "auto",
    },
    "windows": {
        "sill_height_mm": 900,
        "head_height_mm": 2100,
        "frame_thickness_mm": 75,
    },
    "walls": {
        "internal_stack": [
            {"material": "Plaster", "thickness_mm": 12},
            {"material": "Brick", "thickness_mm": 115},
            {"material": "Plaster", "thickness_mm": 12},
        ],
        "external_stack": [
            {"material": "Plaster", "thickness_mm": 15},
            {"material": "Brick", "thickness_mm": 230},
            {"material": "Plaster", "thickness_mm": 15},
        ],
        "u_value_internal_W_m2K": 2.00,
        "u_value_external_W_m2K": 1.80,
    },
    "stairs": {
        "riser_height_mm": 167,
        "tread_depth_mm": 270,
        "flight_width_mm": 1000,
        "headroom_mm": 2100,
        "handrail_height_mm": 900,
    },
    "roof": {
        "type": "Flat",
        "slope_percent": 1.0,
        "waterproofing": "APP_Membrane",
    },
}

IFC_MAP = {
    "walls": "IfcWallStandardCase",
    "columns": "IfcColumn",
    "beams": "IfcBeam",
    "slabs": "IfcSlab",
    "doors": "IfcDoor",
    "windows": "IfcWindow",
    "stairs": "IfcStair",
}

DXF_LAYERS = {
    "WALLS": {"color": 7, "lineweight": 0.35},
    "DOORS": {"color": 3, "lineweight": 0.25},
    "WINDOWS": {"color": 4, "lineweight": 0.25},
    "COLUMNS": {"color": 2, "lineweight": 0.35},
    "BEAMS": {"color": 6, "lineweight": 0.30},
    "SLABS": {"color": 1, "lineweight": 0.30},
    "STAIRS": {"color": 5, "lineweight": 0.25},
    "DIMENSIONS": {"color": 8, "lineweight": 0.18},
    "TEXT": {"color": 9, "lineweight": 0.18},
}


def stable_guid(*parts: str) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(p.encode("utf-8"))
    return m.hexdigest()[0:16]


def add_project_metadata(sample: Dict[str, Any]) -> None:
    out = sample.get("output", {})
    meta = out.setdefault("metadata_augmented_v1_1", {})
    meta["units"] = DEFAULTS["units"]
    meta["project_origin"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    meta["north_angle_deg"] = DEFAULTS["north_angle_deg"]
    meta["elevation_datum"] = {"name": "FFL_0", "elevation_mm": DEFAULTS["elevation"]["ground_ffl_mm"]}
    meta["dxf_layers"] = DXF_LAYERS
    meta["ifc_map"] = IFC_MAP


def add_levels(sample: Dict[str, Any]) -> None:
    out = sample.get("output", {})
    inp = sample.get("input", {})
    floors = int(inp.get("requirements", {}).get("floors", 1) or 1)
    ff = DEFAULTS["elevation"]["floor_to_floor_mm"]
    slab = DEFAULTS["elevation"]["slab_thickness_mm"]
    levels = []
    for i in range(floors):
        ffl = i * ff
        levels.append({
            "floor_index": i,
            "name": f"Level_{i}",
            "ffl_mm": ffl,
            "tos_mm": ffl + slab,
            "floor_to_floor_mm": ff,
        })
    out["levels"] = levels


def enrich_openings(obj_list: List[Dict[str, Any]], defaults: Dict[str, Any], base_id: str, kind: str) -> None:
    if not isinstance(obj_list, list):
        return
    for idx, d in enumerate(obj_list):
        d.setdefault("sill_height_mm", defaults.get("sill_height_mm", 0))
        d.setdefault("head_height_mm", defaults.get("head_height_mm", 2100))
        d.setdefault("frame_thickness_mm", defaults.get("frame_thickness_mm", 75))
        if kind == "doors":
            d.setdefault("swing_direction", defaults.get("swing_direction", "auto"))
        # IDs
        elem_id = d.get("id") or f"{kind}_{idx}"
        d["guid"] = stable_guid(base_id, kind, str(elem_id))
        d.setdefault("ifc_class", IFC_MAP.get(kind, "IfcBuildingElementProxy"))


def enrich_walls(walls: List[Dict[str, Any]], base_id: str) -> None:
    if not isinstance(walls, list):
        return
    for idx, w in enumerate(walls):
        wtype = (w.get("type") or "Internal").lower()
        if "ext" in wtype:
            w["layer_stack"] = DEFAULTS["walls"]["external_stack"]
            w["u_value_W_m2K"] = DEFAULTS["walls"]["u_value_external_W_m2K"]
        else:
            w["layer_stack"] = DEFAULTS["walls"]["internal_stack"]
            w["u_value_W_m2K"] = DEFAULTS["walls"]["u_value_internal_W_m2K"]
        elem_id = w.get("id") or f"wall_{idx}"
        w["guid"] = stable_guid(base_id, "wall", str(elem_id))
        w.setdefault("ifc_class", IFC_MAP.get("walls"))


def enrich_structural(elements: List[Dict[str, Any]], base_id: str, kind: str) -> None:
    if not isinstance(elements, list):
        return
    for idx, e in enumerate(elements):
        elem_id = e.get("id") or f"{kind}_{idx}"
        e["guid"] = stable_guid(base_id, kind, str(elem_id))
        e.setdefault("ifc_class", IFC_MAP.get(kind, "IfcBuildingElementProxy"))


def add_stairs_and_roof(out: Dict[str, Any], base_id: str) -> None:
    # Stairs enrichment if present
    stairs = out.get("construction_geometry", {}).get("stairs") or out.get("geometric_data", {}).get("stairs")
    if isinstance(stairs, dict):
        for k, v in DEFAULTS["stairs"].items():
            stairs.setdefault(k, v)
        stairs.setdefault("guid", stable_guid(base_id, "stairs"))
        stairs.setdefault("ifc_class", IFC_MAP.get("stairs"))
    # Roof metadata (add if not present)
    out.setdefault("roof", {}).setdefault("metadata", DEFAULTS["roof"])


def add_dimension_scaffold(out: Dict[str, Any]) -> None:
    # Adds a minimal dimension scaffold for 2D export (consumers can compute precise dims)
    dims = out.setdefault("dimensions_2d", {})
    dims.setdefault("units", DEFAULTS["units"])
    if "plot_dimensions" in out.get("spatial_design", {}):
        pd = out["spatial_design"]["plot_dimensions"]
        dims.setdefault("overall", [
            {"label": "Plot_Width", "value": pd.get("width_ft"), "units": "ft"},
            {"label": "Plot_Length", "value": pd.get("length_ft"), "units": "ft"},
        ])


def process_file(args) -> bool:
    in_path, out_path = args
    try:
        with open(in_path, "r") as f:
            sample = json.load(f)
        base_id = sample.get("metadata", {}).get("sample_id") or os.path.splitext(os.path.basename(in_path))[0]
        add_project_metadata(sample)
        add_levels(sample)
        out = sample.get("output", {})
        geo = out.get("geometric_data", {})
        # Enrich elements where present
        enrich_walls(geo.get("walls") or out.get("walls", []), base_id)
        enrich_openings(geo.get("doors") or out.get("doors", []), DEFAULTS["doors"], base_id, "doors")
        enrich_openings(geo.get("windows") or out.get("windows", []), DEFAULTS["windows"], base_id, "windows")
        enrich_structural(geo.get("columns") or out.get("columns", []), base_id, "columns")
        enrich_structural(geo.get("beams") or out.get("beams", []), base_id, "beams")
        enrich_structural(geo.get("slabs") or out.get("slabs", []), base_id, "slabs")
        add_stairs_and_roof(out, base_id)
        add_dimension_scaffold(out)
        # Ensure directories
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(sample, f, indent=2)
        return True
    except Exception as e:
        # Log minimal info
        print(f"Error processing {in_path}: {e}")
        return False


def gather_files(input_dir: Path, only_shard: str = "") -> List[Path]:
    files: List[Path] = []
    for split in ["train", "validation"]:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        for shard in sorted(split_dir.iterdir()):
            if not shard.is_dir():
                continue
            if only_shard and shard.name != only_shard:
                continue
            files.extend(sorted(shard.glob("*.json")))
    return files


def main():
    parser = argparse.ArgumentParser(description="Augment HouseBrain dataset with v1.1 precision metadata")
    parser.add_argument("--input", required=True, help="Input dataset root")
    parser.add_argument("--output", required=True, help="Output dataset root (augmented)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--only-shard", type=str, default="", help="Process only a specific shard (e.g., shard_01)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = gather_files(input_dir, args.only_shard)
    if not files:
        print("No files found to process.")
        return

    tasks = []
    for in_file in files:
        rel = in_file.relative_to(input_dir)
        out_file = output_dir / rel
        tasks.append((in_file, out_file))

    total = len(tasks)
    print(f"Augmenting {total:,} files using {args.workers} workers...")

    done = 0
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_file, t) for t in tasks]
        for fut in as_completed(futures):
            done += 1
            if fut.result():
                ok += 1
            if done % 1000 == 0:
                print(f"Processed {done:,}/{total:,} | OK: {ok:,}")

    print(f"\n✅ Augmentation complete. Success: {ok:,}/{total:,}")

if __name__ == "__main__":
    main()
