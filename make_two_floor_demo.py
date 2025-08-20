#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def rect(x: float, y: float, w: float, h: float):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def make_two_floor_demo() -> dict:
    # Overall footprint 12m x 9m
    W, H = 12000.0, 9000.0
    ox, oy = 0.0, 0.0

    levels = [
        {"id": "L1", "name": "Level 1", "elevation_mm": 0},
        {"id": "L2", "name": "Level 2", "elevation_mm": 3000},
    ]

    # Exterior walls for both levels
    ext_ids_L1 = ["L1_ext_1", "L1_ext_2", "L1_ext_3", "L1_ext_4"]
    ext_ids_L2 = ["L2_ext_1", "L2_ext_2", "L2_ext_3", "L2_ext_4"]

    walls = [
        {"id": ext_ids_L1[0], "level_id": "L1", "start": [ox, oy], "end": [ox + W, oy], "type": "exterior", "thickness": 230, "height": 3000},
        {"id": ext_ids_L1[1], "level_id": "L1", "start": [ox + W, oy], "end": [ox + W, oy + H], "type": "exterior", "thickness": 230, "height": 3000},
        {"id": ext_ids_L1[2], "level_id": "L1", "start": [ox + W, oy + H], "end": [ox, oy + H], "type": "exterior", "thickness": 230, "height": 3000},
        {"id": ext_ids_L1[3], "level_id": "L1", "start": [ox, oy + H], "end": [ox, oy], "type": "exterior", "thickness": 230, "height": 3000},

        {"id": ext_ids_L2[0], "level_id": "L2", "start": [ox, oy], "end": [ox + W, oy], "type": "exterior", "thickness": 230, "height": 3000},
        {"id": ext_ids_L2[1], "level_id": "L2", "start": [ox + W, oy], "end": [ox + W, oy + H], "type": "exterior", "thickness": 230, "height": 3000},
        {"id": ext_ids_L2[2], "level_id": "L2", "start": [ox + W, oy + H], "end": [ox, oy + H], "type": "exterior", "thickness": 230, "height": 3000},
        {"id": ext_ids_L2[3], "level_id": "L2", "start": [ox, oy + H], "end": [ox, oy], "type": "exterior", "thickness": 230, "height": 3000},
    ]

    # Interior walls Level 1
    int_x = ox + W * 0.6
    walls += [
        {"id": "L1_int_1", "level_id": "L1", "start": [int_x, oy], "end": [int_x, oy + H], "type": "interior", "thickness": 115, "height": 3000},
        {"id": "L1_int_2", "level_id": "L1", "start": [ox, oy + H * 0.5], "end": [ox + W, oy + H * 0.5], "type": "interior", "thickness": 115, "height": 3000},
        {"id": "L1_int_pl", "level_id": "L1", "start": [int_x + 200, oy + H * 0.5], "end": [int_x + 200, oy + H], "type": "interior", "thickness": 150, "height": 3000, "subtype": "plumbing"},
    ]

    # Interior walls Level 2 (simple layout)
    walls += [
        {"id": "L2_int_1", "level_id": "L2", "start": [int_x, oy], "end": [int_x, oy + H], "type": "interior", "thickness": 115, "height": 3000},
        {"id": "L2_int_2", "level_id": "L2", "start": [ox, oy + H * 0.5], "end": [ox + W, oy + H * 0.5], "type": "interior", "thickness": 115, "height": 3000},
    ]

    # Spaces Level 1
    spaces = [
        {"id": "L1_liv", "name": "Living Room", "type": "living", "level_id": "L1", "boundary": rect(ox, oy, W * 0.6, H * 0.5)},
        {"id": "L1_kit", "name": "Kitchen", "type": "kitchen", "level_id": "L1", "boundary": rect(ox, oy + H * 0.5, W * 0.6, H * 0.5)},
        {"id": "L1_bed", "name": "Bedroom", "type": "bedroom", "level_id": "L1", "boundary": rect(int_x, oy, W * 0.4, H * 0.5)},
        {"id": "L1_bath", "name": "Bathroom", "type": "bathroom", "level_id": "L1", "boundary": rect(int_x, oy + H * 0.5, W * 0.4, H * 0.5)},
        {"id": "L1_stair", "name": "Stair", "type": "stair", "level_id": "L1", "boundary": rect(ox + W * 0.3, oy + H * 0.5 - 900, 1200, 1800)},
    ]

    # Spaces Level 2
    spaces += [
        {"id": "L2_bed1", "name": "Bedroom", "type": "bedroom", "level_id": "L2", "boundary": rect(ox, oy, W * 0.6, H * 0.5)},
        {"id": "L2_bed2", "name": "Bedroom", "type": "bedroom", "level_id": "L2", "boundary": rect(ox, oy + H * 0.5, W * 0.6, H * 0.5)},
        {"id": "L2_bath", "name": "Bathroom", "type": "bathroom", "level_id": "L2", "boundary": rect(int_x, oy + H * 0.5, W * 0.4, H * 0.5)},
        {"id": "L2_study", "name": "Study", "type": "study", "level_id": "L2", "boundary": rect(int_x, oy, W * 0.4, H * 0.5)},
        {"id": "L2_stair", "name": "Stair", "type": "stair", "level_id": "L2", "boundary": rect(ox + W * 0.3, oy + H * 0.5 - 900, 1200, 1800)},
    ]

    # Openings (doors/windows)
    openings = [
        # Main entry door (>= 910mm)
        {"id": "door_main", "level_id": "L1", "wall_id": ext_ids_L1[0], "type": "door", "position": 0.5, "width": 1000, "height": 2100, "sill_height": 0, "head_height": 2100, "handing": "RHR", "swing": "in", "frame_depth": 100, "frame_width": 50},
        # Interior doors
        {"id": "door_bed", "level_id": "L1", "wall_id": "L1_int_2", "type": "door", "position": 0.65, "width": 813, "height": 2100, "sill_height": 0, "head_height": 2100, "handing": "LHR", "swing": "in", "frame_depth": 90, "frame_width": 45},
        {"id": "door_bath", "level_id": "L1", "wall_id": "L1_int_2", "type": "door", "position": 0.85, "width": 711, "height": 2100, "sill_height": 0, "head_height": 2100, "handing": "RHR", "swing": "in", "frame_depth": 90, "frame_width": 45},

        # Windows Level 1
        {"id": "win_l1_liv", "level_id": "L1", "wall_id": ext_ids_L1[2], "type": "window", "position": 0.3, "width": 1500, "height": 1200, "sill_height": 900, "head_height": 2100, "window_operation": "slider_left", "frame_depth": 100, "frame_width": 50},
        {"id": "win_l1_bed", "level_id": "L1", "wall_id": ext_ids_L1[2], "type": "window", "position": 0.75, "width": 1200, "height": 1200, "sill_height": 900, "head_height": 2100, "window_operation": "casement_right_out", "frame_depth": 100, "frame_width": 50},

        # L2 doors
        {"id": "door_l2_bed1", "level_id": "L2", "wall_id": "L2_int_2", "type": "door", "position": 0.35, "width": 813, "height": 2100, "sill_height": 0, "head_height": 2100, "handing": "LHR", "swing": "in", "frame_depth": 90, "frame_width": 45},
        {"id": "door_l2_bed2", "level_id": "L2", "wall_id": "L2_int_2", "type": "door", "position": 0.65, "width": 813, "height": 2100, "sill_height": 0, "head_height": 2100, "handing": "RHR", "swing": "in", "frame_depth": 90, "frame_width": 45},
        # L2 windows
        {"id": "win_l2_study", "level_id": "L2", "wall_id": ext_ids_L2[0], "type": "window", "position": 0.25, "width": 1200, "height": 1200, "sill_height": 900, "head_height": 2100, "window_operation": "double_hung", "frame_depth": 100, "frame_width": 50},
        {"id": "win_l2_bed2", "level_id": "L2", "wall_id": ext_ids_L2[0], "type": "window", "position": 0.75, "width": 1500, "height": 1200, "sill_height": 900, "head_height": 2100, "window_operation": "slider_right", "frame_depth": 100, "frame_width": 50},
    ]

    plan = {
        "metadata": {"project_name": "Two Floor Demo", "units": "mm"},
        "levels": levels,
        "walls": walls,
        "openings": openings,
        "spaces": spaces,
        "stairs": [],
        "electrical": {},
        "schedules": {},
    }
    return plan


if __name__ == "__main__":
    out_dir = Path("out_v2/two_floor_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = make_two_floor_demo()
    out_path = out_dir / "two_floor_demo.json"
    out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_path}")


