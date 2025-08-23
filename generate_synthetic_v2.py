#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple


def rand_range(a: float, b: float) -> float:
    return a + random.random() * (b - a)


# ----------------------------
# v2 helpers and vocabularies
# ----------------------------
CLIMATE_ZONES = [
    "Hot-Dry", "Warm-Humid", "Composite", "Temperate", "Cold"
]

DOOR_WIDTHS = [762, 813, 865, 915, 1000]
WINDOW_WIDTHS = [900, 1200, 1500, 1800]
WALL_THICKNESS_EXT = [200, 230, 250]
WALL_THICKNESS_INT = [100, 115, 150]

WINDOW_OPS = [
    "fixed",
    "casement_left_out",
    "casement_right_out",
    "awning_top_out",
    "hopper_bottom_in",
    "slider_left",
    "slider_right",
    "double_hung",
]


def make_level() -> Dict[str, Any]:
    return {"id": "L1", "name": "Level 1", "elevation_mm": 0}


def make_rect(x: float, y: float, w: float, h: float) -> List[List[float]]:
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _wall_layers(ext: bool) -> List[Dict[str, Any]]:
    """Return a simple wall assembly per v2 schema."""
    if ext:
        return [
            {"material": "plaster", "thickness": 12},
            {"material": "brick", "thickness": 200},
            {"material": "cement_render", "thickness": 18},
        ]
    return [
        {"material": "gypsum_board", "thickness": 12},
        {"material": "stud", "thickness": 90},
        {"material": "gypsum_board", "thickness": 12},
    ]


def _make_metadata() -> Dict[str, Any]:
    return {
        "project_name": "Synthetic House",
        "units": "mm",
        "true_north_deg": round(rand_range(0, 359), 1),
        "climate_zone": random.choice(CLIMATE_ZONES),
        "codes": {"nbc_version": "2016", "local_amendments": "state_rules"},
    }


def make_simple_house(plot_w: float, plot_h: float) -> Dict[str, Any]:
    ox, oy = 0.0, 0.0
    ext_t = random.choice(WALL_THICKNESS_EXT)
    int_t = random.choice(WALL_THICKNESS_INT)
    walls = [
        {
            "id": "ext_1",
            "level_id": "L1",
            "start": [ox, oy],
            "end": [ox + plot_w, oy],
            "type": "exterior",
            "thickness": ext_t,
            "height": 3000,
            "load_bearing": True,
            "layers": _wall_layers(True),
        },
        {
            "id": "ext_2",
            "level_id": "L1",
            "start": [ox + plot_w, oy],
            "end": [ox + plot_w, oy + plot_h],
            "type": "exterior",
            "thickness": ext_t,
            "height": 3000,
            "load_bearing": True,
            "layers": _wall_layers(True),
        },
        {
            "id": "ext_3",
            "level_id": "L1",
            "start": [ox + plot_w, oy + plot_h],
            "end": [ox, oy + plot_h],
            "type": "exterior",
            "thickness": ext_t,
            "height": 3000,
            "load_bearing": True,
            "layers": _wall_layers(True),
        },
        {
            "id": "ext_4",
            "level_id": "L1",
            "start": [ox, oy + plot_h],
            "end": [ox, oy],
            "type": "exterior",
            "thickness": ext_t,
            "height": 3000,
            "load_bearing": True,
            "layers": _wall_layers(True),
        },
    ]
    # simple partitions
    int_x = ox + plot_w * 0.6
    walls.append({
        "id": "int_1",
        "level_id": "L1",
        "start": [int_x, oy],
        "end": [int_x, oy + plot_h],
        "type": "interior",
        "thickness": int_t,
        "height": 2700,
        "load_bearing": False,
        "layers": _wall_layers(False),
    })
    # horizontal partition
    int_y = oy + plot_h * 0.5
    walls.append({
        "id": "int_2",
        "level_id": "L1",
        "start": [ox, int_y],
        "end": [ox + plot_w, int_y],
        "type": "interior",
        "thickness": int_t,
        "height": 2700,
        "load_bearing": False,
        "layers": _wall_layers(False),
    })
    # a plumbing wall segment for bathrooms/kitchen (use type=plumbing to match schema)
    pl_x = ox + plot_w * 0.62
    walls.append({
        "id": "int_pl",
        "level_id": "L1",
        "start": [pl_x, int_y],
        "end": [pl_x, oy + plot_h],
        "type": "plumbing",
        "thickness": max(int_t, 150),
        "height": 2700,
        "load_bearing": False,
        "layers": _wall_layers(False),
        "fire_rating": "1hr",
    })

    spaces = [
        {
            "id": "living",
            "name": "Living Room",
            "type": "living",
            "level_id": "L1",
            "boundary": make_rect(ox, oy, plot_w * 0.6, plot_h * 0.5),
            "ceiling_height": 2700,
            "finish_schedule": {"floor": "wood", "wall": "paint", "ceiling": "gyp"},
            "occupancy_load": 4,
            "egress_required": False,
        },
        {
            "id": "kitchen",
            "name": "Kitchen",
            "type": "kitchen",
            "level_id": "L1",
            "boundary": make_rect(ox, int_y, plot_w * 0.6, plot_h * 0.5),
            "ceiling_height": 2700,
            "finish_schedule": {"floor": "tile", "wall": "paint", "ceiling": "gyp"},
            "occupancy_load": 3,
            "egress_required": False,
        },
        {
            "id": "bed1",
            "name": "Bedroom",
            "type": "bedroom",
            "level_id": "L1",
            "boundary": make_rect(int_x, oy, plot_w * 0.4, plot_h * 0.6),
            "ceiling_height": 2700,
            "finish_schedule": {"floor": "wood", "wall": "paint", "ceiling": "gyp"},
            "occupancy_load": 2,
            "egress_required": True,
        },
        {
            "id": "bath1",
            "name": "Bathroom",
            "type": "bathroom",
            "level_id": "L1",
            "boundary": make_rect(int_x, int_y, plot_w * 0.4, plot_h * 0.4),
            "ceiling_height": 2700,
            "finish_schedule": {"floor": "tile", "wall": "tile", "ceiling": "gyp"},
            "occupancy_load": 1,
            "egress_required": False,
        },
    ]

    openings = [
        {
            "id": "door_main",
            "level_id": "L1",
            "wall_id": random.choice(["ext_1", "ext_3"]),
            "type": "door",
            "position": rand_range(0.2, 0.8),
            "width": random.choice(DOOR_WIDTHS),
            "height": 2100,
            "sill_height": 0,
            "head_height": 2100,
            "handing": random.choice(["LHR", "RHR"]),
            "swing": random.choice(["in", "out"]),
            "frame_depth": 100,
            "frame_width": 50,
            "mullion_pattern": "none",
        },
        {
            "id": "win_liv",
            "level_id": "L1",
            "wall_id": "ext_1",
            "type": "window",
            "position": rand_range(0.15, 0.3),
            "width": random.choice(WINDOW_WIDTHS),
            "height": 1200,
            "sill_height": 900,
            "head_height": 2100,
            "window_operation": random.choice(WINDOW_OPS),
            "frame_depth": 100,
            "frame_width": 50,
            "mullion_pattern": "none",
        },
        {
            "id": "win_bed",
            "level_id": "L1",
            "wall_id": "ext_3",
            "type": "window",
            "position": rand_range(0.6, 0.9),
            "width": random.choice([900, 1200, 1500]),
            "height": 1200,
            "sill_height": 900,
            "head_height": 2100,
            "window_operation": random.choice(WINDOW_OPS),
            "frame_depth": 100,
            "frame_width": 50,
            "mullion_pattern": "none",
        },
        {
            "id": "win_kit",
            "level_id": "L1",
            "wall_id": "ext_1",
            "type": "window",
            "position": rand_range(0.4, 0.55),
            "width": random.choice([900, 1200]),
            "height": 1050,
            "sill_height": 1050,
            "head_height": 2100,
            "window_operation": random.choice(WINDOW_OPS),
            "frame_depth": 100,
            "frame_width": 50,
            "mullion_pattern": "none",
        },
    ]

    # Straight stair (optional)
    stairs: List[Dict[str, Any]] = []
    if plot_h >= 8500 and random.random() < 0.4:
        stairs.append({
            "start": [int_x + 500, oy + 500],
            "end": [int_x + 500, oy + 2500],
            "width": 1000,
            "riser_height": 165,
            "tread_depth": 270,
            "flights": 1,
            "landings": 0,
            "headroom": 2100,
        })

    # Electrical layout (simple but structured)
    def _pt_inside(rect: List[List[float]]) -> Tuple[float, float]:
        (x0, y0), (_, _), (x1, y1), _ = rect
        return rand_range(x0 + 300, x1 - 300), rand_range(y0 + 300, y1 - 300)

    outlets, switches, fixtures, circuits = [], [], [], []
    circuit_id = 1
    for sp in spaces:
        bx = sp["boundary"]
        # a ceiling fixture per room
        fx_x, fx_y = _pt_inside(bx)
        fixtures.append({
            "id": f"fx_{sp['id']}",
            "level_id": sp["level_id"],
            "type": "ceiling_light",
            "position": [fx_x, fx_y],
        })
        # a switch near one corner
        sw_x, sw_y = bx[0][0] + 150, bx[0][1] + 150
        switches.append({
            "id": f"sw_{sp['id']}",
            "level_id": sp["level_id"],
            "type": "1_way",
            "position": [sw_x, sw_y],
            "mount_height": 1200,
        })
        # a couple of outlets
        for j in range(2):
            oxp, oyp = _pt_inside(bx)
            outlets.append({
                "id": f"ol_{sp['id']}_{j}",
                "level_id": sp["level_id"],
                "type": "13A",
                "position": [oxp, oyp],
                "mount_height": 300,
            })
        # circuit connects switch to fixture
        circuits.append({
            "id": f"c{circuit_id}",
            "switch_ids": [f"sw_{sp['id']}",],
            "fixture_ids": [f"fx_{sp['id']}",],
            "breaker": 10,
        })
        circuit_id += 1

    # Schedules from elements
    door_schedule = [
        {
            "id": op["id"],
            "type": "door",
            "width": op.get("width"),
            "height": op.get("height"),
            "handing": op.get("handing"),
            "swing": op.get("swing"),
        }
        for op in openings if op["type"] == "door"
    ]
    window_schedule = [
        {
            "id": op["id"],
            "type": "window",
            "width": op.get("width"),
            "height": op.get("height"),
            "operation": op.get("window_operation"),
        }
        for op in openings if op["type"] == "window"
    ]
    room_finish_schedule = [
        {
            "room_id": sp["id"],
            "floor": sp["finish_schedule"]["floor"],
            "wall": sp["finish_schedule"]["wall"],
            "ceiling": sp["finish_schedule"]["ceiling"],
        }
        for sp in spaces
    ]

    plan = {
        "metadata": _make_metadata(),
        "levels": [make_level()],
        "walls": walls,
        "openings": openings,
        "spaces": spaces,
        "stairs": stairs,
        "electrical": {
            "outlets": outlets,
            "switches": switches,
            "fixtures": fixtures,
            "circuits": circuits,
        },
        "schedules": {
            "door_schedule": door_schedule,
            "window_schedule": window_schedule,
            "room_finish_schedule": room_finish_schedule,
        },
    }
    return plan


def generate_synthetic_v2(out_dir: str, n: int = 100) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        W = random.choice([9000, 10000, 12000, 14000])
        H = random.choice([7000, 8000, 9000, 10000])
        plan = make_simple_house(W, H)
        path = out / f"syn_{i:06d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f, separators=(",", ":"))
    print(f"✅ Generated {n} synthetic v2 plans at {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate synthetic HouseBrain v2 plans")
    # Support both canonical flags and common aliases used in docs/Colab
    ap.add_argument("--out", dest="out")
    ap.add_argument("--out_dir", dest="out")
    ap.add_argument("--num", type=int, dest="num", default=100)
    ap.add_argument("--n", type=int, dest="num")
    args = ap.parse_args()
    if not args.out:
        ap.error("--out (or --out_dir) is required")
    generate_synthetic_v2(args.out, args.num)\n