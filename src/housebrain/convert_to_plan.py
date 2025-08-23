from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _edge_key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    ax, ay = int(round(a[0])), int(round(a[1]))
    bx, by = int(round(b[0])), int(round(b[1]))
    return ((ax, ay), (bx, by)) if (ax, ay) <= (bx, by) else ((bx, by), (ax, ay))


def _line_dir(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L == 0:
        return 0.0, 0.0, 0.0
    return dx / L, dy / L, L


def _project_param(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    ux, uy, L = _line_dir(a, b)
    if L == 0:
        return 0.0
    return (p[0] - a[0]) * ux + (p[1] - a[1]) * uy


def house_to_plan(house_path: str, out_path: str) -> None:
    with open(house_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    floors = data.get("geometry", {}).get("floors", [])
    if not floors:
        raise ValueError("No floors found in house JSON")

    # Build spaces and edge counts across all rooms (first floor only for now)
    spaces: List[Dict[str, Any]] = []
    edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]] = {}
    for room in floors[0].get("rooms", []):
        poly: List[Tuple[float, float]] = room.get("polygon", [])
        if len(poly) < 3:
            continue
        rname = room.get("name", room.get("type", "room")) or "room"
        spaces.append({
            "id": rname,
            "name": (rname or "Room").replace("_", " ").title(),
            "type": room.get("type", "room"),
            "boundary": poly,
        })
        ring = poly if poly[0] == poly[-1] else poly + [poly[0]]
        for a, b in zip(ring[:-1], ring[1:]):
            k = _edge_key(tuple(a), tuple(b))
            rec = edge_counts.get(k)
            if not rec:
                edge_counts[k] = {"count": 1, "start": tuple(a), "end": tuple(b), "adj": set([rname.lower()])}
            else:
                rec["count"] += 1
                rec["adj"].add(rname.lower())

    # Create unique walls with type inferred by count (1=exterior, >1=interior)
    walls: List[Dict[str, Any]] = []
    for idx, (k, info) in enumerate(edge_counts.items(), start=1):
        wtype = "exterior" if info["count"] == 1 else "interior"
        thickness = 230.0 if wtype == "exterior" else 115.0
        subtype = None
        wet_tokens = ["bath", "toilet", "wash", "wc", "utility"]
        if wtype == "interior" and any(any(tok in name for tok in wet_tokens) for name in info.get("adj", [])):
            subtype = "plumbing"
        walls.append({
            "id": f"w{idx}",
            "start": info["start"],
            "end": info["end"],
            "type": wtype,
            "thickness": thickness,
            "subtype": subtype,
        })

    # Map from wall id to tuple
    wall_by_id = {w["id"]: (tuple(w["start"]), tuple(w["end"])) for w in walls}

    # Build openings: assign to best-matching wall by projection
    openings: List[Dict[str, Any]] = []
    def add_opening(op_type: str, p1: Tuple[float, float], p2: Tuple[float, float]):
        best_id = None
        best_dist = 1e9
        best_pos = 0.5
        for wid, (a, b) in wall_by_id.items():
            ux, uy, L = _line_dir(a, b)
            if L == 0:
                continue
            # distances as perpendicular from each endpoint to the line
            def perp_dist(p: Tuple[float, float]) -> float:
                # vector ap
                apx, apy = p[0] - a[0], p[1] - a[1]
                # component perpendicular: magnitude of ap - (ap·u)u
                proj = apx * ux + apy * uy
                rx = apx - proj * ux
                ry = apy - proj * uy
                return (rx * rx + ry * ry) ** 0.5
            dsum = perp_dist(p1) + perp_dist(p2)
            # compute center position along wall
            t1 = _project_param(a, b, p1)
            t2 = _project_param(a, b, p2)
            t_center = (t1 + t2) / 2.0
            if 0.0 <= t_center <= L and dsum < best_dist:
                best_dist = dsum
                best_id = wid
                best_pos = t_center / L
        if best_id is not None:
            width = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            openings.append({
                "id": f"op{len(openings)+1}",
                "wall_id": best_id,
                "type": op_type,
                "position": max(0.0, min(1.0, best_pos)),
                "width": width,
                "metadata": {}
            })

    for room in floors[0].get("rooms", []):
        for d in room.get("doors", []):
            if len(d) >= 2:
                add_opening("door", tuple(d[0]), tuple(d[1]))
        for w in room.get("windows", []):
            if len(w) >= 2:
                add_opening("window", tuple(w[0]), tuple(w[1]))

    plan = {
        "metadata": {"units": "mm", "scale": 100, "project": "HouseBrain", "level": "Ground"},
        "walls": walls,
        "openings": openings,
        "spaces": spaces,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"✅ Wrote plan JSON to {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert legacy House JSON to plan schema")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    house_to_plan(args.input, args.output)