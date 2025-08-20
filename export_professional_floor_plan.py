"""
Professional Floor Plan Renderer - Matches Real Architectural Standards

Enhancements:
- Outer vs inner walls with proper thickness
- Room labels + areas + basic room dimensions
- Door openings with hinge + leaf + swing arc; wall gaps at doors/windows
- Window openings as double-line symbol inside wall gap
- Per-frame north arrow & scale bar
- Overall building dimensions outside plan
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

# --------------------------- geometry helpers ---------------------------

def _collect_points(floor: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for room in floor.get("rooms", []):
        for x, y in room.get("polygon", []) or []:
            pts.append((x, y))
        for opening in room.get("doors", []):
            for x, y in opening:
                pts.append((x, y))
        for opening in room.get("windows", []):
            for x, y in opening:
                pts.append((x, y))
    return pts


def _edge_key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    ax, ay = int(round(a[0])), int(round(a[1]))
    bx, by = int(round(b[0])), int(round(b[1]))
    if (ax, ay) <= (bx, by):
        return (ax, ay), (bx, by)
    return (bx, by), (ax, ay)


def _fit_transform(points: List[Tuple[float, float]], box_w: float, box_h: float, margin: float = 24.0):
    if not points:
        return 1.0, 0.0, 0.0
    minx = min(p[0] for p in points)
    maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxy = max(p[1] for p in points)
    w = maxx - minx if maxx > minx else 1.0
    h = maxy - miny if maxy > miny else 1.0
    sx = (box_w - 2 * margin) / w
    sy = (box_h - 2 * margin) / h
    s = min(sx, sy)
    tx = margin - minx * s
    ty = margin + maxy * s
    return s, tx, ty


def _t(x: float, y: float, s: float, tx: float, ty: float) -> Tuple[float, float]:
    return x * s + tx, -y * s + ty


def _centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not poly:
        return 0.0, 0.0
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return cx, cy


def _bbox(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


# Architectural wall thicknesses (millimetres)
OUTER_WALL_THICK_MM = 230.0
INNER_WALL_THICK_MM = 115.0


def _build_wall_strip_polygon(x1: float, y1: float, x2: float, y2: float, thickness_px: float) -> str:
    """Return an SVG path polygon string representing a wall strip rectangle for the segment.
    Coordinates are already in SVG space. thickness_px is perpendicular thickness in pixels.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6 or thickness_px <= 0:
        return ""
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    off = thickness_px / 2.0
    ax = x1 + px * off
    ay = y1 + py * off
    bx = x2 + px * off
    by = y2 + py * off
    cx = x2 - px * off
    cy = y2 - py * off
    dxp = x1 - px * off
    dyp = y1 - py * off
    return (
        f"M {ax:.1f} {ay:.1f} L {bx:.1f} {by:.1f} L {cx:.1f} {cy:.1f} L {dxp:.1f} {dyp:.1f} Z"
    )


def _dim_line(x1: float, y1: float, x2: float, y2: float, text: str) -> str:
    """Professional dimension with extension lines and arrowheads via marker."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return ""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    # Perpendicular offset for extension lines and text placement
    ext_len = 12.0
    perp_angle = math.radians(angle + 90.0)
    dx = ext_len * math.cos(perp_angle)
    dy = ext_len * math.sin(perp_angle)
    # Midpoint for text
    midx = (x1 + x2) / 2.0
    midy = (y1 + y2) / 2.0
    txtx = midx + dx * 0.5
    txty = midy + dy * 0.5
    return "".join([
        # Extension lines
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x1+dx:.1f}' y2='{y1+dy:.1f}' class='dim-ext'/>",
        f"<line x1='{x2:.1f}' y1='{y2:.1f}' x2='{x2+dx:.1f}' y2='{y2+dy:.1f}' class='dim-ext'/>",
        # Dimension line with arrow markers
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' class='dim-line' marker-start='url(#dim-arrow)' marker-end='url(#dim-arrow)'/>",
        f"<text x='{txtx:.1f}' y='{txty-2:.1f}' class='dimtxt' text-anchor='middle'>{text}</text>",
    ])


def _north_arrow(x: float, y: float, size: float = 28) -> str:
    return (
        f"<g transform='translate({x:.1f},{y:.1f})'>"
        f"<polygon points='0,-{size:.1f} {size/2:.1f},0 0,{size/4:.1f} -{size/2:.1f},0' fill='#FF0000' stroke='#000' stroke-width='1'/>"
        f"<text x='0' y='{size+14:.1f}' class='dimtxt' text-anchor='middle'>N</text>"
        "</g>"
    )


def _scale_bar(x: float, y: float, length: float = 100) -> str:
    seg = length / 4
    lines = [
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{length:.1f}' height='8' fill='none' stroke='#000' stroke-width='1'/>",
    ]
    for i in range(5):
        xx = x + i * seg
        lines.append(f"<line x1='{xx:.1f}' y1='{y:.1f}' x2='{xx:.1f}' y2='{y+12:.1f}' stroke='#000' stroke-width='1'/>")
    lines.append(f"<text x='{x+length/2:.1f}' y='{y+28:.1f}' class='dimtxt' text-anchor='middle'>Scale</text>")
    return "".join(lines)


def _is_horizontal(x1: float, y1: float, x2: float, y2: float, tol: float = 1.0) -> bool:
    return abs(y1 - y2) <= tol and abs(x2 - x1) > tol


def _is_vertical(x1: float, y1: float, x2: float, y2: float, tol: float = 1.0) -> bool:
    return abs(x1 - x2) <= tol and abs(y2 - y1) > tol


def _cut_segment_with_opening(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ox1: float,
    oy1: float,
    ox2: float,
    oy2: float,
    gap: float = 18.0,
) -> List[Tuple[float, float, float, float]]:
    """Cut a horizontal/vertical wall segment by an opening using the opening span.
    If the opening endpoints align with the wall within tolerance, remove that exact span
    (with a small padding). Otherwise, fall back to a fixed-size gap around opening center.
    Returns 0-2 remaining segments.
    """
    tol = 8.0
    pad = 2.0
    segs: List[Tuple[float, float, float, float]] = []
    # Horizontal wall
    if _is_horizontal(x1, y1, x2, y2):
        y = y1
        a, b = sorted((x1, x2))
        # If opening aligned horizontally to wall
        if abs(oy1 - y) <= tol and abs(oy2 - y) <= tol:
            open_min = min(ox1, ox2) - pad
            open_max = max(ox1, ox2) + pad
            left = max(a, open_min)
            right = min(b, open_max)
            if left > a:
                segs.append((a, y, left, y))
            if right < b:
                segs.append((right, y, b, y))
            return segs
        # Fallback: use center-based fixed gap
        cx = (ox1 + ox2) / 2
        left = max(a, cx - gap / 2)
        right = min(b, cx + gap / 2)
        if left > a:
            segs.append((a, y, left, y))
        if right < b:
            segs.append((right, y, b, y))
        return segs
    # Vertical wall
    if _is_vertical(x1, y1, x2, y2):
        x = x1
        a, b = sorted((y1, y2))
        if abs(ox1 - x) <= tol and abs(ox2 - x) <= tol:
            open_min = min(oy1, oy2) - pad
            open_max = max(oy1, oy2) + pad
            bottom = max(a, open_min)
            top = min(b, open_max)
            if bottom > a:
                segs.append((x, a, x, bottom))
            if top < b:
                segs.append((x, top, x, b))
            return segs
        # Fallback center-based
        cy = (oy1 + oy2) / 2
        bottom = max(a, cy - gap / 2)
        top = min(b, cy + gap / 2)
        if bottom > a:
            segs.append((x, a, x, bottom))
        if top < b:
            segs.append((x, top, x, b))
        return segs
    # Non-axis-aligned, do not cut
    return [(x1, y1, x2, y2)]


# --------------------------- professional symbols ---------------------------

def _draw_professional_door(x1: float, y1: float, x2: float, y2: float) -> str:
    """Professional door: panel rectangle, hinge dot, 90° swing arc (dashed)."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0.1:
        return ""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return (
        f"<g transform='translate({x1:.1f},{y1:.1f}) rotate({angle:.1f})'>"
        f"<rect x='0' y='-3' width='{length:.1f}' height='6' fill='none' stroke='#8B4513' stroke-width='1.2'/>"
        f"<circle cx='0' cy='0' r='2' fill='#8B4513'/>"
        f"<path d='M 0 0 A {length:.1f} {length:.1f} 0 0 1 {length:.1f} {length:.1f}' stroke='#8B4513' stroke-width='1' fill='none' stroke-dasharray='5,2'/>"
        "</g>"
    )


def _draw_professional_window(x1: float, y1: float, x2: float, y2: float) -> str:
    """High-visibility window with white core and muntins for contrast."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 6.0:
        return ""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    h = 12.0
    return (
        f"<g transform='translate({x1:.1f},{y1:.1f}) rotate({angle:.1f})'>"
        f"<rect x='0' y='-{h/2:.1f}' width='{length:.1f}' height='{h:.1f}' fill='white' stroke='#0066CC' stroke-width='1.6'/>"
        f"<line x1='{length/3:.1f}' y1='-{h/2:.1f}' x2='{length/3:.1f}' y2='{h/2:.1f}' stroke='#0066CC' stroke-width='1.0'/>"
        f"<line x1='{2*length/3:.1f}' y1='-{h/2:.1f}' x2='{2*length/3:.1f}' y2='{h/2:.1f}' stroke='#0066CC' stroke-width='1.0'/>"
        "</g>"
    )


# Additional professional symbols: fixtures and stairs
def _draw_fixture(fixture_type: str, x: float, y: float) -> str:
    """Draw compact professional fixture symbols at SVG coords (x, y)."""
    ftype = (fixture_type or "").lower()
    if "toilet" in ftype or ftype == "wc":
        return (
            f"<g transform='translate({x:.1f},{y:.1f})'>"
            "<ellipse cx='0' cy='-2' rx='8' ry='4' fill='white' stroke='#333' stroke-width='0.6'/>"
            "<rect x='-4' y='0' width='8' height='7' fill='white' stroke='#333' stroke-width='0.6'/>"
            "</g>"
        )
    if "sink" in ftype or "basin" in ftype:
        return (
            f"<g transform='translate({x:.1f},{y:.1f})'>"
            "<circle cx='0' cy='0' r='6' fill='white' stroke='#333' stroke-width='0.6'/>"
            "<line x1='-3' y1='0' x2='3' y2='0' stroke='#333' stroke-width='0.9'/>"
            "<line x1='0' y1='-3' x2='0' y2='3' stroke='#333' stroke-width='0.9'/>"
            "</g>"
        )
    if "shower" in ftype:
        return (
            f"<g transform='translate({x:.1f},{y:.1f})'>"
            "<rect x='-8' y='-8' width='16' height='16' fill='none' stroke='#333' stroke-width='0.6'/>"
            "<circle cx='0' cy='0' r='3' fill='none' stroke='#333' stroke-width='0.9'/>"
            "</g>"
        )
    if "bath" in ftype:
        return (
            f"<g transform='translate({x:.1f},{y:.1f})'>"
            "<rect x='-12' y='-7' width='24' height='14' rx='3' fill='white' stroke='#333' stroke-width='0.6'/>"
            "</g>"
        )
    if "cooktop" in ftype or "hob" in ftype:
        return (
            f"<g transform='translate({x:.1f},{y:.1f})'>"
            "<rect x='-10' y='-7' width='20' height='14' fill='none' stroke='#333' stroke-width='0.6'/>"
            "<circle cx='-5' cy='-4' r='2.2' fill='none' stroke='#333' stroke-width='0.6'/>"
            "<circle cx='5' cy='-4' r='2.2' fill='none' stroke='#333' stroke-width='0.6'/>"
            "<circle cx='-5' cy='4' r='2.2' fill='none' stroke='#333' stroke-width='0.6'/>"
            "<circle cx='5' cy='4' r='2.2' fill='none' stroke='#333' stroke-width='0.6'/>"
            "</g>"
        )
    return ""


def _draw_stairs(room_poly: List[Tuple[float, float]], s: float, tx: float, ty: float) -> str:
    """Draw stair treads and a direction arrow inside given room polygon."""
    if not room_poly:
        return ""
    minx, miny, maxx, maxy = _bbox(room_poly)
    width = max(12.0, (maxx - minx) * s)
    height = max(12.0, (maxy - miny) * s)
    cx, cy = _centroid(room_poly)
    CX, CY = _t(cx, cy, s, tx, ty)
    svg = [
        f"<rect x='{CX - width/2:.1f}' y='{CY - height/2:.1f}' width='{width:.1f}' height='{height:.1f}' fill='none' stroke='#333' stroke-width='0.6'/>"
    ]
    # 7 treads evenly spaced
    steps = 7
    for i in range(1, steps):
        y_pos = CY - height/2 + (i * height / steps)
        svg.append(f"<line x1='{CX - width/2:.1f}' y1='{y_pos:.1f}' x2='{CX + width/2:.1f}' y2='{y_pos:.1f}' stroke='#333' stroke-width='0.6'/>")
    # simple upward arrow
    svg.append(
        f"<path d='M {CX:.1f} {CY - height*0.3:.1f} L {CX + width*0.18:.1f} {CY - height*0.1:.1f} L {CX - width*0.18:.1f} {CY - height*0.1:.1f} Z' fill='#333'/>"
    )
    return "".join(svg)

# --------------------------- styles ---------------------------

essential_styles = (
    """
<style>
.frame { stroke: #000; stroke-width: 2; fill: none; }
.outer-wall { stroke: #000; stroke-width: 5; fill: none; }
.inner-wall { stroke: #333; stroke-width: 2.5; fill: none; }
.door { stroke: #8B4513; stroke-width: 2; fill: none; }
.window { stroke: #0066CC; stroke-width: 1.5; fill: none; }
.room-fill { fill: #FCFCFC; }
.label { font-family: Arial, Helvetica, sans-serif; font-size: 12px; fill: #111; }
.area { font-family: Arial, Helvetica, sans-serif; font-size: 10px; fill: #666; }
.sub { font-family: Arial, Helvetica, sans-serif; font-size: 10px; fill: #666; }
.title { font-family: Arial, Helvetica, sans-serif; font-size: 18px; font-weight: bold; fill: #000; }
.floor-title { font-family: Arial, Helvetica, sans-serif; font-size: 14px; font-weight: bold; fill: #000; }
.dim-line { stroke: #111; stroke-width: 1; fill: none; }
.dim-ext { stroke: #888; stroke-width: 0.8; stroke-dasharray: 4,2; fill: none; }
.dimtxt { font-family: Arial, Helvetica, sans-serif; font-size: 10px; fill: #111; }
.wall-label { font-family: Arial, Helvetica, sans-serif; font-size: 8px; fill: #777; }
</style>
"""
)


# --------------------------- main renderer ---------------------------

def create_professional_floor_plan(house_data: Dict[str, Any], width: int = 1400, height: int = 1000) -> str:
    floors = house_data.get("geometry", {}).get("floors", [])
    num = max(1, len(floors))
    cols = min(2, num)
    rows = (num + cols - 1) // cols
    gutter = 40
    margin_page = 40
    frame_w = (width - margin_page * 2 - gutter * (cols - 1)) / cols
    frame_h = (height - margin_page * 2 - gutter * (rows - 1)) / rows

    out: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs>",
        essential_styles,
        "<pattern id='grid' width='100' height='100' patternUnits='userSpaceOnUse'>",
        "<path d='M 100 0 L 0 0 0 100' fill='none' stroke='#eeeeee' stroke-width='0.6'/>",
        "</pattern>",
        # Dimension arrow marker
        "<marker id='dim-arrow' markerWidth='8' markerHeight='8' refX='4' refY='4' orient='auto' markerUnits='strokeWidth'>",
        "<path d='M 0 0 L 8 4 L 0 8 z' fill='#333'/>",
        "</marker>",
        # Materials
        "<pattern id='brick' width='20' height='10' patternUnits='userSpaceOnUse'>",
        "<rect width='20' height='10' fill='#f9f9f9'/>",
        "<path d='M0 0 L20 0 M0 5 L20 5 M0 10 L20 10 M10 0 L10 10' stroke='#aaaaaa' stroke-width='0.5'/>",
        "</pattern>",
        "<pattern id='tile' width='14' height='14' patternUnits='userSpaceOnUse'>",
        "<rect width='14' height='14' fill='#f0f8ff'/>",
        "<path d='M0 0 L14 14 M14 0 L0 14' stroke='#aaccff' stroke-width='0.6'/>",
        "</pattern>",
        "</defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='url(#grid)'/>",
        f"<text x='{width/2:.1f}' y='{margin_page/2:.1f}' class='title' text-anchor='middle'>Professional Floor Plan</text>",
    ]

    for idx, floor in enumerate(floors if floors else [{}]):
        r = idx // cols
        c = idx % cols
        fx = margin_page + c * (frame_w + gutter)
        fy = margin_page + r * (frame_h + gutter)

        out.append(f"<rect class='frame' x='{fx:.1f}' y='{fy:.1f}' width='{frame_w:.1f}' height='{frame_h:.1f}'/>")
        title = floor.get("name", f"Floor {idx}")
        out.append(f"<text x='{fx+8:.1f}' y='{fy+18:.1f}' class='floor-title'>{title}</text>")

        pts = _collect_points(floor)
        s, tx, ty = _fit_transform(pts, frame_w, frame_h, margin=28)
        tx += fx
        ty += fy

        # Room fills + labels + basic room dimensions
        for room in floor.get("rooms", []):
            poly = room.get("polygon") or []
            if len(poly) >= 4:
                pts_t = [_t(x, y, s, tx, ty) for x, y in poly]
                d = "M " + " L ".join(f"{X:.1f} {Y:.1f}" for X, Y in pts_t) + " Z"
                rtype = (room.get("name", room.get("type", "")).lower())
                if "bath" in rtype or "toilet" in rtype or "wash" in rtype:
                    out.append(f"<path d='{d}' fill='url(#tile)' stroke='none'/>")
                else:
                    out.append(f"<path d='{d}' class='room-fill'/>")
                cx, cy = _centroid(poly)
                CX, CY = _t(cx, cy, s, tx, ty)
                name = room.get("name", room.get("type", "Room")).replace("_", " ").title()
                area = room.get("area_sqm", None)
                out.append(f"<text x='{CX:.1f}' y='{CY-4:.1f}' class='label' text-anchor='middle'>{name}</text>")
                if area is not None:
                    out.append(f"<text x='{CX:.1f}' y='{CY+10:.1f}' class='area' text-anchor='middle'>{area:.1f} sqm</text>")
                minx, miny, maxx, maxy = _bbox(poly)
                x1, y1 = _t(minx, miny, s, tx, ty)
                x2, y2 = _t(maxx, miny, s, tx, ty)
                out.append(_dim_line(x1, y1 - 12, x2, y2 - 12, f"{(maxx-minx)/1000:.2f}m"))
                x3, y3 = _t(maxx, miny, s, tx, ty)
                x4, y4 = _t(maxx, maxy, s, tx, ty)
                out.append(_dim_line(x3 + 12, y3, x4 + 12, y4, f"{(maxy-miny)/1000:.2f}m"))

                # Fixtures if present
                for fix in room.get("fixtures", []):
                    pos = fix.get("position") or fix.get("center") or [None, None]
                    if pos and len(pos) >= 2 and pos[0] is not None and pos[1] is not None:
                        FX, FY = _t(pos[0], pos[1], s, tx, ty)
                        out.append(_draw_fixture(fix.get("type", ""), FX, FY))

                # Stairs if a staircase room
                rtype_norm = room.get("type", "").lower()
                if "stair" in rtype_norm:
                    out.append(_draw_stairs(poly, s, tx, ty))

        # Wall edges
        edge_count: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)
        for room in floor.get("rooms", []):
            poly = room.get("polygon") or []
            if len(poly) >= 2:
                ring = poly if poly[0] == poly[-1] else poly + [poly[0]]
                for a, b in zip(ring[:-1], ring[1:]):
                    edge_count[_edge_key(a, b)] += 1

        # Collect openings (transformed)
        openings: List[Tuple[float, float, float, float]] = []
        for room in floor.get("rooms", []):
            for pair in room.get("doors", []) + room.get("windows", []):
                if len(pair) >= 2:
                    (x1, y1), (x2, y2) = pair[0], pair[1]
                    X1, Y1 = _t(x1, y1, s, tx, ty)
                    X2, Y2 = _t(x2, y2, s, tx, ty)
                    openings.append((X1, Y1, X2, Y2))

        # Build wall segments in transformed coords
        wall_segments: List[Tuple[float, float, float, float, str]] = []
        for (a_i, b_i), count in edge_count.items():
            ax, ay = a_i
            bx, by = b_i
            X1, Y1 = _t(ax, ay, s, tx, ty)
            X2, Y2 = _t(bx, by, s, tx, ty)
            klass = "outer-wall" if count == 1 else "inner-wall"
            wall_segments.append((X1, Y1, X2, Y2, klass))

        # Cut wall segments by openings to create gaps
        cut_segments: List[Tuple[float, float, float, float, str]] = []
        for X1, Y1, X2, Y2, klass in wall_segments:
            segments = [(X1, Y1, X2, Y2)]
            for ox1, oy1, ox2, oy2 in openings:
                new_segments: List[Tuple[float, float, float, float]] = []
                for sx1, sy1, sx2, sy2 in segments:
                    new_segments.extend(_cut_segment_with_opening(sx1, sy1, sx2, sy2, ox1, oy1, ox2, oy2))
                segments = new_segments
            for sx1, sy1, sx2, sy2 in segments:
                cut_segments.append((sx1, sy1, sx2, sy2, klass))

        # Draw walls as filled strips with true thickness
        for sx1, sy1, sx2, sy2, klass in cut_segments:
            thickness_mm = OUTER_WALL_THICK_MM if klass == "outer-wall" else INNER_WALL_THICK_MM
            # Convert mm to px in current frame scale (geom units presumed to be mm)
            thickness_px = (thickness_mm / 1000.0) * s
            d_strip = _build_wall_strip_polygon(sx1, sy1, sx2, sy2, thickness_px)
            if d_strip:
                out.append(
                    f"<path d='{d_strip}' fill='#000' stroke='none' opacity='0.9'/>"
                )

        # Doors (professional symbol) adjusted to wall thickness
        for room in floor.get("rooms", []):
            for door_poly in room.get("doors", []):
                if len(door_poly) >= 2:
                    (x1, y1), (x2, y2) = door_poly[0], door_poly[1]
                    X1, Y1 = _t(x1, y1, s, tx, ty)
                    X2, Y2 = _t(x2, y2, s, tx, ty)
                    out.append(_draw_professional_door(X1, Y1, X2, Y2))

        # Windows (professional symbol inside the gap), shortened to fit wall thickness
        for room in floor.get("rooms", []):
            for win_poly in room.get("windows", []):
                if len(win_poly) >= 2:
                    (x1, y1), (x2, y2) = win_poly[0], win_poly[1]
                    X1, Y1 = _t(x1, y1, s, tx, ty)
                    X2, Y2 = _t(x2, y2, s, tx, ty)
                    t_short = 0.2
                    XX1 = X1 + (X2 - X1) * t_short
                    YY1 = Y1 + (Y2 - Y1) * t_short
                    XX2 = X2 - (X2 - X1) * t_short
                    YY2 = Y2 - (Y2 - Y1) * t_short
                    out.append(_draw_professional_window(XX1, YY1, XX2, YY2))

        # overall dimensions outside plan (based on all points)
        if pts:
            minx = min(p[0] for p in pts)
            maxx = max(p[0] for p in pts)
            miny = min(p[1] for p in pts)
            maxy = max(p[1] for p in pts)
            TX1, TY1 = _t(minx, miny, s, tx, ty)
            TX2, TY2 = _t(maxx, miny, s, tx, ty)
            out.append(_dim_line(TX1, TY1 - 24, TX2, TY2 - 24, f"{(maxx-minx)/1000:.2f}m overall"))
            RX1, RY1 = _t(maxx, miny, s, tx, ty)
            RX2, RY2 = _t(maxx, maxy, s, tx, ty)
            out.append(_dim_line(RX1 + 24, RY1, RX2 + 24, RY2, f"{(maxy-miny)/1000:.2f}m overall"))

        # scale + north arrow per frame
        out.append(_north_arrow(fx + frame_w - 40, fy + 50))
        out.append(_scale_bar(fx + frame_w - 140, fy + frame_h - 50, 100))

        # floor area label
        area = sum(r.get("area_sqm", 0.0) for r in floor.get("rooms", []))
        out.append(f"<text x='{fx+8:.1f}' y='{fy+frame_h-8:.1f}' class='sub'>Approx. {area:.1f} sq. metres</text>")

    total_area = house_data.get("total_area", 0)
    # Title block
    out.append(f"<rect x='{width-260:.1f}' y='{height-110:.1f}' width='240' height='100' fill='white' stroke='#000' stroke-width='1'/>")
    out.append(f"<text x='{width-140:.1f}' y='{height-90:.1f}' class='title' text-anchor='middle'>ARCHITECTURAL FLOOR PLAN</text>")
    out.append(f"<text x='{width-140:.1f}' y='{height-70:.1f}' class='label' text-anchor='middle'>SCALE: 1:100</text>")
    out.append(f"<text x='{width-140:.1f}' y='{height-52:.1f}' class='label' text-anchor='middle'>PROJECT: HOUSEBRAIN</text>")
    out.append(f"<text x='{width-140:.1f}' y='{height-34:.1f}' class='label' text-anchor='middle'>DRAWN BY: HB-AI</text>")
    out.append(f"<text x='{width/2:.1f}' y='{height-8:.1f}' class='sub' text-anchor='middle'>Total area: approx. {total_area:.1f} sq. metres • Generated by HouseBrain v1.1</text>")
    out.append("</svg>")
    return "".join(out)


# --------------------------- CLI ---------------------------

def export_professional_floor_plan(input_path: str, output_path: str, width: int = 1400, height: int = 1000) -> None:
    def _convert_advanced_sample_to_house(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an advanced dataset sample (input/output) into renderer house_data format."""
        out: Dict[str, Any] = sample.get("output", {})
        geo = out.get("geometric_data") or {}
        rooms = geo.get("rooms", [])
        if not rooms:
            return {}
        floors_map: Dict[str, Dict[str, Any]] = {}
        total_area = 0.0
        for r in rooms:
            floor_name = r.get("floor") or "Floor"
            if floor_name not in floors_map:
                floors_map[floor_name] = {"name": floor_name, "rooms": []}
            b = r.get("bounds", {})
            x = float(b.get("x", 0))
            y = float(b.get("y", 0))
            w = float(b.get("width", 0))
            h = float(b.get("height", 0))
            # rectangle polygon (clockwise), origin top-left in dataset; our transform handles orientation
            poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            area = float(b.get("area", w * h))
            total_area += area
            # doors/windows lines from center+width when present
            doors_list = []
            for d in r.get("doors", []):
                dx = float(d.get("x", x + w / 2))
                dy = float(d.get("y", y))
                dw = float(d.get("width", min(w, h) * 0.2))
                doors_list.append([(dx - dw / 2, dy), (dx + dw / 2, dy)])
            windows_list = []
            for win in r.get("windows", []):
                wx = float(win.get("x", x + w - 1))
                wy = float(win.get("y", y + h / 2))
                ww = float(win.get("width", min(w, h) * 0.25))
                windows_list.append([(wx, wy - ww / 2), (wx, wy + ww / 2)])
            floors_map[floor_name]["rooms"].append({
                "name": r.get("id", r.get("type", "Room")),
                "type": r.get("type", "room"),
                "polygon": poly,
                "doors": doors_list,
                "windows": windows_list,
                "area_sqm": area * 0.092903,  # if area in sqft, approx convert to sqm; else acts as scale
            })
        return {"geometry": {"floors": list(floors_map.values())}, "total_area": total_area * 0.092903}

    with open(input_path, 'r', encoding='utf-8') as f:
        house_data = json.load(f)
    # Auto-convert advanced dataset sample structure
    if "geometry" not in house_data and isinstance(house_data, dict) and "output" in house_data:
        converted = _convert_advanced_sample_to_house(house_data)
        if converted:
            house_data = converted
    svg_content = create_professional_floor_plan(house_data, width, height)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"🏗️ Professional floor plan created: {output_path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Create professional floor plans matching real architectural standards")
    ap.add_argument("--input", required=True, help="HouseBrain JSON file")
    ap.add_argument("--output", required=True, help="Output SVG file")
    ap.add_argument("--width", type=int, default=1400, help="SVG width")
    ap.add_argument("--height", type=int, default=1000, help="SVG height")
    args = ap.parse_args()
    export_professional_floor_plan(args.input, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
