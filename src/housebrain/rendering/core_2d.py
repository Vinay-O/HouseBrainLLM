from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.housebrain.schema import HouseOutput, Level, Room, Point2D, Door, Window


class Professional2DRenderer:
    """
    A new renderer designed to work directly with the HouseOutput Pydantic schema.
    It infers wall layouts from room boundaries.
    """

    def __init__(self, house_plan: HouseOutput, sheet_mode: str = "floor"):
        self.plan = house_plan
        self.sheet_mode = sheet_mode
        self.walls: List[Dict] = []
        self.openings: List[Dict] = []
        self.spaces: List[Dict] = []

        # Conversion factor from feet (schema) to mm (legacy renderer logic)
        self.ft_to_mm = 304.8

        # Process the first level for now
        if self.plan.levels:
            self._process_level(self.plan.levels[0])

    def _process_level(self, level: Level):
        """Infers walls and populates internal structures from a Level object."""
        
        # 1. Convert Rooms from schema to a simpler dict format for processing
        self.spaces = [
            {
                "id": room.id,
                "name": room.type.value.replace('_', ' ').title(),
                "type": room.type.value,
                "boundary": [
                    (room.bounds.x * self.ft_to_mm, room.bounds.y * self.ft_to_mm),
                    ((room.bounds.x + room.bounds.width) * self.ft_to_mm, room.bounds.y * self.ft_to_mm),
                    ((room.bounds.x + room.bounds.width) * self.ft_to_mm, (room.bounds.y + room.bounds.height) * self.ft_to_mm),
                    (room.bounds.x * self.ft_to_mm, (room.bounds.y + room.bounds.height) * self.ft_to_mm),
                ]
            }
            for room in level.rooms
        ]

        # 2. Infer walls from room boundaries
        edge_counts: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
        
        for room in level.rooms:
            b = room.bounds
            points = [
                (b.x, b.y), (b.x + b.width, b.y),
                (b.x + b.width, b.y + b.height), (b.x, b.y + b.height)
            ]
            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                # Sort points to make edges canonical
                edge = tuple(sorted((p1, p2)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        wall_id_counter = 0
        for edge, count in edge_counts.items():
            p1, p2 = edge
            wall_type = "interior" if count > 1 else "exterior"
            self.walls.append({
                "id": f"W{wall_id_counter}",
                "start": (p1[0] * self.ft_to_mm, p1[1] * self.ft_to_mm),
                "end": (p2[0] * self.ft_to_mm, p2[1] * self.ft_to_mm),
                "type": wall_type,
                "thickness": (115 if wall_type == "interior" else 230), # in mm
            })
            wall_id_counter += 1

        # 3. Process doors and windows (needs more logic to map to inferred walls)
        # This is a placeholder and a complex step.
        # For now, we are just creating the structures.
        opening_id_counter = 0
        for room in level.rooms:
            for door in room.doors:
                self.openings.append({
                    "id": f"D{opening_id_counter}",
                    "wall_id": "TODO",  # This needs to be calculated
                    "type": "door",
                    "position": 0.5,  # Placeholder
                    "width": door.width * self.ft_to_mm,
                    "metadata": {}
                })
                opening_id_counter += 1
            for window in room.windows:
                self.openings.append({
                    "id": f"W{opening_id_counter}",
                    "wall_id": "TODO",  # This needs to be calculated
                    "type": "window",
                    "position": 0.5,  # Placeholder
                    "width": window.width * self.ft_to_mm,
                    "metadata": {}
                })
                opening_id_counter += 1
    
    def _line_dir(self, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        L = math.hypot(dx, dy)
        if L == 0:
            return 0.0, 0.0, 0.0
        return dx / L, dy / L, L

    def _wall_strip(self, a: Tuple[float, float], b: Tuple[float, float], t: float) -> str:
        ux, uy, L = self._line_dir(a, b)
        if L == 0:
            return ""
        px, py = -uy, ux
        off = t / 2.0
        ax = a[0] + px * off
        ay = a[1] + py * off
        bx = b[0] + px * off
        by = b[1] + py * off
        cx = b[0] - px * off
        cy = b[1] - py * off
        dx = a[0] - px * off
        dy = a[1] - py * off
        return f"M {ax:.1f} {ay:.1f} L {bx:.1f} {by:.1f} L {cx:.1f} {cy:.1f} L {dx:.1f} {dy:.1f} Z"

    def render(self, width: int = 1800, height: int = 1200) -> str:
        """
        Main rendering method.
        """
        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<defs>",
            "<style>",
            "/* Professional Line Weight Hierarchy (CAD Standard) */",
            ".wall-exterior { fill: #111; opacity: 1.0; stroke: #000; stroke-width: 2.5; }",
            ".wall-interior { fill: #111; opacity: 1.0; stroke: #000; stroke-width: 1.8; }",
            ".label { font-family: Arial, Helvetica, sans-serif; font-size: 11px; fill: #111; }",
            ".dim { stroke: #111; stroke-width: 0.75; fill: none; }",
            ".dimtxt { font-family: Arial, Helvetica, sans-serif; font-size: 10px; fill: #111; paint-order: stroke fill; stroke: #fff; stroke-width: 2px; }",
            ".roomfill { fill: #F8F8F8; }",
            "</style>",
            "</defs>",
            f"<rect width='{width}' height='{height}' fill='white'/>",
        ]

        # Simple fit: compute bounds of all points
        pts: List[Tuple[float, float]] = []
        for w in self.walls:
            pts.extend([w["start"], w["end"]])
        for s in self.spaces:
            pts.extend(s["boundary"])

        if not pts:
            svg.append("</svg>")
            return "".join(svg)

        minx = min(p[0] for p in pts)
        miny = min(p[1] for p in pts)
        maxx = max(p[0] for p in pts)
        maxy = max(p[1] for p in pts)
        w = max(1.0, maxx - minx)
        h = max(1.0, maxy - miny)
        margin = 80.0
        sx = (width - margin * 2) / w
        sy = (height - margin * 2) / h
        s = min(sx, sy)
        tx = margin - minx * s
        ty = margin + maxy * s

        def T(x: float, y: float) -> Tuple[float, float]:
            return (x * s + tx, -y * s + ty)

        # Render spaces (fills)
        svg.append("<g id='spaces'>")
        for sp in self.spaces:
            if len(sp["boundary"]) >= 3:
                pts_t = [T(x, y) for x, y in sp["boundary"]]
                d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts_t) + " Z"
                svg.append(f"<path d='{d}' class='roomfill' stroke='none'/>")
        svg.append("</g>")

        # Render walls
        svg.append("<g id='walls'>")
        # Visual wall thickness in pixels
        outer_wall_px = 8.0
        inner_wall_px = 5.0
        for wll in self.walls:
            X1, Y1 = T(*wll["start"])
            X2, Y2 = T(*wll["end"])
            thickness = outer_wall_px if wll["type"] == "exterior" else inner_wall_px
            d = self._wall_strip((X1, Y1), (X2, Y2), thickness)
            if d:
                klass = f"wall-{wll['type']}"
                svg.append(f"<path d='{d}' class='{klass}'/>")
        svg.append("</g>")
        
        # Render labels
        svg.append("<g id='annotations'>")
        for sp in self.spaces:
             if not sp["boundary"]:
                continue
             cx = sum(p[0] for p in sp["boundary"]) / len(sp["boundary"])
             cy = sum(p[1] for p in sp["boundary"]) / len(sp["boundary"])
             CX, CY = T(cx, cy)
             svg.append(f"<text x='{CX:.1f}' y='{CY-10:.1f}' class='label' text-anchor='middle' font-weight='bold'>{sp['name']}</text>")
        svg.append("</g>")


        svg.append("</svg>")
        return "".join(svg)

def render_2d_plan(house_plan: HouseOutput, output_dir: Path, base_filename: str):
    """
    High-level function to generate and save 2D floor plans.
    """
    logger = logging.getLogger(__name__)
    renderer = Professional2DRenderer(house_plan)
    
    # For now, just a basic render
    svg_content = renderer.render()
    
    output_path = output_dir / f"{base_filename}_2d_floor_plan.svg"
    try:
        output_path.write_text(svg_content, encoding="utf-8")
        logger.info(f"✅ Successfully rendered 2D plan to {output_path.resolve()}")
    except Exception as e:
        logger.error(f"❌ Failed to write 2D plan SVG: {e}", exc_info=True)
