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

        # 3. Process doors and windows
        self._process_openings(level)
    
    def _find_closest_wall_segment(self, point: Point2D, room_id: str, adjacent_room_id: str = None) -> Tuple[str, float] | None:
        """
        Finds the closest wall segment to a given point.

        For doors, it looks for a wall shared between room_id and adjacent_room_id.
        For windows, it looks for an exterior wall of room_id.

        Returns: A tuple of (wall_id, position_along_wall) or None if not found.
        """
        candidate_walls = []
        
        # This is a simplification. A robust solution would use a spatial index (e.g., quadtree)
        # and a more complex geometric analysis to find which rooms an edge belongs to.
        # For now, we find walls that are "close" to the room's boundary.
        
        # Find the room's bounds to narrow down the search
        target_room = next((r for r in self.plan.levels[0].rooms if r.id == room_id), None)
        if not target_room:
            return None

        px, py = point.x * self.ft_to_mm, point.y * self.ft_to_mm
        min_dist = float('inf')
        best_wall = None
        
        for wall in self.walls:
            w_start = wall['start']
            w_end = wall['end']
            
            # Vector from wall start to end
            wx, wy = w_end[0] - w_start[0], w_end[1] - w_start[1]
            
            # Vector from wall start to point
            px_rel, py_rel = px - w_start[0], py - w_start[1]
            
            dot_product = px_rel * wx + py_rel * wy
            len_sq = wx*wx + wy*wy
            
            if len_sq == 0: # Should not happen for a valid wall
                continue
                
            t = max(0, min(1, dot_product / len_sq))
            
            # Closest point on the line segment
            closest_x = w_start[0] + t * wx
            closest_y = w_start[1] + t * wy
            
            # Distance from point to the wall segment
            dist_sq = (px - closest_x)**2 + (py - closest_y)**2
            
            if dist_sq < min_dist:
                # Check if this wall is a plausible candidate for the room.
                # This is a heuristic: we check if the wall's midpoint is near the room's boundary.
                wall_mid_x = (w_start[0] + w_end[0]) / 2
                wall_mid_y = (w_start[1] + w_end[1]) / 2
                
                room_b = target_room.bounds
                room_b_mm = {
                    'x': room_b.x * self.ft_to_mm,
                    'y': room_b.y * self.ft_to_mm,
                    'width': room_b.width * self.ft_to_mm,
                    'height': room_b.height * self.ft_to_mm,
                }
                
                # Check if the wall's midpoint lies on one of the room's four edges (with a tolerance)
                tolerance = 10.0 # mm
                on_horizontal = (abs(wall_mid_y - room_b_mm['y']) < tolerance or abs(wall_mid_y - (room_b_mm['y'] + room_b_mm['height'])) < tolerance) and \
                                (room_b_mm['x'] - tolerance <= wall_mid_x <= room_b_mm['x'] + room_b_mm['width'] + tolerance)
                on_vertical = (abs(wall_mid_x - room_b_mm['x']) < tolerance or abs(wall_mid_x - (room_b_mm['x'] + room_b_mm['width'])) < tolerance) and \
                              (room_b_mm['y'] - tolerance <= wall_mid_y <= room_b_mm['y'] + room_b_mm['height'] + tolerance)

                if on_horizontal or on_vertical:
                    min_dist = dist_sq
                    # Position is the fractional distance 't' along the wall
                    best_wall = (wall['id'], t)

        # We accept a match if it's very close to the point (e.g., within ~6 inches)
        if min_dist < (150.0)**2:
             return best_wall
        
        return None

    def _process_openings(self, level: Level):
        """Processes doors and windows, mapping them to inferred walls."""
        opening_id_counter = 0
        for room in level.rooms:
            # Process doors associated with this room
            for door in room.doors:
                 # Doors are often listed in both rooms they connect. We only process them once.
                if any(o.get('metadata', {}).get('original_id') == door.position for o in self.openings):
                    continue

                wall_info = self._find_closest_wall_segment(door.position, door.room1, door.room2)
                if wall_info:
                    wall_id, position_on_wall = wall_info
                    self.openings.append({
                        "id": f"D{opening_id_counter}",
                        "wall_id": wall_id,
                        "type": "door",
                        "position": position_on_wall,
                        "width": door.width * self.ft_to_mm,
                        "metadata": {
                            "swing": "in", # Default, can be refined
                            "handing": "RHR", # Default
                            "original_id": door.position # To prevent duplicates
                        }
                    })
                    opening_id_counter += 1
                else:
                    logging.warning(f"Could not find a wall for door at ({door.position.x}, {door.position.y})")

            # Process windows
            for window in room.windows:
                wall_info = self._find_closest_wall_segment(window.position, window.room_id)
                if wall_info:
                    wall_id, position_on_wall = wall_info
                    self.openings.append({
                        "id": f"W{opening_id_counter}",
                        "wall_id": wall_id,
                        "type": "window",
                        "position": position_on_wall,
                        "width": window.width * self.ft_to_mm,
                        "metadata": {
                           "window_operation": "fixed" # Default
                        }
                    })
                    opening_id_counter += 1
                else:
                     logging.warning(f"Could not find a wall for window at ({window.position.x}, {window.position.y})")

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

    def _opening_span(self, wall: Dict, opening: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        ux, uy, L = self._line_dir(wall["start"], wall["end"])
        oL = opening["width"]
        # position along wall measured from start
        s = opening["position"] * L
        cx = wall["start"][0] + ux * s
        cy = wall["start"][1] + uy * s
        ax = cx - ux * (oL / 2)
        ay = cy - uy * (oL / 2)
        bx = cx + ux * (oL / 2)
        by = cy + uy * (oL / 2)
        return (ax, ay), (bx, by)

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
            ".door { stroke: #8B4513; stroke-width: 1.2; fill: none; }",
            ".door-swing { stroke: #8B4513; stroke-width: 0.8; fill: none; stroke-dasharray: 3,1; }",
            ".window { stroke: #0066CC; stroke-width: 1.2; fill: white; stroke-opacity: 0.9; }",
            ".window-sill { stroke: #0066CC; stroke-width: 2.0; fill: none; }",
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

        walls_by_id = {w['id']: w for w in self.walls}

        for wll in self.walls:
            # Collect openings on this wall
            wall_openings = [op for op in self.openings if op['wall_id'] == wll['id']]
            
            # Start with the full wall segment
            spans = [(wll["start"], wll["end"])]
            
            # Sequentially cut out each opening from the spans
            for op in wall_openings:
                (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
                new_spans = []
                for (sx1, sy1), (sx2, sy2) in spans:
                    ux, uy, L = self._line_dir((sx1, sy1), (sx2, sy2))
                    if L == 0: continue
                    
                    def proj(p): return (p[0] - sx1) * ux + (p[1] - sy1) * uy
                    
                    op_start_proj = proj((ox1, oy1))
                    op_end_proj = proj((ox2, oy2))
                    
                    lo = min(op_start_proj, op_end_proj)
                    hi = max(op_start_proj, op_end_proj)
                    
                    # If the opening is completely outside the span, keep the span
                    if hi < 0 or lo > L:
                        new_spans.append(((sx1, sy1), (sx2, sy2)))
                        continue
                        
                    # Left segment
                    if lo > 0:
                        new_spans.append(((sx1, sy1), (sx1 + ux * lo, sy1 + uy * lo)))
                    # Right segment
                    if hi < L:
                        new_spans.append(((sx1 + ux * hi, sy1 + uy * hi), (sx2, sy2)))
                
                spans = new_spans
            
            # Render the remaining wall segments
            for (sx1, sy1), (sx2, sy2) in spans:
                X1, Y1 = T(sx1, sy1)
                X2, Y2 = T(sx2, sy2)
                thickness = outer_wall_px if wll["type"] == "exterior" else inner_wall_px
                d = self._wall_strip((X1, Y1), (X2, Y2), thickness)
                if d:
                    klass = f"wall-{wll['type']}"
                    svg.append(f"<path d='{d}' class='{klass}'/>")
        svg.append("</g>")

        # Render Openings
        svg.append("<g id='openings'>")
        for op in self.openings:
            wll = walls_by_id.get(op['wall_id'])
            if not wll:
                continue

            (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
            X1, Y1 = T(ox1, oy1)
            X2, Y2 = T(ox2, oy2)
            angle = math.degrees(math.atan2(Y2 - Y1, X2 - X1))
            length = math.hypot(X2 - X1, Y2 - Y1)
            thickness_px = outer_wall_px if wll["type"] == "exterior" else inner_wall_px
            
            if op['type'] == 'door':
                h = max(6.0, min(12.0, 0.9 * thickness_px))
                handing = op['metadata'].get("handing", "RHR").upper()
                swing_dir = op['metadata'].get("swing", "in").lower()
                
                hinge_at_end = handing.startswith("R")
                
                svg.append(f"<g transform='translate({X1:.1f},{Y1:.1f}) rotate({angle:.1f})'>")
                inner_tx = length if hinge_at_end else 0.0
                inner_sx = -1 if hinge_at_end else 1
                svg.append(f"<g transform='translate({inner_tx:.1f},0) scale({inner_sx},1)'>")
                
                # Door leaf
                svg.append(f"<rect x='0' y='-{h/4:.1f}' width='{length:.1f}' height='{h/2:.1f}' class='door'/>")
                
                # Swing arc
                swing_radius = length
                sy = 1.0 if swing_dir == "in" else -1.0
                end_x_arc = swing_radius * math.cos(math.radians(-90))
                end_y_arc = swing_radius * math.sin(math.radians(-90))
                svg.append(f"<path d='M 0 0 A {swing_radius:.1f} {swing_radius:.1f} 0 0 {1 if sy > 0 else 0} {end_x_arc:.1f} {sy*end_y_arc:.1f}' class='door-swing'/>")

                svg.append("</g></g>") # Close transforms

            elif op['type'] == 'window':
                h = max(8.0, min(16.0, 0.9 * thickness_px))
                svg.append(f"<g transform='translate({X1:.1f},{Y1:.1f}) rotate({angle:.1f})'>")
                # Window frame
                svg.append(f"<rect x='0' y='-{h/2:.1f}' width='{length:.1f}' height='{h:.1f}' class='window'/>")
                # Center mullion
                svg.append(f"<line x1='{length/2:.1f}' y1='-{h/2:.1f}' x2='{length/2:.1f}' y2='{h/2:.1f}' stroke='#0066CC' stroke-width='1.0'/>")
                svg.append("</g>")

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
