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

    def __init__(self, house_plan: HouseOutput, sheet_mode: str = "floor", level_to_render_idx: int = 0):
        self.plan = house_plan
        self.sheet_mode = sheet_mode
        self.level_to_render_idx = level_to_render_idx
        self.walls: List[Dict] = []
        self.openings: List[Dict] = []
        self.spaces: List[Dict] = []

        # Conversion factor from feet (schema) to mm (legacy renderer logic)
        self.ft_to_mm = 304.8

        # Process the specified level
        if self.plan.levels and len(self.plan.levels) > self.level_to_render_idx:
            self._process_level(self.plan.levels[self.level_to_render_idx])

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

    def _mm_to_feet_inches_str(self, mm: float) -> str:
        """Convert millimeters to feet and inches string format."""
        if mm < 0:
            return "-" + self._mm_to_feet_inches_str(abs(mm))
        inches = mm / 25.4
        feet = int(inches // 12)
        remaining_inches = inches % 12
        
        if feet > 0:
            return f"{feet}'-{remaining_inches:.0f}\""
        else:
            return f"{remaining_inches:.1f}\""
            
    def _add_chained_dimensions(self, svg: List[str], T, margin: float):
        """Add chained dimensioning system for the building exterior."""
        if not self.walls:
            return

        all_points = []
        for w in self.walls:
            all_points.extend([w["start"], w["end"]])

        minx = min(p[0] for p in all_points)
        miny = min(p[1] for p in all_points)
        maxx = max(p[0] for p in all_points)
        maxy = max(p[1] for p in all_points)
        
        # --- HORIZONTAL CHAINED DIMENSIONS (Bottom of plan) ---
        x_points = sorted(list(set(p[0] for p in all_points)))
        y_dim_line_world = miny - 600 # 600mm below the building
        TY_DIM = T(0, y_dim_line_world)[1]
        
        for i in range(len(x_points) - 1):
            x1, x2 = x_points[i], x_points[i+1]
            dim_mm = x2 - x1
            if dim_mm < 50: continue # Skip tiny dimensions
            
            TX1, _ = T(x1, miny)
            TX2, _ = T(x2, miny)
            
            # Dimension line segment
            svg.append(f"<line x1='{TX1:.1f}' y1='{TY_DIM:.1f}' x2='{TX2:.1f}' y2='{TY_DIM:.1f}' class='dim'/>")
            # Extension lines from building to dimension line
            svg.append(f"<line x1='{TX1:.1f}' y1='{T(x1, miny)[1]:.1f}' x2='{TX1:.1f}' y2='{TY_DIM + 5:.1f}' class='dim'/>")
            svg.append(f"<line x1='{TX2:.1f}' y1='{T(x2, miny)[1]:.1f}' x2='{TX2:.1f}' y2='{TY_DIM + 5:.1f}' class='dim'/>")
            # Dimension text
            dim_text = self._mm_to_feet_inches_str(dim_mm)
            text_x = (TX1 + TX2) / 2
            svg.append(f"<text x='{text_x:.1f}' y='{TY_DIM - 8:.1f}' class='dimtxt' text-anchor='middle'>{dim_text}</text>")
            
        # Overall Horizontal Dimension
        TX1, _ = T(minx, miny)
        TX2, _ = T(maxx, miny)
        TY_OVERALL = TY_DIM + 40
        svg.append(f"<line x1='{TX1:.1f}' y1='{TY_OVERALL:.1f}' x2='{TX2:.1f}' y2='{TY_OVERALL:.1f}' class='dim'/>")
        svg.append(f"<line x1='{TX1:.1f}' y1='{T(minx, miny)[1]:.1f}' x2='{TX1:.1f}' y2='{TY_OVERALL + 5:.1f}' class='dim'/>")
        svg.append(f"<line x1='{TX2:.1f}' y1='{T(maxx, miny)[1]:.1f}' x2='{TX2:.1f}' y2='{TY_OVERALL + 5:.1f}' class='dim'/>")
        dim_text = self._mm_to_feet_inches_str(maxx - minx)
        svg.append(f"<text x='{(TX1 + TX2) / 2:.1f}' y='{TY_OVERALL - 8:.1f}' class='dimtxt' text-anchor='middle'>{dim_text}</text>")

        # --- VERTICAL CHAINED DIMENSIONS (Left of plan) ---
        y_points = sorted(list(set(p[1] for p in all_points)))
        x_dim_line_world = minx - 600 # 600mm left of the building
        TX_DIM = T(x_dim_line_world, 0)[0]
        
        for i in range(len(y_points) - 1):
            y1, y2 = y_points[i], y_points[i+1]
            dim_mm = y2 - y1
            if dim_mm < 50: continue

            _, TY1 = T(minx, y1)
            _, TY2 = T(minx, y2)
            
            svg.append(f"<line x1='{TX_DIM:.1f}' y1='{TY1:.1f}' x2='{TX_DIM:.1f}' y2='{TY2:.1f}' class='dim'/>")
            svg.append(f"<line x1='{T(minx, y1)[0]:.1f}' y1='{TY1:.1f}' x2='{TX_DIM - 5:.1f}' y2='{TY1:.1f}' class='dim'/>")
            svg.append(f"<line x1='{T(minx, y2)[0]:.1f}' y1='{TY2:.1f}' x2='{TX_DIM - 5:.1f}' y2='{TY2:.1f}' class='dim'/>")
            dim_text = self._mm_to_feet_inches_str(dim_mm)
            text_y = (TY1 + TY2) / 2
            svg.append(f"<text transform='translate({TX_DIM - 8:.1f}, {text_y:.1f}) rotate(-90)' class='dimtxt' text-anchor='middle'>{dim_text}</text>")

        # Overall Vertical Dimension
        _, TY1 = T(minx, miny)
        _, TY2 = T(minx, maxy)
        TX_OVERALL = TX_DIM - 40
        svg.append(f"<line x1='{TX_OVERALL:.1f}' y1='{TY1:.1f}' x2='{TX_OVERALL:.1f}' y2='{TY2:.1f}' class='dim'/>")
        svg.append(f"<line x1='{T(minx, miny)[0]:.1f}' y1='{TY1:.1f}' x2='{TX_OVERALL - 5:.1f}' y2='{TY1:.1f}' class='dim'/>")
        svg.append(f"<line x1='{T(minx, maxy)[0]:.1f}' y1='{TY2:.1f}' x2='{TX_OVERALL - 5:.1f}' y2='{TY2:.1f}' class='dim'/>")
        dim_text = self._mm_to_feet_inches_str(maxy - miny)
        svg.append(f"<text transform='translate({TX_OVERALL - 8:.1f}, {(TY1+TY2)/2:.1f}) rotate(-90)' class='dimtxt' text-anchor='middle'>{dim_text}</text>")

    def _add_room_details(self, svg: List[str], T):
        """Adds fixtures, furniture, and technical annotations to each room."""
        
        # Add a new layer group for fixtures
        svg.append("<g id='fixtures'>")

        for sp in self.spaces:
            if not sp["boundary"] or len(sp["boundary"]) < 3:
                continue
            
            # --- 1. Calculate Room Geometry ---
            xs = [p[0] for p in sp["boundary"]]
            ys = [p[1] for p in sp["boundary"]]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            w_mm = maxx - minx
            h_mm = maxy - miny
            
            CX, CY = T(cx, cy)
            
            # --- 2. Add Fixtures based on Room Type ---
            space_type = sp["type"].lower()
            
            # Simple furniture/fixture placeholders
            if space_type == "kitchen":
                # Countertop
                svg.append(f"<rect x='{T(minx, maxy - 600)[0]:.1f}' y='{T(minx, maxy - 600)[1]:.1f}' width='{w_mm*self.s:.1f}' height='{600*self.s:.1f}' fill='none' stroke='#aaa' stroke-width='1'/>")
                # Sink
                svg.append(f"<rect x='{T(cx - 300, maxy - 450)[0]:.1f}' y='{T(cx - 300, maxy - 450)[1]:.1f}' width='{600*self.s:.1f}' height='{300*self.s:.1f}' fill='white' stroke='#333' stroke-width='1'/>")
            elif space_type == "bathroom":
                # Toilet
                svg.append(f"<circle cx='{T(minx + 400, miny + 400)[0]:.1f}' cy='{T(minx + 400, miny + 400)[1]:.1f}' r='{200*self.s:.1f}' class='fixture' fill='white'/>")
                # Sink
                svg.append(f"<rect x='{T(maxx - 800, miny + 200)[0]:.1f}' y='{T(maxx - 800, miny + 200)[1]:.1f}' width='{600*self.s:.1f}' height='{400*self.s:.1f}' fill='white' class='fixture'/>")
            elif "bedroom" in space_type:
                 # Bed (Queen size ~ 5'x6.6' -> 1524mm x 2032mm)
                bed_w, bed_h = 1524 * self.s, 2032 * self.s
                svg.append(f"<rect x='{CX - bed_w/2:.1f}' y='{CY - bed_h/2:.1f}' width='{bed_w:.1f}' height='{bed_h:.1f}' fill='none' stroke='#888' stroke-width='1.5'/>")
                # Pillow
                svg.append(f"<rect x='{CX - bed_w/2 + 5:.1f}' y='{CY - bed_h/2 + 5:.1f}' width='{bed_w - 10:.1f}' height='{bed_h * 0.2:.1f}' fill='none' stroke='#aaa' stroke-width='1'/>")

            # --- 3. Add Technical Annotations ---
            
            # Area calculation
            area_m2 = (w_mm / 1000) * (h_mm / 1000)
            area_sqft = area_m2 * 10.7639
            
            # Update main label with area
            svg.append(f"<text x='{CX:.1f}' y='{CY:.1f}' class='sub' text-anchor='middle'>{area_sqft:.0f} sq ft</text>")

        svg.append("</g>") # Close fixtures group

    def _add_title_block(self, svg: List[str], width: int, height: int):
        """Adds a professional title block to the bottom right of the sheet."""
        block_x = width - 350
        block_y = height - 150
        block_w = 330
        block_h = 130
        
        svg.append(f"<g id='title-block' transform='translate({block_x}, {block_y})'>")
        svg.append(f"<rect x='0' y='0' width='{block_w}' height='{block_h}' fill='white' stroke='#333' stroke-width='1.5'/>")
        
        # Main Title
        svg.append(f"<text x='{block_w/2}' y='30' class='label' font-size='16' font-weight='bold' text-anchor='middle'>ARCHITECTURAL FLOOR PLAN</text>")
        
        # Project Info
        svg.append(f"<line x1='10' y1='50' x2='{block_w-10}' y2='50' stroke='#333' stroke-width='0.5'/>")
        svg.append(f"<text x='15' y='68' class='label' font-size='10'>PROJECT:</text>")
        svg.append(f"<text x='100' y='68' class='label' font-size='10' font-weight='bold'>HouseBrain AI Residence</text>")
        
        svg.append(f"<text x='15' y='88' class='label' font-size='10'>CLIENT:</text>")
        svg.append(f"<text x='100' y='88' class='label' font-size='10'>[Client Name]</text>")

        # Drawing Info
        svg.append(f"<line x1='10' y1='100' x2='{block_w-10}' y2='100' stroke='#333' stroke-width='0.5'/>")
        svg.append(f"<text x='15' y='118' class='label' font-size='9'>SCALE: As Noted</text>")
        svg.append(f"<text x='150' y='118' class='label' font-size='9'>DRAWN BY: HB-AI</text>")
        svg.append(f"<text x='{block_w-15}' y='118' class='label' font-size='12' font-weight='bold' text-anchor='end'>A-101</text>")
        
        svg.append("</g>")

    def _add_north_arrow(self, svg: List[str], width: int, height: int):
        """Adds a North arrow to the top right."""
        NAx, NAy = width - 100, 100
        na_size = 30
        svg.append(f"<g id='north-arrow' transform='translate({NAx},{NAy})'>")
        svg.append(f"<circle cx='0' cy='0' r='{na_size}' fill='white' stroke='#333' stroke-width='1'/>")
        svg.append(f"<polygon points='0,{-na_size+5} {na_size/4},0 0,{na_size-15} {-na_size/4},0' fill='#333'/>")
        svg.append(f"<text x='0' y='{-na_size-8}' text-anchor='middle' class='label' font-weight='bold' font-size='14'>N</text>")
        svg.append("</g>")

    def _add_stairs(self, svg: List[str], T):
        """Draws staircases on the plan."""
        svg.append("<g id='stairs'>")
        # Access the stairs from the specific level being rendered.
        current_level_number = self.plan.levels[self.level_to_render_idx].level_number
        current_level_stairs = self.plan.levels[self.level_to_render_idx].stairs
        
        for stair in current_level_stairs:
            # We only draw the stair representation on its starting floor.
            if stair.floor_from != current_level_number:
                continue

            pos = (stair.position.x * self.ft_to_mm, stair.position.y * self.ft_to_mm)
            w = stair.width * self.ft_to_mm
            l = stair.length * self.ft_to_mm
            
            # For simplicity, we assume stairs are axis-aligned and find the nearest stairwell room boundary.
            # A more robust solution would use rotation.
            stair_room = next((s for s in self.spaces if s["type"] == "stairwell"), None)
            if not stair_room:
                continue # Cannot draw stairs without a stairwell

            # Simplified: Assume stair aligns with the shorter dimension of the stairwell
            xs = [p[0] for p in stair_room["boundary"]]
            ys = [p[1] for p in stair_room["boundary"]]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            
            # Draw treads
            treads = 12
            if (maxx - minx) < (maxy - miny): # Vertical stairwell
                for i in range(1, treads):
                    y = miny + (i * (maxy - miny) / treads)
                    X1, Y1 = T(minx + 50, y)
                    X2, Y2 = T(maxx - 50, y)
                    svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#888' stroke-width='1'/>")
                # Arrow UP
                AX, AY = T((minx+maxx)/2, miny + (maxy-miny)*0.25)
                svg.append(f"<path d='M {AX:.1f} {AY:.1f} l 0 -15 l -5 5 l 5 -5 l 5 5' stroke='#111' stroke-width='1.5' fill='none'/>")
                svg.append(f"<text x='{AX+8:.1f}' y='{AY-8:.1f}' class='sub' font-size='9'>UP</text>")
            else: # Horizontal stairwell
                for i in range(1, treads):
                    x = minx + (i * (maxx-minx) / treads)
                    X1, Y1 = T(x, miny + 50)
                    X2, Y2 = T(x, maxy - 50)
                    svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#888' stroke-width='1'/>")
                 # Arrow UP
                AX, AY = T(minx + (maxx-minx)*0.25, (miny+maxy)/2)
                svg.append(f"<path d='M {AX:.1f} {AY:.1f} l -15 0 l 5 -5 l -5 5 l 5 5' stroke='#111' stroke-width='1.5' fill='none'/>")
                svg.append(f"<text x='{AX-20:.1f}' y='{AY+4:.1f}' class='sub' font-size='9'>UP</text>")

        svg.append("</g>")

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
        
        # Increase margin dynamically to make space for dimensions
        # Check if dimensions will be added. For now, we assume they always are.
        margin_x_left = 180.0
        margin_x_right = 80.0
        margin_y_top = 80.0
        margin_y_bottom = 180.0

        sx = (width - margin_x_left - margin_x_right) / w
        sy = (height - margin_y_top - margin_y_bottom) / h
        s = min(sx, sy)
        self.s = s # Store scale factor for use in other methods
        tx = margin_x_left - minx * s
        ty = margin_y_top + maxy * s

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

        # Render Room Details (Fixtures, internal annotations)
        self._add_room_details(svg, T)

        # Render Stairs
        self._add_stairs(svg, T)

        # Render Dimensions
        svg.append("<g id='dimensions'>")
        self._add_chained_dimensions(svg, T, margin)
        svg.append("</g>")

        # Add Title Block and North Arrow
        self._add_north_arrow(svg, width, height)
        self._add_title_block(svg, width, height)


        svg.append("</svg>")
        return "".join(svg)

def render_2d_plan(house_plan: HouseOutput, output_dir: Path, base_filename: str):
    """
    High-level function to generate and save 2D floor plans for each level.
    """
    logger = logging.getLogger(__name__)

    if not house_plan.levels:
        logger.warning("No levels found in the house plan. Nothing to render.")
        return

    for i, level in enumerate(house_plan.levels):
        logger.info(f"Rendering Level {level.level_number}...")
        
        # Pass the full plan, but also the index of the level to render
        renderer = Professional2DRenderer(house_plan, level_to_render_idx=i)
        
        svg_content = renderer.render()
        
        output_path = output_dir / f"{base_filename}_level_{level.level_number}.svg"
        try:
            output_path.write_text(svg_content, encoding="utf-8")
            logger.info(f"✅ Successfully rendered Level {level.level_number} to {output_path.resolve()}")
        except Exception as e:
            logger.error(f"❌ Failed to write SVG for Level {level.level_number}: {e}", exc_info=True)
