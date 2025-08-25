from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
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
    
    @staticmethod
    def _get_room_edges(room: Room) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns a list of canonical edges for a room's bounding box."""
        b = room.bounds
        points = [
            (b.x, b.y), (b.x + b.width, b.y),
            (b.x + b.width, b.y + b.height), (b.x, b.y + b.height)
        ]
        edges = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            edge = tuple(sorted((p1, p2)))  # Canonical representation
            edges.append(edge)
        return edges

    def _find_wall_from_edge(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> Dict | None:
        """Finds a wall in self.walls that corresponds to a given geometric edge."""
        edge_mm = tuple(sorted(((edge[0][0] * self.ft_to_mm, edge[0][1] * self.ft_to_mm),
                                (edge[1][0] * self.ft_to_mm, edge[1][1] * self.ft_to_mm))))
        
        tolerance = 1.0 # mm
        for wall in self.walls:
            wall_edge = tuple(sorted((wall['start'], wall['end'])))
            
            dist1 = math.hypot(wall_edge[0][0] - edge_mm[0][0], wall_edge[0][1] - edge_mm[0][1])
            dist2 = math.hypot(wall_edge[1][0] - edge_mm[1][0], wall_edge[1][1] - edge_mm[1][1])

            if dist1 < tolerance and dist2 < tolerance:
                return wall
        return None

    def _find_host_wall_for_opening(
        self,
        opening: Union[Door, Window],
        level: Level
    ) -> Tuple[Dict, float] | None:
        """
        Determines the correct host wall for a door or window and its position along it.
        This new method uses topology (shared walls) instead of just proximity.
        """
        all_rooms_by_id = {room.id: room for room in level.rooms}
        
        target_edge = None
        
        # --- 1. Identify the correct geometric edge for the opening ---
        if isinstance(opening, Door):
            room1 = all_rooms_by_id.get(opening.room1)
            room2 = all_rooms_by_id.get(opening.room2)
            if not room1 or not room2:
                logging.warning(f"Door {opening.position} references non-existent room.")
                return None
            
            edges1 = set(self._get_room_edges(room1))
            edges2 = set(self._get_room_edges(room2))
            
            shared_edges = edges1.intersection(edges2)
            if not shared_edges:
                logging.warning(f"Door between {opening.room1} and {opening.room2} has no shared wall.")
                return None
            target_edge = shared_edges.pop()

        elif isinstance(opening, Window):
            room = all_rooms_by_id.get(opening.room_id)
            if not room:
                logging.warning(f"Window {opening.position} references non-existent room.")
                return None
            
            room_edges = self._get_room_edges(room)
            other_rooms = [r for r in level.rooms if r.id != room.id]
            
            exterior_edges = []
            for edge in room_edges:
                is_exterior = True
                for other_room in other_rooms:
                    if edge in self._get_room_edges(other_room):
                        is_exterior = False
                        break
                if is_exterior:
                    exterior_edges.append(edge)

            if not exterior_edges:
                logging.warning(f"Window in {opening.room_id} has no exterior wall.")
                return None
            
            # Find the closest exterior edge to the window's position
            min_dist_sq = float('inf')
            best_edge = None
            px = opening.position.x
            py = opening.position.y

            for edge in exterior_edges:
                p1, p2 = edge
                dist_sq = self._point_segment_dist_sq((px, py), p1, p2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_edge = edge
            target_edge = best_edge
            
        if not target_edge:
            return None
            
        # --- 2. Find the Wall Dictionary corresponding to the edge ---
        wall = self._find_wall_from_edge(target_edge)
        if not wall:
            logging.warning(f"Could not find wall object for edge {target_edge}")
            return None

        # --- 3. Project the opening's point onto the wall to get its fractional position ---
        w_start_ft = (wall['start'][0] / self.ft_to_mm, wall['start'][1] / self.ft_to_mm)
        w_end_ft = (wall['end'][0] / self.ft_to_mm, wall['end'][1] / self.ft_to_mm)
        
        wx, wy = w_end_ft[0] - w_start_ft[0], w_end_ft[1] - w_start_ft[1]
        px_rel, py_rel = opening.position.x - w_start_ft[0], opening.position.y - w_start_ft[1]
        
        len_sq = wx*wx + wy*wy
        if len_sq == 0: return None
        
        t = (px_rel * wx + py_rel * wy) / len_sq
        position_on_wall = max(0, min(1, t)) # Clamp to [0, 1]

        return wall, position_on_wall

    @staticmethod
    def _point_segment_dist_sq(p, a, b):
        """Calculates the squared distance from point p to line segment ab."""
        px, py = p
        ax, ay = a
        bx, by = b

        # Vector from a to b
        ab_x, ab_y = bx - ax, by - ay
        # Vector from a to p
        ap_x, ap_y = px - ax, py - ay

        ab_len_sq = ab_x*ab_x + ab_y*ab_y
        if ab_len_sq == 0:
            return ap_x*ap_x + ap_y*ap_y

        t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
        t = max(0, min(1, t)) # Clamp to segment

        closest_x = ax + t * ab_x
        closest_y = ay + t * ab_y
        
        return (px - closest_x)**2 + (py - closest_y)**2


    def _process_openings(self, level: Level):
        """Processes doors and windows, mapping them to inferred walls."""
        opening_id_counter = 0
        
        # Create a set of processed opening positions to avoid duplicates
        # (e.g., a door defined in two adjacent rooms)
        processed_positions = set()

        all_openings = []
        for room in level.rooms:
            for door in room.doors:
                all_openings.append(door)
            for window in room.windows:
                all_openings.append(window)

        for opening in all_openings:
            pos_tuple = (opening.position.x, opening.position.y)
            if pos_tuple in processed_positions:
                continue
            
            processed_positions.add(pos_tuple)

            wall_info = self._find_host_wall_for_opening(opening, level)
            
            if wall_info:
                wall, position_on_wall = wall_info
                
                if isinstance(opening, Door):
                    op_type = "door"
                    op_id = f"D{opening_id_counter}"
                else:
                    op_type = "window"
                    op_id = f"W{opening_id_counter}"
                
                self.openings.append({
                    "id": op_id,
                    "wall_id": wall['id'],
                    "type": op_type,
                    "position": position_on_wall,
                    "width": opening.width * self.ft_to_mm,
                })
                opening_id_counter += 1
            else:
                op_type = "Door" if isinstance(opening, Door) else "Window"
                logging.warning(f"Could not find a host wall for {op_type} at ({opening.position.x}, {opening.position.y})")

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

    def _add_furniture(self, svg: List[str], T):
        """Adds furniture items to the SVG."""
        svg.append("<g id='furniture'>")
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

            for furniture_item in sp.get("furniture", []):
                # Center of furniture relative to the SVG canvas origin (top-left)
                center_x_abs = furniture_item.center.x * self.s + CX
                center_y_abs = (self.drawing_size - furniture_item.center.y) * self.s + CY

                width = furniture_item.width * self.s
                height = furniture_item.height * self.s

                # Top-left corner for the rect element
                rect_x = center_x_abs - width / 2
                rect_y = center_y_abs - height / 2

                transform = f"rotate({-furniture_item.angle}, {center_x_abs}, {center_y_abs})"
                
                svg.append(
                    f"<rect x='{rect_x:.1f}' y='{rect_y:.1f}' width='{width:.1f}' height='{height:.1f}' class='furniture' transform='{transform}'/>"
                )
        svg.append("</g>")

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
        svg.append(f"<text x='{block_w-15}' y='118' class='label' font-size='12' font-weight='bold' text-anchor='end'>A-{101 + self.level_to_render_idx}</text>")
        
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
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            
            # Draw treads
            treads = 16 # More realistic number for a 10ft ceiling
            is_vertical = (maxx - minx) < (maxy - miny)
            
            if is_vertical: # Vertical stairwell
                for i in range(1, treads):
                    y = miny + (i * (maxy - miny) / treads)
                    X1, Y1 = T(minx + 50, y)
                    X2, Y2 = T(maxx - 50, y)
                    svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' class='stair-tread'/>")
                # Arrow UP
                AX_start, AY_start = T(cx, maxy - 500)
                AX_end, AY_end = T(cx, miny + 500)
                svg.append(f"<line x1='{AX_start:.1f}' y1='{AY_start:.1f}' x2='{AX_end:.1f}' y2='{AY_end:.1f}' class='stair-arrow'/>")
                svg.append(f"<polygon points='{AX_end-4},{AY_end+8} {AX_end+4},{AY_end+8} {AX_end},{AY_end}' class='stair-arrow-head'/>")
                svg.append(f"<text x='{AX_start + 8:.1f}' y='{AY_start - 8:.1f}' class='sub' font-size='9'>UP</text>")
            else: # Horizontal stairwell
                for i in range(1, treads):
                    x = minx + (i * (maxx-minx) / treads)
                    X1, Y1 = T(x, miny + 50)
                    X2, Y2 = T(x, maxy - 50)
                    svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' class='stair-tread'/>")
                 # Arrow UP
                AX_start, AY_start = T(maxx - 500, cy)
                AX_end, AY_end = T(minx + 500, cy)
                svg.append(f"<line x1='{AX_start:.1f}' y1='{AY_start:.1f}' x2='{AX_end:.1f}' y2='{AY_end:.1f}' class='stair-arrow'/>")
                svg.append(f"<polygon points='{AX_end+8},{AY_end-4} {AX_end+8},{AY_end+4} {AX_end},{AY_end}' class='stair-arrow-head'/>")
                svg.append(f"<text x='{AX_start + 8:.1f}' y='{AY_start - 12:.1f}' class='sub' font-size='9'>UP</text>")

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
            ".wall-exterior { fill: #333; stroke: #333; stroke-width: 0.5; }",
            ".wall-interior { fill: #555; stroke: #555; stroke-width: 0.5; }",
            ".door { stroke: #8B4513; stroke-width: 1.2; fill: none; }",
            ".door-swing { stroke: #8B4513; stroke-width: 0.8; fill: none; stroke-dasharray: 3,1; }",
            ".window { stroke: #0066CC; stroke-width: 1.2; fill: white; stroke-opacity: 0.9; }",
            ".window-sill { stroke: #0066CC; stroke-width: 2.0; fill: none; }",
            ".label { font-family: Arial, Helvetica, sans-serif; font-size: 11px; fill: #111; }",
            ".dim { stroke: #111; stroke-width: 0.75; fill: none; }",
            ".dimtxt { font-family: Arial, Helvetica, sans-serif; font-size: 10px; fill: #111; paint-order: stroke fill; stroke: #fff; stroke-width: 2px; }",
            ".roomfill { fill: #F8F8F8; }",
            ".stair-tread { stroke: #999; stroke-width: 0.8; }",
            ".stair-arrow { stroke: #333; stroke-width: 1.5; marker-end: url(#arrowhead); }",
            ".stair-arrow-head { fill: #333; }",
            ".fixture { stroke: #333; stroke-width: 1; }",
            ".furniture { fill: #FFD700; stroke: #FFD700; stroke-width: 1; }", # Gold color for furniture
            "</style>",
            "<marker id='arrowhead' markerWidth='10' markerHeight='7' refX='0' refY='3.5' orient='auto'>",
            "  <polygon points='0 0, 10 3.5, 0 7'/>",
            "</marker>",
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
                # The thickness is now based on the real-world mm value, scaled to the drawing
                thickness_px = wll["thickness"] * s
                d = self._wall_strip((X1, Y1), (X2, Y2), thickness_px)
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
            thickness_px = wll["thickness"] * s
            
            if op['type'] == 'door':
                # Frame thickness is a fraction of the wall thickness
                frame_h = max(2.0, min(8.0, 0.1 * thickness_px))
                
                # Defaults because schema does not support this yet
                handing = "RHR" 
                swing_dir = "in"
                
                hinge_at_end = handing.startswith("R")
                
                svg.append(f"<g transform='translate({X1:.1f},{Y1:.1f}) rotate({angle:.1f})'>")
                inner_tx = length if hinge_at_end else 0.0
                inner_sx = -1 if hinge_at_end else 1
                svg.append(f"<g transform='translate({inner_tx:.1f},0) scale({inner_sx},1)'>")
                
                # Door leaf
                svg.append(f"<rect x='0' y='-{frame_h:.1f}' width='{length:.1f}' height='{frame_h*2:.1f}' class='door'/>")
                
                # Swing arc
                swing_radius = length
                sy = 1.0 if swing_dir == "in" else -1.0
                end_x_arc = swing_radius * math.cos(math.radians(-90))
                end_y_arc = swing_radius * math.sin(math.radians(-90))
                svg.append(f"<path d='M 0 0 A {swing_radius:.1f} {swing_radius:.1f} 0 0 {1 if sy > 0 else 0} {end_x_arc:.1f} {sy*end_y_arc:.1f}' class='door-swing'/>")

                svg.append("</g></g>") # Close transforms

            elif op['type'] == 'window':
                # Window sits in the middle of the wall thickness
                svg.append(f"<g transform='translate({X1:.1f},{Y1:.1f}) rotate({angle:.1f})'>")
                # Window frame/sill
                svg.append(f"<rect x='0' y='-{thickness_px/2:.1f}' width='{length:.1f}' height='{thickness_px:.1f}' class='window-sill'/>")
                # Glass pane
                svg.append(f"<line x1='0' y1='0' x2='{length:.1f}' y2='0' stroke='#88CCFF' stroke-width='{max(1, 0.3*thickness_px)}' />")
                
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

        # Render Furniture
        self._add_furniture(svg, T)

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
