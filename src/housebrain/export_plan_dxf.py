from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import ezdxf
import math


def export_wall_3d(msp, wall_data: Dict[str, Any], thickness: float, wall_type: str) -> None:
    """Export wall as 3D solid with proper thickness and height."""
    (x1, y1) = wall_data["start"]
    (x2, y2) = wall_data["end"]
    
    # Calculate wall normal vector for thickness
    wall_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if wall_length == 0:
        return
        
    # Unit vector along wall
    ux = (x2 - x1) / wall_length
    uy = (y2 - y1) / wall_length
    
    # Normal vector (perpendicular to wall)
    nx = -uy
    ny = ux
    
    # Wall height based on type
    wall_height = 3000 if wall_type == "exterior" else 2700  # mm
    
    # Half thickness offset
    half_thickness = thickness / 2
    
    # Wall corner points
    corners = [
        (x1 + nx * half_thickness, y1 + ny * half_thickness, 0),
        (x2 + nx * half_thickness, y2 + ny * half_thickness, 0),
        (x2 - nx * half_thickness, y2 - ny * half_thickness, 0),
        (x1 - nx * half_thickness, y1 - ny * half_thickness, 0),
        (x1 + nx * half_thickness, y1 + ny * half_thickness, wall_height),
        (x2 + nx * half_thickness, y2 + ny * half_thickness, wall_height),
        (x2 - nx * half_thickness, y2 - ny * half_thickness, wall_height),
        (x1 - nx * half_thickness, y1 - ny * half_thickness, wall_height),
    ]
    
    # Create 3D solid mesh
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 7, 6, 5],  # Top face
        [0, 4, 5, 1],  # Front face
        [1, 5, 6, 2],  # Right face
        [2, 6, 7, 3],  # Back face
        [3, 7, 4, 0],  # Left face
    ]
    
    mesh = msp.add_mesh(dxfattribs={"layer": "A-WALL-3D"})
    mesh.vertices = corners
    mesh.faces = faces


def export_foundation_3d(msp, plan: Dict[str, Any]) -> None:
    """Export foundation elements with proper depth and footing details."""
    spaces = plan.get("spaces", [])
    if not spaces:
        return
        
    # Calculate building footprint
    all_points = []
    for space in spaces:
        all_points.extend(space.get("boundary", []))
    
    if not all_points:
        return
        
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    
    # Foundation wall depth
    foundation_depth = -2400  # mm below grade
    foundation_thickness = 200  # mm
    
    # Foundation footprint (extended beyond building)
    footing_extend = 300  # mm
    
    # Foundation corners
    foundation_corners = [
        (minx - footing_extend, miny - footing_extend, foundation_depth),
        (maxx + footing_extend, miny - footing_extend, foundation_depth),
        (maxx + footing_extend, maxy + footing_extend, foundation_depth),
        (minx - footing_extend, maxy + footing_extend, foundation_depth),
        (minx - footing_extend, miny - footing_extend, 0),
        (maxx + footing_extend, miny - footing_extend, 0),
        (maxx + footing_extend, maxy + footing_extend, 0),
        (minx - footing_extend, maxy + footing_extend, 0),
    ]
    
    # Foundation mesh
    foundation_faces = [
        [0, 1, 2, 3],  # Bottom
        [4, 7, 6, 5],  # Top
        [0, 4, 5, 1],  # Front
        [1, 5, 6, 2],  # Right
        [2, 6, 7, 3],  # Back
        [3, 7, 4, 0],  # Left
    ]
    
    mesh = msp.add_mesh(dxfattribs={"layer": "S-FOUN"})
    mesh.vertices = foundation_corners
    mesh.faces = foundation_faces


def export_roof_structure_3d(msp, plan: Dict[str, Any]) -> None:
    """Export roof structure with proper pitch and framing."""
    spaces = plan.get("spaces", [])
    if not spaces:
        return
        
    # Calculate building footprint
    all_points = []
    for space in spaces:
        all_points.extend(space.get("boundary", []))
    
    if not all_points:
        return
        
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    
    # Roof parameters
    wall_height = 2700  # mm
    roof_pitch = 30  # degrees (6/12 pitch)
    roof_overhang = 600  # mm
    ridge_height = wall_height + (maxy - miny) / 2 * math.tan(math.radians(roof_pitch))
    
    # Extended roof footprint
    roof_minx = minx - roof_overhang
    roof_maxx = maxx + roof_overhang
    roof_miny = miny - roof_overhang
    roof_maxy = maxy + roof_overhang
    
    # Gable roof (simple)
    roof_ridge_y = (miny + maxy) / 2
    
    # Roof vertices
    roof_corners = [
        (roof_minx, roof_miny, wall_height),  # Eave front left
        (roof_maxx, roof_miny, wall_height),  # Eave front right
        (roof_maxx, roof_maxy, wall_height),  # Eave back right
        (roof_minx, roof_maxy, wall_height),  # Eave back left
        (roof_minx, roof_ridge_y, ridge_height),  # Ridge front left
        (roof_maxx, roof_ridge_y, ridge_height),  # Ridge front right
    ]
    
    # Roof faces (gable ends and slopes)
    roof_faces = [
        [0, 1, 5, 4],  # Front slope
        [2, 3, 4, 5],  # Back slope
        [0, 4, 3],     # Left gable end
        [1, 2, 5],     # Right gable end
    ]
    
    mesh = msp.add_mesh(dxfattribs={"layer": "A-ROOF"})
    mesh.vertices = roof_corners
    mesh.faces = roof_faces


def export_structural_elements_3d(msp, plan: Dict[str, Any]) -> None:
    """Export structural beams and columns."""
    walls = plan.get("walls", [])
    
    # Add structural columns at wall intersections
    wall_intersections = find_wall_intersections(walls)
    
    for intersection in wall_intersections:
        x, y = intersection
        # Standard column size 150x150mm
        column_size = 150
        column_height = 2700
        
        # Column corners
        half_size = column_size / 2
        column_corners = [
            (x - half_size, y - half_size, 0),
            (x + half_size, y - half_size, 0),
            (x + half_size, y + half_size, 0),
            (x - half_size, y + half_size, 0),
            (x - half_size, y - half_size, column_height),
            (x + half_size, y - half_size, column_height),
            (x + half_size, y + half_size, column_height),
            (x - half_size, y + half_size, column_height),
        ]
        
        column_faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 7, 6, 5],  # Top
            [0, 4, 5, 1],  # Front
            [1, 5, 6, 2],  # Right
            [2, 6, 7, 3],  # Back
            [3, 7, 4, 0],  # Left
        ]
        
        mesh = msp.add_mesh(dxfattribs={"layer": "S-COLS"})
        mesh.vertices = column_corners
        mesh.faces = column_faces


def find_wall_intersections(walls: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Find intersection points between walls for structural column placement."""
    intersections = []
    tolerance = 10.0  # mm
    
    for i, wall1 in enumerate(walls):
        for j, wall2 in enumerate(walls):
            if i >= j:
                continue
                
            # Check if walls intersect
            p1_start = wall1["start"]
            p1_end = wall1["end"]
            p2_start = wall2["start"]
            p2_end = wall2["end"]
            
            # Check if endpoints are close (T-junction or corner)
            endpoints = [p1_start, p1_end, p2_start, p2_end]
            for k, pt1 in enumerate(endpoints):
                for l, pt2 in enumerate(endpoints):
                    if k >= l:
                        continue
                    
                    dist = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    if dist < tolerance:
                        # Found intersection
                        avg_point = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
                        if avg_point not in intersections:
                            intersections.append(avg_point)
    
    return intersections


def export_plan_to_dxf(plan_path: str, out_path: str, units: str = "mm") -> None:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan: Dict[str, Any] = json.load(f)

    doc = ezdxf.new("R2018")
    # Set units: 4=inches, 6=meters, 1=mm
    insunits = 1 if units == "mm" else 4
    doc.header["$INSUNITS"] = insunits
    msp = doc.modelspace()

    # Layers per CAD convention
    def add_layer(name: str, color: int = 7, ltype: str | None = None):
        if name not in doc.layers:
            if ltype:
                if ltype not in doc.linetypes:
                    doc.linetypes.add(ltype, pattern=[0.25, 0.125, -0.0625])
                doc.layers.add(name, color=color, linetype=ltype)
            else:
                doc.layers.add(name, color=color)

    # Enhanced layer structure for 3D structural accuracy
    add_layer("A-WALL", color=7)
    add_layer("A-WALL-PLUMB", color=4, ltype="DASHED")
    add_layer("A-WALL-3D", color=7)  # 3D wall solids
    add_layer("A-DOOR", color=30)
    add_layer("A-DOOR-3D", color=30)  # 3D door components
    add_layer("A-GLAZ", color=140)
    add_layer("A-GLAZ-3D", color=140)  # 3D window components
    add_layer("A-ANNO-DIMS", color=2)
    add_layer("A-AREA", color=8)
    add_layer("S-BEAM", color=1)  # Structural beams
    add_layer("S-COLS", color=1)  # Structural columns
    add_layer("S-FOUN", color=3)  # Foundation elements
    add_layer("A-ROOF", color=5)  # Roof structure
    add_layer("A-FLOR", color=6)  # Floor slabs

    # Walls with 3D structural accuracy
    for w in plan.get("walls", []):
        (x1, y1) = w["start"]
        (x2, y2) = w["end"]
        thickness = w.get("thickness", 115.0)  # mm
        wall_type = w.get("type", "interior")
        
        # 2D wall centerline
        layer = "A-WALL-PLUMB" if w.get("subtype") == "plumbing" else "A-WALL"
        msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer})
        
        # 3D wall solid with actual thickness
        export_wall_3d(msp, w, thickness, wall_type)

    # Spaces (closed polylines)
    for s in plan.get("spaces", []):
        bnd: List[Tuple[float, float]] = [tuple(p) for p in s.get("boundary", [])]
        if len(bnd) >= 3:
            if bnd[0] != bnd[-1]:
                bnd.append(bnd[0])
            msp.add_lwpolyline(bnd, dxfattribs={"layer": "A-AREA", "closed": True})

    # Openings (projected as lines)
    def line_from_opening(wall: Dict[str, Any], opening: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        (ax, ay) = tuple(wall["start"]) ; (bx, by) = tuple(wall["end"])
        dx, dy = bx - ax, by - ay
        L = (dx*dx + dy*dy) ** 0.5 or 1.0
        ux, uy = dx / L, dy / L
        s = opening["position"] * L
        cx, cy = ax + ux * s, ay + uy * s
        half = opening["width"] / 2.0
        p1 = (cx - ux * half, cy - uy * half)
        p2 = (cx + ux * half, cy + uy * half)
        return p1, p2

    walls_by_id = {w["id"]: w for w in plan.get("walls", [])}
    for op in plan.get("openings", []):
        wall = walls_by_id.get(op["wall_id"]) ;
        if not wall:
            continue
        p1, p2 = line_from_opening(wall, op)
        layer = "A-DOOR" if op["type"] == "door" else "A-GLAZ"
        msp.add_line(p1, p2, dxfattribs={"layer": layer})

    # STRUCTURAL ELEMENTS (3D)
    export_foundation_3d(msp, plan)
    export_roof_structure_3d(msp, plan)
    export_structural_elements_3d(msp, plan)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path)
    print(f"✅ DXF exported to {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export plan schema JSON to CAD-ready DXF")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--units", default="mm", choices=["mm", "inch"]) 
    args = ap.parse_args()
    export_plan_to_dxf(args.input, args.output, args.units)


