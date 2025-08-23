from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# --- Data Structures for 3D Geometry ---

@dataclass
class Vertex:
    """Represents a 3D vertex with position, normal, and UV coordinates."""
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    uv: Tuple[float, float]

@dataclass
class Triangle:
    """Represents a triangle defined by three vertex indices."""
    indices: Tuple[int, int, int]

@dataclass
class Mesh:
    """Represents a mesh with a list of vertices and triangles, and an optional material."""
    id: str
    vertices: List[Vertex] = field(default_factory=list)
    triangles: List[Triangle] = field(default_factory=list)
    material_id: str | None = None

@dataclass
class SceneGraph:
    """Represents the entire 3D scene as a collection of meshes."""
    meshes: List[Mesh] = field(default_factory=list)

# --- Geometry Building Logic ---

def _triangulate_polygon(polygon: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    """
    Triangulates a simple convex polygon using fan triangulation.
    Note: This is a basic implementation and will not work for concave polygons.
    A more robust solution would use an ear-clipping algorithm.
    """
    if len(polygon) < 3:
        return []
    
    triangles = []
    for i in range(1, len(polygon) - 1):
        triangles.append((0, i, i + 1))
    return triangles

def _extrude_wall(start: Tuple[float, float], end: Tuple[float, float], height: float, thickness: float) -> Mesh:
    """Creates a 3D cuboid mesh for a wall segment."""
    x1, z1 = start
    x2, z2 = end

    # Simplified normal calculation (assumes walls are axis-aligned, needs improvement for angled walls)
    # A proper implementation would use cross products to find the perpendicular direction.
    dx, dz = x2 - x1, z2 - z1
    wall_len = (dx**2 + dz**2)**0.5
    nx, nz = -dz/wall_len, dx/wall_len # Perpendicular vector for thickness

    half_thick_x, half_thick_z = (nx * thickness) / 2, (nz * thickness) / 2

    # Define 8 vertices of the cuboid
    verts = [
        Vertex(position=(x1 - half_thick_x, 0, z1 - half_thick_z), normal=(0,0,0), uv=(0,0)), # 0
        Vertex(position=(x1 + half_thick_x, 0, z1 + half_thick_z), normal=(0,0,0), uv=(0,0)), # 1
        Vertex(position=(x2 + half_thick_x, 0, z2 + half_thick_z), normal=(0,0,0), uv=(0,0)), # 2
        Vertex(position=(x2 - half_thick_x, 0, z2 - half_thick_z), normal=(0,0,0), uv=(0,0)), # 3
        Vertex(position=(x1 - half_thick_x, height, z1 - half_thick_z), normal=(0,0,0), uv=(0,0)), # 4
        Vertex(position=(x1 + half_thick_x, height, z1 + half_thick_z), normal=(0,0,0), uv=(0,0)), # 5
        Vertex(position=(x2 + half_thick_x, height, z2 + half_thick_z), normal=(0,0,0), uv=(0,0)), # 6
        Vertex(position=(x2 - half_thick_x, height, z2 - half_thick_z), normal=(0,0,0), uv=(0,0)), # 7
    ]

    # Define 12 triangles (2 for each face of the cuboid)
    # Normals are not calculated yet, placeholder (0,0,0)
    tris = [
        Triangle(indices=(0, 1, 2)), Triangle(indices=(0, 2, 3)), # Bottom
        Triangle(indices=(4, 7, 6)), Triangle(indices=(4, 6, 5)), # Top
        Triangle(indices=(0, 3, 7)), Triangle(indices=(0, 7, 4)), # Side 1
        Triangle(indices=(1, 5, 6)), Triangle(indices=(1, 6, 2)), # Side 2
        Triangle(indices=(0, 4, 5)), Triangle(indices=(0, 5, 1)), # End 1
        Triangle(indices=(3, 2, 6)), Triangle(indices=(3, 6, 7)), # End 2
    ]
    
    # TODO: Calculate proper normals for each face and assign to vertices.
    # For now, this provides the basic geometry.

    return Mesh(id="wall_mesh", vertices=verts, triangles=tris)

def build_geometry_from_plan(plan_data: Dict[str, Any]) -> SceneGraph:
    """
    Constructs a 3D scene graph from a validated HouseBrain v2 plan JSON.
    """
    scene = SceneGraph()
    
    # 1. Create Floor Meshes
    for i, space in enumerate(plan_data.get("spaces", [])):
        boundary_2d = [tuple(p) for p in space.get("boundary", [])]
        if not boundary_2d:
            continue
            
        elevation = 0.0 # TODO: Get from level info
        
        verts = [Vertex(position=(p[0], elevation, p[1]), normal=(0, 1, 0), uv=(p[0]/10000, p[1]/10000)) for p in boundary_2d]
        tri_indices = _triangulate_polygon(boundary_2d)
        tris = [Triangle(indices=ti) for ti in tri_indices]
        
        floor_mesh = Mesh(id=f"floor_{space.get('id', i)}", vertices=verts, triangles=tris, material_id="floor_default")
        scene.meshes.append(floor_mesh)

    # 2. Create Wall Meshes
    for i, wall in enumerate(plan_data.get("walls", [])):
        start_2d = tuple(wall["start"])
        end_2d = tuple(wall["end"])
        height = wall.get("height", 3000.0)
        thickness = wall.get("thickness", 115.0)

        wall_mesh = _extrude_wall(start_2d, end_2d, height, thickness)
        wall_mesh.id = f"wall_{wall.get('id', i)}"
        wall_mesh.material_id = "wall_default"
        scene.meshes.append(wall_mesh)
        
    # 3. Punch Openings (Placeholder Logic)
    # A production implementation requires a robust CSG or mesh re-tessellation algorithm.
    # This section is a placeholder to show where the logic would go.
    for opening in plan_data.get("openings", []):
        wall_id = opening.get("wall_id")
        # Find the corresponding wall mesh in the scene...
        # ...then apply a mesh boolean/subtraction operation.
        # print(f"INFO: Opening '{opening.get('id')}' needs to be cut from wall '{wall_id}'.")

    return scene
