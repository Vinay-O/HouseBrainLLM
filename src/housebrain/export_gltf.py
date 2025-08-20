from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import base64
import math
import struct


def _make_unit_cube_geometry() -> Dict[str, Any]:
    """Create a unit cube (centered at origin) geometry with positions, normals, indices.
    Returns dict with buffer (data URI), bufferViews, accessors, and meshes referencing material index 0.
    """
    # Define 24 vertices (4 per face) with normals per face
    # Faces: +X, -X, +Y, -Y, +Z, -Z
    positions: List[float] = []
    normals: List[float] = []
    indices: List[int] = []

    def add_face(px: float, py: float, pz: float, nx: float, ny: float, nz: float, ux: float, uy: float, uz: float, vx: float, vy: float, vz: float):
        base_index = len(positions) // 3
        # Square face corners: -u -v, +u -v, +u +v, -u +v
        corners = [
            (-0.5, -0.5),
            (0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5),
        ]
        for cu, cv in corners:
            x = px + cu * ux + cv * vx
            y = py + cu * uy + cv * vy
            z = pz + cu * uz + cv * vz
            positions.extend([x, y, z])
            normals.extend([nx, ny, nz])
        # Two triangles
        indices.extend([
            base_index + 0, base_index + 1, base_index + 2,
            base_index + 0, base_index + 2, base_index + 3,
        ])

    # +X face
    add_face(0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # -X face
    add_face(-0.5, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    # +Y face
    add_face(0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    # -Y face
    add_face(0.0, -0.5, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # +Z face
    add_face(0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # -Z face
    add_face(0.0, 0.0, -0.5, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    # Pack into a single buffer
    pos_bytes = b"".join(struct.pack("<fff", *positions[i:i+3]) for i in range(0, len(positions), 3))
    nrm_bytes = b"".join(struct.pack("<fff", *normals[i:i+3]) for i in range(0, len(normals), 3))
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)  # unsigned short

    # Align bufferViews to 4 bytes
    def pad4(b: bytes) -> bytes:
        return b + (b"\x00" * ((4 - (len(b) % 4)) % 4))

    pos_off = 0
    pos_bytes = pad4(pos_bytes)
    nrm_off = pos_off + len(pos_bytes)
    nrm_bytes = pad4(nrm_bytes)
    idx_off = nrm_off + len(nrm_bytes)
    idx_bytes = pad4(idx_bytes)
    buffer_bytes = pos_bytes + nrm_bytes + idx_bytes
    buffer_b64 = base64.b64encode(buffer_bytes).decode("ascii")

    buffer = {
        "byteLength": len(buffer_bytes),
        "uri": f"data:application/octet-stream;base64,{buffer_b64}",
    }

    bufferViews = [
        {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_bytes), "target": 34962},  # ARRAY_BUFFER
        {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_bytes), "target": 34962},
        {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_bytes), "target": 34963},  # ELEMENT_ARRAY_BUFFER
    ]

    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(positions)//3, "type": "VEC3", "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
        {"bufferView": 1, "componentType": 5126, "count": len(normals)//3, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
    ]

    # Two meshes referencing different materials (0: exterior, 1: interior)
    meshes = [
        {
            "name": "CubeExterior",
            "primitives": [
                {
                    "attributes": {"POSITION": 0, "NORMAL": 1},
                    "indices": 2,
                    "mode": 4,
                    "material": 0,
                }
            ],
        },
        {
            "name": "CubeInterior",
            "primitives": [
                {
                    "attributes": {"POSITION": 0, "NORMAL": 1},
                    "indices": 2,
                    "mode": 4,
                    "material": 1,
                }
            ],
        },
    ]

    return {
        "buffer": buffer,
        "bufferViews": bufferViews,
        "accessors": accessors,
        "meshes": meshes,
    }


def plan_to_simple_gltf(plan: Dict[str, Any], sheet_mode: str = "floor") -> Dict[str, Any]:
    # Construct a glTF with a single embedded unit-cube geometry and instanced components
    geo = _make_unit_cube_geometry()

    # PBR materials
    materials = [
        {  # Exterior wall
            "name": "ExteriorWallPBR",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.78, 0.75, 0.72, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.95,
            }
        },
        {  # Interior wall (painted)
            "name": "InteriorWallPBR",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.94, 0.94, 0.94, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            }
        },
        {  # Concrete
            "name": "ConcretePBR",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.72, 0.72, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.95,
            }
        },
        {  # Floor slab
            "name": "FloorPBR",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.85, 0.85, 0.84, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            }
        },
    ]

    # Extend meshes to include concrete and floor variants using same accessors
    meshes: List[Dict[str, Any]] = list(geo["meshes"])  # 0: exterior, 1: interior
    meshes.append({
        "name": "CubeConcrete",
        "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4, "material": 2}]
    })
    meshes.append({
        "name": "CubeFloor",
        "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4, "material": 3}]
    })

    nodes: List[Dict[str, Any]] = []
    for w in plan.get("walls", []):
        (x1, y1) = w["start"]
        (x2, y2) = w["end"]
        thickness = float(w.get("thickness", 115))
        height = float(w.get("height", 2700))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length = math.hypot(x2 - x1, y2 - y1)
        # Rotate around Y so that X-axis aligns with wall vector (X,Z plane)
        phi = math.atan2((y2 - y1), (x2 - x1))
        q = [0.0, math.sin(phi / 2.0), 0.0, math.cos(phi / 2.0)]
        mesh_index = 0 if w.get("type") == "exterior" else 1
        nodes.append(
            {
                "name": f"wall_{w.get('id')}",
                "mesh": mesh_index,
                # Map plan: X->X (m), Y->Z (m), height->Y (m)
                "translation": [cx / 1000.0, height / 2000.0, cy / 1000.0],
                "rotation": q,
                "scale": [max(length / 1000.0, 0.001), max(height / 1000.0, 0.001), max(thickness / 1000.0, 0.001)],
            }
        )

    # Utility: segment intersection
    def seg_intersect(a1: Tuple[float, float], a2: Tuple[float, float], b1: Tuple[float, float], b2: Tuple[float, float]):
        (x1, y1), (x2, y2) = a1, a2
        (x3, y3), (x4, y4) = b1, b2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        def within(a, b, p):
            return min(a, b) - 1e-6 <= p <= max(a, b) + 1e-6
        if within(x1, x2, px) and within(y1, y2, py) and within(x3, x4, px) and within(y3, y4, py):
            return (px, py)
        return None

    # Columns at wall intersections
    walls = plan.get("walls", [])
    intersections: List[Tuple[float, float]] = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            p = seg_intersect(tuple(walls[i]["start"]), tuple(walls[i]["end"]), tuple(walls[j]["start"]), tuple(walls[j]["end"]))
            if p is not None:
                intersections.append(p)

    column_size_mm = 300.0
    col_h_mm = 3000.0
    for idx, (px, py) in enumerate(intersections):
        nodes.append({
            "name": f"column_{idx}",
            "mesh": 2,  # concrete
            "translation": [px / 1000.0, col_h_mm / 2000.0, py / 1000.0],
            "rotation": [0, 0, 0, 1],
            "scale": [column_size_mm / 1000.0, col_h_mm / 1000.0, column_size_mm / 1000.0],
        })

    # Floor slab based on bounds of geometry
    xs: List[float] = []
    ys: List[float] = []
    for w in walls:
        xs.extend([w["start"][0], w["end"][0]])
        ys.extend([w["start"][1], w["end"][1]])
    for s in plan.get("spaces", []):
        for (x, y) in s.get("boundary", []):
            xs.append(x)
            ys.append(y)
    if xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        w_m = max((maxx - minx) / 1000.0, 0.001)
        d_m = max((maxy - miny) / 1000.0, 0.001)
        t_m = 0.12  # 120mm slab
        nodes.append({
            "name": "floor_slab",
            "mesh": 3,  # floor
            "translation": [cx / 1000.0, t_m / 2.0, cy / 1000.0],
            "rotation": [0, 0, 0, 1],
            "scale": [w_m, t_m, d_m],
        })

    # Simple directional light (KHR_lights_punctual)
    lights_ext = {"KHR_lights_punctual": {"lights": [{"type": "directional", "color": [1.0, 1.0, 1.0], "intensity": 750.0, "name": "Sun"}]}}
    nodes.append({
        "name": "Sun",
        "translation": [0.0, 10.0, 0.0],
        "rotation": [0.2, -0.2, 0.0, 0.96],
        "extensions": {"KHR_lights_punctual": {"light": 0}},
    })

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0"},
        "buffers": [geo["buffer"]],
        "bufferViews": geo["bufferViews"],
        "accessors": geo["accessors"],
        "materials": materials,
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "scene": 0,
        "extensionsUsed": ["KHR_lights_punctual"],
        "extensions": lights_ext,
    }
    return gltf


def export_gltf(input_path: str, output_path: str, sheet_mode: str = "floor") -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    gltf = plan_to_simple_gltf(plan, sheet_mode)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gltf, f, separators=(",", ":"))
    print(f"✅ glTF exported to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export plan schema JSON to glTF 2.0 (placeholder)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    export_gltf(args.input, args.output)


