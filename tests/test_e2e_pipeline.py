from __future__ import annotations

import json
import sys
from pathlib import Path


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_dir))

    from housebrain.pipeline_v2 import run_pipeline

    plan = {
        "metadata": {"project_name": "e2e_house", "units": "mm"},
        "levels": [{"id": "L1", "name": "Ground", "elevation_mm": 0}],
        "walls": [
            {
                "id": "w1",
                "level_id": "L1",
                "start": [0, 0],
                "end": [3000, 0],
                "type": "exterior",
                "thickness": 230.0,
                "height": 3000.0,
            }
        ],
        "openings": [],
        "spaces": [
            {
                "id": "s1",
                "name": "Living Room",
                "type": "living_room",
                "level_id": "L1",
                "boundary": [[0, 0], [3000, 0], [3000, 3000], [0, 3000]],
            }
        ],
        "stairs": [],
    }

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    out_dir = tmp_path
    run_pipeline(str(plan_path), str(out_dir))

    base = plan_path.stem
    assert (out_dir / f"{base}_floor.svg").exists()
    assert (out_dir / f"{base}.dxf").exists()
    
    gltf_path = out_dir / f"{base}_3d.gltf"
    assert gltf_path.exists()

    # --- Add assertions for glTF content ---
    with open(gltf_path, "r") as f:
        gltf_data = json.load(f)

    # 1. Assert mesh count (1 space + 1 wall = 2 meshes)
    assert "meshes" in gltf_data and len(gltf_data["meshes"]) == 2, "Should be one mesh for the floor and one for the wall"

    # 2. Assert triangle budget
    total_triangles = 0
    if "meshes" in gltf_data and "accessors" in gltf_data:
        accessors = gltf_data["accessors"]
        for mesh in gltf_data["meshes"]:
            for primitive in mesh.get("primitives", []):
                if "indices" in primitive:
                    indices_accessor_idx = primitive["indices"]
                    indices_accessor = accessors[indices_accessor_idx]
                    # Count is number of indices. Divide by 3 for triangles.
                    total_triangles += indices_accessor.get("count", 0) // 3
    
    assert 0 < total_triangles < 1000, f"Triangle count ({total_triangles}) is out of budget for this simple model"

    assert (out_dir / f"{base}_boq.json").exists()
    assert (out_dir / "index.html").exists()
