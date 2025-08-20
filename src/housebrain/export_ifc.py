from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def export_ifc(input_path: str, output_path: str) -> None:
    try:
        import ifcopenshell
        import ifcopenshell.api
    except Exception:
        # Graceful fallback: write a stub file noting missing dependency
        Path(output_path).write_text("IFC export requires ifcopenshell. Please install it.", encoding="utf-8")
        print(f"⚠️ ifcopenshell not installed. Wrote note to {output_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        plan: Dict[str, Any] = json.load(f)

    # Minimal IFC project with site/building/storey and walls (with basic materials)
    model = ifcopenshell.file()
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name=plan.get("metadata", {}).get("project_name", "HouseBrain"))
    ifcopenshell.api.run("unit.assign_unit", model, length_units="MILLIMETRE")
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model", context_identifier="Body")
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1")

    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, product=site)
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, product=building)
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, product=storey)

    # Define simple material layers
    def make_material(name: str):
        mat = ifcopenshell.api.run("material.add_material", model, name=name, category="MATERIAL")
        return mat

    mat_ext = make_material("Masonry")
    mat_int = make_material("GypsumBoard")

    for w in plan.get("walls", []):
        wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWallStandardCase", name=w.get("id"))
        ifcopenshell.api.run("spatial.assign_container", model, product=wall, relating_structure=storey)
        # Assign material (best-effort)
        if w.get("type") == "exterior":
            ifcopenshell.api.run("material.assign_material", model, product=wall, material=mat_ext)
        else:
            ifcopenshell.api.run("material.assign_material", model, product=wall, material=mat_int)

    model.write(output_path)
    print(f"✅ IFC exported to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export plan schema JSON to IFC (placeholder)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    export_ifc(args.input, args.output)


