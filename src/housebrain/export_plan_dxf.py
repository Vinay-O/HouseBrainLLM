from __future__ import annotations


def export_plan_to_dxf(input_path: str, out_path: str, units: str = "mm") -> None:
    """Write a minimal ASCII DXF container so downstream tools have a file.
    """
    dxf = """0
SECTION
2
HEADER
0
ENDSEC
0
SECTION
2
ENTITIES
0
ENDSEC
0
EOF
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dxf)
    print(f"✅ DXF written to {out_path}")
