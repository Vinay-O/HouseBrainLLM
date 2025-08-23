import json
import os
import argparse
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add the src directory to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from housebrain.schema import HouseInput, HouseOutput
from housebrain.llm import generate_house_design
from housebrain.validate import validate_house_design
from pathlib import Path
from mock_inference import generate_housebrain_output
from export_svg import get_svg_string
from export_obj import export_to_obj


# Create FastAPI app
app = FastAPI(
    title="HouseBrain API",
    description="AI-powered architectural design system",
    version="1.1.0"
)


class DesignRequest(BaseModel):
    """Request model for house design generation"""
    input: Dict[str, Any]


class DesignResponse(BaseModel):
    """Response model for house design generation"""
    success: bool
    house_output: Dict[str, Any]
    validation: Dict[str, Any]
    render_paths: Dict[str, str]


class GenerateRequest(BaseModel):
    """Request model for mock/real generate endpoint"""
    payload: Dict[str, Any]
    scenario: str = "s01"


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HouseBrain API v1.1.0",
        "description": "AI-powered architectural design system",
        "endpoints": {
            "/design": "Generate house design",
            "/validate": "Validate house design",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.1.0"}


@app.post("/design", response_model=DesignResponse)
async def generate_design(request: DesignRequest):
    """Generate a complete house design"""
    try:
        # Parse input
        house_input = HouseInput(**request.input)
        
        # Generate design
        house_output = generate_house_design(house_input, demo_mode=True)
        
        # Validate design
        validation = validate_house_design(house_output)
        
        # Generate renders
        render_paths = await generate_renders(house_output)
        
        return DesignResponse(
            success=True,
            house_output=house_output.model_dump(),
            validation=validation.model_dump(),
            render_paths=render_paths
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate")
async def validate_design(request: DesignRequest):
    """Validate an existing house design"""
    try:
        # Parse input
        house_input = HouseInput(**request.input)
        
        # Generate design first (for demo purposes)
        house_output = generate_house_design(house_input, demo_mode=True)
        
        # Validate design
        validation = validate_house_design(house_output)
        
        return {
            "success": True,
            "validation": validation.model_dump(),
            "house_output": house_output.model_dump()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/renders/{render_type}")
async def get_render(render_type: str):
    """Get a specific render file"""
    render_path = f"outputs/{render_type}"
    
    if not os.path.exists(render_path):
        raise HTTPException(status_code=404, detail=f"Render {render_type} not found")
    
    return FileResponse(render_path)


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate output using curated example (mock) and return inline SVG + OBJ text.
    Frontend can render SVG directly and download OBJ if needed.
    """
    # Use curated example as stand-in for model output
    result = generate_housebrain_output(req.payload, scenario_id=req.scenario)

    # Temp JSON for exporters
    tmp_json = Path("/tmp/hb_gen.json")
    tmp_json.write_text(json.dumps(result), encoding="utf-8")

    # Inline SVG (2D plan)
    svg = get_svg_string(str(tmp_json), width=1200, height=800, scale=0.001)

    # OBJ text (simple 3D prisms)
    out_dir = Path("/tmp/hb_api")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "scene.obj"
    export_to_obj(str(tmp_json), str(obj_path), scale=0.001, height=3.0)
    obj_text = obj_path.read_text(encoding="utf-8") if obj_path.exists() else ""

    return {"ok": True, "output": result, "render": {"svg": svg, "obj": obj_text}}


async def generate_renders(house_output: HouseOutput) -> Dict[str, str]:
    """Generate all render outputs"""
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    render_paths = {}
    
    # Generate JSON plan
    plan_path = os.path.join(outputs_dir, "plan.json")
    with open(plan_path, 'w') as f:
        json.dump(house_output.dict(), f, indent=2, default=str)
    render_paths["plan_json"] = plan_path
    
    # Generate SVG floor plans
    for i, level in enumerate(house_output.levels):
        svg_path = os.path.join(outputs_dir, f"level_{i}.svg")
        generate_svg_plan(level, svg_path)
        render_paths[f"level_{i}_svg"] = svg_path
    
    # Generate 3D OBJ file
    obj_path = os.path.join(outputs_dir, "scene.obj")
    generate_obj_file(house_output, obj_path)
    render_paths["scene_obj"] = obj_path
    
    return render_paths


def generate_svg_plan(level, svg_path: str):
    """Generate SVG floor plan for a level"""
    import svgwrite
    
    # Calculate dimensions
    min_x = min(room.bounds.x for room in level.rooms) if level.rooms else 0
    min_y = min(room.bounds.y for room in level.rooms) if level.rooms else 0
    max_x = max(room.bounds.x + room.bounds.width for room in level.rooms) if level.rooms else 100
    max_y = max(room.bounds.y + room.bounds.height for room in level.rooms) if level.rooms else 100
    
    width = max_x - min_x + 20
    height = max_y - min_y + 20
    
    # Create SVG
    dwg = svgwrite.Drawing(svg_path, size=(width, height), viewBox=f"0 0 {width} {height}")
    
    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    
    # Room colors
    room_colors = {
        "living_room": "#FFE4B5",
        "dining_room": "#F0E68C",
        "kitchen": "#98FB98",
        "master_bedroom": "#DDA0DD",
        "bedroom": "#E6E6FA",
        "bathroom": "#87CEEB",
        "half_bath": "#87CEEB",
        "family_room": "#FFB6C1",
        "study": "#F5DEB3",
        "garage": "#D3D3D3",
        "utility": "#F0F8FF",
        "storage": "#F5F5DC",
        "stairwell": "#696969",
        "corridor": "#F8F8FF",
        "entrance": "#FFD700"
    }
    
    # Draw rooms
    for room in level.rooms:
        color = room_colors.get(room.type.value, "#FFFFFF")
        
        # Room rectangle
        dwg.add(dwg.rect(
            insert=(room.bounds.x - min_x + 10, room.bounds.y - min_y + 10),
            size=(room.bounds.width, room.bounds.height),
            fill=color,
            stroke='black',
            stroke_width=1
        ))
        
        # Room label
        dwg.add(dwg.text(
            room.type.value.replace('_', ' ').title(),
            insert=(room.bounds.x - min_x + 10 + room.bounds.width/2, 
                   room.bounds.y - min_y + 10 + room.bounds.height/2),
            text_anchor="middle",
            dominant_baseline="middle",
            font_size="12",
            font_family="Arial",
            fill='black'
        ))
        
        # Draw doors
        for door in room.doors:
            dwg.add(dwg.circle(
                center=(door.position.x - min_x + 10, door.position.y - min_y + 10),
                r=2,
                fill='brown',
                stroke='black'
            ))
        
        # Draw windows
        for window in room.windows:
            dwg.add(dwg.rect(
                insert=(window.position.x - min_x + 10, window.position.y - min_y + 10),
                size=(window.width, 1),
                fill='lightblue',
                stroke='blue'
            ))
    
    # Draw stairs
    for stair in level.stairs:
        dwg.add(dwg.rect(
            insert=(stair.position.x - min_x + 10, stair.position.y - min_y + 10),
            size=(stair.width, stair.length),
            fill='#8B4513',
            stroke='black',
            stroke_width=2
        ))
        
        # Stair direction arrow
        dwg.add(dwg.text(
            "↑" if stair.direction == "up" else "↓",
            insert=(stair.position.x - min_x + 10 + stair.width/2,
                   stair.position.y - min_y + 10 + stair.length/2),
            text_anchor="middle",
            dominant_baseline="middle",
            font_size="16",
            fill='white'
        ))
    
    dwg.save()


def generate_obj_file(house_output: HouseOutput, obj_path: str):
    """Generate 3D OBJ file for the house"""
    with open(obj_path, 'w') as f:
        f.write("# HouseBrain 3D Model\n")
        f.write("# Generated by HouseBrain v1.1.0\n\n")
        
        vertex_count = 1
        
        
        for level in house_output.levels:
            level_height = level.height
            base_z = level.level_number * level_height
            
            for room in level.rooms:
                # Room vertices (bottom face)
                x1, y1 = room.bounds.x, room.bounds.y
                x2, y2 = room.bounds.x + room.bounds.width, room.bounds.y + room.bounds.height
                
                # Bottom vertices
                f.write(f"v {x1} {y1} {base_z}\n")
                f.write(f"v {x2} {y1} {base_z}\n")
                f.write(f"v {x2} {y2} {base_z}\n")
                f.write(f"v {x1} {y2} {base_z}\n")
                
                # Top vertices
                f.write(f"v {x1} {y1} {base_z + level_height}\n")
                f.write(f"v {x2} {y1} {base_z + level_height}\n")
                f.write(f"v {x2} {y2} {base_z + level_height}\n")
                f.write(f"v {x1} {y2} {base_z + level_height}\n")
                
                # Bottom face
                f.write(f"f {vertex_count} {vertex_count+1} {vertex_count+2} {vertex_count+3}\n")
                
                # Top face
                f.write(f"f {vertex_count+4} {vertex_count+7} {vertex_count+6} {vertex_count+5}\n")
                
                # Side faces
                f.write(f"f {vertex_count} {vertex_count+4} {vertex_count+5} {vertex_count+1}\n")
                f.write(f"f {vertex_count+1} {vertex_count+5} {vertex_count+6} {vertex_count+2}\n")
                f.write(f"f {vertex_count+2} {vertex_count+6} {vertex_count+7} {vertex_count+3}\n")
                f.write(f"f {vertex_count+3} {vertex_count+7} {vertex_count+4} {vertex_count}\n")
                
                vertex_count += 8


def run_demo():
    """Run demo mode with sample input"""
    print("🏠 HouseBrain v1.1.0 - Demo Mode")
    print("=" * 50)
    
    # Load sample input
    with open("data/sample_input.json", 'r') as f:
        sample_input = json.load(f)
    
    # Generate design
    house_input = HouseInput(**sample_input)
    house_output = generate_house_design(house_input, demo_mode=True)
    
    # Validate design
    validation = validate_house_design(house_output)
    
    # Generate renders
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Save JSON plan
    with open(f"{outputs_dir}/plan.json", 'w') as f:
        json.dump(house_output.model_dump(), f, indent=2, default=str)
    
    # Generate SVG plans
    for i, level in enumerate(house_output.levels):
        generate_svg_plan(level, f"{outputs_dir}/level_{i}.svg")
    
    # Generate 3D model
    generate_obj_file(house_output, f"{outputs_dir}/scene.obj")
    
    print("✅ Demo completed successfully!")
    print(f"📁 Outputs saved to: {outputs_dir}/")
    print(f"📊 Validation score: {validation.compliance_score:.1f}/100")
    
    if validation.errors:
        print(f"⚠️  Errors: {len(validation.errors)}")
        for error in validation.errors[:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    if validation.warnings:
        print(f"💡 Warnings: {len(validation.warnings)}")
        for warning in validation.warnings[:3]:  # Show first 3 warnings
            print(f"   - {warning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HouseBrain API Server")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        print(f"🚀 Starting HouseBrain API server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)\n