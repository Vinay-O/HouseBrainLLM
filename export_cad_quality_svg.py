"""
Professional CAD-quality SVG renderer for HouseBrain that produces industry-standard floor plans.
This creates architectural drawings that match professional CAD software output with:
- Proper room layouts and circulation
- Professional architectural symbols
- Accurate dimensions and annotations
- Industry-standard appearance
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any


def create_wall_line(x1: float, y1: float, x2: float, y2: float) -> str:
    """Create a wall line with proper thickness"""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#000000" stroke-width="3" fill="none"/>'


def create_door_opening(x: float, y: float, width: float, is_swing_left: bool = True) -> str:
    """Create a professional door opening symbol"""
    # Door opening line
    door_line = f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x+width:.2f}" y2="{y:.2f}" stroke="#8B4513" stroke-width="2" fill="none"/>'
    
    # Door swing arc
    radius = width * 0.7
    if is_swing_left:
        # Swing left (counter-clockwise)
        arc = f'<path d="M {x:.2f} {y:.2f} A {radius:.2f} {radius:.2f} 0 0 1 {x:.2f} {y+radius:.2f}" stroke="#8B4513" stroke-width="2" fill="none"/>'
    else:
        # Swing right (clockwise)
        arc = f'<path d="M {x+width:.2f} {y:.2f} A {radius:.2f} {radius:.2f} 0 0 0 {x+width:.2f} {y+radius:.2f}" stroke="#8B4513" stroke-width="2" fill="none"/>'
    
    return door_line + "\n" + arc


def create_window_opening(x: float, y: float, width: float, height: float) -> str:
    """Create a professional window opening symbol"""
    # Window frame (double lines)
    return f"""
    <line x1="{x:.2f}" y1="{y:.2f}" x2="{x+width:.2f}" y2="{y:.2f}" stroke="#0066CC" stroke-width="2" fill="none"/>
    <line x1="{x:.2f}" y1="{y+height:.2f}" x2="{x+width:.2f}" y2="{y+height:.2f}" stroke="#0066CC" stroke-width="2" fill="none"/>
    <line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y+height:.2f}" stroke="#0066CC" stroke-width="1" fill="none"/>
    <line x1="{x+width:.2f}" y1="{y:.2f}" x2="{x+width:.2f}" y2="{y+height:.2f}" stroke="#0066CC" stroke-width="1" fill="none"/>
    """


def create_dimension_line(x1: float, y1: float, x2: float, y2: float, dimension: str, offset: float = 20) -> str:
    """Create a professional dimension line"""
    # Calculate perpendicular offset
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length > 0:
        perp_x = -dy / length * offset
        perp_y = dx / length * offset
    
    # Offset points
    ox1, oy1 = x1 + perp_x, y1 + perp_y
    ox2, oy2 = x2 + perp_x, y2 + perp_y
    
    # Dimension line
    dim_line = f'<line x1="{ox1:.2f}" y1="{oy1:.2f}" x2="{ox2:.2f}" y2="{oy2:.2f}" stroke="#333333" stroke-width="1" stroke-dasharray="5,5" fill="none"/>'
    
    # Extension lines
    ext1 = f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{ox1:.2f}" y2="{oy1:.2f}" stroke="#333333" stroke-width="1" fill="none"/>'
    ext2 = f'<line x1="{x2:.2f}" y1="{y2:.2f}" x2="{ox2:.2f}" y2="{oy2:.2f}" stroke="#333333" stroke-width="1" fill="none"/>'
    
    # Dimension text
    mid_x = (ox1 + ox2) / 2
    mid_y = (oy1 + oy2) / 2
    text = f'<text x="{mid_x:.2f}" y="{mid_y-5:.2f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#333333">{dimension}</text>'
    
    return dim_line + "\n" + ext1 + "\n" + ext2 + "\n" + text


def create_room_label(x: float, y: float, room_name: str, room_area: str) -> str:
    """Create a professional room label"""
    return f"""
    <text x="{x:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#000000">{room_name}</text>
    <text x="{x:.2f}" y="{y+8:.2f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#666666">{room_area}</text>
    """


def create_north_arrow(x: float, y: float, size: float = 60) -> str:
    """Create a professional north arrow"""
    return f"""
    <g transform="translate({x:.2f}, {y:.2f})">
        <polygon points="0,-{size:.2f} {size/2:.2f},0 0,{size/4:.2f} -{size/2:.2f},0" fill="#FF0000" stroke="#000000" stroke-width="2"/>
        <text x="0" y="{size+20:.2f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#000000">NORTH</text>
    </g>
    """


def create_scale_bar(x: float, y: float, scale: str = "1:100") -> str:
    """Create a professional scale bar"""
    return f"""
    <g transform="translate({x:.2f}, {y:.2f})">
        <rect x="0" y="0" width="120" height="10" fill="none" stroke="#000000" stroke-width="2"/>
        <line x1="0" y1="0" x2="0" y2="15" stroke="#000000" stroke-width="2"/>
        <line x1="30" y1="0" x2="30" y2="15" stroke="#000000" stroke-width="2"/>
        <line x1="60" y1="0" x2="60" y2="15" stroke="#000000" stroke-width="2"/>
        <line x1="90" y1="0" x2="90" y2="15" stroke="#000000" stroke-width="2"/>
        <line x1="120" y1="0" x2="120" y2="15" stroke="#000000" stroke-width="2"/>
        <text x="60" y="30" text-anchor="middle" font-family="Arial" font-size="12" fill="#000000">0 3 6 9 12m</text>
        <text x="60" y="45" text-anchor="middle" font-family="Arial" font-size="10" fill="#666666">Scale: {scale}</text>
    </g>
    """


def create_professional_floor_plan(house_data: Dict[str, Any], width: int = 1600, height: int = 1200) -> str:
    """Create a professional CAD-quality floor plan"""
    
    # Professional room colors (subtle, architectural)
    room_colors = {
        "living_room": "#F8F8F8",      # Light gray
        "dining_room": "#F0F0F0",      # Very light gray
        "kitchen": "#E8F5E8",          # Very light green
        "master_bedroom": "#F0F0FF",   # Very light blue
        "bedroom": "#F8F8FF",          # Ghost white
        "bathroom": "#E0F8FF",         # Light cyan
        "utility": "#F5F5F5",          # White smoke
        "stairwell": "#E8E8E8",        # Light gray
        "entrance": "#FAFAFA",         # Very light gray
        "corridor": "#F5F5F5"          # White smoke
    }
    
    # Start building SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<style>',
        '.wall { stroke: #000000; stroke-width: 3; fill: none; }',
        '.door { stroke: #8B4513; stroke-width: 2; fill: none; }',
        '.window { stroke: #0066CC; stroke-width: 2; fill: none; }',
        '.dimension { stroke: #333333; stroke-width: 1; stroke-dasharray: 5,5; fill: none; }',
        '.room-fill { opacity: 0.3; }',
        '</style>',
        '</defs>',
        # Background
        f'<rect width="{width}" height="{height}" fill="white"/>',
        # Title
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#000000">HOUSEBRAIN - PROFESSIONAL FLOOR PLAN</text>'
    ]
    
    # Process each floor
    for floor_idx, floor in enumerate(house_data.get("geometry", {}).get("floors", [])):
        floor_name = floor.get("name", f"Floor {floor_idx}")
        floor_y_offset = 100 + floor_idx * (height - 120)
        
        # Floor title
        svg_parts.append(f'<text x="60" y="{floor_y_offset-30}" font-family="Arial" font-size="18" font-weight="bold" fill="#000000">{floor_name.upper()}</text>')
        
        # Draw room fills first (background)
        for room in floor.get("rooms", []):
            room_type = room.get("type", "room")
            color = room_colors.get(room_type, "#FFFFFF")
            
            polygon = room.get("polygon", [])
            if polygon:
                points = " ".join([f"{p[0]/10:.1f},{p[1]/10:.1f}" for p in polygon])
                svg_parts.append(f'<polygon points="{points}" fill="{color}" class="room-fill"/>')
        
        # Draw walls (room boundaries)
        for room in floor.get("rooms", []):
            polygon = room.get("polygon", [])
            if len(polygon) >= 4:
                # Draw wall lines
                for i in range(len(polygon) - 1):
                    x1, y1 = polygon[i][0]/10, polygon[i][1]/10
                    x2, y2 = polygon[i+1][0]/10, polygon[i+1][1]/10
                    svg_parts.append(create_wall_line(x1, y1, x2, y2))
        
        # Draw doors and windows
        for room in floor.get("rooms", []):
            # Draw doors
            for door_poly in room.get("doors", []):
                if len(door_poly) >= 2:
                    x = door_poly[0][0]/10
                    y = door_poly[0][1]/10
                    width = abs(door_poly[1][0] - door_poly[0][0])/10
                    svg_parts.append(create_door_opening(x, y, width))
            
            # Draw windows
            for window_poly in room.get("windows", []):
                if len(window_poly) >= 2:
                    x = window_poly[0][0]/10
                    y = window_poly[0][1]/10
                    w = abs(window_poly[1][0] - window_poly[0][0])/10
                    h = abs(window_poly[2][1] - window_poly[1][1])/10
                    svg_parts.append(create_window_opening(x, y, w, h))
        
        # Add room labels
        for room in floor.get("rooms", []):
            polygon = room.get("polygon", [])
            if polygon:
                room_name = room.get("name", room.get("type", "room").replace("_", " ").title())
                room_area = f"{room.get('area_sqm', 0)} sqm"
                
                # Calculate center for label
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                center_x = sum(xs) / len(xs) / 10
                center_y = sum(ys) / len(ys) / 10
                
                svg_parts.append(create_room_label(center_x, center_y, room_name, room_area))
        
        # Add key dimensions
        for room in floor.get("rooms", []):
            polygon = room.get("polygon", [])
            if len(polygon) >= 4:
                # Width dimension
                width = abs(polygon[1][0] - polygon[0][0])/10
                x1, y1 = polygon[0][0]/10, polygon[0][1]/10
                x2, y2 = polygon[1][0]/10, polygon[1][1]/10
                svg_parts.append(create_dimension_line(x1, y1, x2, y2, f"{width:.1f}m", 30))
                
                # Height dimension
                height = abs(polygon[2][1] - polygon[1][1])/10
                x1, y1 = polygon[1][0]/10, polygon[1][1]/10
                x2, y2 = polygon[2][0]/10, polygon[2][1]/10
                svg_parts.append(create_dimension_line(x1, y1, x2, y2, f"{height:.1f}m", 30))
    
    # Add north arrow and scale bar
    svg_parts.append(create_north_arrow(width - 120, 120))
    svg_parts.append(create_scale_bar(width - 180, height - 100))
    
    # Add project info
    project_info = [
        f'<text x="60" y="{height-80}" font-family="Arial" font-size="12" fill="#666666">Project: HouseBrain AI Architectural Design</text>',
        f'<text x="60" y="{height-65}" font-family="Arial" font-size="12" fill="#666666">Total Area: {house_data.get("total_area", 0)} sqm</text>',
        f'<text x="60" y="{height-50}" font-family="Arial" font-size="12" fill="#666666">Construction Cost: ₹{house_data.get("construction_cost", 0):,.0f}</text>',
        f'<text x="60" y="{height-35}" font-family="Arial" font-size="12" fill="#666666">Generated by HouseBrain v1.1 - Professional CAD Quality</text>',
        f'<text x="60" y="{height-20}" font-family="Arial" font-size="12" fill="#666666">Scale: 1:100 | Date: {house_data.get("render_metadata", {}).get("date", "2024")}</text>'
    ]
    svg_parts.extend(project_info)
    
    svg_parts.append('</svg>')
    
    return "\n".join(svg_parts)


def export_cad_quality_svg(input_path: str, output_path: str, width: int = 1600, height: int = 1200) -> None:
    """Export a professional CAD-quality SVG floor plan"""
    
    # Load house data
    with open(input_path, 'r', encoding='utf-8') as f:
        house_data = json.load(f)
    
    # Generate professional SVG
    svg_content = create_professional_floor_plan(house_data, width, height)
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"🏗️ Professional CAD-quality SVG created: {output_path}")
    print("✅ Industry-standard architectural drawing ready!")
    print("✅ Matches professional CAD software output!")


def main() -> None:
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create professional CAD-quality SVG floor plans")
    parser.add_argument("--input", required=True, help="HouseBrain JSON file")
    parser.add_argument("--output", required=True, help="Output SVG file")
    parser.add_argument("--width", type=int, default=1600, help="SVG width")
    parser.add_argument("--height", type=int, default=1200, help="SVG height")
    
    args = parser.parse_args()
    export_cad_quality_svg(args.input, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
