"""
Professional SVG renderer for HouseBrain that produces CAD-quality 2D floor plans.
This creates industry-standard architectural drawings with doors, windows, dimensions,
room labels, and annotations that rival professional CAD software.

Features:
- Professional architectural symbols
- Door and window openings with proper representation
- Room dimensions and area labels
- North arrow and scale bar
- Room names and types
- Professional color scheme and line weights
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any


def create_door_symbol(x: float, y: float, width: float, angle: float = 0) -> str:
    """Create SVG path for a door symbol (90-degree arc)"""
    # Door swing arc
    radius = width * 0.8
    # 90-degree swing
    end_angle = 90
    
    # Calculate arc points
    # arc start computed but unused in simplified symbol
    # arc start computed but unused in simplified symbol
    x2 = x + radius * math.cos(math.radians(end_angle))
    y2 = y + radius * math.sin(math.radians(end_angle))
    
    # Large arc flag (1 for angles > 180 degrees)
    large_arc = 0
    
    return f"M {x:.2f} {y:.2f} A {radius:.2f} {radius:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"


def create_window_symbol(x: float, y: float, width: float, height: float) -> str:
    """Create SVG path for a window symbol (double line)"""
    # Window opening with double lines
    return f"""
    <line x1="{x:.2f}" y1="{y:.2f}" x2="{x+width:.2f}" y2="{y:.2f}" stroke="#0066CC" stroke-width="2"/>
    <line x1="{x:.2f}" y1="{y+height:.2f}" x2="{x+width:.2f}" y2="{y+height:.2f}" stroke="#0066CC" stroke-width="2"/>
    <line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y+height:.2f}" stroke="#0066CC" stroke-width="1"/>
    <line x1="{x+width:.2f}" y1="{y:.2f}" x2="{x+width:.2f}" y2="{y+height:.2f}" stroke="#0066CC" stroke-width="1"/>
    """


def create_dimension_line(x1: float, y1: float, x2: float, y2: float, dimension: str) -> str:
    """Create a dimension line with measurement text"""
    # Dimension line
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    return f"""
    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#333333" stroke-width="1" stroke-dasharray="5,5"/>
    <text x="{mid_x:.2f}" y="{mid_y-5:.2f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333333">{dimension}</text>
    """


def create_north_arrow(x: float, y: float, size: float = 50) -> str:
    """Create a north arrow symbol"""
    return f"""
    <g transform="translate({x:.2f}, {y:.2f})">
        <polygon points="0,-{size:.2f} {size/2:.2f},0 0,{size/4:.2f} -{size/2:.2f},0" fill="#FF0000" stroke="#000000" stroke-width="1"/>
        <text x="0" y="{size+15:.2f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#000000">N</text>
    </g>
    """


def create_scale_bar(x: float, y: float, scale: str = "1:100") -> str:
    """Create a scale bar"""
    return f"""
    <g transform="translate({x:.2f}, {y:.2f})">
        <rect x="0" y="0" width="100" height="8" fill="none" stroke="#000000" stroke-width="1"/>
        <line x1="0" y1="0" x2="0" y2="12" stroke="#000000" stroke-width="1"/>
        <line x1="25" y1="0" x2="25" y2="12" stroke="#000000" stroke-width="1"/>
        <line x1="50" y1="0" x2="50" y2="12" stroke="#000000" stroke-width="1"/>
        <line x1="75" y1="0" x2="75" y2="12" stroke="#000000" stroke-width="1"/>
        <line x1="100" y1="0" x2="100" y2="12" stroke="#000000" stroke-width="1"/>
        <text x="50" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#000000">0 2.5 5 7.5 10m</text>
        <text x="50" y="40" text-anchor="middle" font-family="Arial" font-size="10" fill="#666666">Scale: {scale}</text>
    </g>
    """


def create_room_label(x: float, y: float, room_name: str, room_area: str) -> str:
    """Create a room label with name and area"""
    return f"""
    <text x="{x:.2f}" y="{y-5:.2f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#000000">{room_name}</text>
    <text x="{x:.2f}" y="{y+10:.2f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#666666">{room_area}</text>
    """


def create_professional_svg(house_data: Dict[str, Any], width: int = 1400, height: int = 1000) -> str:
    """Create a professional CAD-quality SVG floor plan"""
    
    # Room colors for different types (professional architectural colors)
    room_colors = {
        "living_room": "#F5F5DC",      # Beige
        "dining_room": "#F0E68C",      # Khaki
        "kitchen": "#98FB98",          # Pale green
        "master_bedroom": "#E6E6FA",   # Lavender
        "bedroom": "#F0F8FF",          # Alice blue
        "bathroom": "#E0F6FF",         # Light cyan
        "utility": "#F8F8FF",          # Ghost white
        "stairwell": "#F5F5F5",        # White smoke
        "entrance": "#FFFACD",         # Lemon chiffon
        "corridor": "#FAF0E6"          # Linen
    }
    
    # Start building SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<style>',
        '.room { stroke: #000000; stroke-width: 2; }',
        '.door { stroke: #8B4513; stroke-width: 3; fill: none; }',
        '.window { stroke: #0066CC; stroke-width: 2; }',
        '.dimension { stroke: #333333; stroke-width: 1; stroke-dasharray: 5,5; }',
        '.annotation { font-family: Arial; font-size: 12; fill: #333333; }',
        '.wall { stroke: #000000; stroke-width: 3; }',
        '</style>',
        '</defs>',
        # Background
        f'<rect width="{width}" height="{height}" fill="white"/>',
        # Title
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold" fill="#000000">HouseBrain - Professional Floor Plan</text>'
    ]
    
    # Process each floor
    for floor_idx, floor in enumerate(house_data.get("geometry", {}).get("floors", [])):
        floor_name = floor.get("name", f"Floor {floor_idx}")
        floor_y_offset = 80 + floor_idx * (height - 100)
        
        # Floor title
        svg_parts.append(f'<text x="50" y="{floor_y_offset-20}" font-family="Arial" font-size="16" font-weight="bold" fill="#000000">{floor_name}</text>')
        
        # Draw rooms
        for room in floor.get("rooms", []):
            room_type = room.get("type", "room")
            color = room_colors.get(room_type, "#FFFFFF")
            
            # Room polygon
            polygon = room.get("polygon", [])
            if polygon:
                points = " ".join([f"{p[0]/10:.1f},{p[1]/10:.1f}" for p in polygon])
                svg_parts.append(f'<polygon points="{points}" fill="{color}" class="room"/>')
                
                # Room label
                room_name = room.get("name", room_type.replace("_", " ").title())
                room_area = f"{room.get('area_sqm', 0)} sqm"
                
                # Calculate center for label
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                center_x = sum(xs) / len(xs) / 10
                center_y = sum(ys) / len(ys) / 10
                
                svg_parts.append(create_room_label(center_x, center_y, room_name, room_area))
                
                # Draw doors
                for door_poly in room.get("doors", []):
                    if len(door_poly) >= 2:
                        door_path = "M " + " L ".join([f"{p[0]/10:.1f} {p[1]/10:.1f}" for p in door_poly])
                        svg_parts.append(f'<path d="{door_path}" class="door"/>')
                        
                        # Door swing arc
                        if len(door_poly) >= 2:
                            x, y = door_poly[0][0]/10, door_poly[0][1]/10
                            width = abs(door_poly[1][0] - door_poly[0][0])/10
                            svg_parts.append(f'<path d="{create_door_symbol(x, y, width)}" class="door"/>')
                
                # Draw windows
                for window_poly in room.get("windows", []):
                    if len(window_poly) >= 2:
                        x = window_poly[0][0]/10
                        y = window_poly[0][1]/10
                        w = abs(window_poly[1][0] - window_poly[0][0])/10
                        h = abs(window_poly[2][1] - window_poly[1][1])/10
                        svg_parts.append(create_window_symbol(x, y, w, h))
        
        # Add dimensions for key rooms
        for room in floor.get("rooms", []):
            polygon = room.get("polygon", [])
            if len(polygon) >= 4:
                # Width dimension
                width = abs(polygon[1][0] - polygon[0][0])/10
                x1, y1 = polygon[0][0]/10, polygon[0][1]/10
                x2, y2 = polygon[1][0]/10, polygon[1][1]/10
                svg_parts.append(create_dimension_line(x1, y1-20, x2, y2-20, f"{width:.1f}m"))
                
                # Height dimension
                height = abs(polygon[2][1] - polygon[1][1])/10
                x1, y1 = polygon[1][0]/10, polygon[1][1]/10
                x2, y2 = polygon[2][0]/10, polygon[2][1]/10
                svg_parts.append(create_dimension_line(x1+20, y1, x2+20, y2, f"{height:.1f}m"))
    
    # Add north arrow and scale bar
    svg_parts.append(create_north_arrow(width - 100, 100))
    svg_parts.append(create_scale_bar(width - 150, height - 80))
    
    # Add project info
    project_info = [
        f'<text x="50" y="{height-60}" font-family="Arial" font-size="12" fill="#666666">Project: HouseBrain AI Design</text>',
        f'<text x="50" y="{height-45}" font-family="Arial" font-size="12" fill="#666666">Total Area: {house_data.get("total_area", 0)} sqm</text>',
        f'<text x="50" y="{height-30}" font-family="Arial" font-size="12" fill="#666666">Construction Cost: ₹{house_data.get("construction_cost", 0):,.0f}</text>',
        f'<text x="50" y="{height-15}" font-family="Arial" font-size="12" fill="#666666">Generated by HouseBrain v1.1</text>'
    ]
    svg_parts.extend(project_info)
    
    svg_parts.append('</svg>')
    
    return "\n".join(svg_parts)


def export_professional_svg(input_path: str, output_path: str, width: int = 1400, height: int = 1000) -> None:
    """Export a professional CAD-quality SVG floor plan"""
    
    # Load house data
    with open(input_path, 'r', encoding='utf-8') as f:
        house_data = json.load(f)
    
    # Generate professional SVG
    svg_content = create_professional_svg(house_data, width, height)
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"🏗️ Professional CAD-quality SVG created: {output_path}")
    print("✅ Industry-standard architectural drawing ready!")


def main() -> None:
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create professional CAD-quality SVG floor plans")
    parser.add_argument("--input", required=True, help="HouseBrain JSON file")
    parser.add_argument("--output", required=True, help="Output SVG file")
    parser.add_argument("--width", type=int, default=1400, help="SVG width")
    parser.add_argument("--height", type=int, default=1000, help="SVG height")
    
    args = parser.parse_args()
    export_professional_svg(args.input, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
