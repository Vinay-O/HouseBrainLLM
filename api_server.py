#!/usr/bin/env python3
"""
HouseBrain API Server for Frontend Integration
Provides RESTful endpoints for all frontend sections
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from housebrain.pipeline_v2 import run_pipeline
from housebrain.reports import boq_from_plan
from generate_synthetic_v2 import make_simple_house

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

@app.route('/api/generate-house', methods=['POST'])
def generate_house():
    """
    Generate complete house design from user requirements
    Input: {rooms: [...], style: str, budget: int, location: str}
    Output: {plan_json, svg_urls, gltf_urls, cost_estimate, construction_flow}
    """
    try:
        data = request.json
        
        # Extract user requirements
        rooms = data.get('rooms', [])
        style = data.get('style', 'modern')
        budget = data.get('budget', 100000)
        location = data.get('location', 'suburban')
        
        # Generate plan using LLM (placeholder - integrate your actual LLM here)
        plan = generate_plan_from_requirements(rooms, style, budget, location)
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save plan JSON
            plan_path = os.path.join(temp_dir, 'plan.json')
            with open(plan_path, 'w') as f:
                json.dump(plan, f)
            
            # Run pipeline
            output_dir = os.path.join(temp_dir, 'outputs')
            run_pipeline(plan_path, output_dir, 1800, 1400, ['floor', 'rcp', 'power', 'plumbing'])
            
            # Generate cost estimate
            cost_estimate = generate_detailed_cost_estimate(plan, budget, location)
            
            # Generate construction flow
            construction_flow = generate_construction_sequence(plan)
            
            # Copy outputs to persistent storage (implement your storage strategy)
            persistent_urls = save_outputs_to_storage(output_dir)
            
            return jsonify({
                'success': True,
                'plan_json': plan,
                'outputs': persistent_urls,
                'cost_estimate': cost_estimate,
                'construction_flow': construction_flow
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cost-estimator', methods=['POST'])
def cost_estimator():
    """
    Standalone cost estimation from plan or requirements
    Input: {plan_json} or {basic_requirements}
    Output: {detailed_costs, material_breakdown, labor_costs, timeline}
    """
    try:
        data = request.json
        
        if 'plan_json' in data:
            plan = data['plan_json']
        else:
            # Generate simple plan from requirements for costing
            plan = generate_simple_plan_for_costing(data)
        
        # Enhanced cost estimation
        cost_breakdown = generate_detailed_cost_estimate(
            plan, 
            data.get('budget', 100000),
            data.get('location', 'suburban')
        )
        
        return jsonify({
            'success': True,
            'cost_breakdown': cost_breakdown
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/interior-design', methods=['POST'])
def interior_design():
    """
    Generate interior design images and layouts
    Input: {room_type, style, color_scheme, furniture_preferences}
    Output: {generated_images, layout_suggestions, material_palette}
    """
    try:
        data = request.json
        
        # Generate interior design using AI image generation
        interior_results = generate_interior_design(
            data.get('room_type'),
            data.get('style'),
            data.get('color_scheme'),
            data.get('furniture_preferences')
        )
        
        return jsonify({
            'success': True,
            'interior_design': interior_results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_plan_from_requirements(rooms, style, budget, location):
    """
    Generate architectural plan from user requirements
    This is where your trained LLM would be integrated
    """
    # Placeholder: Use simple house generator with requirements
    # TODO: Replace with actual LLM inference
    
    # Estimate dimensions based on rooms and budget
    total_area_sqft = min(budget / 150, 3000)  # $150/sqft estimate
    plot_w = (total_area_sqft * 92.9) ** 0.5  # Convert to mm
    plot_h = plot_w * 0.8  # Rectangular ratio
    
    plan = make_simple_house(plot_w, plot_h)
    
    # Customize based on requirements
    plan['metadata']['style'] = style
    plan['metadata']['budget'] = budget
    plan['metadata']['location'] = location
    
    return plan

def generate_detailed_cost_estimate(plan, budget, location):
    """
    Enhanced cost estimation with regional pricing
    """
    # Get basic BoQ
    boq = boq_from_plan(plan)
    
    # Regional cost multipliers
    location_multipliers = {
        'urban': 1.3,
        'suburban': 1.0,
        'rural': 0.8
    }
    base_multiplier = location_multipliers.get(location, 1.0)
    
    # Detailed cost breakdown
    costs = {
        'foundation': {
            'area_sqft': boq['total_area_sqft'],
            'cost_per_sqft': 15 * base_multiplier,
            'total': boq['total_area_sqft'] * 15 * base_multiplier
        },
        'framing': {
            'wall_length_ft': (boq['wall_lengths_mm']['exterior'] + boq['wall_lengths_mm']['interior']) / 304.8,
            'cost_per_ft': 25 * base_multiplier,
            'total': ((boq['wall_lengths_mm']['exterior'] + boq['wall_lengths_mm']['interior']) / 304.8) * 25 * base_multiplier
        },
        'roofing': {
            'area_sqft': boq['total_area_sqft'] * 1.2,  # Roof area estimate
            'cost_per_sqft': 12 * base_multiplier,
            'total': boq['total_area_sqft'] * 1.2 * 12 * base_multiplier
        },
        'electrical': {
            'outlets': sum(boq['door_counts'].values()) * 5,  # Estimate outlets
            'cost_per_outlet': 150 * base_multiplier,
            'total': sum(boq['door_counts'].values()) * 5 * 150 * base_multiplier
        },
        'plumbing': {
            'fixtures': 8,  # Estimate fixtures
            'cost_per_fixture': 800 * base_multiplier,
            'total': 8 * 800 * base_multiplier
        }
    }
    
    # Calculate totals
    material_total = sum(item['total'] for item in costs.values())
    labor_total = material_total * 0.6  # 60% labor overhead
    permit_total = material_total * 0.05  # 5% permits
    contingency = (material_total + labor_total + permit_total) * 0.1  # 10% contingency
    
    total_cost = material_total + labor_total + permit_total + contingency
    
    return {
        'breakdown': costs,
        'summary': {
            'materials': material_total,
            'labor': labor_total,
            'permits': permit_total,
            'contingency': contingency,
            'total': total_cost
        },
        'timeline_months': max(4, total_cost / 50000),  # Rough timeline estimate
        'cost_per_sqft': total_cost / boq['total_area_sqft']
    }

def generate_construction_sequence(plan):
    """
    Generate construction timeline and sequence
    """
    phases = [
        {'phase': 'Site Preparation', 'duration_days': 5, 'description': 'Survey, permits, excavation'},
        {'phase': 'Foundation', 'duration_days': 10, 'description': 'Footings, foundation walls, waterproofing'},
        {'phase': 'Framing', 'duration_days': 15, 'description': 'Wall framing, roof structure'},
        {'phase': 'MEP Rough-In', 'duration_days': 12, 'description': 'Electrical, plumbing, HVAC rough installation'},
        {'phase': 'Insulation & Drywall', 'duration_days': 10, 'description': 'Insulation, drywall installation and finishing'},
        {'phase': 'Flooring', 'duration_days': 8, 'description': 'Flooring installation throughout'},
        {'phase': 'Kitchen & Bath', 'duration_days': 12, 'description': 'Cabinet and fixture installation'},
        {'phase': 'Final MEP', 'duration_days': 8, 'description': 'Final electrical, plumbing, HVAC'},
        {'phase': 'Paint & Trim', 'duration_days': 10, 'description': 'Interior painting and trim work'},
        {'phase': 'Final Inspection', 'duration_days': 3, 'description': 'Final inspections and cleanup'}
    ]
    
    return {
        'phases': phases,
        'total_duration_days': sum(p['duration_days'] for p in phases),
        'critical_path': ['Foundation', 'Framing', 'MEP Rough-In', 'Drywall']
    }

def generate_interior_design(room_type, style, color_scheme, furniture_prefs):
    """
    Generate interior design recommendations
    TODO: Integrate with image generation APIs (DALL-E, Midjourney, etc.)
    """
    # Placeholder implementation
    return {
        'style_recommendations': f"{style} style for {room_type}",
        'color_palette': color_scheme,
        'furniture_suggestions': furniture_prefs,
        'generated_images': [
            f"placeholder_url_for_{room_type}_{style}_1.jpg",
            f"placeholder_url_for_{room_type}_{style}_2.jpg"
        ],
        'material_suggestions': {
            'flooring': 'Hardwood oak' if style == 'traditional' else 'Polished concrete',
            'walls': 'Neutral paint' if style == 'modern' else 'Textured wallpaper',
            'lighting': 'Recessed LED' if style == 'modern' else 'Pendant fixtures'
        }
    }

def generate_simple_plan_for_costing(requirements):
    """Generate minimal plan for cost estimation"""
    area = requirements.get('area_sqft', 1500)
    plot_w = (area * 92.9) ** 0.5
    plot_h = plot_w * 0.8
    return make_simple_house(plot_w, plot_h)

def save_outputs_to_storage(output_dir):
    """
    Save outputs to persistent storage and return URLs
    TODO: Implement your storage strategy (S3, GCS, local, etc.)
    """
    # Placeholder - return mock URLs
    base_name = "example_output"
    return {
        'svg': {
            'floor': f"/outputs/{base_name}_floor.svg",
            'rcp': f"/outputs/{base_name}_rcp.svg",
            'power': f"/outputs/{base_name}_power.svg",
            'plumbing': f"/outputs/{base_name}_plumbing.svg"
        },
        'gltf': {
            'floor': f"/outputs/{base_name}_floor.gltf",
            'rcp': f"/outputs/{base_name}_rcp.gltf",
            'power': f"/outputs/{base_name}_power.gltf",
            'plumbing': f"/outputs/{base_name}_plumbing.gltf"
        },
        'dxf': f"/outputs/{base_name}.dxf",
        'boq': f"/outputs/{base_name}_boq.json",
        'index': f"/outputs/index.html"
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
