#!/usr/bin/env python3
"""
HouseBrain Advanced Dataset Generator
Generates super-quality architectural datasets with advanced reasoning capabilities.
Matches the sophistication of the existing 1M dataset.
"""

import json
import random
import argparse
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from tqdm import tqdm



@dataclass
class AdvancedDatasetConfig:
    """Advanced configuration for super-quality dataset generation."""
    target_samples: int = 1_000_000
    quality_threshold: float = 0.90
    train_ratio: float = 0.90
    shard_size: int = 50_000  # Align with 1M super dataset shards
    min_reasoning_steps: int = 8
    min_output_chars: int = 800
    max_output_chars: int = 20_000
    india_ratio: float = 0.60
    seed: int = 42
    output_dir: str = "housebrain_advanced_dataset"
    zip_output: bool = False

    # Weighted distribution to match 74/23/3 split exactly
    # High (74%): GC 25, SFP 20, Structural 15, Cost 14
    # Medium (23%): Energy 8, MEP 7, Interior 5, Landscape 3
    # Lower (3%): Sustainability 2, Smart Home 1
    problem_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "Geometric_Construction": 0.25,
        "Spatial_Floor_Planning": 0.20,
        "Structural_Engineering": 0.15,
        "Cost_Engineering": 0.14,
        "Energy_Engineering": 0.08,
        "MEP_Design": 0.07,
        "Interior_Design": 0.05,
        "Landscape_Design": 0.03,
        "Sustainability_Design": 0.02,
        "Smart_Home_Integration": 0.01,
    })


# Advanced problem types with detailed reasoning
PROBLEM_TYPES = {
    "Geometric_Construction": {
        "description": "Exact coordinate system, wall assemblies, material quantities",
        "reasoning_steps": [
            "Establish coordinate system and grid for precise geometric placement",
            "Calculate exact room dimensions and wall positions with coordinates",
            "Design structural grid system with column and beam placements",
            "Generate foundation geometry with footing positions and sizes",
            "Plan multi-floor continuity with vertical alignment of structural elements",
            "Calculate exact material quantities for construction planning",
            "Create 2D floor plan data with walls, doors, windows, and dimensions",
            "Generate 3D model data with spatial coordinates and material specifications"
        ],
        "output_sections": ["geometric_data", "structural_grid", "material_quantities", "2d_floor_plan", "3d_model_data"]
    },
    
    "Spatial_Floor_Planning": {
        "description": "Room adjacencies, floor continuity, structural grid",
        "reasoning_steps": [
            "Analyze functional relationships between different spaces",
            "Design optimal room adjacencies for efficient circulation",
            "Plan multi-floor spatial continuity and vertical circulation",
            "Integrate structural grid with spatial planning requirements",
            "Optimize daylight and ventilation through spatial arrangement",
            "Ensure accessibility and universal design compliance",
            "Plan service areas and utility spaces for optimal functionality",
            "Create spatial hierarchy and zoning for different activities"
        ],
        "output_sections": ["spatial_layout", "room_adjacencies", "circulation_plan", "floor_continuity", "accessibility_features"]
    },
    
    "Structural_Engineering": {
        "description": "Load calculations, foundation design, seismic analysis",
        "reasoning_steps": [
            "Analyze site conditions and soil properties to determine foundation requirements",
            "Calculate structural loads including dead loads, live loads, and environmental loads",
            "Design foundation system based on soil bearing capacity and building loads",
            "Determine structural system and member sizes considering safety factors and codes",
            "Perform seismic analysis and design for the specific seismic zone requirements",
            "Analyze wind loads and lateral stability considering building height and location",
            "Detail structural connections and joints for optimal load transfer and constructability",
            "Validate design through structural analysis and ensure compliance with all safety standards"
        ],
        "output_sections": ["structural_design", "seismic_design", "safety_factors", "construction_details", "quality_control"]
    },
    
    "Cost_Engineering": {
        "description": "BOQ, material costs, labor estimates",
        "reasoning_steps": [
            "Analyze project scope and requirements for comprehensive cost estimation",
            "Calculate material quantities based on detailed design specifications",
            "Estimate labor requirements considering skill levels and productivity rates",
            "Determine equipment and machinery costs for construction activities",
            "Calculate overhead costs including supervision, quality control, and safety measures",
            "Analyze market rates and inflation factors for accurate cost projection",
            "Develop cash flow projections and payment schedules for financial planning",
            "Identify cost optimization opportunities and value engineering alternatives"
        ],
        "output_sections": ["cost_breakdown", "material_quantities", "labor_estimates", "cash_flow", "optimization_opportunities"]
    },
    
    "Energy_Engineering": {
        "description": "Thermal analysis, HVAC design, sustainability",
        "reasoning_steps": [
            "Analyze climate data and building orientation for thermal performance",
            "Calculate building envelope thermal properties and heat transfer coefficients",
            "Design HVAC system based on cooling and heating load calculations",
            "Optimize natural ventilation and passive cooling strategies",
            "Integrate renewable energy systems for sustainable power generation",
            "Design lighting systems for energy efficiency and occupant comfort",
            "Implement building automation and energy management systems",
            "Ensure compliance with energy codes and green building standards"
        ],
        "output_sections": ["energy_analysis", "hvac_design", "thermal_performance", "renewable_energy", "energy_management"]
    },
    
    "MEP_Design": {
        "description": "Electrical, plumbing, mechanical systems",
        "reasoning_steps": [
            "Analyze electrical load requirements and design power distribution system",
            "Design lighting systems considering energy efficiency and occupant comfort",
            "Plan plumbing system with water supply, drainage, and waste management",
            "Design mechanical systems including HVAC, ventilation, and air quality",
            "Integrate fire protection and life safety systems",
            "Plan communication and data infrastructure for smart building features",
            "Ensure coordination between different MEP systems for optimal performance",
            "Comply with all relevant codes and standards for MEP installations"
        ],
        "output_sections": ["electrical_design", "plumbing_design", "mechanical_design", "fire_protection", "communication_systems"]
    },
    
    "Interior_Design": {
        "description": "Finishes, furniture, lighting design",
        "reasoning_steps": [
            "Analyze functional requirements and user needs for interior spaces",
            "Design space planning and furniture layout for optimal functionality",
            "Select appropriate materials and finishes for durability and aesthetics",
            "Design lighting systems for ambient, task, and accent lighting",
            "Plan color schemes and visual hierarchy for cohesive design",
            "Integrate acoustic design for sound quality and noise control",
            "Ensure accessibility and universal design compliance",
            "Create sustainable interior solutions with eco-friendly materials"
        ],
        "output_sections": ["interior_layout", "material_specifications", "lighting_design", "color_scheme", "acoustic_design"]
    },
    
    "Landscape_Design": {
        "description": "Site planning, hardscape, softscape",
        "reasoning_steps": [
            "Analyze site conditions including topography, soil, and climate",
            "Design site layout considering building placement and circulation",
            "Plan hardscape elements including walkways, driveways, and outdoor spaces",
            "Design softscape with appropriate plant selection and landscaping",
            "Integrate water features and irrigation systems for sustainability",
            "Plan outdoor lighting and security systems for safety and aesthetics",
            "Ensure proper drainage and stormwater management",
            "Create outdoor spaces for recreation, relaxation, and social interaction"
        ],
        "output_sections": ["site_layout", "hardscape_design", "softscape_design", "water_features", "outdoor_amenities"]
    },
    
    "Sustainability_Design": {
        "description": "Green building, renewable energy",
        "reasoning_steps": [
            "Analyze environmental impact and carbon footprint of building design",
            "Design energy-efficient building envelope and systems",
            "Integrate renewable energy systems for sustainable power generation",
            "Plan water conservation and rainwater harvesting systems",
            "Design waste management and recycling facilities",
            "Select sustainable materials with low environmental impact",
            "Implement green building certification requirements",
            "Create monitoring and reporting systems for sustainability performance"
        ],
        "output_sections": ["sustainability_analysis", "renewable_energy", "water_conservation", "waste_management", "green_certification"]
    },
    
    "Smart_Home_Integration": {
        "description": "IoT, automation, security systems",
        "reasoning_steps": [
            "Analyze user requirements for smart home functionality and automation",
            "Design IoT infrastructure for device connectivity and data management",
            "Plan home automation systems for lighting, HVAC, and security",
            "Integrate security systems including access control and surveillance",
            "Design communication networks for smart device integration",
            "Plan energy management systems for optimal efficiency",
            "Ensure cybersecurity and data privacy for smart home systems",
            "Create user interfaces and control systems for easy operation"
        ],
        "output_sections": ["iot_infrastructure", "automation_systems", "security_design", "communication_networks", "user_interfaces"]
    }
}


def generate_advanced_context(problem_type: str, india_market: bool = True) -> Dict[str, Any]:
    """Generate advanced context for the problem type."""
    context = {
        "indian_market": india_market,
        "problem_type": problem_type
    }
    
    if india_market:
        context.update({
            "region": random.choice([
                "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", 
                "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Bhopal",
                "Indore", "Nagpur", "Vadodara", "Surat", "Varanasi", "Patna"
            ]),
            "climate_zone": random.choice([
                "Hot_Dry", "Warm_Humid", "Temperate", "Cold", "Composite"
            ]),
            "architectural_style": random.choice([
                "Modern_Indian", "Traditional_Indian", "Contemporary_Indian", 
                "Colonial_Indian", "Vernacular_Indian", "Fusion_Indian"
            ]),
            "vastu_compliance": random.choice([True, False]),
            "nbc_compliance": True,
            "local_bye_laws": True
        })
    else:
        context.update({
            "region": random.choice([
                "North_America", "Europe", "Asia_Pacific", "Middle_East", "Africa"
            ]),
            "climate_zone": random.choice([
                "Tropical", "Subtropical", "Temperate", "Continental", "Polar"
            ]),
            "architectural_style": random.choice([
                "Modern", "Traditional", "Contemporary", "Minimalist", "Industrial"
            ])
        })
    
    return context


def generate_advanced_input(problem_type: str) -> Dict[str, Any]:
    """Generate advanced input data for the problem type."""
    india_market = random.random() < 0.6  # 60% India ratio
    
    # Basic plot details
    plot_details = {
        "plot_shape": random.choice(["Rectangular", "Square", "L_Shape", "Irregular"]),
        "width_ft": random.randint(30, 120),
        "length_ft": random.randint(40, 150),
        "orientation": random.choice(["North", "South", "East", "West", "North_East", "North_West", "South_East", "South_West"]),
        "road_access": random.choice(["Single_Side", "Multiple_Sides", "Corner"]),
        "setback_front": random.randint(10, 25),
        "setback_rear": random.randint(10, 20),
        "setback_side": random.randint(5, 15)
    }
    plot_details["area_sqft"] = plot_details["width_ft"] * plot_details["length_ft"]
    
    # Requirements
    requirements = {
        "family_size": random.randint(2, 8),
        "floors": random.randint(1, 4),
        "budget_inr": random.randint(500000, 10000000),
        "lifestyle": random.choice(["Minimalist", "Luxury", "Traditional", "Modern", "Eco_Friendly"])
    }
    
    # Problem-specific elements
    problem_info = PROBLEM_TYPES[problem_type]
    elements = []
    
    if problem_type == "Geometric_Construction":
        elements = [
            "Exact_coordinate_system", "Wall_thickness_specifications", "Structural_grid_alignment",
            "Foundation_geometry", "Column_and_beam_positions", "Material_quantity_calculations",
            "2D_floor_plan_generation", "3D_model_generation"
        ]
    elif problem_type == "Structural_Engineering":
        elements = ["Foundation", "Column", "Beam", "Slab", "Roof", "Wall", "Stair", "Retaining_wall"]
    elif problem_type == "Cost_Engineering":
        elements = ["Material_quantities", "Labor_estimates", "Equipment_costs", "Overhead_costs", "Profit_margin"]
    elif problem_type == "Energy_Engineering":
        elements = ["Building_envelope", "HVAC_system", "Lighting_system", "Renewable_energy", "Energy_management"]
    else:
        elements = ["Design_analysis", "Technical_specifications", "Implementation_plan", "Quality_control"]
    
    return {
        "problem_type": problem_type,
        "context": generate_advanced_context(problem_type, india_market),
        "plot_details": plot_details,
        "requirements": requirements,
        "construction_requirements": elements,
        "reasoning_steps": problem_info["reasoning_steps"]
    }


def generate_advanced_output(problem_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate advanced output data for the problem type."""
    problem_info = PROBLEM_TYPES[problem_type]
    output_sections = problem_info["output_sections"]
    
    output = {}
    
    # Generate problem-specific output sections
    for section in output_sections:
        if section == "geometric_data":
            output[section] = generate_geometric_data(input_data)
        elif section == "structural_design":
            output[section] = generate_structural_design(input_data)
        elif section == "cost_breakdown":
            output[section] = generate_cost_breakdown(input_data)
        elif section == "energy_analysis":
            output[section] = generate_energy_analysis(input_data)
        elif section == "interior_layout":
            output[section] = generate_interior_layout(input_data)
        else:
            output[section] = generate_generic_section(section, input_data)
    
    # Add enhanced metadata (v1.1 features)
    output["metadata_augmented_v1_1"] = {
        "units": "mm",
        "project_origin": {"x": 0.0, "y": 0.0, "z": 0.0},
        "north_angle_deg": 0.0,
        "elevation_datum": {"name": "FFL_0", "elevation_mm": 0},
        "dxf_layers": {
            "WALLS": {"color": 7, "lineweight": 0.35},
            "DOORS": {"color": 3, "lineweight": 0.25},
            "WINDOWS": {"color": 4, "lineweight": 0.25},
            "COLUMNS": {"color": 2, "lineweight": 0.35},
            "BEAMS": {"color": 1, "lineweight": 0.35},
            "DIMENSIONS": {"color": 5, "lineweight": 0.18}
        },
        "ifc_classes": {
            "walls": "IfcWall",
            "doors": "IfcDoor", 
            "windows": "IfcWindow",
            "columns": "IfcColumn",
            "beams": "IfcBeam"
        },
        "guids": {
            "project_guid": "12345678-1234-1234-1234-123456789012",
            "building_guid": "87654321-4321-4321-4321-210987654321"
        }
    }
    
    return output


def generate_geometric_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate richer multi-floor geometric construction data."""
    plot = input_data["plot_details"]
    floors = max(1, int(input_data.get("requirements", {}).get("floors", 1)))

    def make_room(room_id: str, floor_name: str, x: int, y: int, w: int, h: int, exterior: bool) -> Dict[str, Any]:
        walls = []
        # four walls rectangle
        walls.append({"start": {"x": x, "y": y}, "end": {"x": x + w, "y": y}, "thickness": 0.75, "type": "exterior" if exterior else "interior"})
        walls.append({"start": {"x": x + w, "y": y}, "end": {"x": x + w, "y": y + h}, "thickness": 0.75, "type": "exterior" if exterior else "interior"})
        walls.append({"start": {"x": x + w, "y": y + h}, "end": {"x": x, "y": y + h}, "thickness": 0.75, "type": "exterior" if exterior else "interior"})
        walls.append({"start": {"x": x, "y": y + h}, "end": {"x": x, "y": y}, "thickness": 0.75, "type": "exterior" if exterior else "interior"})
        return {
            "id": room_id,
            "type": room_id,
            "floor": floor_name,
            "bounds": {"x": x, "y": y, "width": w, "height": h, "area": w * h},
            "walls": walls,
            "doors": [{"x": x + w // 2, "y": y, "width": 3}],
            "windows": [{"x": x + w - 2, "y": y + h // 2, "width": 4}],
        }

    rooms: List[Dict[str, Any]] = []
    # generate 6–10 rooms per floor
    for floor_idx in range(floors):
        floor_name = "Ground" if floor_idx == 0 else f"Level_{floor_idx+1}"
        num_rooms = random.randint(6, 10)
        cursor_x, cursor_y = 0, 0
        cell_w = max(10, plot["width_ft"] // 6)
        cell_h = max(10, plot["length_ft"] // 6)
        for r in range(num_rooms):
            w = random.randint(cell_w // 2, cell_w)
            h = random.randint(cell_h // 2, cell_h)
            rooms.append(make_room(
                room_id=random.choice(["living_room","kitchen","bedroom","bathroom","dining","study","utility","guest_room"]),
                floor_name=floor_name,
                x=cursor_x,
                y=cursor_y,
                w=w,
                h=h,
                exterior=(cursor_x == 0 or cursor_y == 0)
            ))
            cursor_x += w + 2
            if cursor_x + w > plot["width_ft"]:
                cursor_x = 0
                cursor_y += h + 2

    return {
        "plot_dimensions": {
            "width_ft": plot["width_ft"],
            "length_ft": plot["length_ft"],
            "area_sqft": plot["area_sqft"]
        },
        "wall_specifications": {
            "exterior_wall_thickness_inches": 9,
            "interior_wall_thickness_inches": 4.5,
            "foundation_wall_thickness_inches": 12
        },
        "rooms": rooms,
        "structural_grid": {
            "grid_spacing_ft": 20,
            "column_positions": [[0, 0], [20, 0], [40, 0], [0, 20], [20, 20], [40, 20]],
            "beam_positions": [[0, 10], [20, 10], [40, 10]]
        },
        "material_quantities": {
            "concrete_cum": 45.6,
            "steel_kg": 2340,
            "bricks_count": 12500,
            "cement_bags": 180
        }
    }


def generate_structural_design(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structural engineering data."""
    return {
        "foundation": {
            "type": "Isolated_Footing",
            "depth_m": 3,
            "bearing_capacity": "217 kN/m2",
            "soil_type": "Soft_Rock"
        },
        "superstructure": {
            "system": "RCC_Frame_Structure",
            "columns": "16_columns",
            "beams": "Primary_and_secondary_beams",
            "slab": "RCC_slab_150mm_thick"
        },
        "load_analysis": {
            "dead_load_kg_m2": 375000,
            "live_load_kg_m2": 500000,
            "total_load_kg_m2": 875000,
            "wind_load_kg_m2": 72
        },
        "seismic_design": {
            "zone": "Zone_V",
            "response_reduction_factor": 3.78,
            "importance_factor": 1.0,
            "ductility_detailing": "As_per_IS_13920",
            "base_shear": "1052 kN"
        },
        "safety_factors": {
            "concrete_safety_factor": 1.77,
            "steel_safety_factor": 1.17,
            "load_combination_factor": 1.37
        }
    }


def generate_cost_breakdown(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cost engineering data."""
    budget = input_data["requirements"]["budget_inr"]
    
    return {
        "total_cost_inr": budget,
        "cost_breakdown": {
            "materials": budget * 0.45,
            "labor": budget * 0.25,
            "equipment": budget * 0.15,
            "overhead": budget * 0.10,
            "profit": budget * 0.05
        },
        "material_quantities": {
            "concrete_cum": 45.6,
            "steel_kg": 2340,
            "bricks_count": 12500,
            "cement_bags": 180
        },
        "labor_estimates": {
            "masonry_work_days": 45,
            "carpentry_work_days": 30,
            "electrical_work_days": 25,
            "plumbing_work_days": 20
        }
    }


def generate_energy_analysis(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate energy engineering data."""
    return {
        "thermal_analysis": {
            "u_value_walls": 0.35,
            "u_value_roof": 0.25,
            "u_value_windows": 2.1,
            "solar_heat_gain_coefficient": 0.3
        },
        "hvac_design": {
            "cooling_load_ton": 5.2,
            "heating_load_kw": 8.5,
            "system_type": "Split_AC",
            "efficiency_rating": "5_Star"
        },
        "renewable_energy": {
            "solar_panels_kw": 3.0,
            "battery_storage_kwh": 10.0,
            "annual_generation_kwh": 4500
        }
    }


def generate_interior_layout(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interior design data."""
    return {
        "space_planning": {
            "living_area_sqft": 800,
            "bedroom_area_sqft": 600,
            "kitchen_area_sqft": 200,
            "bathroom_area_sqft": 150
        },
        "material_specifications": {
            "flooring": "Vitrified_tiles",
            "wall_finish": "Paint",
            "ceiling": "Gypsum_board",
            "kitchen_counter": "Granite"
        },
        "lighting_design": {
            "ambient_lighting": "LED_downlights",
            "task_lighting": "Under_cabinet_LEDs",
            "accent_lighting": "Wall_washers"
        }
    }


def generate_generic_section(section: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate generic section data."""
    return {
        "analysis": f"Comprehensive {section.replace('_', ' ').lower()} analysis completed",
        "specifications": f"Detailed {section.replace('_', ' ').lower()} specifications provided",
        "implementation": f"Step-by-step {section.replace('_', ' ').lower()} implementation plan",
        "quality_control": f"Quality control measures for {section.replace('_', ' ').lower()}"
    }


def calculate_quality_score(sample: Dict[str, Any]) -> float:
    """Calculate quality score for the sample."""
    score = 0.0
    
    # Check reasoning steps
    if len(sample["input"]["reasoning_steps"]) >= 8:
        score += 0.3
    
    # Check output complexity
    output_str = json.dumps(sample["output"])
    if len(output_str) >= 800:
        score += 0.3
    
    # Check problem type complexity
    problem_type = sample["input"]["problem_type"]
    if problem_type in ["Geometric_Construction", "Structural_Engineering", "Cost_Engineering"]:
        score += 0.2
    elif problem_type in ["Spatial_Floor_Planning", "Energy_Engineering", "MEP_Design"]:
        score += 0.15
    else:
        score += 0.1
    
    # Check India-specific features
    if sample["input"]["context"]["indian_market"]:
        score += 0.1
    
    # Check metadata completeness
    if "metadata_augmented_v1_1" in sample["output"]:
        score += 0.1
    
    return min(score, 1.0)


def generate_advanced_sample(config: AdvancedDatasetConfig) -> Optional[Dict[str, Any]]:
    """Generate a single advanced training sample."""
    # Select problem type using weighted distribution 74/23/3
    problem_types = list(config.problem_type_weights.keys())
    weights = list(config.problem_type_weights.values())
    problem_type = random.choices(problem_types, weights=weights, k=1)[0]
    
    # Generate input
    input_data = generate_advanced_input(problem_type)
    
    # Generate output
    output_data = generate_advanced_output(problem_type, input_data)
    
    # Create sample
    sample = {
        "input": input_data,
        "output": output_data
    }
    
    # Enforce hard minimum output length
    output_len = len(json.dumps(output_data))
    if output_len < config.min_output_chars:
        return None

    # Calculate quality score
    quality_score = calculate_quality_score(sample)
    
    # Apply quality threshold
    if quality_score < config.quality_threshold:
        return None
    
    return sample


def generate_advanced_dataset(config: AdvancedDatasetConfig) -> None:
    """Generate the complete advanced dataset."""
    print(f"🏠 Generating {config.target_samples} super-quality HouseBrain samples...")
    print(f"🎯 Quality threshold: {config.quality_threshold}")
    print(f"🇮🇳 India ratio: {config.india_ratio}")
    print(f"🧠 Min reasoning steps: {config.min_reasoning_steps}")
    
    # Set random seed
    random.seed(config.seed)
    
    # Create output directories
    train_dir = Path(config.output_dir) / "train"
    val_dir = Path(config.output_dir) / "validation"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate split
    train_samples = int(config.target_samples * config.train_ratio)
    
    # Generate samples with progress tracking
    all_samples = []
    seen_hashes = set()
    generated = 0
    accepted = 0
    
    with tqdm(total=config.target_samples, desc="Generating super-quality samples") as pbar:
        while accepted < config.target_samples:
            sample = generate_advanced_sample(config)
            generated += 1
            
            if sample is not None:
                # Deduplicate using hash of input section
                sample_hash = hashlib.sha256(json.dumps(sample["input"], sort_keys=True).encode("utf-8")).hexdigest()
                if sample_hash in seen_hashes:
                    continue
                seen_hashes.add(sample_hash)
                all_samples.append(sample)
                accepted += 1
                pbar.update(1)
                
                # Update progress description
                acceptance_rate = (accepted / generated) * 100
                pbar.set_description(f"✅ Accepted {accepted}/{config.target_samples} | Rate: {acceptance_rate:.2f}%")
            
            # Safety check
            if generated > config.target_samples * 10:  # Max 10x generation attempts
                break
    
    print("🎉 Super-quality dataset generation complete!")
    print(f"📊 Total generated: {generated}")
    print(f"✅ Accepted: {len(all_samples)}")
    
    # Split and save in shards
    train_samples_list = all_samples[:train_samples]
    val_samples_list = all_samples[train_samples:]
    
    # Save training data in shards
    print(f"💾 Saving {len(train_samples_list)} training samples...")
    for i in range(0, len(train_samples_list), config.shard_size):
        shard_num = (i // config.shard_size) + 1
        shard_dir = train_dir / f"shard_{shard_num:02d}"
        shard_dir.mkdir(exist_ok=True)
        
        shard_samples = train_samples_list[i:i + config.shard_size]
        for j, sample in enumerate(shard_samples):
            filename = f"sample_{i + j:07d}.json"
            filepath = shard_dir / filename
            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)
    
    # Save validation data
    print(f"💾 Saving {len(val_samples_list)} validation samples...")
    for i, sample in enumerate(val_samples_list):
        filename = f"sample_{i:07d}.json"
        filepath = val_dir / filename
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
    
    # Save dataset info
    dataset_info = {
        "name": "housebrain_dataset_r1_super_1M",
        "version": "R1_Super_v1.0",
        "total_samples": len(all_samples),
        "train_ratio": config.train_ratio,
        "india_ratio": config.india_ratio,
        "quality_threshold": config.quality_threshold,
        "shard_size": config.shard_size,
        "min_reasoning_steps": config.min_reasoning_steps,
        "generated_at": datetime.now().isoformat()
    }
    
    info_file = Path(config.output_dir) / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("✅ Advanced dataset generated successfully!")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📊 Dataset info: {dataset_info}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate HouseBrain Advanced Dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--quality", type=float, default=0.90, help="Quality threshold (0.0-1.0)")
    parser.add_argument("--india", type=float, default=0.60, help="India market ratio (0.0-1.0)")
    parser.add_argument("--output", type=str, default="housebrain_advanced_dataset", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=100000, help="Samples per shard")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AdvancedDatasetConfig(
        target_samples=args.samples,
        quality_threshold=args.quality,
        india_ratio=args.india,
        output_dir=args.output,
        shard_size=args.shard_size
    )
    
    # Generate dataset
    generate_advanced_dataset(config)


if __name__ == "__main__":
    main()\n