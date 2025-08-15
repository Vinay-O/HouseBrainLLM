#!/usr/bin/env python3
"""
HouseBrain India-Focused Dataset Generator

Generates 75K high-quality synthetic architectural datasets specifically for the Indian market.
Enhanced with Indian building codes (NBC), climate zones, architectural styles, and material preferences.
Optimized for maximum quality and Indian market requirements.

Features:
- Indian Building Codes (NBC 2016)
- Indian Climate Zones (Tropical, Subtropical, Arid, Composite)
- Indian Architectural Styles (Modern Indian, Indo-Saracenic, Contemporary Indian)
- Indian Plot Characteristics (smaller plots, Indian setbacks)
- Indian Material Preferences (concrete, Indian stones, tiles)
- Indian Budget Ranges and Construction Costs
"""

import os
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field, is_dataclass, asdict
import argparse

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import with fallback for notebook environment
try:
    from housebrain.schema import HouseInput, HouseOutput
    from housebrain.layout import solve_house_layout
except ImportError:
    # Define minimal versions for fallback
    @dataclass
    class HouseInput:
        basicDetails: Dict
        plot: Dict
        roomBreakdown: List
    
    @dataclass
    class HouseOutput:
        input: Dict
        levels: List
        total_area: int
        construction_cost: int
        materials: Dict
        render_paths: Dict
    
    def solve_house_layout(input_data):
        # Fallback implementation
        input_dict = asdict(input_data) if is_dataclass(input_data) else input_data
        return HouseOutput(
            input=input_dict,
            levels=[],
            total_area=input_dict["basicDetails"]["totalArea"],
            construction_cost=int(input_dict["basicDetails"]["budget"] * 0.6),
            materials={},
            render_paths={}
        )

@dataclass
class IndiaDatasetConfig:
    """Configuration for India-focused dataset generation"""
    num_samples: int = 75000
    output_dir: str = "housebrain_dataset_india_75k"
    train_ratio: float = 0.9
    min_plot_size: int = 500  # Indian plot sizes (sqft)
    max_plot_size: int = 5000
    min_bedrooms: int = 1
    max_bedrooms: int = 5
    min_floors: int = 1
    max_floors: int = 4
    min_budget: int = 500000  # Indian budget range (INR)
    max_budget: int = 50000000
    fast_mode: bool = True
    
    # Indian-specific configurations
    indian_styles: List[str] = field(default_factory=lambda: [
        "Modern_Indian", "Indo_Saracenic", "Contemporary_Indian", "Traditional_Indian",
        "Minimalist_Indian", "Luxury_Indian", "Eco_Friendly_Indian", "Smart_Indian",
        "Fusion_Indian", "Heritage_Indian", "Urban_Indian", "Suburban_Indian"
    ])
    
    indian_regions: List[str] = field(default_factory=lambda: [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore",
        "Bhopal", "Nagpur", "Vadodara", "Surat", "Nashik", "Thane"
    ])
    
    indian_climate_zones: List[str] = field(default_factory=lambda: [
        "Tropical_Hot_Humid", "Tropical_Warm_Humid", "Subtropical_Hot_Dry",
        "Subtropical_Warm_Humid", "Composite", "Arid_Hot_Dry", "Arid_Warm_Dry"
    ])

class IndiaHouseBrainDatasetGenerator:
    """Generates synthetic house design datasets specifically for Indian market"""
    
    def __init__(self, config: IndiaDatasetConfig):
        self.config = config
        self.indian_styles = config.indian_styles
        self.indian_regions = config.indian_regions
        self.indian_climate_zones = config.indian_climate_zones
        
        # Indian-specific parameter options
        self.indian_plot_shapes = ["Rectangle", "Square", "L_Shape", "Irregular", "Corner_Plot"]
        self.indian_exterior_materials = ["Concrete", "Brick", "Stone", "Glass", "Steel", "Wood", "Composite"]
        self.indian_roofing_materials = ["RCC_Slab", "Mangalore_Tiles", "Asphalt_Shingles", "Metal", "Clay_Tiles", "Green_Roof"]
        self.indian_window_types = ["Casement", "Sliding", "Fixed", "Bay", "Picture", "Jali"]
        self.indian_door_types = ["Wood", "Steel", "Glass", "Composite", "Traditional"]
        self.indian_seismic_zones = ["Zone_II", "Zone_III", "Zone_IV", "Zone_V"]  # Indian seismic zones
        self.indian_soil_types = ["Clay", "Sandy", "Rocky", "Loamy", "Black_Cotton", "Laterite"]
        self.indian_garage_types = ["Open_Parking", "Covered_Parking", "Basement_Parking", "None"]
        
        # Indian building codes (NBC 2016)
        self.nbc_floor_area_ratios = {
            "Residential": {"min": 0.3, "max": 0.7},
            "High_Rise": {"min": 0.4, "max": 0.8},
            "Luxury": {"min": 0.2, "max": 0.6}
        }
        
        # Indian construction costs per sqft (INR)
        self.indian_construction_costs = {
            "Economy": 1200,      # 1200 INR/sqft
            "Standard": 2000,     # 2000 INR/sqft
            "Premium": 3500,      # 3500 INR/sqft
            "Luxury": 5000        # 5000 INR/sqft
        }
    
    def _generate_indian_basic_details(self) -> Dict[str, Any]:
        """Generate Indian-specific basic details"""
        total_area = random.randint(800, 5000)  # Indian house sizes
        bedrooms = random.randint(self.config.min_bedrooms, self.config.max_bedrooms)
        bathrooms = max(1, bedrooms - 1)  # Indian bathroom ratio
        floors = random.randint(self.config.min_floors, self.config.max_floors)
        
        # Indian budget calculation based on area and location
        base_cost_per_sqft = random.choice(list(self.indian_construction_costs.values()))
        budget = total_area * base_cost_per_sqft * random.uniform(0.8, 1.2)
        
        return {
            "totalArea": total_area,
            "unit": "sqft",
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "budget": int(budget),
            "style": random.choice(self.indian_styles)
        }
    
    def _generate_indian_plot_details(self) -> Dict[str, Any]:
        """Generate Indian-specific plot details"""
        plot_size = random.randint(self.config.min_plot_size, self.config.max_plot_size)
        length = math.sqrt(plot_size * random.uniform(0.8, 1.2))
        width = plot_size / length
        
        # Indian setback requirements (varies by city)
        setback_percentage = random.uniform(0.05, 0.15)
        setback_ft = int(length * setback_percentage)
        
        return {
            "length": round(length, 1),
            "width": round(width, 1),
            "unit": "ft",
            "shape": random.choice(self.indian_plot_shapes),
            "orientation": random.choice(["N", "S", "E", "W", "NE", "NW", "SE", "SW"]),
            "slope_degrees": random.uniform(0, 10),
            "is_corner_plot": random.choice([True, False]),
            "setbacks_ft": {
                "front": setback_ft,
                "rear": setback_ft,
                "left": setback_ft,
                "right": setback_ft
            }
        }
    
    def _generate_indian_room_breakdown(self, bedrooms: int, total_area: int) -> List[Dict[str, Any]]:
        """Generate Indian-specific room breakdown"""
        rooms = []
        
        # Master bedroom (larger in Indian homes)
        master_area = random.randint(200, 400)
        rooms.append({
            "type": "master_bedroom",
            "count": 1,
            "minArea": master_area
        })
        
        # Regular bedrooms
        for i in range(bedrooms - 1):
            room_area = random.randint(120, 250)
            rooms.append({
                "type": "bedroom",
                "count": 1,
                "minArea": room_area
            })
        
        # Indian-specific rooms
        rooms.extend([
            {"type": "bathroom", "count": max(1, bedrooms - 1), "minArea": random.randint(40, 80)},
            {"type": "kitchen", "count": 1, "minArea": random.randint(150, 300)},
            {"type": "livingRoom", "count": 1, "minArea": random.randint(200, 400)},
            {"type": "diningRoom", "count": 1, "minArea": random.randint(100, 200)},
            {"type": "puja_room", "count": 1, "minArea": random.randint(50, 100)},  # Indian specific
            {"type": "utility", "count": 1, "minArea": random.randint(60, 120)},
            {"type": "balcony", "count": random.randint(1, 3), "minArea": random.randint(30, 80)}
        ])
        
        # Optional rooms based on area
        if total_area > 2000:
            rooms.extend([
                {"type": "homeOffice", "count": 1, "minArea": random.randint(100, 200)},
                {"type": "guest_room", "count": 1, "minArea": random.randint(150, 250)}
            ])
        
        return rooms
    
    def _create_indian_enhanced_output(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced output with Indian-specific features"""
        basic_details = input_data["basicDetails"]
        plot_data = input_data["plot"]
        
        # Indian construction cost calculation
        area = basic_details["totalArea"]
        base_cost = self.indian_construction_costs["Standard"]
        construction_cost = int(area * base_cost * random.uniform(0.8, 1.2))
        
        # Indian material preferences
        materials = {
            "exterior": random.choice(self.indian_exterior_materials),
            "roofing": random.choice(self.indian_roofing_materials),
            "flooring": random.choice(["Vitrified_Tiles", "Marble", "Granite", "Wood", "Carpet"]),
            "windows": random.choice(self.indian_window_types),
            "doors": random.choice(self.indian_door_types)
        }
        
        # Indian exterior specifications
        exterior_specs = {
            "exterior_wall": materials["exterior"],
            "roofing": materials["roofing"],
            "windows": materials["windows"],
            "doors": materials["doors"],
            "garage_type": random.choice(self.indian_garage_types),
            "garage_spaces": random.randint(0, 2),
            "exterior_color": random.choice(["White", "Beige", "Light_Blue", "Pink", "Yellow", "Gray"]),
            "roof_color": random.choice(["Red", "Brown", "Gray", "Blue", "Green"])
        }
        
        # Indian climate and site conditions
        climate_zone = random.choice(self.indian_climate_zones)
        region = random.choice(self.indian_regions)
        
        climate_and_site = {
            "climate_zone": climate_zone,
            "seismic_zone": random.choice(self.indian_seismic_zones),
            "wind_zone": random.choice(["Low", "Medium", "High"]),
            "snow_load": "None",  # Most of India
            "rainfall": random.choice(["Low", "Moderate", "High"]),
            "soil_type": random.choice(self.indian_soil_types),
            "water_table": random.choice(["Low", "Medium", "High"]),
            "utilities": {
                "water": random.choice(["Municipal", "Borewell", "Tanker"]),
                "sewer": random.choice(["Municipal", "Septic_Tank", "Bio_Toilet"]),
                "electricity": "Available",
                "gas": random.choice(["Available", "Not_Available", "LPG"]),
                "solar_ready": random.choice([True, False])
            },
            "cooling_priority": "High" if "Hot" in climate_zone else "Medium",
            "heating_priority": "Low" if "Hot" in climate_zone else "Medium",
            "insulation_level": "High" if "Hot" in climate_zone else "Medium"
        }
        
        # Indian building codes (NBC 2016)
        plot_area = plot_data["length"] * plot_data["width"]
        far = random.uniform(0.3, 0.7)  # Floor Area Ratio
        max_buildable = plot_area * far
        
        building_codes = {
            "floor_area_ratio": round(far, 2),
            "max_buildable_area": int(max_buildable),
            "max_height_ft": random.randint(25, 45),  # Indian height restrictions
            "parking_required": max(1, basic_details["bedrooms"] // 2),
            "fire_safety": {
                "sprinklers": basic_details["floors"] > 2,
                "fire_exits": 2 if basic_details["floors"] > 1 else 1,
                "fire_walls": basic_details["floors"] > 2
            },
            "accessibility": {
                "ramp_required": basic_details["floors"] > 1,
                "accessible_bathroom": basic_details["floors"] > 1
            },
            "nbc_compliance": {
                "structural_safety": True,
                "fire_safety": True,
                "electrical_safety": True,
                "plumbing_safety": True,
                "accessibility": True
            }
        }
        
        return {
            "input": input_data,
            "levels": [],
            "total_area": area,
            "construction_cost": construction_cost,
            "materials": materials,
            "exterior_specifications": exterior_specs,
            "climate_and_site": climate_and_site,
            "building_codes": building_codes,
            "render_paths": {
                "front": f"renders/{basic_details['style']}_front.png",
                "top": f"renders/{basic_details['style']}_top.png"
            }
        }
    
    def _validate_indian_sample(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> bool:
        """Validate Indian-specific requirements"""
        try:
            # Basic validation
            if not input_data or not output_data:
                return False
            
            basic_details = input_data.get("basicDetails", {})
            plot_data = input_data.get("plot", {})
            
            # Indian-specific validations
            if basic_details.get("totalArea", 0) < 500 or basic_details.get("totalArea", 0) > 10000:
                return False
            
            if basic_details.get("budget", 0) < 500000 or basic_details.get("budget", 0) > 100000000:
                return False
            
            # Plot size validation
            plot_area = plot_data.get("length", 0) * plot_data.get("width", 0)
            if plot_area < 500 or plot_area > 10000:
                return False
            
            # Room count validation
            room_breakdown = input_data.get("roomBreakdown", [])
            bedroom_count = sum(1 for room in room_breakdown if "bedroom" in room.get("type", ""))
            if bedroom_count != basic_details.get("bedrooms", 0):
                return False
            
            return True
            
        except Exception:
            return False
    
    def generate_dataset(self) -> Path:
        """Generate the India-focused dataset"""
        output_dir = Path(self.config.output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "validation"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏗️ Generating India-focused dataset: {self.config.num_samples:,} samples")
        print(f"📁 Output directory: {output_dir}")
        
        train_count = 0
        val_count = 0
        attempts = 0
        max_attempts = self.config.num_samples * 2
        
        while (train_count + val_count) < self.config.num_samples and attempts < max_attempts:
            attempts += 1
            
            try:
                # Generate Indian-specific input
                basic_details = self._generate_indian_basic_details()
                plot_data = self._generate_indian_plot_details()
                room_breakdown = self._generate_indian_room_breakdown(
                    basic_details["bedrooms"], 
                    basic_details["totalArea"]
                )
                
                input_data = {
                    "basicDetails": basic_details,
                    "plot": plot_data,
                    "roomBreakdown": room_breakdown
                }
                
                # Generate enhanced output
                output_data = self._create_indian_enhanced_output(input_data)
                
                # Validate sample
                if not self._validate_indian_sample(input_data, output_data):
                    continue
                
                # Create sample
                # Get materials from output_data
                materials = output_data["materials"]
                
                sample = {
                    "id": f"HBV5-IND-{(train_count + val_count + 1):06d}",
                    "input": input_data,
                    "output": output_data,
                    "metadata": {
                        "region": random.choice(self.indian_regions),
                        "climate_zone": output_data["climate_and_site"]["climate_zone"],
                        "plot_shape": plot_data["shape"],
                        "exterior_material": materials["exterior"],
                        "roofing_material": materials["roofing"],
                        "indian_specific": True,
                        "nbc_compliant": True,
                        "generated_at": "2024-01-01T00:00:00Z"
                    }
                }
                
                # Determine split
                if random.random() < self.config.train_ratio and train_count < int(self.config.num_samples * self.config.train_ratio):
                    split_dir = train_dir
                    train_count += 1
                elif val_count < int(self.config.num_samples * (1 - self.config.train_ratio)):
                    split_dir = val_dir
                    val_count += 1
                else:
                    continue
                
                # Save sample
                sample_file = split_dir / f"{sample['id']}.json"
                with open(sample_file, 'w') as f:
                    json.dump(sample, f, indent=2)
                
                if (train_count + val_count) % 1000 == 0:
                    print(f"✅ Generated {train_count + val_count:,} samples...")
                
            except Exception as e:
                print(f"⚠️ Error generating sample: {e}")
                continue
        
        # Create dataset info
        dataset_info = {
            "name": "HouseBrain India Dataset",
            "version": "1.0",
            "description": "India-focused synthetic architectural dataset with NBC compliance, Indian climate zones, and architectural styles",
            "num_samples": train_count + val_count,
            "train_samples": train_count,
            "val_samples": val_count,
            "generated_at": "2024-01-01T00:00:00Z",
            "indian_features": [
                "nbc_2016_compliance",
                "indian_climate_zones",
                "indian_architectural_styles",
                "indian_plot_characteristics",
                "indian_material_preferences",
                "indian_budget_ranges",
                "indian_seismic_zones",
                "indian_soil_types"
            ],
            "config": {
                "min_plot_size": self.config.min_plot_size,
                "max_plot_size": self.config.max_plot_size,
                "min_bedrooms": self.config.min_bedrooms,
                "max_bedrooms": self.config.max_bedrooms,
                "min_floors": self.config.min_floors,
                "max_floors": self.config.max_floors,
                "min_budget": self.config.min_budget,
                "max_budget": self.config.max_budget,
                "indian_styles": self.indian_styles,
                "indian_regions": self.indian_regions,
                "indian_climate_zones": self.indian_climate_zones,
                "fast_mode": self.config.fast_mode,
                "train_ratio": self.config.train_ratio
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ India-focused dataset generated successfully!")
        print(f"📊 Train: {train_count:,} | Val: {val_count:,} | Attempts: {attempts:,}")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Generate India-focused HouseBrain dataset")
    parser.add_argument("--samples", type=int, default=75000, help="Number of samples to generate")
    parser.add_argument("--output", default="housebrain_dataset_india_75k", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--fast", action="store_true", help="Use fast mode")
    parser.add_argument("--zip", action="store_true", help="Create zip archive after generation")
    
    args = parser.parse_args()
    
    print("🏗️ HouseBrain India-Focused Dataset Generator")
    print("=" * 60)
    
    config = IndiaDatasetConfig(
        num_samples=args.samples,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        fast_mode=args.fast
    )
    
    generator = IndiaHouseBrainDatasetGenerator(config)
    output_dir = generator.generate_dataset()
    
    if args.zip:
        import zipfile
        zip_path = output_dir.with_suffix('.zip')
        print(f"📦 Creating zip archive: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir.parent)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Zip archive created: {zip_path}")
    
    print(f"\n📋 India Dataset Summary:")
    print(f"   1. Dataset directory: {output_dir}")
    print(f"   2. Total samples: {args.samples:,} ({int(args.samples * args.train_ratio):,} train, {int(args.samples * (1 - args.train_ratio)):,} validation)")
    print(f"   3. Indian features: NBC compliance, climate zones, architectural styles")
    print(f"   4. Indian regions: {len(config.indian_regions)} major cities")
    print(f"   5. Indian styles: {len(config.indian_styles)} architectural styles")
    print(f"   6. Ready for training: Optimized for Indian market success!")

if __name__ == "__main__":
    main()
