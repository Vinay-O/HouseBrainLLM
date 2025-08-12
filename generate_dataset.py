#!/usr/bin/env python3
"""
HouseBrain Dataset Generator - Enhanced Version

Generates synthetic architectural datasets for training the HouseBrain LLM.
Enhanced with crucial parameters: plot shape, exterior finishes, climate, building codes.
Can generate up to 200K test cases with realistic architectural parameters.
"""

import os
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import argparse

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from housebrain.schema import HouseInput, HouseOutput
from housebrain.layout import solve_house_layout


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    num_samples: int = 1000
    output_dir: str = "housebrain_dataset_v5_150k"
    train_ratio: float = 0.9
    min_plot_size: int = 1000
    max_plot_size: int = 10000
    min_bedrooms: int = 1
    max_bedrooms: int = 6
    min_floors: int = 1
    max_floors: int = 4
    min_budget: int = 100000
    max_budget: int = 2000000
    fast_mode: bool = True  # Skip layout solving for speed
    styles: List[str] = field(default_factory=lambda: [
        "Modern", "Contemporary", "Traditional", "Colonial", "Mediterranean",
        "Craftsman", "Victorian", "Minimalist", "Scandinavian", "Industrial",
        "Tropical", "Rustic", "Art Deco", "Mid-Century Modern", "Gothic"
    ])
    regions: List[str] = field(default_factory=lambda: [
        "US_Northeast", "US_Southeast", "US_Midwest", "US_Southwest", "US_West",
        "EU_UK", "EU_Germany", "EU_France", "EU_Italy", "EU_Spain",
        "Asia_India", "Asia_China", "Asia_Japan", "Asia_Singapore", "Asia_Australia"
    ])


class HouseBrainDatasetGenerator:
    """Generates synthetic house design datasets with enhanced parameters"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.styles = config.styles
        self.regions = config.regions
        
        # Enhanced parameter options
        self.plot_shapes = ["Rectangle", "L_Shape", "Irregular", "Corner_Plot", "Square"]
        self.exterior_materials = ["Brick", "Stone", "Stucco", "Vinyl", "Wood", "Concrete", "Fiber_Cement"]
        self.roofing_materials = ["Asphalt_Shingles", "Metal", "Tile", "Slate", "Flat_Roof", "Wood_Shingles"]
        self.window_types = ["Single_Hung", "Double_Hung", "Casement", "Picture", "Sliding", "Bay"]
        self.door_types = ["Wood", "Steel", "Fiberglass", "Sliding_Glass", "French"]
        self.climate_zones = ["Hot_Dry", "Hot_Humid", "Cold", "Temperate", "Tropical", "Mediterranean"]
        self.seismic_zones = ["Low", "Medium", "High"]
        self.soil_types = ["Clay", "Sandy", "Rocky", "Loamy", "Silty"]
        self.garage_types = ["Attached", "Detached", "Carport", "None"]
        
    def generate_plot_dimensions(self) -> Dict[str, Any]:
        """Generate realistic plot dimensions with enhanced parameters"""
        # Generate plot area
        plot_area = random.randint(self.config.min_plot_size, self.config.max_plot_size)
        
        # Choose plot shape
        plot_shape = random.choice(self.plot_shapes)
        
        # Generate dimensions based on shape
        if plot_shape == "Square":
            aspect_ratio = random.uniform(0.8, 1.2)
        elif plot_shape == "L_Shape":
            aspect_ratio = random.uniform(0.3, 0.7)  # More elongated
        elif plot_shape == "Corner_Plot":
            aspect_ratio = random.uniform(0.6, 1.4)  # Slightly varied
        else:  # Rectangle or Irregular
            aspect_ratio = random.uniform(0.4, 2.5)
        
        # Calculate dimensions
        width = math.sqrt(plot_area * aspect_ratio)
        length = plot_area / width
        
        # Round to reasonable values
        width = round(width, 1)
        length = round(length, 1)
        
        # Generate slope (0-15 degrees)
        slope = random.uniform(0, 15)
        
        # Determine if corner plot
        is_corner_plot = plot_shape == "Corner_Plot" or random.random() > 0.8
        
        return {
            "length": length,
            "width": width,
            "unit": "ft",
            "shape": plot_shape,
            "orientation": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "slope_degrees": round(slope, 1),
            "is_corner_plot": is_corner_plot,
            "setbacks_ft": {
                "front": random.randint(3, 12) if not is_corner_plot else random.randint(2, 8),
                "rear": random.randint(3, 12),
                "left": random.randint(2, 8) if not is_corner_plot else random.randint(1, 5),
                "right": random.randint(2, 8) if not is_corner_plot else random.randint(1, 5)
            }
        }
    
    def generate_exterior_specifications(self) -> Dict[str, Any]:
        """Generate exterior finishes and materials"""
        return {
            "exterior_wall": random.choice(self.exterior_materials),
            "roofing": random.choice(self.roofing_materials),
            "windows": random.choice(self.window_types),
            "doors": random.choice(self.door_types),
            "garage_type": random.choice(self.garage_types),
            "garage_spaces": random.randint(0, 3) if random.choice(self.garage_types) != "None" else 0,
            "exterior_color": random.choice([
                "White", "Beige", "Gray", "Brown", "Blue", "Green", "Red", "Yellow"
            ]),
            "roof_color": random.choice([
                "Black", "Gray", "Brown", "Red", "Green", "Blue"
            ])
        }
    
    def generate_climate_and_site(self) -> Dict[str, Any]:
        """Generate climate and site conditions"""
        climate_zone = random.choice(self.climate_zones)
        
        # Climate-specific parameters
        if climate_zone in ["Hot_Dry", "Hot_Humid"]:
            cooling_priority = "High"
            heating_priority = "Low"
            insulation_level = "High"
        elif climate_zone == "Cold":
            cooling_priority = "Low"
            heating_priority = "High"
            insulation_level = "High"
        else:
            cooling_priority = "Medium"
            heating_priority = "Medium"
            insulation_level = "Medium"
        
        return {
            "climate_zone": climate_zone,
            "seismic_zone": random.choice(self.seismic_zones),
            "wind_zone": random.choice(["Low", "Medium", "High"]),
            "snow_load": random.choice(["None", "Light", "Heavy"]),
            "rainfall": random.choice(["Low", "Moderate", "High"]),
            "soil_type": random.choice(self.soil_types),
            "water_table": random.choice(["Low", "Medium", "High"]),
            "utilities": {
                "water": random.choice(["City", "Well"]),
                "sewer": random.choice(["City", "Septic"]),
                "electricity": "Available",
                "gas": random.choice(["Available", "Not_Available"]),
                "solar_ready": random.choice([True, False])
            },
            "cooling_priority": cooling_priority,
            "heating_priority": heating_priority,
            "insulation_level": insulation_level
        }
    
    def generate_building_codes(self, plot_area: int, floors: int) -> Dict[str, Any]:
        """Generate building code requirements"""
        # Calculate FAR (Floor Area Ratio) - typically 0.2 to 0.8
        far = random.uniform(0.2, 0.8)
        max_buildable_area = int(plot_area * far)
        
        # Height restrictions (typically 25-35 feet for residential)
        max_height_ft = random.randint(25, 35)
        
        # Parking requirements (typically 1-2 spaces per bedroom)
        parking_spaces = random.randint(1, 3)
        
        return {
            "floor_area_ratio": round(far, 2),
            "max_buildable_area": max_buildable_area,
            "max_height_ft": max_height_ft,
            "parking_required": parking_spaces,
            "fire_safety": {
                "sprinklers": random.choice([True, False]),
                "fire_exits": random.randint(1, 3),
                "fire_walls": random.choice([True, False])
            },
            "accessibility": {
                "ramp_required": random.choice([True, False]),
                "accessible_bathroom": random.choice([True, False])
            }
        }
    
    def generate_basic_details(self, plot_area: int) -> Dict[str, Any]:
        """Generate basic house details with enhanced parameters"""
        bedrooms = random.randint(self.config.min_bedrooms, self.config.max_bedrooms)
        floors = random.randint(self.config.min_floors, self.config.max_floors)
        
        # Calculate realistic budget based on area, region, and materials
        base_cost_per_sqft = random.randint(80, 200)
        
        # Adjust for floors (multi-story costs more)
        floor_multiplier = 1.0 if floors == 1 else (1.2 if floors == 2 else 1.4)
        
        budget = plot_area * base_cost_per_sqft * floor_multiplier * random.uniform(0.8, 1.2)
        budget = int(budget)
        
        # Ensure budget is within limits
        budget = max(self.config.min_budget, min(self.config.max_budget, budget))
        
        # Calculate bathrooms based on bedrooms
        bathrooms = max(1, bedrooms // 2)  # At least 1 bathroom, 1 per 2 bedrooms
        
        return {
            "totalArea": plot_area,
            "unit": "sqft",
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "budget": budget,
            "style": random.choice(self.styles)
        }
    
    def generate_room_breakdown(self, bedrooms: int, floors: int) -> List[Dict[str, Any]]:
        """Generate room breakdown with enhanced room types"""
        rooms = []
        
        # Calculate bathrooms (typically 1-2 per floor)
        bathrooms_per_floor = max(1, bedrooms // 3)
        total_bathrooms = bathrooms_per_floor * floors
        
        # Add bedrooms
        for i in range(bedrooms):
            room_type = "master_bedroom" if i == 0 else "bedroom"
            rooms.append({
                "type": room_type,
                "count": 1,
                "minArea": random.randint(120, 300) if room_type == "master_bedroom" else random.randint(100, 200)
            })
        
        # Add bathrooms
        rooms.append({
            "type": "bathroom",
            "count": total_bathrooms,
            "minArea": random.randint(40, 80)
        })
        
        # Add kitchen
        rooms.append({
            "type": "kitchen",
            "count": 1,
            "minArea": random.randint(120, 250)
        })
        
        # Add living room
        rooms.append({
            "type": "livingRoom",
            "count": 1,
            "minArea": random.randint(200, 400)
        })
        
        # Add dining room (optional)
        if random.random() > 0.3:  # 70% chance
            rooms.append({
                "type": "diningRoom",
                "count": 1,
                "minArea": random.randint(120, 250)
            })
        
        # Add utility room (optional)
        if random.random() > 0.5:  # 50% chance
            rooms.append({
                "type": "utility",
                "count": 1,
                "minArea": random.randint(50, 100)
            })
        
        # Add home office (optional)
        if random.random() > 0.6:  # 40% chance
            rooms.append({
                "type": "homeOffice",
                "count": 1,
                "minArea": random.randint(80, 150)
            })
        
        # Add balcony/patio (optional, especially for multi-story)
        if floors > 1 and random.random() > 0.7:  # 30% chance for multi-story
            rooms.append({
                "type": "balcony",
                "count": random.randint(1, 3),
                "minArea": random.randint(30, 80)
            })
        
        return rooms
    
    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single sample with enhanced parameters"""
        # Generate plot with enhanced parameters
        plot = self.generate_plot_dimensions()
        plot_area = int(plot["length"] * plot["width"])
        
        # Generate basic details
        basic_details = self.generate_basic_details(plot_area)
        
        # Generate room breakdown
        room_breakdown = self.generate_room_breakdown(
            basic_details["bedrooms"], 
            basic_details["floors"]
        )
        
        # Generate enhanced specifications
        exterior_specs = self.generate_exterior_specifications()
        climate_site = self.generate_climate_and_site()
        building_codes = self.generate_building_codes(plot_area, basic_details["floors"])
        
        # Create enhanced HouseInput
        house_input = HouseInput(
            basicDetails=basic_details,
            plot=plot,
            roomBreakdown=room_breakdown
        )
        
        # Generate output
        if self.config.fast_mode:
            output_json = self._create_enhanced_output(house_input, exterior_specs, climate_site, building_codes)
        else:
            try:
                house_output = solve_house_layout(house_input)
                output_json = house_output.model_dump()
                # Add enhanced parameters to output
                output_json.update({
                    "exterior_specifications": exterior_specs,
                    "climate_and_site": climate_site,
                    "building_codes": building_codes
                })
            except Exception as e:
                output_json = self._create_enhanced_output(house_input, exterior_specs, climate_site, building_codes)
        
        return {
            "id": f"HBV5-{sample_id:06d}",
            "input": house_input.model_dump(),
            "output": output_json,
            "metadata": {
                "region": random.choice(self.regions),
                "climate_zone": climate_site["climate_zone"],
                "plot_shape": plot["shape"],
                "exterior_material": exterior_specs["exterior_wall"],
                "roofing_material": exterior_specs["roofing"],
                "generated_at": "2024-01-01T00:00:00Z"
            }
        }
    
    def _create_enhanced_output(self, house_input: HouseInput, exterior_specs: Dict, climate_site: Dict, building_codes: Dict) -> Dict[str, Any]:
        """Create enhanced output when layout solver fails"""
        return {
            "input": house_input.model_dump(),
            "levels": [],
            "total_area": house_input.basicDetails["totalArea"],
            "construction_cost": int(house_input.basicDetails["budget"] * 0.6),
            "materials": {
                "exterior": exterior_specs["exterior_wall"],
                "roofing": exterior_specs["roofing"],
                "flooring": random.choice(["Hardwood", "Tile", "Carpet", "Laminate"]),
                "windows": exterior_specs["windows"],
                "doors": exterior_specs["doors"]
            },
            "exterior_specifications": exterior_specs,
            "climate_and_site": climate_site,
            "building_codes": building_codes,
            "render_paths": {
                "front": f"renders/{house_input.basicDetails['style']}_front.png",
                "top": f"renders/{house_input.basicDetails['style']}_top.png"
            }
        }
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        print(f"🏗️ Generating {self.config.num_samples:,} enhanced house design samples...")
        output_dir = Path(self.config.output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "validation"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        samples = []
        for i in range(self.config.num_samples):
            if (i + 1) % 1000 == 0 or (i + 1) == self.config.num_samples:
                print(f"   Generated {i+1:,}/{self.config.num_samples:,} samples...")
            
            sample = self.generate_sample(i + 1)
            samples.append(sample)
        
        # Split into train/validation
        random.shuffle(samples)
        split_idx = int(len(samples) * self.config.train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # Save training samples
        print(f"💾 Saving {len(train_samples):,} training samples...")
        for sample in train_samples:
            filename = f"{sample['id']}.json"
            filepath = train_dir / filename
            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Save validation samples
        print(f"💾 Saving {len(val_samples):,} validation samples...")
        for sample in val_samples:
            filename = f"{sample['id']}.json"
            filepath = val_dir / filename
            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Create dataset info
        dataset_info = {
            "name": "HouseBrain Dataset v5 Enhanced",
            "version": "5.1",
            "description": "Enhanced synthetic architectural dataset with plot shape, exterior finishes, climate, and building codes",
            "num_samples": self.config.num_samples,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "generated_at": "2024-01-01T00:00:00Z",
            "enhanced_features": [
                "plot_shape_and_orientation",
                "exterior_finishes_and_materials",
                "climate_and_site_conditions",
                "building_codes_and_regulations",
                "garage_and_parking",
                "utilities_and_accessibility"
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
                "styles": self.styles,
                "regions": self.regions,
                "fast_mode": self.config.fast_mode
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ Enhanced dataset generated successfully!")
        print(f"📊 Dataset features: {len(dataset_info['enhanced_features'])} enhanced parameters")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced HouseBrain dataset")
    parser.add_argument("--samples", type=int, default=150000, help="Number of samples to generate")
    parser.add_argument("--output", default="housebrain_dataset_v5_150k", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (skip layout solving)")
    parser.add_argument("--zip", action="store_true", help="Create zip archive after generation")
    
    args = parser.parse_args()
    
    print("🏠 HouseBrain Enhanced Dataset Generator")
    print("=" * 50)
    
    config = DatasetConfig(
        num_samples=args.samples,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        fast_mode=args.fast
    )
    
    generator = HouseBrainDatasetGenerator(config)
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
    
    print(f"\n📋 Next steps:")
    print(f"   1. Dataset directory: {output_dir}")
    print(f"   2. Total samples: {args.samples:,} ({int(args.samples * args.train_ratio):,} train, {int(args.samples * (1 - args.train_ratio)):,} validation)")
    print(f"   3. Enhanced features: Plot shape, exterior finishes, climate, building codes")
    print(f"   4. Use for training: Follow COLAB_KAGGLE_WORKFLOW.md")


if __name__ == "__main__":
    main()
