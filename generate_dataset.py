#!/usr/bin/env python3
"""
HouseBrain Dataset Generator

Generates synthetic architectural datasets for training the HouseBrain LLM.
Can generate up to 100K test cases with realistic architectural parameters.
"""

import os
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
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
    output_dir: str = "housebrain_dataset_v5_100k"
    train_ratio: float = 0.9
    min_plot_size: int = 1000
    max_plot_size: int = 10000
    min_bedrooms: int = 1
    max_bedrooms: int = 6
    min_floors: int = 1
    max_floors: int = 4
    min_budget: int = 100000
    max_budget: int = 2000000
    styles: List[str] = None
    regions: List[str] = None


class HouseBrainDatasetGenerator:
    """Generates synthetic house design datasets"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.styles = config.styles or [
            "Modern", "Contemporary", "Traditional", "Colonial", "Mediterranean",
            "Craftsman", "Victorian", "Minimalist", "Scandinavian", "Industrial",
            "Tropical", "Rustic", "Art Deco", "Mid-Century Modern", "Gothic"
        ]
        self.regions = config.regions or [
            "US_Northeast", "US_Southeast", "US_Midwest", "US_Southwest", "US_West",
            "EU_UK", "EU_Germany", "EU_France", "EU_Italy", "EU_Spain",
            "Asia_India", "Asia_China", "Asia_Japan", "Asia_Singapore", "Asia_Australia"
        ]
        
    def generate_plot_dimensions(self) -> Dict[str, Any]:
        """Generate realistic plot dimensions"""
        # Generate plot area
        plot_area = random.randint(self.config.min_plot_size, self.config.max_plot_size)
        
        # Generate aspect ratio (width to length)
        aspect_ratio = random.uniform(0.5, 2.0)  # Rectangular plots
        
        # Calculate dimensions
        width = math.sqrt(plot_area * aspect_ratio)
        length = plot_area / width
        
        # Round to reasonable values
        width = round(width, 1)
        length = round(length, 1)
        
        return {
            "length": length,
            "width": width,
            "unit": "ft",
            "orientation": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "setbacks_ft": {
                "front": random.randint(3, 8),
                "rear": random.randint(3, 8),
                "left": random.randint(2, 6),
                "right": random.randint(2, 6)
            }
        }
    
    def generate_basic_details(self, plot_area: int) -> Dict[str, Any]:
        """Generate basic house details"""
        bedrooms = random.randint(self.config.min_bedrooms, self.config.max_bedrooms)
        floors = random.randint(self.config.min_floors, self.config.max_floors)
        
        # Calculate realistic budget based on area and region
        base_cost_per_sqft = random.randint(80, 200)
        budget = plot_area * base_cost_per_sqft * random.uniform(0.8, 1.2)
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
        """Generate room breakdown"""
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
        
        return rooms
    
    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single sample"""
        # Generate plot
        plot = self.generate_plot_dimensions()
        plot_area = int(plot["length"] * plot["width"])
        
        # Generate basic details
        basic_details = self.generate_basic_details(plot_area)
        
        # Generate room breakdown
        room_breakdown = self.generate_room_breakdown(
            basic_details["bedrooms"], 
            basic_details["floors"]
        )
        
        # Create HouseInput
        house_input = HouseInput(
            basicDetails=basic_details,
            plot=plot,
            roomBreakdown=room_breakdown
        )
        
        # Generate output using layout solver
        try:
            house_output = solve_house_layout(house_input)
            output_json = house_output.model_dump()
        except Exception as e:
            # If layout solver fails, create basic output
            output_json = self._create_basic_output(house_input)
        
        # Create training prompt
        prompt = self._create_training_prompt(house_input, output_json)
        
        return {
            "id": f"HBV5-{sample_id:06d}",
            "text": prompt,
            "input": house_input.model_dump(),
            "output": output_json,
            "metadata": {
                "region": random.choice(self.regions),
                "climate_zone": random.choice(["Tropical", "Subtropical", "Temperate", "Cold"]),
                "generated_at": "2024-01-01T00:00:00Z"
            }
        }
    
    def _create_basic_output(self, house_input: HouseInput) -> Dict[str, Any]:
        """Create basic output when layout solver fails"""
        return {
            "input": house_input.model_dump(),
            "levels": [],
            "total_area": house_input.basicDetails["totalArea"],
            "construction_cost": int(house_input.basicDetails["budget"] * 0.6),
            "materials": {},
            "render_paths": {}
        }
    
    def _create_training_prompt(self, house_input: HouseInput, output_json: Dict) -> str:
        """Create training prompt"""
        return f"""You are HouseBrain, an expert architectural AI that designs residential houses.

Given the following house requirements, generate a complete house design:

INPUT:
{json.dumps(house_input.model_dump(), indent=2)}

OUTPUT:
{json.dumps(output_json, indent=2)}

The design must be:
- Functional and practical
- Code compliant
- Cost-effective
- Aesthetically pleasing
- Optimized for the given plot and requirements

Think like an experienced architect who has designed hundreds of successful homes."""
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        print(f"🏗️  Generating {self.config.num_samples} house design samples...")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "validation"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        samples = []
        for i in range(self.config.num_samples):
            if i % 1000 == 0:
                print(f"   Generated {i}/{self.config.num_samples} samples...")
            
            sample = self.generate_sample(i + 1)
            samples.append(sample)
        
        # Split into train/validation
        random.shuffle(samples)
        split_idx = int(len(samples) * self.config.train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # Save training samples
        print(f"💾 Saving {len(train_samples)} training samples...")
        for i, sample in enumerate(train_samples):
            filename = f"HBV5-{sample['id']}.json"
            filepath = train_dir / filename
            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Save validation samples
        print(f"💾 Saving {len(val_samples)} validation samples...")
        for i, sample in enumerate(val_samples):
            filename = f"HBV5-{sample['id']}.json"
            filepath = val_dir / filename
            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Create dataset info
        dataset_info = {
            "name": "HouseBrain Dataset v5",
            "version": "5.0",
            "description": "Synthetic architectural dataset for HouseBrain LLM training",
            "num_samples": self.config.num_samples,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "generated_at": "2024-01-01T00:00:00Z",
            "config": {
                "min_plot_size": self.config.min_plot_size,
                "max_plot_size": self.config.max_plot_size,
                "min_bedrooms": self.config.min_bedrooms,
                "max_bedrooms": self.config.max_bedrooms,
                "styles": self.styles,
                "regions": self.regions
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ Dataset generated successfully!")
        print(f"   - Training samples: {len(train_samples)}")
        print(f"   - Validation samples: {len(val_samples)}")
        print(f"   - Output directory: {output_dir}")
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate HouseBrain dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", default="housebrain_dataset_v5_100k", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split ratio")
    
    args = parser.parse_args()
    
    print("🏠 HouseBrain Dataset Generator")
    print("=" * 50)
    
    config = DatasetConfig(
        num_samples=args.samples,
        output_dir=args.output,
        train_ratio=args.train_ratio
    )
    
    generator = HouseBrainDatasetGenerator(config)
    output_dir = generator.generate_dataset()
    
    print(f"\n📋 Next steps:")
    print(f"   1. Upload to Google Colab: {output_dir}")
    print(f"   2. Use for training: python finetune_m2pro.py --dataset {args.output}")
    print(f"   3. Push to Git: git add {args.output} && git commit -m 'Add dataset'")


if __name__ == "__main__":
    main()
