#!/usr/bin/env python3
"""
Enhanced Dataset Generation for HouseBrain
Target: 1M high-quality samples with quality gates
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from tqdm import tqdm

@dataclass
class EnhancedDatasetConfig:
    """Configuration for enhanced dataset generation"""
    
    # Target sizes
    target_samples: int = 1000000  # 1M samples
    current_samples: int = 425000   # Current dataset size
    daily_generation: int = 50000   # Generate 50K per day
    quality_threshold: float = 0.8  # Minimum quality score
    
    # Indian market focus
    india_ratio: float = 0.4  # 40% India-specific data
    regional_focus: List[str] = field(default_factory=lambda: [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore"
    ])
    
    # Building types and styles
    building_types: List[str] = field(default_factory=lambda: [
        "Residential", "Commercial", "Mixed_Use", "Industrial", "Institutional"
    ])
    
    indian_styles: List[str] = field(default_factory=lambda: [
        "Modern_Indian", "Indo_Saracenic", "Contemporary_Indian", "Traditional_Indian",
        "Minimalist_Indian", "Luxury_Indian", "Eco_Friendly_Indian", "Smart_Indian",
        "Fusion_Indian", "Heritage_Indian", "Urban_Indian", "Suburban_Indian"
    ])
    
    # Climate zones (India-specific)
    climate_zones: List[str] = field(default_factory=lambda: [
        "Tropical_Hot_Humid", "Tropical_Warm_Humid", "Subtropical_Hot_Dry",
        "Subtropical_Warm_Humid", "Composite", "Arid_Hot_Dry", "Arid_Warm_Dry"
    ])
    
    # Budget ranges (INR)
    budget_ranges: List[Dict] = field(default_factory=lambda: [
        {"min": 500000, "max": 2000000, "label": "Economy"},
        {"min": 2000000, "max": 5000000, "label": "Mid_Range"},
        {"min": 5000000, "max": 15000000, "label": "Premium"},
        {"min": 15000000, "max": 50000000, "label": "Luxury"},
        {"min": 50000000, "max": 200000000, "label": "Ultra_Luxury"}
    ])
    
    # Plot characteristics
    plot_shapes: List[str] = field(default_factory=lambda: [
        "Rectangular", "Square", "L_Shape", "T_Shape", "Irregular", "Corner_Plot"
    ])
    
    # Quality gates
    min_rooms: int = 1
    max_rooms: int = 15
    min_area: int = 500  # sq ft
    max_area: int = 10000  # sq ft

class EnhancedHouseBrainDatasetGenerator:
    """Enhanced dataset generator with quality gates"""
    
    def __init__(self, config: EnhancedDatasetConfig):
        self.config = config
        self.generated_samples = 0
        self.quality_gates = []
        
    def generate_plot_details(self) -> Dict[str, Any]:
        """Generate realistic plot details"""
        plot_shape = random.choice(self.config.plot_shapes)
        
        # Generate dimensions based on shape
        if plot_shape == "Square":
            width = random.randint(30, 100)
            length = width
        elif plot_shape == "Rectangular":
            width = random.randint(25, 80)
            length = random.randint(width + 10, width * 2)
        else:
            width = random.randint(30, 90)
            length = random.randint(35, 95)
        
        area = width * length
        
        return {
            "plot_shape": plot_shape,
            "width_ft": width,
            "length_ft": length,
            "area_sqft": area,
            "orientation": random.choice(["North", "South", "East", "West", "North_East", "North_West", "South_East", "South_West"]),
            "road_access": random.choice(["Front", "Side", "Corner", "Multiple"]),
            "setback_front": random.randint(10, 25),
            "setback_rear": random.randint(10, 20),
            "setback_side": random.randint(5, 15)
        }
    
    def generate_room_breakdown(self, total_area: int) -> Dict[str, Any]:
        """Generate detailed room breakdown"""
        num_rooms = random.randint(self.config.min_rooms, min(self.config.max_rooms, total_area // 200))
        
        # Standard room types and their area ranges
        room_types = {
            "bedroom": {"min": 120, "max": 300, "count": random.randint(1, 4)},
            "bathroom": {"min": 60, "max": 150, "count": random.randint(1, 3)},
            "kitchen": {"min": 100, "max": 250, "count": 1},
            "living_room": {"min": 150, "max": 400, "count": 1},
            "dining_room": {"min": 80, "max": 200, "count": random.randint(0, 1)},
            "study_room": {"min": 80, "max": 200, "count": random.randint(0, 2)},
            "puja_room": {"min": 40, "max": 100, "count": random.randint(0, 1)},
            "store_room": {"min": 50, "max": 150, "count": random.randint(0, 2)},
            "balcony": {"min": 30, "max": 100, "count": random.randint(0, 3)},
            "terrace": {"min": 100, "max": 300, "count": random.randint(0, 1)}
        }
        
        rooms = []
        allocated_area = 0
        
        for room_type, config in room_types.items():
            for i in range(config["count"]):
                if allocated_area >= total_area * 0.95:  # Leave 5% for circulation
                    break
                    
                room_area = random.randint(config["min"], min(config["max"], total_area - allocated_area))
                room_name = f"{room_type}_{i+1}" if config["count"] > 1 else room_type
                
                rooms.append({
                    "name": room_name,
                    "type": room_type,
                    "area_sqft": room_area,
                    "dimensions": f"{random.randint(8, 20)}x{room_area // random.randint(8, 20)}",
                    "floor": random.choice(["Ground", "First", "Second", "Third"]),
                    "furnishing": random.choice(["Unfurnished", "Semi_Furnished", "Fully_Furnished"])
                })
                
                allocated_area += room_area
        
        return {
            "total_rooms": len(rooms),
            "rooms": rooms,
            "total_area": allocated_area,
            "circulation_area": total_area - allocated_area
        }
    
    def generate_indian_specific_features(self) -> Dict[str, Any]:
        """Generate India-specific architectural features"""
        region = random.choice(self.config.regional_focus)
        climate_zone = random.choice(self.config.climate_zones)
        style = random.choice(self.config.indian_styles)
        
        # Regional variations
        regional_features = {
            "Mumbai": ["High_Rise", "Compact_Design", "Balcony_Garden", "Rainwater_Harvesting"],
            "Delhi": ["Traditional_Courtyard", "Modern_Facade", "Solar_Panels", "Air_Purification"],
            "Bangalore": ["Garden_Integration", "Natural_Ventilation", "Rainwater_Harvesting", "Solar_Water_Heating"],
            "Chennai": ["Coastal_Design", "High_Ceiling", "Cross_Ventilation", "Terrace_Garden"],
            "Kolkata": ["Colonial_Influence", "Balcony_Design", "Natural_Lighting", "Traditional_Materials"]
        }
        
        features = regional_features.get(region, ["Modern_Design", "Energy_Efficient", "Natural_Lighting"])
        
        return {
            "region": region,
            "climate_zone": climate_zone,
            "architectural_style": style,
            "regional_features": random.sample(features, random.randint(2, 4)),
            "vastu_compliant": random.choice([True, False]),
            "green_building": random.choice([True, False]),
            "solar_panels": random.choice([True, False]),
            "rainwater_harvesting": random.choice([True, False]),
            "sewage_treatment": random.choice([True, False])
        }
    
    def generate_building_codes(self) -> Dict[str, Any]:
        """Generate building code compliance details"""
        return {
            "nbc_2016_compliant": True,
            "fire_safety": random.choice(["Type_1", "Type_2", "Type_3", "Type_4"]),
            "earthquake_resistant": random.choice([True, False]),
            "accessibility_compliant": random.choice([True, False]),
            "energy_efficiency": random.choice(["1_Star", "2_Star", "3_Star", "4_Star", "5_Star"]),
            "water_efficiency": random.choice(["1_Star", "2_Star", "3_Star", "4_Star", "5_Star"]),
            "waste_management": random.choice(["Basic", "Advanced", "Zero_Waste"]),
            "parking_requirements": random.choice(["1_Car", "2_Cars", "3_Cars", "Visitor_Parking"])
        }
    
    def generate_materials_and_finishes(self, budget_range: Dict) -> Dict[str, Any]:
        """Generate materials and finishes based on budget"""
        budget_level = budget_range["label"]
        
        material_quality = {
            "Economy": "Standard",
            "Mid_Range": "Premium",
            "Premium": "Luxury",
            "Luxury": "Ultra_Luxury",
            "Ultra_Luxury": "Bespoke"
        }
        
        quality = material_quality.get(budget_level, "Standard")
        
        return {
            "foundation": random.choice(["RCC_Footing", "Pile_Foundation", "Raft_Foundation"]),
            "structure": random.choice(["RCC_Frame", "Steel_Frame", "Load_Bearing"]),
            "walls": random.choice(["Brick_Masonry", "Concrete_Blocks", "AAC_Blocks", "Stone_Masonry"]),
            "roofing": random.choice(["RCC_Slab", "Steel_Truss", "Wooden_Truss", "Precast_Slabs"]),
            "flooring": random.choice(["Vitrified_Tiles", "Marble", "Granite", "Wooden", "Carpet"]),
            "painting": random.choice(["Emulsion", "Textured", "Premium_Emulsion", "Wallpaper"]),
            "electrical": random.choice(["Standard", "Modular", "Smart_Home", "Premium_Modular"]),
            "plumbing": random.choice(["Standard", "Premium", "Luxury", "Smart_Plumbing"]),
            "quality_level": quality
        }
    
    def calculate_quality_score(self, sample: Dict[str, Any]) -> float:
        """Calculate quality score for the sample"""
        score = 0.0
        total_checks = 0
        
        # Area consistency check
        plot_area = sample["input"]["plot_details"]["area_sqft"]
        room_area = sample["output"]["room_breakdown"]["total_area"]
        if room_area <= plot_area * 0.95:  # Rooms should not exceed plot area
            score += 1.0
        total_checks += 1
        
        # Room count consistency
        num_rooms = sample["output"]["room_breakdown"]["total_rooms"]
        if self.config.min_rooms <= num_rooms <= self.config.max_rooms:
            score += 1.0
        total_checks += 1
        
        # Budget consistency
        budget = sample["input"]["budget_range"]
        estimated_cost = plot_area * random.randint(800, 2000)  # Rough estimate
        if estimated_cost <= budget["max"]:
            score += 1.0
        total_checks += 1
        
        # Code compliance
        if sample["output"]["building_codes"]["nbc_2016_compliant"]:
            score += 1.0
        total_checks += 1
        
        # Regional consistency
        region = sample["input"]["indian_features"]["region"]
        climate = sample["input"]["indian_features"]["climate_zone"]
        if self._check_regional_consistency(region, climate):
            score += 1.0
        total_checks += 1
        
        return score / total_checks
    
    def _check_regional_consistency(self, region: str, climate: str) -> bool:
        """Check if region and climate are consistent"""
        regional_climate_map = {
            "Mumbai": ["Tropical_Hot_Humid", "Tropical_Warm_Humid"],
            "Delhi": ["Composite", "Subtropical_Hot_Dry"],
            "Chennai": ["Tropical_Hot_Humid", "Tropical_Warm_Humid"],
            "Kolkata": ["Tropical_Hot_Humid", "Tropical_Warm_Humid"],
            "Bangalore": ["Composite", "Subtropical_Warm_Humid"]
        }
        
        expected_climates = regional_climate_map.get(region, [])
        return climate in expected_climates if expected_climates else True
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate a single high-quality sample"""
        # Generate input
        budget_range = random.choice(self.config.budget_ranges)
        plot_details = self.generate_plot_details()
        indian_features = self.generate_indian_specific_features()
        
        input_data = {
            "plot_details": plot_details,
            "budget_range": budget_range,
            "indian_features": indian_features,
            "building_type": random.choice(self.config.building_types),
            "floors": random.randint(1, 4),
            "family_size": random.randint(2, 8),
            "lifestyle": random.choice(["Minimalist", "Traditional", "Modern", "Luxury", "Eco_Friendly"])
        }
        
        # Generate output
        room_breakdown = self.generate_room_breakdown(plot_details["area_sqft"])
        building_codes = self.generate_building_codes()
        materials = self.generate_materials_and_finishes(budget_range)
        
        output_data = {
            "room_breakdown": room_breakdown,
            "building_codes": building_codes,
            "materials_and_finishes": materials,
            "construction_timeline": random.randint(8, 24),  # months
            "estimated_cost": plot_details["area_sqft"] * random.randint(800, 2000),
            "energy_efficiency_rating": random.choice(["A", "B", "C", "D"]),
            "maintenance_requirements": random.choice(["Low", "Medium", "High"]),
            "resale_value_potential": random.choice(["Good", "Very_Good", "Excellent"])
        }
        
        sample = {
            "input": input_data,
            "output": output_data,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "v6.0",
                "quality_gates_passed": True,
                "sample_id": f"HBV6-ENH-{self.generated_samples:06d}"
            }
        }
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(sample)
        sample["metadata"]["quality_score"] = quality_score
        
        return sample
    
    def generate_dataset(self, output_dir: str, num_samples: int):
        """Generate enhanced dataset with quality gates"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_dir = output_path / "train"
        val_dir = output_path / "validation"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Generating {num_samples} enhanced samples...")
        print(f"📊 Quality threshold: {self.config.quality_threshold}")
        
        generated = 0
        accepted = 0
        
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            while accepted < num_samples:
                sample = self.generate_sample()
                generated += 1
                
                # Quality gate
                if sample["metadata"]["quality_score"] >= self.config.quality_threshold:
                    # Determine train/val split
                    is_train = random.random() < 0.9  # 90% train, 10% val
                    target_dir = train_dir if is_train else val_dir
                    
                    # Save sample
                    filename = f"sample_{accepted:06d}.json"
                    with open(target_dir / filename, 'w') as f:
                        json.dump(sample, f, indent=2)
                    
                    accepted += 1
                    pbar.update(1)
                    
                    if accepted % 1000 == 0:
                        print(f"✅ Generated {accepted}/{num_samples} samples (Quality: {sample['metadata']['quality_score']:.3f})")
                
                # Progress update
                if generated % 100 == 0:
                    pbar.set_postfix({
                        "Generated": generated,
                        "Accepted": accepted,
                        "Acceptance_Rate": f"{accepted/generated*100:.1f}%"
                    })
        
        # Save dataset info
        dataset_info = {
            "total_samples": accepted,
            "train_samples": len(list(train_dir.glob("*.json"))),
            "validation_samples": len(list(val_dir.glob("*.json"))),
            "quality_threshold": self.config.quality_threshold,
            "generated_at": datetime.now().isoformat(),
            "version": "v6.0",
            "description": "Enhanced HouseBrain dataset with quality gates"
        }
        
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n🎉 Dataset generation complete!")
        print(f"📊 Total generated: {generated}")
        print(f"✅ Accepted: {accepted}")
        print(f"📈 Acceptance rate: {accepted/generated*100:.1f}%")
        print(f"📁 Saved to: {output_path}")

def main():
    """Main function"""
    config = EnhancedDatasetConfig()
    generator = EnhancedHouseBrainDatasetGenerator(config)
    
    # Generate additional samples to reach 1M
    additional_samples = config.target_samples - config.current_samples
    print(f"🎯 Target: {config.target_samples:,} samples")
    print(f"📊 Current: {config.current_samples:,} samples")
    print(f"➕ Additional needed: {additional_samples:,} samples")
    
    # Generate in batches
    batch_size = 100000  # 100K per batch
    for i in range(0, additional_samples, batch_size):
        current_batch = min(batch_size, additional_samples - i)
        output_dir = f"housebrain_dataset_v6_enhanced_batch_{i//batch_size + 1}"
        
        print(f"\n🏗️ Generating batch {i//batch_size + 1}: {current_batch:,} samples")
        generator.generate_dataset(output_dir, current_batch)

if __name__ == "__main__":
    main()
