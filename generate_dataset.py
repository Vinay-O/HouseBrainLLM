#!/usr/bin/env python3
"""
HouseBrain Dataset Generator
Generates synthetic architectural datasets for training the HouseBrain LLM.
"""

import json
import os
import random
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass
import zipfile
from pathlib import Path

from src.housebrain.schema import HouseInput, HouseOutput, RoomType, ArchitecturalStyle
from src.housebrain.layout import solve_house_layout


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    samples: int = 1000
    train_ratio: float = 0.8
    output_dir: str = "housebrain_dataset"
    zip_output: bool = False
    fast_mode: bool = False
    styles: List[str] = None
    regions: List[str] = None

    def __post_init__(self):
        if self.styles is None:
            self.styles = ["modern", "traditional", "contemporary", "minimalist", "colonial"]
        if self.regions is None:
            self.regions = ["north", "south", "east", "west", "central"]


def generate_basic_details() -> Dict[str, Any]:
    """Generate basic house input details."""
    # Plot dimensions (in feet)
    plot_length = random.randint(30, 100)
    plot_width = random.randint(25, 80)
    
    # Ensure reasonable aspect ratio
    aspect_ratio = plot_length / plot_width
    if aspect_ratio > 3:
        plot_length = plot_width * 2.5
    elif aspect_ratio < 0.5:
        plot_width = plot_length * 0.6
    
    # Setbacks (in feet)
    front_setback = random.randint(10, 20)
    rear_setback = random.randint(10, 20)
    side_setback = random.randint(5, 15)
    
    # House specifications
    bedrooms = random.randint(1, 5)
    bathrooms = max(1, bedrooms // 2 + random.randint(0, 1))
    floors = random.randint(1, 3)
    
    # Budget (in INR)
    base_cost_per_sqft = random.randint(1500, 3000)
    total_area = plot_length * plot_width * 0.6  # Assuming 60% built-up area
    budget = int(total_area * base_cost_per_sqft * random.uniform(0.8, 1.2))
    
    return {
        "plot": {
            "length": plot_length,
            "width": plot_width,
            "unit": "ft"
        },
        "setbacks_ft": {
            "front": front_setback,
            "rear": rear_setback,
            "left": side_setback,
            "right": side_setback
        },
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "budget_estimate": budget,
        "style": random.choice(["modern", "traditional", "contemporary", "minimalist", "colonial"]),
        "region": random.choice(["north", "south", "east", "west", "central"])
    }


def generate_sample(config: DatasetConfig) -> Dict[str, Any]:
    """Generate a single training sample."""
    # Generate input
    input_data = generate_basic_details()
    
    # Create HouseInput object
    house_input = HouseInput(
        plot=input_data["plot"],
        setbacks_ft=input_data["setbacks_ft"],
        bedrooms=input_data["bedrooms"],
        bathrooms=input_data["bathrooms"],
        floors=input_data["floors"],
        budget_inr=input_data["budget_estimate"],
        style=input_data["style"],
        region=input_data["region"]
    )
    
    # Generate output using layout solver
    if config.fast_mode:
        # Fast mode: generate basic output without layout solving
        house_output = _create_basic_output(house_input)
    else:
        try:
            house_output = solve_house_layout(house_input)
        except Exception as e:
            print(f"Layout solving failed, using basic output: {e}")
            house_output = _create_basic_output(house_input)
    
    return {
        "input": house_input.model_dump(),
        "output": house_output.model_dump()
    }


def _create_basic_output(house_input: HouseInput) -> HouseOutput:
    """Create basic output without layout solving."""
    from src.housebrain.schema import Level, Room, Point2D, Rectangle, RoomType
    
    # Create basic rooms
    rooms = []
    room_types = [RoomType.BEDROOM, RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BATHROOM]
    
    for i in range(house_input.bedrooms):
        rooms.append(Room(
            type=RoomType.BEDROOM,
            bounds=Rectangle(
                x=10 + i * 15,
                y=10,
                width=12,
                height=15
            )
        ))
    
    # Add other essential rooms
    rooms.append(Room(
        type=RoomType.LIVING_ROOM,
        bounds=Rectangle(x=10, y=30, width=20, height=15)
    ))
    
    rooms.append(Room(
        type=RoomType.KITCHEN,
        bounds=Rectangle(x=35, y=30, width=12, height=10)
    ))
    
    rooms.append(Room(
        type=RoomType.BATHROOM,
        bounds=Rectangle(x=35, y=45, width=8, height=8)
    ))
    
    # Create level
    level = Level(
        level_number=0,
        rooms=rooms,
        height_ft=10
    )
    
    # Create basic output
    return HouseOutput(
        levels=[level],
        total_cost_estimate=house_input.budget_inr,
        construction_sequence=["Foundation", "Structure", "Roofing", "Finishing"],
        timeline_weeks=random.randint(20, 40),
        optimization_notes=["Standard construction", "Cost-effective materials"]
    )


def _create_enhanced_output(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Add enhanced features to the output."""
    input_data = sample["input"]
    output_data = sample["output"]
    
    # Add exterior specifications
    output_data["exterior_specifications"] = {
        "roof_type": random.choice(["flat", "sloped", "gable", "hip"]),
        "exterior_finish": random.choice(["brick", "stone", "stucco", "siding"]),
        "window_type": random.choice(["casement", "sliding", "double-hung", "picture"]),
        "door_type": random.choice(["wooden", "steel", "fiberglass", "aluminum"])
    }
    
    # Add climate and site considerations
    output_data["climate_and_site"] = {
        "orientation": random.choice(["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]),
        "sunlight_exposure": random.choice(["full", "partial", "minimal"]),
        "wind_direction": random.choice(["north", "south", "east", "west"]),
        "soil_type": random.choice(["clay", "sandy", "loamy", "rocky"])
    }
    
    # Add building codes
    output_data["building_codes"] = {
        "far_ratio": round(random.uniform(0.5, 2.0), 2),
        "height_restriction": random.randint(30, 100),
        "parking_required": random.randint(1, 4),
        "fire_safety": random.choice(["basic", "enhanced", "commercial"]),
        "accessibility": random.choice(["basic", "ada_compliant", "universal_design"])
    }
    
    return sample


def generate_dataset(config: DatasetConfig) -> None:
    """Generate the complete dataset."""
    print(f"Generating {config.samples} samples...")
    
    # Create output directories
    train_dir = os.path.join(config.output_dir, "train")
    val_dir = os.path.join(config.output_dir, "validation")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Calculate split
    train_samples = int(config.samples * config.train_ratio)
    val_samples = config.samples - train_samples
    
    # Generate samples
    all_samples = []
    for i in range(config.samples):
        if i % 100 == 0:
            print(f"Generated {i}/{config.samples} samples...")
        
        sample = generate_sample(config)
        sample = _create_enhanced_output(sample)
        all_samples.append(sample)
    
    # Split and save
    train_samples_list = all_samples[:train_samples]
    val_samples_list = all_samples[train_samples:]
    
    # Save training data
    print(f"Saving {len(train_samples_list)} training samples...")
    for i, sample in enumerate(train_samples_list):
        filename = f"sample_{i:07d}.json"
        filepath = os.path.join(train_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
    
    # Save validation data
    print(f"Saving {len(val_samples_list)} validation samples...")
    for i, sample in enumerate(val_samples_list):
        filename = f"sample_{i:07d}.json"
        filepath = os.path.join(val_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
    
    # Create zip if requested
    if config.zip_output:
        print("Creating zip archive...")
        zip_path = f"{config.output_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(config.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, config.output_dir)
                    zipf.write(file_path, arcname)
        print(f"Dataset saved as {zip_path}")
    
    print(f"Dataset generation complete!")
    print(f"Training samples: {len(train_samples_list)}")
    print(f"Validation samples: {len(val_samples_list)}")
    print(f"Output directory: {config.output_dir}")
    
    if not config.zip_output:
        print(f"\nNext steps:")
        print(f"1. Upload {config.output_dir} to Colab/Kaggle")
        print(f"2. Use the training scripts to fine-tune the model")
        print(f"3. Monitor training progress")


def main():
    parser = argparse.ArgumentParser(description="Generate HouseBrain training dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training/validation split ratio")
    parser.add_argument("--output", type=str, default="housebrain_dataset", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Create zip archive")
    parser.add_argument("--fast", action="store_true", help="Fast mode (skip layout solving)")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        samples=args.samples,
        train_ratio=args.train_ratio,
        output_dir=args.output,
        zip_output=args.zip,
        fast_mode=args.fast
    )
    
    generate_dataset(config)


if __name__ == "__main__":
    main()
