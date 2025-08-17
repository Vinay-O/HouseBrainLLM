#!/usr/bin/env python3
"""
HouseBrain Dataset Generator for Colab
Generates datasets of various sizes directly in Colab

Usage: 
  python colab_generate_10k.py                    # 10K samples
  python colab_generate_10k.py --samples 100000   # 100K samples
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Use standard generator
from generate_dataset import generate_dataset, DatasetConfig

def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description="Generate HouseBrain dataset for Colab")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="housebrain_dataset_colab", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (skip layout solving)")
    
    args = parser.parse_args()
    
    print(f"🏠 Generating {args.samples} HouseBrain samples for Colab...")
    
    # Create configuration
    config = DatasetConfig(
        samples=args.samples,
        train_ratio=0.9,  # 90% train, 10% validation
        output_dir=args.output,
        zip_output=False,  # Don't zip in Colab
        fast_mode=args.fast
    )
    
    # Generate dataset
    generate_dataset(config)
    
    print(f"✅ {args.samples} dataset generated successfully!")
    print(f"📁 Output directory: {args.output}")
    print(f"🎯 Ready for training!")

if __name__ == "__main__":
    main()
