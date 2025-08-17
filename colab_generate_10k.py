#!/usr/bin/env python3
"""
HouseBrain 10K Dataset Generator for Colab
Generates 10K super-quality samples directly in Colab

Usage: python colab_generate_10k.py
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
import time

# Import the generator from the main script
from generate_1m_super_quality import SuperQualityGenerator, SuperQualityConfig

def generate_10k_dataset():
    """Generate 10K dataset for Colab training"""
    print("🏠 Generating 10K HouseBrain Dataset for Colab...")
    
    # Configuration for 10K generation
    config = SuperQualityConfig()
    config.TOTAL_SAMPLES = 10000
    config.OUTPUT_DIR = "housebrain_dataset_r1_super_10k_aug"
    config.TRAIN_RATIO = 0.9  # 9K train, 1K validation
    
    # Create generator
    generator = SuperQualityGenerator(config)
    
    # Generate dataset
    print(f"📊 Generating {config.TOTAL_SAMPLES} samples...")
    generator.generate_dataset()
    
    # Augment the dataset
    print("🔧 Augmenting dataset with geometric metadata...")
    os.system(f"python augment_dataset_v1_1.py --input {config.OUTPUT_DIR} --output {config.OUTPUT_DIR} --workers 4")
    
    print("✅ 10K dataset generated successfully!")
    print(f"📁 Output directory: {config.OUTPUT_DIR}")
    
    # Show dataset info
    dataset_info_file = Path(config.OUTPUT_DIR) / "dataset_info.json"
    if dataset_info_file.exists():
        with open(dataset_info_file, 'r') as f:
            info = json.load(f)
        print(f"📊 Dataset info: {info}")

if __name__ == "__main__":
    generate_10k_dataset()
