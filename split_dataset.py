#!/usr/bin/env python3
"""
HouseBrain Dataset Splitter
Splits large datasets into smaller parts for parallel training.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List


def split_dataset(source_dir: str, output_dir: str, splits: int) -> None:
    """Split a dataset into multiple parts."""
    print(f"Splitting dataset from {source_dir} into {splits} parts...")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} not found")
    
    # Get all files in train and validation directories
    train_source = os.path.join(source_dir, "train")
    val_source = os.path.join(source_dir, "validation")
    
    if not os.path.exists(train_source):
        raise FileNotFoundError(f"Training directory {train_source} not found")
    
    train_files = []
    val_files = []
    
    if os.path.exists(train_source):
        train_files = [f for f in os.listdir(train_source) if f.endswith('.json')]
        train_files.sort()
    
    if os.path.exists(val_source):
        val_files = [f for f in os.listdir(val_source) if f.endswith('.json')]
        val_files.sort()
    
    print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
    
    # Calculate files per split
    train_files_per_split = len(train_files) // splits
    val_files_per_split = len(val_files) // splits
    
    # Create output directories
    for i in range(splits):
        split_dir = os.path.join(output_dir, f"split_{i+1}")
        split_train_dir = os.path.join(split_dir, "train")
        split_val_dir = os.path.join(split_dir, "validation")
        
        os.makedirs(split_train_dir, exist_ok=True)
        os.makedirs(split_val_dir, exist_ok=True)
        
        # Copy training files
        start_idx = i * train_files_per_split
        end_idx = start_idx + train_files_per_split if i < splits - 1 else len(train_files)
        
        for j in range(start_idx, end_idx):
            if j < len(train_files):
                src_file = os.path.join(train_source, train_files[j])
                dst_file = os.path.join(split_train_dir, train_files[j])
                shutil.copy2(src_file, dst_file)
        
        # Copy validation files
        start_idx = i * val_files_per_split
        end_idx = start_idx + val_files_per_split if i < splits - 1 else len(val_files)
        
        for j in range(start_idx, end_idx):
            if j < len(val_files):
                src_file = os.path.join(val_source, val_files[j])
                dst_file = os.path.join(split_val_dir, val_files[j])
                shutil.copy2(src_file, dst_file)
        
        # Count files in this split
        train_count = len([f for f in os.listdir(split_train_dir) if f.endswith('.json')])
        val_count = len([f for f in os.listdir(split_val_dir) if f.endswith('.json')])
        
        print(f"Split {i+1}: {train_count} training, {val_count} validation files")
    
    print(f"Dataset splitting complete! Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split HouseBrain dataset for parallel training")
    parser.add_argument("--source", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--splits", type=int, default=6, help="Number of splits to create")
    
    args = parser.parse_args()
    
    split_dataset(args.source, args.output, args.splits)


if __name__ == "__main__":
    main()
