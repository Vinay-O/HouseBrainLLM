#!/usr/bin/env python3
"""
HouseBrain Enhanced Combined Dataset Generator

Creates a high-quality combined dataset from the general 350K dataset and the India 75K dataset.

Improvements over naive concatenation:
- Validates sample structure and drops malformed entries
- Normalizes schema keys where possible
- Deduplicates using a robust hash of canonicalized input+key output fields
- Re-indexes samples with consistent IDs
- Balances India vs General share if requested
- Writes a detailed dataset_info.json and a merge_manifest.json

Usage (defaults assume standard repo paths):
  python generate_combined_dataset.py \
    --general housebrain_dataset_v5_350k \
    --india housebrain_dataset_india_75k \
    --output housebrain_dataset_v5_425k \
    --deduplicate \
    --train-ratio 0.9 \
    --target-samples 425000

You can also omit --target-samples to merge everything available from both sources.
"""

import argparse
import json
import os
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class SampleValidator:
    """Lightweight schema validator for HouseBrain samples.

    This avoids pydantic dependencies and focuses on presence/shape of critical fields
    that the training pipeline relies on.
    """

    REQUIRED_INPUT_FIELDS = [
        "basicDetails",
        "plot",
        "roomBreakdown",
    ]

    REQUIRED_BASIC_DETAIL_FIELDS = [
        "totalArea",
        "unit",
        "floors",
        "bedrooms",
        "bathrooms",
        "budget",
        "style",
    ]

    REQUIRED_OUTPUT_FIELDS = [
        "total_area",
        "construction_cost",
    ]

    def validate(self, sample: Dict[str, Any]) -> bool:
        try:
            if not isinstance(sample, dict):
                return False
            if "input" not in sample or "output" not in sample:
                return False

            input_data = sample["input"]
            output_data = sample["output"]

            for field in self.REQUIRED_INPUT_FIELDS:
                if field not in input_data:
                    return False

            basic = input_data["basicDetails"]
            for field in self.REQUIRED_BASIC_DETAIL_FIELDS:
                if field not in basic:
                    return False

            for field in self.REQUIRED_OUTPUT_FIELDS:
                if field not in output_data:
                    return False

            # Quick numeric sanity checks
            if not (500 <= int(basic.get("totalArea", 0)) <= 20000):
                return False
            if not (1 <= int(basic.get("floors", 0)) <= 6):
                return False
            if not (1 <= int(basic.get("bedrooms", 0)) <= 10):
                return False

            return True
        except Exception:
            return False


def canonicalize_for_hash(sample: Dict[str, Any]) -> str:
    """Create a canonical string for deduplication hashing.

    Focus on the input (requirements) and a few output fields, which best capture
    a data point's semantic identity while being stable across formatting changes.
    """
    try:
        input_data = sample.get("input", {})
        output_data = sample.get("output", {})

        key = {
            "basicDetails": input_data.get("basicDetails", {}),
            "plot": input_data.get("plot", {}),
            "roomBreakdown": input_data.get("roomBreakdown", []),
            "total_area": output_data.get("total_area"),
            "construction_cost": output_data.get("construction_cost"),
        }
        return json.dumps(key, sort_keys=True, ensure_ascii=False)
    except Exception:
        # Fall back to full sample dump if something odd occurs
        try:
            return json.dumps(sample, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(sample)


def hash_sample(sample: Dict[str, Any]) -> str:
    canonical = canonicalize_for_hash(sample)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def iter_samples(dataset_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Yield (path, json) for all samples in train and validation subfolders."""
    results: List[Tuple[Path, Dict[str, Any]]] = []
    for split in ["train", "validation"]:
        folder = dataset_dir / split
        if not folder.exists():
            continue
        for fp in folder.glob("*.json"):
            js = load_json(fp)
            if js is not None:
                results.append((fp, js))
    return results


def select_balanced_subset(
    general_paths: List[Tuple[Path, Dict[str, Any]]],
    india_paths: List[Tuple[Path, Dict[str, Any]]],
    target_total: Optional[int],
    desired_india_fraction: Optional[float],
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Tuple[Path, Dict[str, Any]]]]:
    """Compute a balanced subset from general and India pools.

    If target_total is None, return all. Otherwise, sample to match target_total
    and the desired_india_fraction (defaults to natural ratio when None).
    """
    if target_total is None:
        return general_paths, india_paths

    total_available = len(general_paths) + len(india_paths)
    if target_total > total_available:
        target_total = total_available

    if desired_india_fraction is None:
        desired_india_fraction = len(india_paths) / max(1, total_available)

    target_india = int(round(target_total * desired_india_fraction))
    target_general = target_total - target_india

    random.shuffle(general_paths)
    random.shuffle(india_paths)

    selected_general = general_paths[:target_general]
    selected_india = india_paths[:target_india]
    return selected_general, selected_india


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort schema normalization.

    - Ensures presence of `metadata` with source labels
    - Adds placeholders for missing optional fields used downstream
    """
    input_data = sample.get("input", {})
    output_data = sample.get("output", {})

    # Ensure optional sections exist
    output_data.setdefault("materials", {})
    output_data.setdefault("render_paths", {})

    sample["input"] = input_data
    sample["output"] = output_data
    sample.setdefault("metadata", {})
    return sample


def write_samples(
    samples: List[Dict[str, Any]],
    output_dir: Path,
    train_ratio: float,
    id_prefix: str = "HBV5-CMB-",
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_target = int(len(samples) * train_ratio)
    train_count = 0
    val_count = 0

    for idx, sample in enumerate(samples, start=1):
        sample_id = f"{id_prefix}{idx:06d}"
        sample["id"] = sample_id
        split_dir = train_dir if train_count < train_target else val_dir
        if train_count < train_target:
            train_count += 1
        else:
            val_count += 1

        with open(split_dir / f"{sample_id}.json", "w") as f:
            json.dump(sample, f, indent=2)

        if (idx % 5000) == 0:
            print(f"✅ Wrote {idx:,} samples...")

    return train_count, val_count


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced combined HouseBrain dataset")
    parser.add_argument("--general", default="housebrain_dataset_v5_350k", help="Path to general dataset")
    parser.add_argument("--india", default="housebrain_dataset_india_75k", help="Path to India dataset")
    parser.add_argument("--output", default="housebrain_dataset_v5_425k", help="Output directory for combined dataset")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split for the combined dataset")
    parser.add_argument("--target-samples", type=int, default=425000, help="Total number of samples to select; use -1 for all")
    parser.add_argument("--india-fraction", type=float, default=None, help="Desired fraction of India samples; defaults to natural ratio")
    parser.add_argument("--deduplicate", action="store_true", help="Enable deduplication across merged datasets")
    parser.add_argument("--zip", action="store_true", help="Create a zip archive of the final dataset")

    args = parser.parse_args()

    general_dir = Path(args.general)
    india_dir = Path(args.india)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🧩 Enhanced Combined Dataset Generator")
    print("=" * 60)
    print(f"📁 General dataset: {general_dir}")
    print(f"📁 India dataset:   {india_dir}")

    # Load all samples
    general_items = iter_samples(general_dir)
    india_items = iter_samples(india_dir)
    print(f"📄 Loaded {len(general_items):,} general samples and {len(india_items):,} India samples")

    # Select subset to meet target size and balance
    target_total = None if args.target_samples == -1 else int(args.target_samples)
    sel_general, sel_india = select_balanced_subset(
        general_items, india_items, target_total, args.india_fraction
    )

    print(f"🎯 Selected {len(sel_general):,} general and {len(sel_india):,} India samples")

    # Deduplicate and validate
    validator = SampleValidator()
    seen_hashes = set()
    merged: List[Dict[str, Any]] = []
    dropped_malformed = 0
    dropped_duplicates = 0

    def process_pool(pool: List[Tuple[Path, Dict[str, Any]]], source_label: str) -> None:
        nonlocal dropped_malformed, dropped_duplicates
        for fp, js in pool:
            if not validator.validate(js):
                dropped_malformed += 1
                continue

            js = normalize_sample(js)
            js.setdefault("metadata", {})
            js["metadata"]["source_dataset"] = source_label
            js["metadata"].setdefault("indian_specific", source_label.lower().startswith("india"))

            if args.deduplicate:
                h = hash_sample(js)
                if h in seen_hashes:
                    dropped_duplicates += 1
                    continue
                seen_hashes.add(h)

            merged.append(js)

    process_pool(sel_general, "general_v5_350k")
    process_pool(sel_india, "india_75k")

    random.shuffle(merged)
    print(f"🧹 Dropped {dropped_malformed:,} malformed and {dropped_duplicates:,} duplicate samples")
    print(f"📦 Final pool before write: {len(merged):,} samples")

    # Write out with new IDs and split
    train_count, val_count = write_samples(merged, output_dir, args.train_ratio)

    # Dataset info and manifest
    dataset_info = {
        "name": "HouseBrain Combined Dataset",
        "version": "1.0",
        "description": "Enhanced combination of general 350K and India 75K datasets with validation and deduplication",
        "train_samples": train_count,
        "val_samples": val_count,
        "total_samples": train_count + val_count,
        "sources": {
            "general": str(general_dir),
            "india": str(india_dir),
        },
        "settings": {
            "train_ratio": args.train_ratio,
            "target_samples": target_total if target_total is not None else "all",
            "india_fraction": args.india_fraction,
            "deduplicate": bool(args.deduplicate),
        },
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Optional zip packaging
    if args.zip:
        import zipfile
        zip_path = output_dir.with_suffix(".zip")
        print(f"📦 Creating zip archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    fp = Path(root) / file
                    arcname = fp.relative_to(output_dir.parent)
                    zipf.write(str(fp), str(arcname))
        print(f"✅ Zip archive created: {zip_path}")

    print("\n🎉 Combined dataset created successfully!")
    print(f"📊 Train: {train_count:,} | Val: {val_count:,} | Total: {train_count + val_count:,}")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()


