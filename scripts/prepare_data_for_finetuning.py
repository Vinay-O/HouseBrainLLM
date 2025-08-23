#!/usr/bin/env python3
"""
Prepares the human-readable Gold Standard dataset for the fine-tuning script
by converting the nested 'output' JSON object into an escaped JSON string.
"""
import json
from pathlib import Path
import argparse
from tqdm import tqdm

def process_file(input_path: Path, output_path: Path):
    """Reads a source file, converts the output object to a string, and saves it."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure the 'output' key exists and is a dictionary
    if "output" not in data or not isinstance(data["output"], dict):
        print(f"Skipping {input_path.name}: 'output' key is missing or not a JSON object.")
        return

    # Convert the nested JSON object for 'output' into a string
    data["output"] = json.dumps(data["output"], separators=(",", ":"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    """Main function to process all files in the directory."""
    parser = argparse.ArgumentParser(description="Prepare Gold Standard data for fine-tuning.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/training/gold_standard",
        help="Directory containing the source Gold Standard files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training/gold_standard_finetune_ready",
        help="Directory to save the formatted, training-ready files."
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_files = sorted(list(input_path.glob("*.json")))

    if not source_files:
        print(f"No source files found in {input_path}. Nothing to do.")
        return

    print(f"Processing {len(source_files)} files from {input_path}...")

    for file_path in tqdm(source_files, desc="Preparing Data"):
        output_file_path = output_path / file_path.name
        process_file(file_path, output_file_path)

    print(f"\n✅ Successfully prepared {len(source_files)} files for fine-tuning.")
    print(f"Training-ready data is located in: {output_path.resolve()}")


if __name__ == "__main__":
    main()
