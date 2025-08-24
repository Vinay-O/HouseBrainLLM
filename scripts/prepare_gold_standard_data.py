import json
from pathlib import Path
import logging
import sys

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def prepare_data():
    """
    Reads migrated Gold Standard JSON files (which are pure HouseOutput schema),
    extracts the original prompt from the 'input' block, and saves them into a 
    format ready for fine-tuning.
    """
    source_dir = Path("data/migrated_gold_standard")
    output_dir = Path("data/training/gold_standard_finetune_ready")

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        logger.error("Please run the `scripts/migrate_legacy_data.py` script first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean the output directory first
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()

    source_files = list(source_dir.glob("*.json"))
    if not source_files:
        logger.warning(f"No JSON files found in {source_dir}")
        return

    logger.info(f"Found {len(source_files)} Migrated Gold Standard files. Preparing for fine-tuning...")

    prepared_count = 0
    for source_file in source_files:
        try:
            with open(source_file, "r") as f:
                data = json.load(f)

            # Extract the original prompt and the full object as the output
            prompt = data.get("input", {}).get("plot", {}).get("prompt")
            
            if prompt:
                finetune_data = {
                    "prompt": prompt,
                    "output": json.dumps(data) # The entire object is the desired output
                }

                # Save to the new directory
                output_file = output_dir / source_file.name
                with open(output_file, "w") as f:
                    json.dump(finetune_data, f, indent=2)
                prepared_count += 1

            else:
                logger.warning(f"Skipping {source_file.name}: missing 'prompt' key in input.plot.")

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {source_file.name}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with {source_file.name}: {e}")

    logger.info(f"✅ Successfully prepared {prepared_count} files in {output_dir}")

if __name__ == "__main__":
    prepare_data()
