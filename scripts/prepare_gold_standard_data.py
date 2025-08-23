import json
from pathlib import Path
import logging
import sys

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def prepare_data():
    """
    Reads Gold Standard JSON files, extracts the 'prompt' and 'output' fields,
    and saves them into a format ready for fine-tuning.
    """
    source_dir = Path("data/training/gold_standard")
    output_dir = Path("data/training/gold_standard_finetune_ready")

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = list(source_dir.glob("*.json"))
    if not source_files:
        logger.warning(f"No JSON files found in {source_dir}")
        return

    logger.info(f"Found {len(source_files)} Gold Standard files. Preparing for fine-tuning...")

    for source_file in source_files:
        try:
            with open(source_file, "r") as f:
                data = json.load(f)

            # Extract the required fields
            if "prompt" in data and "output" in data:
                finetune_data = {
                    "prompt": data["prompt"],
                    "output": data["output"]
                }

                # Save to the new directory
                output_file = output_dir / source_file.name
                with open(output_file, "w") as f:
                    json.dump(finetune_data, f, indent=2)

            else:
                logger.warning(f"Skipping {source_file.name}: missing 'prompt' or 'output' key.")

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {source_file.name}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with {source_file.name}: {e}")

    logger.info(f"✅ Successfully prepared {len(source_files)} files in {output_dir}")

if __name__ == "__main__":
    prepare_data()
