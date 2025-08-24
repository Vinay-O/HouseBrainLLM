import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.housebrain.schema import HouseOutput
# V2 Imports
from src.housebrain.rendering.core_2d import render_2d_plan
# from src.housebrain.rendering.core_3d import generate_3d_model

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(input_path: Path, output_dir: Path, formats: List[str]):
    """
    Main pipeline function to load, validate, and render a house plan.
    """
    logger.info(f"Starting pipeline for {input_path.name}")
    
    # --- 1. Load and Validate Input JSON ---
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        logger.info("Successfully loaded JSON file.")
        
        # Handle both direct HouseOutput and training data format {"output": "{...}"}
        if 'output' in data and isinstance(data['output'], str):
            logger.info("Detected training data format. Parsing 'output' field.")
            house_data = json.loads(data['output'])
        elif 'prompt' in data and 'output' in data:
             logger.info("Detected training data format with object. Parsing 'output' field.")
             house_data = data['output']
        else:
            logger.info("Assuming direct HouseOutput format.")
            house_data = data

        # Validate the data against the Pydantic schema
        house_plan = HouseOutput.model_validate(house_data)
        logger.info("✅ JSON validation successful. The input is compliant with the HouseOutput schema.")

    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON in file: {input_path}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)
        return

    # --- 2. Create Output Directory ---
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir.resolve()}")

    # --- 3. Run Selected Renderers ---
    base_filename = input_path.stem

    if "2d" in formats:
        logger.info("Starting 2D rendering...")
        render_2d_plan(house_plan, output_dir, base_filename)
        # logger.info("... (2D rendering placeholder) ...")


    if "3d" in formats:
        logger.info("Starting 3D model generation...")
        # generate_3d_model(house_plan, output_dir, base_filename) # Placeholder
        logger.info("... (3D model generation placeholder) ...")

    logger.info(f"✅ Pipeline finished for {input_path.name}")


def main():
    """
    Command-line interface for the HouseBrain rendering pipeline.
    """
    parser = argparse.ArgumentParser(description="HouseBrain Professional Rendering Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        # Make input optional for easy testing
        default="data/gold_standard/gold_standard_22_curated_llama_3.json",
        help="Path to the input HouseBrain JSON file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the output files. Defaults to './output'."
    )
    parser.add_argument(
        "--formats",
        nargs='+',
        choices=['2d', '3d'],
        default=['2d', '3d'],
        help="List of output formats to generate. Can be '2d', '3d', or both. Defaults to both."
    )
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Error: Input file not found at {input_path}")
        return
        # logger.error("Attempting to run migration script...")
        # try:
        #     from scripts.migrate_legacy_data import main as migrate_main
        #     logger.info("Running migration...")
        #     migrate_main()
        #     if not input_path.exists():
        #          logger.error("Migration complete, but test file still not found. Please check paths.")
        #          return
        #     logger.info("Migration successful, retrying pipeline.")
        # except Exception as e:
        #     logger.error(f"Failed to run migration script: {e}")
        #     return

    run_pipeline(input_path, output_dir, args.formats)

if __name__ == "__main__":
    # This allows running the script directly for a quick test
    # without needing to pass command-line arguments.
    main()
