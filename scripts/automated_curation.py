import argparse
import logging
import subprocess
import sys
from pathlib import Path
import shutil

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_command(command, error_message):
    """Runs a command in a subprocess and handles errors."""
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        logger.info(f"Successfully ran: {' '.join(command)}")
        if process.stdout:
            logger.info(f"STDOUT:\n{process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_message}")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running {' '.join(command)}: {e}")
        return False

def automated_curation_pipeline(
    prompt: str,
    output_dir: Path,
    max_retries: int,
    model: str,
    repair_model: str
):
    """
    Manages the full Generate -> Analyze -> Repair loop.
    """
    logger.info("--- Starting Automated Curation Pipeline ---")

    # --- Setup working directory and paths ---
    run_id = Path(prompt).stem if Path(prompt).exists() else "custom_prompt"
    work_dir = output_dir / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_file = work_dir / "prompt.txt"
    if Path(prompt).exists():
        shutil.copy(prompt, prompt_file)
    else:
        with open(prompt_file, 'w') as f:
            f.write(prompt)

    draft_file = work_dir / "draft_0.json"
    error_file = work_dir / "errors_0.json"

    # --- 1. Initial Generation ---
    logger.info("Step 1: Generating initial draft...")
    gen_success = run_command(
        [
            sys.executable, "scripts/generate_draft_from_prompt.py",
            "--prompt-file", str(prompt_file),
            "--output-file", str(draft_file),
            "--model", model
        ],
        "Initial draft generation failed."
    )
    if not gen_success or not draft_file.exists():
        logger.error("Initial draft generation failed to produce an output file. Pipeline cannot continue. Exiting.")
        return

    # --- 2. Analysis and Repair Loop ---
    for i in range(max_retries + 1):
        logger.info(f"--- Iteration {i}: Analyzing draft '{draft_file.name}' ---")
        
        # Analyze the current draft
        validation_success = run_command(
            [
                sys.executable, "run_pipeline.py",
                "--input", str(draft_file),
                "--validate-only",
                "--error-file", str(error_file)
            ],
            f"Validation failed for {draft_file.name}."
        )

        if validation_success:
            logger.info(f"✅ Curation successful after {i} repair attempts!")
            final_path = output_dir / "gold_standard" / f"{run_id}_curated.json"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(draft_file, final_path)
            logger.info(f"Final validated plan saved to: {final_path}")
            # shutil.rmtree(work_dir) # Optional: clean up working directory
            logger.info("--- Automated Curation Pipeline Finished Successfully ---")
            return

        # If validation fails and we have retries left
        if i < max_retries:
            logger.info(f"Step 2.{i+1}: Attempting repair...")
            repaired_draft_file = work_dir / f"draft_{i+1}.json"
            error_file_for_repair = work_dir / f"errors_{i}.json"
            
            repair_success = run_command(
                [
                    sys.executable, "scripts/repair_draft.py",
                    "--draft-file", str(draft_file),
                    "--error-file", str(error_file_for_repair),
                    "--prompt-file", str(prompt_file),
                    "--output-file", str(repaired_draft_file),
                    "--model", repair_model
                ],
                f"Repair attempt {i+1} failed."
            )

            if not repair_success:
                logger.error("Repair script failed to run. Cannot continue this curation. Exiting.")
                return
            
            # Setup for the next iteration
            draft_file = repaired_draft_file
            error_file = work_dir / f"errors_{i+1}.json"
        else:
            logger.error(f"❌ Curation failed after {max_retries} repair attempts.")
            logger.error("The model could not produce a valid plan. Please review the artifacts in:")
            logger.error(f"{work_dir.resolve()}")
            logger.error("--- Automated Curation Pipeline Finished with Failure ---")

def main():
    parser = argparse.ArgumentParser(description="HouseBrain Automated Data Curation Pipeline")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="A string containing the design prompt, or a path to a .txt file with the prompt."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/automated_curation",
        help="Directory to save the final curated files and working artifacts."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of repair attempts before giving up."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Name of the Ollama model to use for initial generation."
    )
    parser.add_argument(
        "--repair-model",
        type=str,
        default=None,
        help="Optional: a more powerful Ollama model for the repair step (e.g., llama3:70b). Defaults to the generation model."
    )
    
    args = parser.parse_args()

    repair_model = args.repair_model if args.repair_model else args.model

    automated_curation_pipeline(
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        max_retries=args.max_retries,
        model=args.model,
        repair_model=repair_model
    )

if __name__ == "__main__":
    main()
