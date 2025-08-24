import argparse
import logging
import subprocess
import sys
from pathlib import Path
import shutil

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_command(command, error_message, timeout=300):
    """Runs a command in a subprocess and handles errors."""
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        logger.info(f"Successfully ran: {' '.join(command)}")
        return True
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_message}")
        logger.error(f"STDERR:\n{e.stderr}")
        return False
    return False

def run_stage_with_retry(stage_function, max_retries, *args):
    """Runs a stage function, retrying on failure."""
    for i in range(max_retries):
        success = stage_function(*args)
        if success:
            return True
        logger.warning(f"Stage failed. Retrying ({i+1}/{max_retries})...")
    return False

def stage_1_layout(prompt_file, stage_1_output, model):
    return run_command(
        [sys.executable, "scripts/stage_1_generate_layout.py", "--prompt-file", str(prompt_file), "--output-file", str(stage_1_output), "--model", model],
        "Stage 1 (Layout Generation) failed."
    )

def stage_2_openings(stage_1_output, prompt_file, stage_2_output, model):
    return run_command(
        [sys.executable, "scripts/stage_2_add_openings.py", "--layout-file", str(stage_1_output), "--prompt-file", str(prompt_file), "--output-file", str(stage_2_output), "--model", model],
        "Stage 2 (Add Openings) failed."
    )

def stage_3_finalize(stage_2_output, prompt_file, final_output):
    return run_command(
        [sys.executable, "scripts/stage_3_finalize_plan.py", "--layout-file", str(stage_2_output), "--prompt-file", str(prompt_file), "--output-file", str(final_output)],
        "Stage 3 (Finalize Plan) failed."
    )

def validate_output(file_to_validate, error_file):
    return run_command(
        [sys.executable, "run_pipeline.py", "--input", str(file_to_validate), "--validate-only", "--error-file", str(error_file)],
        f"Validation failed for {file_to_validate.name}."
    )

def assembly_line_pipeline(prompt: str, output_dir: Path, model: str, run_name: str | None = None, max_retries: int = 3):
    logger.info("--- Starting Architect's Assembly Line ---")
    
    # Setup directories
    run_id = run_name if run_name else Path(prompt).stem
    work_dir = output_dir / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_file = work_dir / "prompt.txt"
    if Path(prompt).is_file():
        shutil.copy(prompt, prompt_file)
    else:
        with open(prompt_file, 'w') as f:
            f.write(prompt)

    # --- Stage 1: Generate Layout ---
    stage_1_output = work_dir / "1_layout.json"
    if not run_stage_with_retry(stage_1_layout, max_retries, prompt_file, stage_1_output, model):
        logger.error("❌ Assembly line failed at Stage 1. Aborting.")
        return

    # --- Stage 2: Add Openings ---
    stage_2_output = work_dir / "2_layout_with_openings.json"
    if not run_stage_with_retry(stage_2_openings, max_retries, stage_1_output, prompt_file, stage_2_output, model):
        logger.error("❌ Assembly line failed at Stage 2. Aborting.")
        return

    # --- Stage 3: Finalize Plan ---
    final_output = work_dir / "3_final_plan.json"
    if not stage_3_finalize(stage_2_output, prompt_file, final_output): # No retry for deterministic stage
        logger.error("❌ Assembly line failed at Stage 3. Aborting.")
        return

    # --- Final Validation ---
    final_error_file = work_dir / "final_validation_errors.json"
    if validate_output(final_output, final_error_file):
        final_path = output_dir / "gold_standard" / f"{run_id}_curated.json"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_output, final_path)
        logger.info(f"✅ Assembly Line Succeeded! Final plan saved to: {final_path}")
    else:
        logger.error(f"❌ Final validation failed. Please check the artifacts in {work_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="HouseBrain Architect's Assembly Line")
    parser.add_argument("--prompt", type=str, required=True, help="A string containing the design prompt or a path to a .txt file.")
    parser.add_argument("--output-dir", type=str, default="output/assembly_line", help="Directory to save the final files.")
    parser.add_argument("--run-name", type=str, help="Optional unique name for this run.")
    parser.add_argument("--model", type=str, default="qwen3:30b", help="Name of the Ollama model to use for generation stages.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for each LLM-based stage.")
    args = parser.parse_args()

    assembly_line_pipeline(
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        model=args.model,
        run_name=args.run_name,
        max_retries=args.max_retries
    )

if __name__ == "__main__":
    main()
