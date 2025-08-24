import subprocess
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Runs a full end-to-end test of the HouseBrain pipeline:
    1. Generates a draft plan from a text prompt using a local LLM.
    2. Runs the validation and rendering pipeline on the generated draft.
    """
    logger.info("--- Starting End-to-End Test ---")

    # --- Configuration ---
    model = "llama3"  # Assumes 'llama3' is available via Ollama
    output_dir = Path("output")
    draft_filename = "e2e_test_draft.json"
    draft_path = output_dir / draft_filename
    prompt_file = Path("temp_e2e_prompt.txt")
    schema_path = Path("src/housebrain/schema.py")
    
    scenario = (
        "A modern, single-story 3BHK house for a 50x80 feet plot. "
        "It must feature an open-plan kitchen and living area, a dedicated home office, "
        "and be Vastu-compliant with a North-facing entrance."
    )

    # --- Step 1: Generate the schema-aware prompt ---
    logger.info(f"Generating schema-aware prompt for scenario: '{scenario}'")
    
    if not schema_path.exists():
        logger.error(f"FATAL: Schema file not found at {schema_path}")
        sys.exit(1)

    schema_definition = schema_path.read_text()

    prompt_header = f"""You are an expert Indian architect specializing in Vastu-compliant residential design.
Your task is to generate a complete, valid, and architecturally sound JSON object
that strictly adheres to the Pydantic schema provided below.

The JSON object must be a single, complete `HouseOutput` object.
Do not add any text or explanation before or after the JSON object.

**Pydantic Schema Definition:**
```python
{schema_definition}
```

**User Design Request:**
"{scenario}"

**Your Output (JSON object conforming to HouseOutput schema only):**
"""
    try:
        prompt_file.write_text(prompt_header, encoding="utf-8")
        logger.info(f"Prompt written to {prompt_file}")
    except Exception as e:
        logger.error(f"Failed to write prompt file: {e}")
        sys.exit(1)

    # --- Step 2: Generate the draft using the LLM ---
    logger.info(f"Generating draft from LLM ('{model}'). This may take a moment...")
    
    generation_command = [
        sys.executable,
        "scripts/generate_draft_from_prompt.py",
        "--model", model,
        "--prompt-file", str(prompt_file),
        "--output-file", str(draft_path)
    ]

    gen_result = subprocess.run(generation_command, capture_output=True, text=True, encoding='utf-8')

    if gen_result.returncode != 0:
        logger.error("❌ Draft generation script failed.")
        logger.error(f"STDOUT:\n{gen_result.stdout}")
        logger.error(f"STDERR:\n{gen_result.stderr}")
        prompt_file.unlink() # Clean up
        sys.exit(1)
    
    logger.info(f"✅ Draft generation script finished. Output saved to {draft_path}")
    logger.info(f"STDOUT:\n{gen_result.stdout}")
    if gen_result.stderr:
        logger.warning(f"STDERR:\n{gen_result.stderr}")

    # --- Step 3: Run the rendering pipeline on the generated draft ---
    logger.info(f"Running rendering pipeline on the generated draft: {draft_path}")

    pipeline_command = [
        sys.executable,
        "run_pipeline.py",
        "--input", str(draft_path)
    ]

    pipe_result = subprocess.run(pipeline_command, capture_output=True, text=True, encoding='utf-8')

    logger.info("--- Pipeline Execution Report ---")
    if pipe_result.returncode != 0:
        logger.error("❌ The rendering pipeline failed for the generated draft.")
    else:
        logger.info("✅ SUCCESS: The rendering pipeline completed for the generated draft.")

    logger.info(f"STDOUT:\n{pipe_result.stdout}")
    if pipe_result.stderr:
        logger.error(f"STDERR:\n{pipe_result.stderr}")
        
    # --- Cleanup ---
    prompt_file.unlink()
    logger.info(f"Cleaned up temporary prompt file: {prompt_file}")
    
    logger.info("--- End-to-End Test Finished ---")

if __name__ == "__main__":
    main()
