#!/bin/bash
set -ex # Exit on error, print commands

echo "--- Starting HouseBrain Local Mixtral Assembly Line Test ---"

# --- Configuration ---
TEST_MODEL="llama3:instruct"
TEST_PROMPT="A modern, single-story 3BHK house for a 50x80 feet plot. It must feature an open-plan kitchen and living area, a dedicated home office, and be Vastu-compliant with a North-facing entrance."
OUTPUT_DIR="output/local_mixtral_test"
RUN_NAME="local_mixtral_test_run"

# --- Execution ---
echo "Ensuring output directory exists..."
mkdir -p "$OUTPUT_DIR/$RUN_NAME"

echo "Using model: $TEST_MODEL"
echo "Using output directory: $OUTPUT_DIR"
echo "Running the Consolidated Assembly Line..."

python scripts/run_complete_assembly_line.py \
    --prompt "$TEST_PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --model "$TEST_MODEL" \
    --max-retries 3

echo ""
echo "--- ✅ Local Mixtral Assembly Line Test Completed ---"
echo "Check the '$OUTPUT_DIR/$RUN_NAME' directory for results."
echo ""
