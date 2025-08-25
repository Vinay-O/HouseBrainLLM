#!/bin/bash
set -ex # Exit on error, print commands

echo "--- Starting HouseBrain Local Llama 3 Assembly Line Test ---"

# --- Configuration ---
TEST_MODEL="llama3:instruct"
TEST_PROMPT="A simple, one-story, 2-bedroom, 1-bathroom rectangular house on a 30x50 feet plot."
OUTPUT_DIR="output/local_llama3_test"
RUN_NAME="local_llama3_test_run"

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
echo "--- ✅ Local Llama 3 Assembly Line Test Completed ---"
echo "Check the '$OUTPUT_DIR/$RUN_NAME' directory for results."
echo ""
