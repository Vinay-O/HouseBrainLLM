#!/bin/bash
set -ex

# Model to test
export TEST_MODEL="llama3:instruct"
export SCRIPT_PATH="scripts/run_complete_assembly_line.py"
export TEST_OUTPUT_DIR="test_output/llama3_instruct_$(date +%Y%m%d_%H%M%S)"

# Create a test prompt
TEST_PROMPT="Design a small, modern 2-bedroom house with an open-plan kitchen and a large balcony. The plot size is 40x60 feet."

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"
echo "Test output will be saved in: $TEST_OUTPUT_DIR"

# Run the pipeline with a single prompt
python3 "$SCRIPT_PATH" \
    --model "$TEST_MODEL" \
    --output-dir "$TEST_OUTPUT_DIR" \
    --prompt "$TEST_PROMPT"

# Verify the output
echo "--- Verification ---"
echo "Listing contents of $TEST_OUTPUT_DIR:"
ls -l "$TEST_OUTPUT_DIR"
echo "Test complete."
