#!/bin/bash

# ==============================================================================
# HouseBrain Local Pipeline Smoke Test (Consolidated Version)
# ==============================================================================
#
# This script runs a single, quick test of the new self-contained assembly line.
#
# Usage:
#   ./test_pipeline_locally.sh
#
# ==============================================================================

set -e 
set -o pipefail
set -x # Print each command before executing it

echo "--- Starting HouseBrain Local Consolidated Assembly Line Test ---"

# --- Configuration ---
TEST_MODEL="mistral:7b-instruct" # Using the stable Mistral model name
TEST_PROMPT="A simple, one-story, 2-bedroom, 1-bathroom rectangular house on a 30x50 feet plot."
OUTPUT_DIR="output/local_consolidated_test"
RUN_NAME="local_mistral_test"

# --- Pre-flight Check ---
echo "Ensuring output directory exists..."
mkdir -p "$OUTPUT_DIR/$RUN_NAME"

# --- Test Execution ---
echo "Using model: $TEST_MODEL"
echo "Using output directory: $OUTPUT_DIR"
echo "Running the Consolidated Assembly Line..."

python scripts/run_complete_assembly_line.py \
    --prompt "$TEST_PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --model "$TEST_MODEL" \
    --max-retries 1

echo ""
echo "--- ✅ Local Consolidated Assembly Line Test Completed ---"
echo "NOTE: This only confirms the script runs. Architectural quality depends on the model's performance."


echo ""
echo "--- ✅ Local Consolidated Assembly Line Test Completed ---"
echo "NOTE: This only confirms the script runs. Architectural quality depends on the model's performance."
