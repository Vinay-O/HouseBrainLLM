#!/bin/bash

# ==============================================================================
# HouseBrain Local Pipeline Smoke Test
# ==============================================================================
#
# This script runs a single, quick test of the automated curation pipeline
# to catch basic errors (e.g., missing files, TypeErrors) before committing
# and pushing code. It's a safety check to ensure the core logic is functional.
#
# It uses a simple prompt and a locally available Ollama model.
#
# Usage:
#   ./test_pipeline_locally.sh
#
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.

echo "--- Starting HouseBrain Local Pipeline Smoke Test ---"

# --- Configuration ---
# You can change this to any model you have available locally with 'ollama list'
# Using a smaller model like llama3:8b can make the test faster.
TEST_MODEL="qwen3:30b" 
TEST_PROMPT="A simple, one-story, 2-bedroom, 1-bathroom rectangular house on a 30x50 feet plot."
OUTPUT_DIR="output/local_test_run"

# --- Test Execution ---
echo "Using model: $TEST_MODEL"
echo "Using output directory: $OUTPUT_DIR"
echo "Running automated_curation.py..."

python scripts/automated_curation.py \
    --prompt "$TEST_PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --model "$TEST_MODEL" \
    --repair-model "$TEST_MODEL" \
    --max-retries 1

echo ""
echo "--- ✅ Local Pipeline Smoke Test Completed Successfully ---"
echo "NOTE: This only confirms the script runs. Architectural quality depends on the model's performance."
