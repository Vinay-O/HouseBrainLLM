#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
MODEL_TO_TEST="phi3:instruct"
SCRIPT_PATH="scripts/run_complete_assembly_line.py"
BATCH_OUTPUT_DIR="test_output/local_batch_$(date +%Y%m%d_%H%M%S)"
SUCCESS_DIR="$BATCH_OUTPUT_DIR/gold_standard"
PROMPT_FILE="local_test_prompts.txt"
NUM_PROMPTS=10

# --- Create Prompts ---
echo "Creating $NUM_PROMPTS test prompts..."
cat > "$PROMPT_FILE" << EOL
Design a luxurious 4-bedroom villa with a swimming pool and a home theater.
Create a compact, eco-friendly 1-bedroom apartment for a narrow city plot.
Design a traditional two-story farmhouse with a wrap-around porch and a large kitchen.
Plan a minimalist open-plan studio loft with high ceilings and industrial features.
Design a 3-bedroom suburban family home with a two-car garage and a backyard patio.
Create a single-story beach house with large windows and direct access to the sand.
Design a 5-bedroom G+2 building on a 30x60 feet plot, with ground floor parking.
Plan a rustic mountain cabin with a stone fireplace and a cozy sleeping loft.
Design a modern duplex with separate entrances and shared garden space.
Create a budget-friendly 2BHK apartment design for a small family, under 1000 sqft.
EOL

# --- Run Batch Test ---
echo "Starting batch test with model: $MODEL_TO_TEST"
echo "Output will be saved in: $BATCH_OUTPUT_DIR"
mkdir -p "$BATCH_OUTPUT_DIR"

while IFS= read -r prompt; do
    echo "--------------------------------------------------"
    echo "Processing Prompt: $prompt"
    echo "--------------------------------------------------"
    # We add '|| true' so the script doesn't exit if one prompt fails
    python3 "$SCRIPT_PATH" \
        --model "$MODEL_TO_TEST" \
        --output-dir "$BATCH_OUTPUT_DIR" \
        --prompt "$prompt" || true
done < "$PROMPT_FILE"

# --- Verification ---
echo "=================================================="
echo "Batch test complete."
echo "=================================================="

if [ -d "$SUCCESS_DIR" ]; then
    SUCCESS_COUNT=$(ls -1q "$SUCCESS_DIR" | wc -l)
    echo "✅ Success! Found $SUCCESS_COUNT validated plans in '$SUCCESS_DIR'."
    ls -l "$SUCCESS_DIR"
else
    SUCCESS_COUNT=0
    echo "❌ No successful plans were generated."
fi

echo "Generated $SUCCESS_COUNT / $NUM_PROMPTS plans successfully."

# --- Cleanup ---
rm "$PROMPT_FILE"

# Exit with success if at least one file was generated
if [ "$SUCCESS_COUNT" -gt 0 ]; then
    exit 0
else
    exit 1
fi

