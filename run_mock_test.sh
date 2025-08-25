#!/bin/bash
set -e
OUTPUT_DIR="output/mock_run_v2"
PROMPTS_FILE="platinum_prompts.txt"
MODEL_NAME="mixtral:instruct" # Make sure this model is available in Ollama

mkdir -p "$OUTPUT_DIR"
echo "--- Starting Mock Run with Upgraded Prompts ---"
echo "Output will be saved in: $OUTPUT_DIR"
echo ""

# Get the first 10 prompts
PROMPTS=($(head -n 10 "$PROMPTS_FILE"))

COUNT=0
head -n 10 "$PROMPTS_FILE" | while IFS= read -r prompt; do
  COUNT=$((COUNT+1))
  # Generate a unique run name from the prompt
  RUN_NAME=$(echo "$prompt" | shasum | head -c 10)
  
  echo "--------------------------------------------------"
  echo "Processing prompt ${COUNT}/10: ${prompt:0:80}..."
  
  python3 scripts/run_complete_assembly_line.py \
    --prompt "$prompt" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --model "$MODEL_NAME" \
    --max-retries 2
    
  echo "✅ Done with prompt ${COUNT}/10."
  echo ""
done

echo "--- Mock Run Complete ---"
echo "Check the results in the '${OUTPUT_DIR}' directory."
