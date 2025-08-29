import ollama
import os
import hashlib
import time
import argparse
import json

def generate_single_plan(prompt, model_name, output_dir):
    """
    Generates a single house plan and saves the raw output.
    """
    print(f"--- Running Single Prompt Test ---")
    print(f"Model: {model_name}")
    print(f"Output Directory: {output_dir}")
    print("-" * 30)

    # --- Create a unique directory for this run ---
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:10]
    timestamp = int(time.time())
    unique_id = f"single_test_{prompt_hash}_{timestamp}"
    
    run_output_dir = os.path.join(output_dir, model_name, unique_id)
    os.makedirs(run_output_dir, exist_ok=True)
    
    output_filename = os.path.join(run_output_dir, "raw_output.json")
    
    try:
        # --- Construct the detailed prompt ---
        structured_prompt = f"""
        Please act as an expert architect. Your task is to generate a detailed JSON representation of a house floor plan based on the following request.

        **Architectural Request:** "{prompt}"

        **Instructions:**
        1.  **Think Step-by-Step:** First, reason about the spatial layout and overall flow.
        2.  **Generate JSON:** After your reasoning, provide ONLY the JSON output. Do not include any other text outside of the JSON block.

        **JSON Schema:**
        {{
          "levels": [
            {{
              "level_id": "ground_floor",
              "rooms": [
                {{"id": "living_room", "bounds": [x1, y1, x2, y2], "label": "Living Room"}}
              ],
              "openings": [
                {{"id": "front_door", "bounds": [x1, y1, x2, y2], "type": "door"}}
              ]
            }}
          ]
        }}
        
        Now, begin.
        """
        
        print(f"Sending prompt to Ollama...")
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': structured_prompt}],
            format='json'
        )
        
        raw_output = response['message']['content']
        
        # --- Save the raw output ---
        with open(output_filename, 'w') as f:
            # Try to pretty-print if it's valid JSON, otherwise save as raw text
            try:
                parsed_json = json.loads(raw_output)
                json.dump(parsed_json, f, indent=4)
            except json.JSONDecodeError:
                f.write(raw_output)

        print(f"\n✅ Success! Raw output saved to:")
        print(f"   {output_filename}")
        return True

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(f"\n❌ Error: {error_message}")
        with open(os.path.join(run_output_dir, "error.log"), 'w') as f:
            f.write(error_message)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single prompt test for house plan generation.")
    parser.add_argument(
        "--model",
        type=str,
        default="housebrain-architect-v1",
        help="The name of the Ollama model to use."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="raw_generated_data",
        help="The base directory to save the output."
    )
    args = parser.parse_args()

    # --- Load the first prompt from the platinum file ---
    try:
        with open("platinum_prompts.txt", 'r') as f:
            # Read the first non-empty line
            test_prompt = next(line for line in f if line.strip())
            test_prompt = test_prompt.strip()
            print(f"Loaded test prompt: {test_prompt}")
    except (FileNotFoundError, StopIteration):
        print("Error: platinum_prompts.txt not found or is empty. Using a default prompt.")
        test_prompt = "Design a modern, 2BHK apartment in Mumbai with a sea-facing balcony."


    generate_single_plan(
        prompt=test_prompt,
        model_name=args.model,
        output_dir=args.output_dir
    )
