import ollama
import json
import sys
import os

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We can try to import the schema for context, but the script will run even if it fails.
try:
    from src.housebrain.schema import HouseOutput
    SCHEMA_DEFINITION = HouseOutput.model_json_schema()
    print("✅ Successfully loaded HouseBrain schema for context.")
except ImportError:
    SCHEMA_DEFINITION = "Schema not available. Generate a house plan with levels, rooms, doors, and windows."
    print("⚠️ Could not import HouseBrain schema. Prompt will use a basic description.")


MODELS_TO_TEST = [
    "qwen2:7b",
    "llama3:instruct",
    "mistral:7b-instruct",
]

COMPLEX_PROMPT = """
Generate a detailed house plan in valid JSON format for a two-story modern minimalist house in Bangalore, India, on a 30x40 feet plot (1200 sq ft).

The ground floor must include:
- A living room.
- A kitchen with an attached utility area.
- A guest bedroom with an attached bathroom.
- A common powder room.
- An internal staircase.

The first floor must have:
- A master bedroom with a walk-in closet and an attached bathroom.
- A children's bedroom with an attached bathroom.
- A small family lounge area that opens to a balcony.

Constraints & Instructions:
1.  All rooms, including bathrooms, closets, and utility areas, must have unique IDs and specified dimensions.
2.  Ensure there is at least one window in every room (except closets/storage).
3.  The output MUST be a single, clean JSON object that aims to satisfy the HouseOutput schema.
4.  Do NOT include any conversational text, markdown formatting (like ```json), or explanations outside of the JSON object itself.
5.  The root of the JSON should contain keys like "house_name", "style", "total_area", and "levels".

Here is a simplified schema definition to guide you:
{schema_definition}
""".format(schema_definition=json.dumps(SCHEMA_DEFINITION, indent=2))

def run_comparison():
    """
    Runs the complex prompt against a list of specified Ollama models and prints their raw output.
    """
    print("="*80)
    print("🏛️  HouseBrain Model Showdown 🏛️")
    print(f"Running a complex prompt against {len(MODELS_TO_TEST)} models.")
    print("="*80)
    print(f"Prompt length: {len(COMPLEX_PROMPT)} characters")
    print("\n--- PROMPT ---")
    print("Generate a detailed house plan in JSON format for a two-story modern minimalist house in Bangalore, India...")
    print("--- END PROMPT ---\n")

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*20} 🤖 Querying: {model_name} {'='*20}\n")
        try:
            client = ollama.Client()
            
            # NOTE: Temporarily removing the model check/pull logic for debugging.
            # This script now assumes you have the models pulled locally.
            # You can pull them with `ollama pull <model_name>`
            
            response = client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': COMPLEX_PROMPT}],
                format='json' # Request JSON output
            )
            
            raw_output = response['message']['content']
            
            # Try to parse and re-indent for readability
            try:
                parsed_json = json.loads(raw_output)
                print(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError:
                print("--- RAW OUTPUT (Not valid JSON) ---")
                print(raw_output)

        except Exception as e:
            print(f"❌ An error occurred while querying {model_name}: {e}")
            print("Please ensure Ollama is running and the model is available.")
        
        print(f"\n{'='*20} ✅ Finished: {model_name} {'='*20}\n")

if __name__ == "__main__":
    run_comparison()
