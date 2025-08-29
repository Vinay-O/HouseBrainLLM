import json
import os
import sys
import ollama
import re

def get_original_prompt(prompt_file_path: str) -> str:
    """Reads the first non-empty line from the prompt file."""
    try:
        with open(prompt_file_path, 'r') as f:
            return next(line.strip() for line in f if line.strip())
    except (FileNotFoundError, StopIteration):
        print(f"Warning: Could not read prompt file at {prompt_file_path}")
        return ""

def analyze_plan(plan_data: dict) -> dict:
    """Analyzes the plan to extract key features like bedroom count."""
    analysis = {"bedroom_count": 0, "level_count": 0}
    analysis["level_count"] = len(plan_data.get("levels", []))
    
    for level in plan_data.get("levels", []):
        for room in level.get("rooms", []):
            # Check for 'bedroom' in the id or label, case-insensitive
            if "bedroom" in room.get("id", "").lower() or "bedroom" in room.get("label", "").lower():
                analysis["bedroom_count"] += 1
    return analysis

def validate_with_llm(original_prompt: str, analysis: dict, judge_model: str) -> bool:
    """
    Asks a "judge" LLM if the plan's features match the original prompt.
    """
    print(f"  > Asking {judge_model} for a semantic check...")
    
    # Extract the requested BHK from the prompt, e.g., "1BHK" -> 1
    bhk_match = re.search(r'(\d+)\s*BHK', original_prompt, re.IGNORECASE)
    requested_bedrooms = int(bhk_match.group(1)) if bhk_match else "an unspecified number of"

    validation_prompt = f"""
    A user requested a floor plan with the following prompt: "{original_prompt}"
    
    An AI generated a plan with the following features:
    - Number of bedrooms: {analysis['bedroom_count']}

    Does the number of bedrooms in the generated plan ({analysis['bedroom_count']}) logically satisfy the user's request for "{requested_bedrooms}" bedrooms?
    
    Answer ONLY with the word YES or NO.
    """
    
    try:
        response = ollama.chat(
            model=judge_model,
            messages=[{'role': 'user', 'content': validation_prompt}]
        )
        answer = response['message']['content'].strip().upper()
        print(f"  > Judge's answer: {answer}")
        return "YES" in answer
    except Exception as e:
        print(f"  > LLM validation failed: {e}")
        return False # Fail safely

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python semantic_validator.py <path_to_enriched_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    PROMPT_FILE = "platinum_prompts.txt"
    JUDGE_MODEL = "llama3:instruct" # A fast and reliable model for simple checks
    
    PROD_DIR = "PROD_DATASET/gold_tier"
    QUARANTINE_DIR = "quarantine/semantic_errors"

    print(f"--- 3. Semantic Validation: {os.path.basename(input_file)} ---")

    with open(input_file, 'r') as f:
        plan_data = json.load(f)
    
    original_prompt = get_original_prompt(PROMPT_FILE)
    plan_analysis = analyze_plan(plan_data)
    
    print(f"  > Original prompt requested a '{original_prompt.split(',')[1].strip()}'.")
    print(f"  > Generated plan has {plan_analysis['bedroom_count']} bedroom(s).")
    
    is_valid = validate_with_llm(original_prompt, plan_analysis, JUDGE_MODEL)

    if is_valid:
        print("✅ PASSED: Plan is semantically consistent with the prompt.")
        os.makedirs(PROD_DIR, exist_ok=True)
        os.rename(input_file, os.path.join(PROD_DIR, os.path.basename(input_file)))
        print(f"\nFile moved to production dataset: {PROD_DIR}")
    else:
        print("❌ FAILED: Plan is NOT semantically consistent with the prompt.")
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        os.rename(input_file, os.path.join(QUARANTINE_DIR, os.path.basename(input_file)))
        print(f"\nFile moved to quarantine: {QUARANTINE_DIR}")
