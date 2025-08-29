import os
import json
import argparse
import random
import copy
from tqdm import tqdm
import re

def load_json_file(filepath):
    """Loads a JSON file and returns its content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        tqdm.write(f"Could not read or parse {filepath}: {e}")
        return None

# --- FLAW INJECTION FUNCTIONS ---

def flaw_remove_all_doors(plan_data):
    """Removes all doors from all rooms on all levels."""
    flawed_plan = copy.deepcopy(plan_data)
    for level in flawed_plan.get("levels", []):
        for room in level.get("rooms", []):
            if "doors" in room:
                room["doors"] = []
    flawed_plan["__flaw_applied__"] = "remove_all_doors"
    return flawed_plan

def flaw_overlap_rooms(plan_data):
    """Finds two rooms on the same level and makes them overlap."""
    flawed_plan = copy.deepcopy(plan_data)
    for level in flawed_plan.get("levels", []):
        if len(level.get("rooms", [])) >= 2:
            # Pick a random room to modify
            room_to_modify_index = random.randrange(len(level["rooms"]))
            room_to_modify = level["rooms"][room_to_modify_index]
            
            # Increase its size to overlap with something
            if "bounds" in room_to_modify:
                room_to_modify["bounds"]["width"] *= 1.5
                flawed_plan["__flaw_applied__"] = "overlap_rooms"
                return flawed_plan
    # If no suitable level was found, return the original with a note
    flawed_plan["__flaw_applied__"] = "none"
    return flawed_plan

def flaw_shrink_critical_room(plan_data):
    """Finds a critical room (bedroom, kitchen, living_room) and shrinks it."""
    flawed_plan = copy.deepcopy(plan_data)
    critical_room_types = ["bedroom", "kitchen", "living_room"]
    
    potential_rooms_to_shrink = []
    for level in flawed_plan.get("levels", []):
        for i, room in enumerate(level.get("rooms", [])):
            if room.get("type") in critical_room_types and "bounds" in room:
                potential_rooms_to_shrink.append(room)

    if potential_rooms_to_shrink:
        room_to_shrink = random.choice(potential_rooms_to_shrink)
        room_to_shrink["bounds"]["width"] = 2
        room_to_shrink["bounds"]["height"] = 2
        flawed_plan["__flaw_applied__"] = "shrink_critical_room"
        return flawed_plan

    flawed_plan["__flaw_applied__"] = "none"
    return flawed_plan

def flaw_delete_random_room(plan_data):
    """Deletes a random room from a random level."""
    flawed_plan = copy.deepcopy(plan_data)
    potential_levels = [level for level in flawed_plan.get("levels", []) if level.get("rooms")]
    
    if potential_levels:
        level_to_modify = random.choice(potential_levels)
        room_index_to_delete = random.randrange(len(level_to_modify["rooms"]))
        del level_to_modify["rooms"][room_index_to_delete]
        flawed_plan["__flaw_applied__"] = "delete_random_room"
        return flawed_plan

    flawed_plan["__flaw_applied__"] = "none"
    return flawed_plan

def main():
    parser = argparse.ArgumentParser(description="Create a 'correction' dataset for training a Senior Architect model.")
    parser.add_argument("--input-dir", required=True, type=str, help="Directory containing the high-quality, correct JSON plan files.")
    parser.add_argument("--output-file", type=str, default="finetune_dataset_senior_architect.jsonl", help="Path to the output JSONL file.")
    parser.add_argument("--prompts-file", type=str, default="prompts_combined_2500.txt", help="Path to the master prompt file.")
    args = parser.parse_args()
    
    flaw_functions = [
        flaw_remove_all_doors,
        flaw_overlap_rooms,
        flaw_shrink_critical_room,
        flaw_delete_random_room
    ]

    # Load all possible prompts
    try:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(all_prompts)} prompts from {args.prompts_file}")
    except FileNotFoundError:
        print(f"❌ FATAL: Prompt file not found at '{args.prompts_file}'")
        return

    all_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.json')]
    
    if not all_files:
        print(f"❌ No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(all_files)} plans to process for the correction dataset.")

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for filepath in tqdm(all_files, desc="Generating Correction Dataset"):
            correct_plan = load_json_file(filepath)
            if not correct_plan:
                continue

            # --- NEW PROMPT FINDING LOGIC ---
            user_prompt = None
            filename = os.path.basename(filepath)
            try:
                # Assumes filename is like 'plan_0001.json'
                prompt_index_str = re.sub(r'\D', '', filename)
                prompt_index = int(prompt_index_str)
                if 0 <= prompt_index < len(all_prompts):
                    user_prompt = all_prompts[prompt_index]
                else:
                    tqdm.write(f"Skipping {filename}: index {prompt_index} out of range.")
                    continue
            except (ValueError, IndexError):
                tqdm.write(f"Skipping {filename}: could not extract valid prompt index.")
                continue
            
            if not user_prompt:
                tqdm.write(f"Skipping {filename} because no prompt could be associated.")
                continue

            # Select a random flaw to introduce
            flaw_function = random.choice(flaw_functions)
            flawed_plan = flaw_function(correct_plan)

            if flawed_plan.get("__flaw_applied__") == "none":
                tqdm.write(f"Skipping {os.path.basename(filepath)} as no flaw could be applied.")
                continue

            # The user message now contains the original prompt AND the flawed plan
            user_content = (
                "Please act as a senior architect. The following JSON house plan was generated for a user request, but it contains one or more significant architectural, geometric, or functional flaws. Your task is to identify and fix the issues to make the plan compliant with the original request and sound architectural principles. Here is the original user request:\n\n"
                f"--- USER REQUEST ---\n{user_prompt}\n\n"
                "--- FLAWED JSON PLAN ---\n"
                f"{json.dumps(flawed_plan, indent=2)}\n\n"
                "--- CRITICAL INSTRUCTIONS ---\n"
                "Your entire response MUST be a single, raw, valid JSON object that represents the corrected plan. Do not add any explanations, apologies, or any characters outside of the main JSON structure."
            )

            # The assistant's response is the original, correct plan
            assistant_content = json.dumps(correct_plan, indent=2)

            # Create the final training record
            output_record = {
                "messages": [
                    {"role": "system", "content": "You are an expert AI architect. Your only task is to review a flawed house plan and output the corrected version as a single, raw JSON object. Do not add any conversational text or markdown."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            outfile.write(json.dumps(output_record) + '\n')

    print("\n" + "="*50)
    print(f"✅ Success! Created correction dataset with ~{len(all_files)} samples.")
    print(f"   Dataset saved to: '{args.output_file}'")
    print("="*50)


if __name__ == "__main__":
    main()
