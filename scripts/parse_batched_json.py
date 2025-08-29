import os
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Parse a JSON file containing an array of plans and split them into individual files.")
    parser.add_argument("--input-file", required=True, type=str, help="Path to the batched JSON file (e.g., perplexity_batch_01.json).")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save the individual plan files.")
    parser.add_argument("--start-index", type=int, default=0, help="The starting number for the output filenames (e.g., plan_0000.json).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_plans = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error reading or parsing the input file: {e}")
        return
        
    if not isinstance(all_plans, list):
        print("❌ Error: The input JSON file is not a list/array of plans.")
        return

    print(f"Found {len(all_plans)} plans in the batch file. Splitting now...")

    for i, plan_data in enumerate(tqdm(all_plans, desc="Splitting Batched Plans")):
        file_index = args.start_index + i
        output_filename = f"plan_{file_index:04d}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2)
        except IOError as e:
            tqdm.write(f"Could not write file {output_path}: {e}")

    print("\n" + "="*50)
    print(f"✅ Success! Split {len(all_plans)} plans into '{args.output_dir}'.")
    print("="*50)

if __name__ == "__main__":
    main()
