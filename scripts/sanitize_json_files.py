import os
import argparse
import json
from tqdm import tqdm
import re

def sanitize_json_string(json_string):
    """
    Removes illegal control characters from a string before JSON parsing.
    """
    # This regex matches control characters except for tab, newline, and carriage return
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_string)

def main(args):
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Input directory not found at '{args.input_dir}'")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    filenames = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    
    if not filenames:
        print(f"🤷 No JSON files found in '{args.input_dir}'.")
        return

    print(f"Found {len(filenames)} files. Starting sanitization...")

    success_count = 0
    fail_count = 0

    for filename in tqdm(filenames, desc="Sanitizing JSON files"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Clean the raw string
            sanitized_content = sanitize_json_string(raw_content)
            
            # Now, try to parse it to ensure it's valid JSON
            json.loads(sanitized_content)

            # If parsing is successful, write the sanitized content to the new file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(sanitized_content)
            
            success_count += 1

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            tqdm.write(f"⚠️ Failed to process {filename}: {e}")
            fail_count += 1
        except Exception as e:
            tqdm.write(f"🚨 An unexpected error occurred with {filename}: {e}")
            fail_count += 1
            
    print("\n" + "="*50)
    print("✅ Sanitization complete!")
    print(f"   Successfully processed: {success_count} files")
    print(f"   Failed to process:      {fail_count} files")
    print(f"   Cleaned files saved in: '{args.output_dir}'")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize raw JSON files by removing illegal control characters.")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True,
        help="Directory containing the raw, potentially invalid JSON files."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Directory where the sanitized JSON files will be saved."
    )
    args = parser.parse_args()
    main(args)
