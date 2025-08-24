import json
import logging
import sys
from pathlib import Path

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Constants ---
# These are all the keys in the schema that should be a list.
# If any of these are `None` (null in JSON), they will be replaced with `[]`.
LIST_KEYS = {
    "levels",
    "rooms",
    "walls",
    "doors",
    "windows",
    "furniture",
    "stairs",
    "columns",
    "beams",
    "annotations",
    "points",
    "components",
    "materials",
    "costs",
    "appliances",
    "electrical_fixtures",
    "plumbing_fixtures"
}

def sanitize_data(data_obj):
    """Recursively traverses a dict/list and replaces None with [] for specific keys."""
    was_modified = False
    if isinstance(data_obj, dict):
        for key, value in data_obj.items():
            if key in LIST_KEYS and value is None:
                data_obj[key] = []
                was_modified = True
            elif isinstance(value, (dict, list)):
                if sanitize_data(value):
                    was_modified = True
    elif isinstance(data_obj, list):
        for item in data_obj:
            if isinstance(item, (dict, list)):
                if sanitize_data(item):
                    was_modified = True
    return was_modified

def sanitize_json_file(file_path: Path) -> bool:
    """Loads, sanitizes, and saves a JSON file if modifications were made."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if sanitize_data(data):
            logger.info(f"Sanitizing inconsistent list values in: {file_path.name}")
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        return False
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not process {file_path.name}: {e}")
        return False

def main():
    """
    Scans all .json files in the gold_standard directory, finds files where
    list-type fields are set to `null` instead of `[]`, and corrects them.
    """
    gold_standard_dir = Path("data/training/gold_standard")
    if not gold_standard_dir.is_dir():
        logger.error(f"Directory not found: {gold_standard_dir}")
        sys.exit(1)

    logger.info(f"Scanning for inconsistent JSON files in {gold_standard_dir}...")

    json_files = [f for f in gold_standard_dir.glob("*.json") if "_draft" not in f.name]
    sanitized_count = sum(1 for file in json_files if sanitize_json_file(file))

    if sanitized_count > 0:
        logger.info(f"✅ Sanitization complete. Corrected {sanitized_count} file(s).")
    else:
        logger.info("✅ No inconsistencies found. All files are correctly formatted.")

if __name__ == "__main__":
    main()
