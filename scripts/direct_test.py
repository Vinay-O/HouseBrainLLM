# direct_test.py
import json
import logging
import sys
from pathlib import Path
import urllib.request
from typing import Optional

# --- Add Project Root to Path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.housebrain.schema import RoomType

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Re-implement necessary functions locally ---
VALID_ROOM_TYPES = [e.value for e in RoomType]

TEST_PROMPT = f"""You are an expert AI architect. Your task is to generate ONLY the high-level geometric layout for a house based on a user's prompt.

**CRITICAL INSTRUCTIONS:**
1.  Focus ONLY on `levels` and `rooms`.
2.  Rooms MUST have an `id`, `type`, and non-overlapping `bounds`.
3.  The `type` for each room MUST be one of the following valid options: `{VALID_ROOM_TYPES}`. Do NOT invent new types.
4.  DO NOT include `doors` or `windows` in this stage.
5.  Your output MUST be a single, valid JSON object with a root "levels" key.

**Golden Example of a perfect room structure:**
```json
{{
  "id": "living_room_0",
  "type": "living_room",
  "bounds": {{"x": 10, "y": 10, "width": 20, "height": 15}}
}}
```
---
**User Prompt:**
Design a simple, single-story 2BHK house with a living room, kitchen, two bedrooms, and one bathroom.
---
Now, generate the JSON for the house layout, adhering strictly to the instructions provided."""


def call_ollama(model_name: str, prompt: str) -> Optional[str]:
    logger.info(f"Sending prompt of length {len(prompt)} to model {model_name}...")
    url = "http://localhost:11434/api/generate"
    data = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}
    encoded_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=encoded_data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            if response.status == 200:
                response_data = json.loads(response.read().decode('utf-8'))
                return response_data.get("response", "")
    except urllib.error.HTTPError as e:
        error_content = e.read().decode('utf-8')
        logger.error(f"HTTP Error from Ollama API: {e.code} {e.reason}")
        logger.error(f"Ollama response: {error_content}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred calling Ollama API: {e}")
        return None

def main():
    print("--- Starting Direct Ollama Test ---")
    model_name = "mixtral:instruct"
    
    raw_response = call_ollama(model_name, TEST_PROMPT)
    
    print("\n--- RAW RESPONSE FROM OLLAMA ---")
    if raw_response:
        print(raw_response)
    else:
        print("!!! FAILED to get a response from Ollama. Check logs for errors. !!!")
    print("--- End of Test ---")

if __name__ == "__main__":
    main()
