import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    """
    This script loads the fine-tuned HouseBrain V1 model and runs inference locally.
    It's optimized to run on Apple Silicon (M-series chips) using the MPS backend.
    """

    # --- 1. Configuration ---
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path = "./housebrain_v1_adapters" # Assumes adapters are in the same folder

    # --- 2. Device Setup (Detect Apple Silicon) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon (MPS) device found. Using MPS for acceleration.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ℹ️ Using device: {device}. Note: MPS not found, performance may vary.")

    # --- 3. Check for Adapters ---
    print(f"🔍 Checking for adapter directory at: {adapter_path}")
    if not os.path.isdir(adapter_path):
        print(f"❌ ERROR: Adapter directory not found at '{adapter_path}'.")
        print("Please make sure you have unzipped your adapters in the same directory as this script.")
        return
    print("✅ Adapter directory found.")

    # --- 4. Load Tokenizer ---
    print(f"⬇️ Loading tokenizer for: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    print("✅ Tokenizer loaded.")

    # --- 5. Load Base Model ---
    # We load in bfloat16 for better performance on modern GPUs/MPS.
    # We are NOT using 4-bit quantization here to ensure compatibility with MPS.
    print(f"⬇️ Loading base model: {base_model_name} (this may take a moment and download ~6GB)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        device_map=device,          # Pin the model to the detected device
        trust_remote_code=True
    )
    print("✅ Base model loaded.")

    # --- 6. Fuse Adapters ---
    print(f"🔧 Fusing adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Adapters loaded.")
    print("🔄 Merging model and adapters for faster inference...")
    model = model.merge_and_unload()
    print("✅ Model merged and ready!")

    # --- 7. Run Inference (with One-Shot Prompting) ---

    GOLDEN_EXAMPLE = """
{
  "input": { "basicDetails": {"bedrooms": 1, "floors": 1, "totalArea": 500, "style": "Modern"}},
  "total_area": 500.0,
  "construction_cost": 75000.0,
  "levels": [
    {
      "level_number": 0,
      "rooms": [
        {
          "id": "living_room_1",
          "type": "living_room",
          "bounds": {"x": 0, "y": 0, "width": 15, "height": 20}
        },
        {
          "id": "kitchen_1",
          "type": "kitchen",
          "bounds": {"x": 15, "y": 0, "width": 10, "height": 12}
        }
      ]
    }
  ]
}
"""

    SYSTEM_PROMPT = f"""You are a world-class AI architect. Your task is to generate a single, complete JSON object representing a house plan based on the user's request. You must adhere strictly to the provided schema. Do not include any conversational text or markdown formatting.

### GOLDEN_EXAMPLE of the required JSON output format:
{GOLDEN_EXAMPLE}

Now, generate the complete JSON for the user request."""


    prompt_text = "Design a modern, single-story 2BHK house for a 30x40 feet plot with a total area of 1200 sqft."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]

    print("\n" + "="*50)
    print(f"🤖 Generating response for prompt: '{prompt_text}'")
    print("="*50 + "\n")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.9,
        temperature=0.6
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("---✨ Generated Plan ✨---")
    print(response)
    print("---✅ End of Plan ✅---")


if __name__ == "__main__":
    main()
