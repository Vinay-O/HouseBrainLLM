import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def main():
    # --- Configuration ---
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path = "./senior_architect_adapters"

    # --- Load Tokenizer and Base Model ---
    print("🚀 Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Use bfloat16 for MPS
        trust_remote_code=True,
    )

    # --- Load and Merge LoRA Adapters ---
    print(f"🧠 Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("✅ Merging adapters into the base model...")
    model = model.merge_and_unload()
    
    print("🤖 Senior Architect model is ready!")

    # --- Prepare a Test Case ---
    # This is a sample flawed plan, similar to what the model was trained on.
    # The flaw: There are no doors, making the house unusable.
    user_prompt = "Design a small, modern 2-bedroom, 1-bathroom single-story house on a rectangular plot."
    
    flawed_plan = {
      "input": {"basicDetails": {"bedrooms": 2, "floors": 1, "totalArea": 840, "style": "Modern"}},
      "total_area": 840.0,
      "construction_cost": 150000.0,
      "levels": [{
          "level_number": 0,
          "rooms": [
            {"id": "living_room", "type": "living_room", "bounds": {"x": 0, "y": 0, "width": 15, "height": 20}, "doors": [], "windows": []},
            {"id": "kitchen", "type": "kitchen", "bounds": {"x": 15, "y": 0, "width": 10, "height": 12}, "doors": [], "windows": []},
            {"id": "bedroom_1", "type": "bedroom", "bounds": {"x": 0, "y": 20, "width": 15, "height": 14}, "doors": [], "windows": []},
            {"id": "bathroom_1", "type": "bathroom", "bounds": {"x": 15, "y": 12, "width": 10, "height": 8}, "doors": [], "windows": []}
          ]
      }],
      "__flaw_applied__": "remove_all_doors"
    }

    # Format the input exactly as it was during training
    user_content = (
        "Please act as a senior architect. The following JSON house plan was generated for a user request, but it contains one or more significant architectural, geometric, or functional flaws. Your task is to identify and fix the issues to make the plan compliant with the original request and sound architectural principles. Here is the original user request:\n\n"
        f"--- USER REQUEST ---\n{user_prompt}\n\n"
        "--- FLAWED JSON PLAN ---\n"
        f"{json.dumps(flawed_plan, indent=2)}"
    )
    
    system_prompt = "You are an expert architect who reviews and corrects flawed house plans."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # --- Run Inference ---
    print("\n\n💬 Sending request to Senior Architect for correction...")
    
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )
    
    response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # --- Display the Result ---
    print("\n\n✅ Senior Architect's Corrected Plan:\n" + "="*50)
    print(response_text)
    print("="*50)

    # Attempt to parse the JSON to validate it
    try:
        json.loads(response_text)
        print("\n🎉 JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"\n❌ Warning: The output is not valid JSON. Error: {e}")


if __name__ == "__main__":
    main()
