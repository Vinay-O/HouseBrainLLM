# Local Model Guide: Merging, Downloading, and Running HouseBrain

## 🎯 Overview

This guide provides the complete workflow for taking your trained model shards, merging them into a final powerful model, downloading it, and running it on your local machine for inference.

The process is broken down into three phases:
1.  **Phase 1: Merge Models in Colab** (Recommended for high RAM)
2.  **Phase 2: Download Final Model**
3.  **Phase 3: Run Inference Locally**

---

## Phase 1: Merge All Shards in Google Colab

This phase combines your original 600k shards and your new 200k v2 shards into a single, optimized "super model".

### **Step 1: Setup Colab Environment**

```python
# Clone repo and set up dependencies
%cd /content
!rm -rf HouseBrainLLM
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd /content/HouseBrainLLM

# This requires a kernel restart
!python - <<'PY'
from housebrain_colab_trainer import fix_dependencies
fix_dependencies()
print("✅ Dependencies fixed. Please RESTART the kernel now.")
PY

# Pull any recent changes after restart
%cd /content/HouseBrainLLM
!git pull
```

### **Step 2: Mount Google Drive and Load Shards**

Make sure your trained shards are in a `checkpoints` folder in your Drive.

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy all trained shards from Drive to the Colab environment for faster access
# This will take a few minutes
print("⏳ Copying all 8 model shards from Google Drive...")
!rsync -ah --progress /content/drive/MyDrive/HouseBrainLLM/checkpoints/ /content/checkpoints/
print("✅ All shards copied to Colab.")
```

### **Step 3: Merge Original 600k Shards**

This combines your first six shards into a single adapter.

```python
# Merge the original 6 shards (s01 to s06)
!python merge_models.py \
  --base "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --out "/content/housebrain_600k_merged" \
  --strategy "average" \
  --models \
    /content/checkpoints/hb_r1_s01 \
    /content/checkpoints/hb_r1_s02 \
    /content/checkpoints/hb_r1_s03 \
    /content/checkpoints/hb_r1_s04 \
    /content/checkpoints/hb_r1_s05 \
    /content/checkpoints/hb_r1_s06

print("✅ Merged original 600k shards.")
```

### **Step 4: Merge New 200k v2 Shards**

This combines your two new high-quality v2 shards.

```python
# Merge the two new v2 shards
!python merge_models.py \
  --base "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --out "/content/housebrain_200k_v2_merged" \
  --strategy "average" \
  --models \
    /content/checkpoints/hb_v2_s01 \
    /content/checkpoints/hb_v2_s02

print("✅ Merged new 200k v2 shards.")
```

### **Step 5: Final Super-Merge (600k + 200k v2)**

This is the final step where we combine the two merged models. We use a **weighted merge** to give more importance to your new, higher-quality v2 data.

```python
# Final weighted merge
# We give the new v2 model twice the weight (0.66) of the original (0.33)
!python merge_models.py \
  --base "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --out "/content/housebrain_800k_super_merged" \
  --strategy "weighted" \
  --models \
    /content/housebrain_600k_merged \
    /content/housebrain_200k_v2_merged \
  --weights 0.33 0.66

print("🏆 Created the final 800k Super Model!")
```

---

## Phase 2: Download the Final Model

Now, we'll compress the final model and download it to your local machine.

### **Step 1: Compress the Model**

```python
# Create a compressed tarball for easy downloading
print("⏳ Compressing the final model...")
!tar -czf /content/housebrain_800k_super_merged.tar.gz -C /content/ housebrain_800k_super_merged
print("✅ Model compressed.")
!ls -lh /content/housebrain_800k_super_merged.tar.gz
```

### **Step 2: Download the File**

```python
# Download the compressed model file
from google.colab import files
files.download('/content/housebrain_800k_super_merged.tar.gz')
```

---

## Phase 3: Run Inference Locally

Follow these steps on your local computer after downloading the model.

### **Step 1: Setup Local Python Environment**

You need `torch`, `transformers`, `peft`, and `accelerate`.

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv housebrain_env
source housebrain_env/bin/activate

# Install required packages
pip install torch torchvision torchaudio
pip install transformers peft accelerate bitsandbytes
```

### **Step 2: Create the Local Inference Script**

Create a file named `run_local_inference.py` and paste the following code into it.

```python
# run_local_inference.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# --- CONFIGURATION ---
# IMPORTANT: Update this path to where you extracted the downloaded model
MODEL_DIR = "./housebrain_800k_super_merged" 
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_FILE = "generated_plan_local.json"

def main():
    if not os.path.exists(MODEL_DIR):
        raise SystemExit(f"❌ Error: Model directory not found at '{MODEL_DIR}'. Please extract the downloaded .tar.gz file.")

    print("🧠 Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16, # Use bfloat16 for better performance
        device_map="auto" # Automatically use GPU if available
    )

    print(f"🚀 Loading HouseBrain adapter from: {MODEL_DIR}")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()

    print("✨ HouseBrain model loaded successfully!")

    # --- Define your architectural requirements ---
    requirements = {
      "project_name": "Modern Suburban Villa",
      "style": "modern_scandinavian", 
      "area_sqft": 2200, 
      "stories": 2, 
      "location": "suburban",
      "rooms": ["living_room", "dining_room", "kitchen", "master_bedroom", "guest_bedroom", "home_office", "laundry", "garage"],
      "features": ["open-concept kitchen", "large windows for natural light", "master suite with walk-in closet", "rooftop terrace"]
    }

    # --- Create the prompt ---
    sys_prompt = "You are HouseBrain. You must only output a single, complete, and strictly valid HouseBrain Plan v2 JSON that fulfills the user's requirements. Do not output any other text, explanation, or markdown."
    user_prompt = f"Generate a HouseBrain Plan v2 for the following requirements:\n{json.dumps(requirements, indent=2)}"
    
    # Using the ChatML format expected by the model
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    print("\n📝 Generating plan for the following requirements:")
    print(json.dumps(requirements, indent=2))
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    print("\n⏳ Generating... (This may take a minute)")
    with torch.no_grad():
       output_tokens = model.generate(
           **inputs, 
           max_new_tokens=4096, 
           temperature=0.6, 
           do_sample=True,
           pad_token_id=tokenizer.eos_token_id
       )
    
    generated_text = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # --- Extract and save the JSON output ---
    print("\n✅ Generation complete. Extracting JSON...")
    try:
        # Cleanly extract the JSON block
        start = generated_text.find("{")
        end = generated_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in the model's output.")
            
        plan_json_str = generated_text[start:end]
        plan_data = json.loads(plan_json_str)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(plan_data, f, indent=2)
            
        print(f"💾 Successfully saved generated plan to '{OUTPUT_FILE}'")
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error: Failed to parse JSON from model output. Error: {e}")
        print("\n--- Raw Model Output ---")
        print(generated_text)
        print("------------------------")

if __name__ == "__main__":
    main()
```

### **Step 3: Run the Local Script**

1.  **Unzip the Model**: Make sure you've extracted `housebrain_800k_super_merged.tar.gz` in the same directory as your script. You should have a folder named `housebrain_800k_super_merged`.
2.  **Run the Python script**:
    ```bash
    python run_local_inference.py
    ```

The script will load your powerful new model, generate a complex house plan based on the requirements, and save it as `generated_plan_local.json`. You can then validate this file using the v2 pipeline tools.
