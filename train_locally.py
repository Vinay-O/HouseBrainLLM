import argparse
import logging
import json
import glob
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_prepare_dataset(dataset_path: Path) -> Dataset:
    """Loads and formats the JSON data from the specified path."""
    logger.info(f"Loading data from: {dataset_path}")
    json_files = glob.glob(f"{dataset_path}/**/*.json", recursive=True)
    
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path}. Please check the path.")

    def format_data(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Make the prompt extraction more robust
            prompt = data.get("input", {}).get("basicDetails", {}).get("prompt")
            if not prompt:
                prompt = data.get("input", {}).get("plot", {}).get("prompt")

            if not prompt: return None
            answer = json.dumps(data, indent=2)
            
            formatted_text = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
            return {"text": formatted_text}
        except Exception as e:
            logger.warning(f"Skipping file {file_path} due to error: {e}")
            return None

    data_list = [format_data(f) for f in json_files]
    data_list = [item for item in data_list if item is not None]

    if not data_list:
        raise ValueError("No valid, formattable data found in the dataset path.")

    logger.info(f"Successfully loaded and formatted {len(data_list)} examples.")
    return Dataset.from_list(data_list)

def run_training(dataset_path: str, model_id: str, output_path: str):
    """Main function to run the fine-tuning process."""
    
    # --- 1. Load Dataset ---
    dataset = load_and_prepare_dataset(Path(dataset_path))

    # --- 2. Load Model and Tokenizer ---
    logger.info(f"Loading base model: {model_id}")
    
    # Note: On CPU/Mac, BitsAndBytes might be slow or unstable.
    # It's heavily optimized for NVIDIA GPUs.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", # This will likely default to CPU on a Mac
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    logger.info("✅ Model and tokenizer loaded.")

    # --- 3. Configure LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    logger.info("✅ LoRA configured.")

    # --- 4. Run Training ---
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        # fp16=True should only be used with CUDA
        fp16=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=4096,
        args=training_args,
        packing=True,
    )
    
    logger.info("Starting training... This may take a very long time on a local machine.")
    trainer.train()
    logger.info("✅ Training complete!")

    # --- 5. Save Final Adapter ---
    final_path = Path(output_path) / "final_adapter"
    logger.info(f"Saving final model adapter to {final_path}")
    trainer.save_model(str(final_path))
    logger.info(f"🎉 All done! Your fine-tuned adapter is saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Run HouseBrain Fine-Tuning Locally")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the 'gold_standard' directory containing the JSON dataset.")
    parser.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="The Hugging Face model ID to fine-tune.")
    parser.add_argument("--output-path", type=str, default="output/local_training_run", help="Directory to save the training checkpoints and final adapter.")
    args = parser.parse_args()

    run_training(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    main()
