#!/usr/bin/env python3
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import json

# Fine-tuning script for HouseBrain LLM

def format_dataset_entry(entry):
    """Converts a dataset entry into the required prompt-completion format."""
    prompt = entry["prompt"]
    completion = json.dumps(entry["output"], indent=2)
    # This is a standard format for many instruction-following models
    return {"text": f"<s>[INST] {prompt} [/INST]\n{completion} </s>"}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for HouseBrain.")
    parser.add_argument("--dataset-path", type=str, default="data/training/gold_standard", help="Path to the training dataset.")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Base model ID from Hugging Face.")
    parser.add_argument("--new-model", type=str, default="housebrain-llama2-7b-v0.1", help="Name for the new fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    args = parser.parse_args()

    # --- 1. Load Dataset ---
    # Use glob to load all JSON files in the directory
    dataset = load_dataset("json", data_files=os.path.join(args.dataset_path, "*.json"), split="train")
    
    # Format the dataset
    formatted_dataset = dataset.map(format_dataset_entry, remove_columns=dataset.column_names)

    # --- 2. Configure Quantization (for memory efficiency) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 3. Load Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto", # Automatically use GPU if available
        trust_remote_code=True
    )
    model.config.use_cache = False

    # --- 4. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token

    # --- 5. Configure LoRA (Parameter-Efficient Fine-Tuning) ---
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 6. Set Training Arguments ---
    training_arguments = TrainingArguments(
        output_dir=f"./models/{args.new_model}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # --- 7. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=4096, # Set a sequence length that can handle the schema + output
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # --- 8. Start Fine-Tuning ---
    trainer.train()

    # --- 9. Save the Fine-Tuned Model ---
    trainer.model.save_pretrained(f"models/{args.new_model}-final")
    
    print(f"✅ Fine-tuning complete. Model saved to models/{args.new_model}-final")

if __name__ == "__main__":
    main()
