#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm


def format_dataset_entry(example):
    """
    Formats a single dataset entry into the required prompt structure.
    This function will be applied to the dataset before training.
    """
    text = (
        f"### Instruction:\n{example['prompt']}\n\n"
        f"### Reasoning:\n{example['scratchpad']}\n\n"
        f"### Response:\n```json\n{example['output']}\n```"
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with specific architect-grade examples.")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct", help="The Hugging Face model ID to fine-tune.")
    parser.add_argument("--dataset_path", type=str, default="data/training/indian_residential", help="Path to the training dataset directory.")
    parser.add_argument("--output_dir", type=str, default="models/housebrain-deepseek-v1-finetuned", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization for memory saving.")
    args = parser.parse_args()

    print("Starting fine-tuning process with the following configuration:")
    print(f"  Model ID: {args.model_id}")
    print(f"  Dataset Path: {args.dataset_path}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Use 4-bit Quantization: {args.use_4bit}")


    # --- Step 1: Loading and Formatting Dataset ---
    # Construct a glob pattern to load all JSON files in the specified directory
    data_files = str(Path(args.dataset_path) / "*.json")
    dataset = load_dataset("json", data_files=data_files, split="train")

    # Apply formatting to the dataset to create the 'text' column
    dataset = dataset.map(format_dataset_entry)
    print(f"Loaded and formatted {len(dataset)} examples from {args.dataset_path}")
    print("\n--- Example formatted prompt ---")
    print(dataset[0]['text'])
    print("------------------------------------")


    # --- Step 2: Loading Tokenizer and Base Model ---
    print("\n--- Step 2: Loading Tokenizer and Base Model ---")
    
    bnb_config = None
    if args.use_4bit:
        print("Using 4-bit quantization (BitsAndBytes) to reduce memory usage.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # Set a padding token if one isn't already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically map model layers to available devices
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    # --- 3. Configure LoRA ---
    print("\n--- Step 3: Configuring LoRA (Low-Rank Adaptation) ---")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] # Specific to Llama/Mistral/DeepSeek arch
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configured. Trainable parameters:")
    model.print_trainable_parameters()

    # --- 4. Set Training Arguments ---
    print("\n--- Step 4: Setting Training Arguments ---")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=2,
        fp16=False, # Set to True if not using 4-bit
        bf16=True if args.use_4bit else False, # Required for 4-bit
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        report_to="tensorboard"
    )

    print("\n--- Step 5: Initializing SFTTrainer and Starting Training ---")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=4096, # Increased for potentially longer examples
        tokenizer=tokenizer,
        args=training_arguments,
    )

    print("Starting training... This may take a significant amount of time.")
    trainer.train()

    # --- 6. Save Final Model ---
    print("\n--- Step 6: Saving Final Fine-Tuned Model ---")
    final_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    print(f"✅ Training complete. Final model saved to: {final_save_path}")

if __name__ == "__main__":
    main()
