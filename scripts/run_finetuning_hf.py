import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig

def main(args):
    # --- 1. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Set padding token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load and Prepare Datasets ---
    print("Loading and preparing datasets...")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")

    def formatting_func(examples):
        texts = []
        for msg_list in examples["messages"]:
            prompt = ""
            response = ""
            for msg in msg_list:
                if msg["role"] == "user":
                    prompt = msg["content"]
                elif msg["role"] == "assistant":
                    response = msg["content"]
            
            if prompt and response:
                text = f"""You are a world-class AI architect. Generate a detailed and accurate JSON representation of a house floor plan based on the user's request.

### Request:
{prompt}

### Response:
{response}"""
                texts.append(text)
        return {"text": texts}

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    train_dataset = train_dataset.map(formatting_func, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(formatting_func, batched=True, remove_columns=val_dataset.column_names)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # --- 3. Configure LoRA ---
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Set Up Trainer ---
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_strategy="steps",
        save_steps=50,
        use_mps_device=torch.backends.mps.is_available(), # Use Apple Silicon GPU
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # --- 5. Start Training ---
    print("Starting training...")
    trainer.train()

    # --- 6. Save Model ---
    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    print("Fine-tuning complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with Hugging Face and PEFT.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="The Hugging Face model to fine-tune.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="finetune_dataset_train.jsonl",
        help="Path to the training data.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="finetune_dataset_val.jsonl",
        help="Path to the validation data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hf_finetune_output",
        help="Directory to save the training outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    
    args = parser.parse_args()
    main(args)
