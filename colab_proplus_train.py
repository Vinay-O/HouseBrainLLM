#!/usr/bin/env python3
"""
HouseBrain Colab Pro+ Training Script
Optimized for A100/V100 with BF16, longer sequences, and advanced features
"""

import os
import json
import torch
import wandb
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np

# Configuration
class TrainingConfig:
    # Model settings
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
    
    # A100 optimized settings
    SEQUENCE_LENGTH = 1024  # Can go up to 1536 on A100
    LORA_R = 16  # Higher rank for better quality
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Training settings
    LEARNING_RATE = 1.5e-4
    BATCH_SIZE = 4  # Effective batch size via gradient accumulation
    GRADIENT_ACCUMULATION_STEPS = 32  # Effective batch = 128
    WARMUP_STEPS = 100
    MAX_STEPS = 5000
    EVAL_STEPS = 500
    SAVE_STEPS = 1000
    
    # Quality settings
    GRADIENT_CLIPPING = 0.5
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 10
    
    # Paths
    DATASET_PATH = "/content/housebrain_dataset_v5_425k"
    OUTPUT_DIR = "/content/drive/MyDrive/housebrain_models"
    WANDB_PROJECT = "housebrain-proplus"

def setup_environment():
    """Setup environment for A100 training"""
    print("🔧 Setting up Colab Pro+ environment...")
    
    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # GPU verification
    assert torch.cuda.is_available(), "❌ GPU required!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"📊 Memory: {gpu_memory:.1f} GB")
    print(f"🔧 Compute Capability: {torch.cuda.get_device_capability()}")
    
    return device

def load_and_prepare_model(config):
    """Load model with A100 optimizations"""
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("📥 Loading model with A100 optimizations...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,  # BF16 for A100
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Enable optimizations
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    print("🔧 Adding LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def load_dataset(config):
    """Load and prepare dataset"""
    print("📂 Loading dataset...")
    
    dataset_path = Path(config.DATASET_PATH)
    train_files = list((dataset_path / "train").glob("*.json"))
    val_files = list((dataset_path / "validation").glob("*.json")) if (dataset_path / "validation").exists() else train_files[:1000]
    
    print(f"📊 Found {len(train_files)} training files")
    print(f"📊 Found {len(val_files)} validation files")
    
    def load_samples(files, max_files=None):
        samples = []
        for file in tqdm(files[:max_files], desc="Loading files"):
            try:
                with open(file, 'r') as f:
                    sample = json.load(f)
                
                # Format for training
                input_text = json.dumps(sample["input"], indent=2)
                output_text = json.dumps(sample["output"], indent=2)
                full_text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
                
                samples.append({"text": full_text})
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
                continue
        
        return samples
    
    train_samples = load_samples(train_files, max_files=100000)  # Start with 100K for testing
    val_samples = load_samples(val_files, max_files=5000)
    
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    return train_dataset, val_dataset

def tokenize_function(examples, tokenizer, config):
    """Tokenize function with proper truncation"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=config.SEQUENCE_LENGTH,
        return_tensors=None,
    )

def setup_training_args(config):
    """Setup training arguments for A100"""
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # Model settings
        model_max_length=config.SEQUENCE_LENGTH,
        
        # Training settings
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        max_steps=config.MAX_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        
        # Optimization
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.GRADIENT_CLIPPING,
        lr_scheduler_type="cosine",
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        
        # Logging
        logging_steps=config.LOGGING_STEPS,
        report_to="wandb",
        
        # Quality
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # A100 optimizations
        fp16=False,  # Use BF16 instead
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )

def compute_metrics(eval_preds, tokenizer):
    """Compute evaluation metrics"""
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=-1)
    
    # Calculate perplexity
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(predictions), torch.tensor(labels))
    perplexity = torch.exp(loss)
    
    return {
        "perplexity": perplexity.item(),
        "eval_loss": loss.item(),
    }

def main():
    """Main training function"""
    print("🚀 Starting HouseBrain Colab Pro+ Training")
    print("=" * 60)
    
    # Setup
    config = TrainingConfig()
    device = setup_environment()
    
    # Initialize wandb
    wandb.init(
        project=config.WANDB_PROJECT,
        config=vars(config),
        name=f"housebrain-a100-{config.LORA_R}-{config.SEQUENCE_LENGTH}"
    )
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(config)
    
    # Load dataset
    train_dataset, val_dataset = load_dataset(config)
    
    # Tokenize datasets
    print("🔧 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Setup training
    training_args = setup_training_args(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    print("🏗️ Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving final model...")
    final_output_dir = f"{config.OUTPUT_DIR}/final"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Final evaluation
    print("📊 Final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"Final eval loss: {final_metrics['eval_loss']:.4f}")
    print(f"Final perplexity: {final_metrics['perplexity']:.4f}")
    
    wandb.finish()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
