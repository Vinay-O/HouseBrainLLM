#!/usr/bin/env python3
"""
HouseBrain Colab Pro+ Training Script (Simple Version)
No WandB required - uses built-in logging and monitoring
"""

import os
import json
import torch
import time
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from datetime import datetime

# Configuration
class TrainingConfig:
    # Model settings
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
    
    # A100 optimized settings
    SEQUENCE_LENGTH = 1024
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Training settings
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 8
    WARMUP_STEPS = 100
    MAX_STEPS = 50000
    SAVE_STEPS = 1000
    EVAL_STEPS = 500
    LOGGING_STEPS = 10
    
    # Dataset
    DATASET_PATH = "housebrain_dataset_v6_1M"
    
    # Output
    OUTPUT_DIR = "housebrain-trained-model"
    
    # Logging
    LOG_FILE = "training_log.txt"

class SimpleLogger:
    """Simple logging without external dependencies"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = time.time()
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "gpu_memory": []
        }
        
    def log(self, message, print_to_console=True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        
        if print_to_console:
            print(log_entry)
    
    def log_metrics(self, step, train_loss=None, eval_loss=None, lr=None):
        """Log training metrics"""
        if train_loss is not None:
            self.metrics["train_loss"].append((step, train_loss))
        if eval_loss is not None:
            self.metrics["eval_loss"].append((step, eval_loss))
        if lr is not None:
            self.metrics["learning_rate"].append((step, lr))
        
        # Get GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            self.metrics["gpu_memory"].append((step, gpu_memory))
        
        # Log summary
        self.log(f"Step {step}: Train Loss={train_loss:.4f}, Eval Loss={eval_loss:.4f}, LR={lr:.2e}, GPU={gpu_memory:.2f}GB")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        with open("training_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        self.log(f"Metrics saved to training_metrics.json")

def setup_environment():
    """Setup training environment"""
    print("🔧 Setting up environment...")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! Please enable GPU in Colab.")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    return gpu_memory

def load_dataset(config):
    """Load and prepare dataset"""
    print("📊 Loading dataset...")
    
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")
    
    # Load training files
    train_files = list((dataset_path / "train").glob("*.json"))
    validation_files = list((dataset_path / "validation").glob("*.json"))
    
    print(f"📈 Training samples: {len(train_files):,}")
    print(f"📉 Validation samples: {len(validation_files):,}")
    
    def load_samples(file_list):
        samples = []
        for file in tqdm(file_list, desc="Loading samples"):
            with open(file, 'r') as f:
                sample = json.load(f)
                # Format for training
                input_text = json.dumps(sample["input"], separators=(",", ":"))
                output_text = json.dumps(sample["output"], separators=(",", ":"))
                full_text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
                samples.append({"text": full_text})
        return samples
    
    train_samples = load_samples(train_files)
    validation_samples = load_samples(validation_files)
    
    return Dataset.from_list(train_samples), Dataset.from_list(validation_samples)

def create_model_and_tokenizer(config):
    """Create model and tokenizer"""
    print("🤖 Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    """Main training function"""
    config = TrainingConfig()
    logger = SimpleLogger(config.LOG_FILE)
    
    # Setup
    gpu_memory = setup_environment()
    logger.log(f"Starting HouseBrain training with {gpu_memory:.1f}GB GPU")
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset(config)
    logger.log(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} validation samples")
    
    # Create model
    model, tokenizer = create_model_and_tokenizer(config)
    logger.log("Model and tokenizer loaded successfully")
    
    # Tokenize datasets
    def tokenize_train(examples):
        return tokenize_function(examples, tokenizer, config.SEQUENCE_LENGTH)
    
    def tokenize_eval(examples):
        return tokenize_function(examples, tokenizer, config.SEQUENCE_LENGTH)
    
    train_dataset = train_dataset.map(tokenize_train, batched=True)
    eval_dataset = eval_dataset.map(tokenize_eval, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=None,
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable WandB
        logging_dir="./logs",
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    
    # Custom callback for logging
    class LoggingCallback:
        def __init__(self, logger):
            self.logger = logger
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                train_loss = logs.get("loss")
                eval_loss = logs.get("eval_loss")
                lr = logs.get("learning_rate")
                
                if train_loss is not None or eval_loss is not None:
                    self.logger.log_metrics(
                        state.global_step,
                        train_loss=train_loss,
                        eval_loss=eval_loss,
                        lr=lr
                    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback(logger)]
    )
    
    # Start training
    logger.log("🚀 Starting training...")
    logger.log(f"📊 Config: LR={config.LEARNING_RATE}, Batch={config.BATCH_SIZE}, SeqLen={config.SEQUENCE_LENGTH}")
    
    try:
        trainer.train()
        logger.log("✅ Training completed successfully!")
        
        # Save final model
        trainer.save_model(f"{config.OUTPUT_DIR}/final")
        tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final")
        logger.log(f"💾 Final model saved to {config.OUTPUT_DIR}/final")
        
        # Save metrics
        logger.save_metrics()
        
        # Print summary
        logger.log("📈 Training Summary:")
        logger.log(f"   - Total steps: {trainer.state.global_step}")
        logger.log(f"   - Final train loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")
        logger.log(f"   - Best eval loss: {trainer.state.best_metric}")
        
    except Exception as e:
        logger.log(f"❌ Training failed: {str(e)}")
        raise
    
    logger.log("🎉 All done! Check the logs and saved model.")

if __name__ == "__main__":
    main()
