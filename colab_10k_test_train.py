#!/usr/bin/env python3
"""
HouseBrain 10K Test Training Script
Optimized for quick validation on 10K augmented dataset

Purpose: Validate training pipeline before full 1M training
Dataset: housebrain_dataset_r1_super_10k_aug (9K train, 1K validation)
Time: ~30-60 minutes on Colab Pro+
"""

import os
import json
import torch
import time
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, 
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from datetime import datetime

# Configuration for 10K Test
class TestConfig:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Original choice with reasoning
    DATASET_PATH = "housebrain_dataset_r1_super_10k_aug"
    
    # Test-optimized parameters
    SEQUENCE_LENGTH = 4096
    LORA_R = 32  # Reduced for faster training
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.1
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8  # Reduced for faster iteration
    WARMUP_STEPS = 100  # Reduced for test
    MAX_STEPS = 1000  # Quick test run
    SAVE_STEPS = 200
    EVAL_STEPS = 100
    LOGGING_STEPS = 10
    
    OUTPUT_DIR = "housebrain-10k-test-trained"
    LOG_FILE = "training_log_10k_test.txt"

SYSTEM_PROMPT = (
    "You are HouseBrain, an expert architectural AI with advanced reasoning capabilities. "
    "Always produce strictly valid JSON that complies with our schema. "
    "Ensure NBC 2016 (India) and general code compliance, and provide detailed step-by-step reasoning. "
    "Specialize in geometric construction with exact coordinates, spatial floor planning, and 2D/3D generation capabilities."
)

class SimpleLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = []
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def log_metrics(self, step: int, **kwargs):
        metric_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.metrics.append(metric_entry)
        
    def save_metrics(self):
        with open("training_metrics_10k_test.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

def setup_environment():
    """Setup GPU environment and check resources"""
    print("🔧 Setting up environment for 10K test...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available. Please use GPU runtime.")
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
    
    # Memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    return {"name": gpu_name, "memory": gpu_memory}

def build_chat_and_labels(tokenizer, user_text: str, assistant_text: str, max_len: int):
    """Builds chat-style input and masks labels for assistant-only training."""
    system_ids = tokenizer.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n", add_special_tokens=False)
    user_ids = tokenizer.encode(f"<|im_start|>user\n{user_text}<|im_end|>\n", add_special_tokens=False)
    assistant_prefix_ids = tokenizer.encode(f"<|im_start|>assistant\n", add_special_tokens=False)
    assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    full_input_ids = system_ids + user_ids + assistant_prefix_ids + assistant_ids
    labels = ([-100] * (len(system_ids) + len(user_ids) + len(assistant_prefix_ids))) + assistant_ids

    if eos_id is not None:
        full_input_ids.append(eos_id)
        labels.append(eos_id)

    # Truncate if too long
    if len(full_input_ids) > max_len:
        full_input_ids = full_input_ids[:max_len]
        labels = labels[:max_len]

    return {"input_ids": full_input_ids, "labels": labels}

def load_dataset(config, tokenizer):
    """Load and prepare 10K test dataset with chat formatting"""
    print("📊 Loading 10K test dataset...")
    dataset_path = Path(config.DATASET_PATH)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ 10K dataset not found: {dataset_path}")
    
    # Load from sharded structure
    train_files = []
    validation_files = []
    
    # Find all shards
    train_shards = list((dataset_path / "train").glob("shard_*"))
    val_shards = list((dataset_path / "validation").glob("shard_*"))
    
    for shard in train_shards:
        train_files.extend(list(shard.glob("*.json")))
    for shard in val_shards:
        validation_files.extend(list(shard.glob("*.json")))
    
    print(f"📈 Training samples: {len(train_files):,}")
    print(f"📉 Validation samples: {len(validation_files):,}")
    
    def load_and_format_samples(file_list):
        samples = []
        for file in tqdm(file_list, desc="Loading samples"):
            try:
                with open(file, 'r') as f:
                    sample = json.load(f)
                    user_text = json.dumps(sample["input"], separators=(",", ":"))
                    assistant_text = json.dumps(sample["output"], separators=(",", ":"))
                    samples.append({"user_text": user_text, "assistant_text": assistant_text})
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
                continue
        return samples

    train_raw = load_and_format_samples(train_files)
    val_raw = load_and_format_samples(validation_files)
    
    if not train_raw or not val_raw:
        raise RuntimeError("❌ No valid samples found in dataset")

    train_dataset = Dataset.from_list(train_raw)
    eval_dataset = Dataset.from_list(val_raw)

    # Tokenize and apply masking
    print("🔧 Tokenizing and applying chat formatting...")
    train_dataset = train_dataset.map(
        lambda examples: build_chat_and_labels(tokenizer, examples["user_text"], examples["assistant_text"], config.SEQUENCE_LENGTH),
        batched=False, remove_columns=["user_text", "assistant_text"]
    )
    eval_dataset = eval_dataset.map(
        lambda examples: build_chat_and_labels(tokenizer, examples["user_text"], examples["assistant_text"], config.SEQUENCE_LENGTH),
        batched=False, remove_columns=["user_text", "assistant_text"]
    )
    
    return train_dataset, eval_dataset

def create_model_and_tokenizer(config):
    """Create model and tokenizer with test optimizations"""
    print("🤖 Loading DeepSeek-R1-Distill-Qwen-7B model and tokenizer...")
    
    # Fix transformers version compatibility issue
    print("📦 Fixing transformers version compatibility...")
    try:
        import subprocess
        # Install a compatible version that works with the model
        subprocess.run(["pip", "install", "transformers==4.36.0", "--force-reinstall"], check=True)
        print("✅ Transformers version fixed successfully")
    except Exception as e:
        print(f"⚠️ Could not fix transformers: {e}")
    
    # Try loading tokenizer with error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    except Exception as e:
        print(f"⚠️ Error loading tokenizer: {e}")
        print("🔄 Trying alternative tokenizer loading...")
        # Try with trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations for test and error handling
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # Disable for training
        )
    except KeyError as e:
        if "qwen2" in str(e):
            print("⚠️ KeyError with qwen2 detected. Trying alternative loading method...")
            # Try loading with different approach
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False,
                ignore_mismatched_sizes=True
            )
        else:
            raise e
    
    # LoRA configuration for test
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    """Main training function for 10K test"""
    config = TestConfig()
    logger = SimpleLogger(config.LOG_FILE)
    
    try:
        # Setup environment
        gpu_info = setup_environment()
        
        # Create output directory
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        
        # Load model and tokenizer
        model, tokenizer = create_model_and_tokenizer(config)
        
        # Load dataset
        train_dataset, eval_dataset = load_dataset(config, tokenizer)
        
        # Training arguments for test
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
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
            bf16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for test
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )
        
        # Start training
        logger.log(f"🚀 Starting HouseBrain 10K Test Training")
        logger.log(f"💻 GPU: {gpu_info['name']} ({gpu_info['memory']:.1f}GB)")
        logger.log(f"📊 Dataset: {config.DATASET_PATH}")
        logger.log(f"📊 Config: LR={config.LEARNING_RATE}, Batch={config.BATCH_SIZE}, SeqLen={config.SEQUENCE_LENGTH}, GradAccum={config.GRADIENT_ACCUMULATION}")
        logger.log(f"🎯 Target: {config.MAX_STEPS} steps (quick test)")
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # Save final model
        trainer.save_model(f"{config.OUTPUT_DIR}/final")
        tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final")
        
        # Log completion
        training_time = (end_time - start_time) / 3600  # hours
        logger.log(f"✅ Training completed in {training_time:.2f} hours")
        logger.log(f"💾 Model saved to {config.OUTPUT_DIR}/final")
        
        # Save metrics
        logger.save_metrics()
        
        print(f"\n🎉 10K Test Training Complete!")
        print(f"⏱️  Time: {training_time:.2f} hours")
        print(f"💾 Model: {config.OUTPUT_DIR}/final")
        print(f"📊 Metrics: training_metrics_10k_test.json")
        print(f"\n✅ If test successful, proceed with full 1M training!")
        
    except Exception as e:
        logger.log(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
