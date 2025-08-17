#!/usr/bin/env python3
"""
HouseBrain Colab Pro+ Training Script (R1 Super-Quality Dataset)
Optimized for DeepSeek-R1-Distill-Qwen-7B with 1M super-quality reasoning dataset

Dataset Features:
- 1M samples with 74% Geometric_Construction focus
- 30K+ characters per geometric sample
- 2D/3D generation ready with exact coordinates
- India-specific (60%) with NBC 2016 compliance
- Quality threshold: 0.85
- 43GB total dataset size
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

# Configuration
class TrainingConfig:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    SEQUENCE_LENGTH = 4096  # Increased for better reasoning
    LORA_R = 64  # Increased for better performance
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.1
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 1  # Reduced for memory efficiency
    GRADIENT_ACCUMULATION = 16  # Increased for effective batch size
    WARMUP_STEPS = 1000
    MAX_STEPS = 100000  # For 1M dataset
    SAVE_STEPS = 2000  # More frequent saves
    EVAL_STEPS = 1000  # More frequent evaluation
    LOGGING_STEPS = 10
    OUTPUT_DIR = "housebrain-r1-super-trained"
    LOG_FILE = "training_log_r1_super.txt"
    
    @classmethod
    def detect_dataset(cls):
        """Auto-detect dataset directory and adjust parameters"""
        possible_dirs = [
            "housebrain_dataset_r1_super_10k_aug",
            "housebrain_dataset_r1_super_1M_aug_v1_1", 
            "housebrain_dataset_r1_super_1M"
        ]
        
        for dir_name in possible_dirs:
            if Path(dir_name).exists():
                cls.DATASET_PATH = dir_name
                print(f"🎯 Detected dataset: {dir_name}")
                
                # Adjust parameters for 10K test dataset
                if "10k" in dir_name:
                    cls.MAX_STEPS = 1000  # Fewer steps for test
                    cls.SAVE_STEPS = 200
                    cls.EVAL_STEPS = 100
                    print("📊 Using 10K test configuration")
                else:
                    print("📊 Using 1M full configuration")
                
                return dir_name
        
        raise FileNotFoundError("❌ No dataset directory found. Please run colab_setup.py first")

SYSTEM_PROMPT = (
    "You are HouseBrain, an expert architectural AI with advanced reasoning capabilities. "
    "Always produce strictly valid JSON that complies with our schema. "
    "Ensure NBC 2016 (India) and general code compliance, and provide detailed step-by-step reasoning. "
    "Focus on complex architectural problem-solving including structural engineering, sustainability, and smart home integration. "
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
        with open("training_metrics_r1_super.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

def setup_environment():
    """Setup GPU environment and check resources"""
    print("🔧 Setting up environment...")
    
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
    """Load and prepare super-quality dataset with chat formatting"""
    print("📊 Loading super-quality dataset...")
    dataset_path = Path(config.DATASET_PATH)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")
    
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
    """Create model and tokenizer with R1 optimizations"""
    print("🤖 Loading DeepSeek-R1-Distill-Qwen-7B model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=False  # Use full precision for better quality
    )
    
    # LoRA configuration optimized for R1
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    config = TrainingConfig()
    logger = SimpleLogger(config.LOG_FILE)
    
    try:
        # Setup
        gpu_info = setup_environment()
        logger.log(f"🚀 Starting HouseBrain R1 Super-Quality training")
        logger.log(f"💻 GPU: {gpu_info['name']} ({gpu_info['memory']:.1f}GB)")
        
        # Detect dataset and adjust config
        config.detect_dataset()
        logger.log(f"📊 Dataset: {config.DATASET_PATH}")
        
        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer(config)
        logger.log("✅ Model and tokenizer loaded successfully")
        
        # Load dataset
        train_dataset, eval_dataset = load_dataset(config, tokenizer)
        logger.log(f"📊 Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} validation samples")
        
        # Training arguments optimized for R1
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
            report_to=None,
            logging_dir="./logs_r1_super",
            save_total_limit=3,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_num_workers=2,
            group_by_length=True,  # Optimize for variable sequence lengths
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
                        self.logger.log_metrics(state.global_step, train_loss=train_loss, eval_loss=eval_loss, lr=lr)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            callbacks=[LoggingCallback(logger)]
        )
        
        # Start training
        logger.log("🚀 Starting training...")
        logger.log(f"📊 Config: LR={config.LEARNING_RATE}, Batch={config.BATCH_SIZE}, SeqLen={config.SEQUENCE_LENGTH}")
        logger.log(f"🎯 Target: {config.MAX_STEPS} steps")
        
        trainer.train()
        
        # Save final model
        logger.log("✅ Training completed successfully!")
        trainer.save_model(f"{config.OUTPUT_DIR}/final")
        tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final")
        logger.log(f"💾 Final model saved to {config.OUTPUT_DIR}/final")
        
        # Save metrics and summary
        logger.save_metrics()
        
        final_loss = trainer.state.log_history[-1].get('loss', 'N/A')
        best_eval_loss = trainer.state.best_metric
        total_steps = trainer.state.global_step
        
        logger.log("📈 Training Summary:")
        logger.log(f"   - Total steps: {total_steps}")
        logger.log(f"   - Final train loss: {final_loss}")
        logger.log(f"   - Best eval loss: {best_eval_loss}")
        logger.log(f"   - Training time: {time.time() - trainer.state.start_time:.1f}s")
        
        # Performance metrics
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            losses = [log.get('loss', 0) for log in trainer.state.log_history if log.get('loss')]
            if losses:
                avg_loss = sum(losses) / len(losses)
                logger.log(f"   - Average train loss: {avg_loss:.4f}")
        
        logger.log("🎉 All done! Check the logs and saved model.")
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        logger.log(error_msg)
        raise

if __name__ == "__main__":
    main()
