#!/usr/bin/env python3
"""
HouseBrain Unified Colab Training Script
Clean, reliable training script for both 10K test and full training

Features:
- Automatic dependency fixing
- GPU detection and optimization
- Progress monitoring
- Checkpoint management
- Error recovery
"""

import os
import sys
import json
import torch
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Fix dependencies first
def fix_dependencies():
    """Fix all dependency conflicts and block TF/JAX imports that break NumPy<2."""
    print("🔧 Fixing dependencies...")
    try:
        # Uninstall TensorFlow/JAX to prevent transformers TF integrations from importing
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                        "tensorflow", "tensorflow-cpu", "tensorflow-gpu",
                        "tensorflow-decision-forests", "jax", "jaxlib",
                        "orbax-checkpoint", "optax", "flax", "chex"],
                       check=False, capture_output=True)

        # Install compatible libs without touching preinstalled torch/transformers in Colab
        # Keep NumPy < 2 to avoid TF/JAX and some ecosystem issues
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir",
            "numpy==1.26.4", "scipy==1.11.4", "contourpy==1.2.1",
            "accelerate>=0.27.0", "peft>=0.8.0", "datasets>=2.16.0", "bitsandbytes>=0.43.1", "tqdm"
        ], check=True, capture_output=True)

        # Inhibit TF integration in transformers
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        # Disable Weights & Biases auto-logging
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        print("✅ Dependencies fixed!")
        return True
    except Exception as e:
        print(f"❌ Dependency fix failed: {e}")
        return False

# Lazy import holders
AutoTokenizer = None
AutoModelForCausalLM = None
Trainer = None
DataCollatorForLanguageModeling = None
TrainingArguments = None
BitsAndBytesConfig = None
LoraConfig = None
get_peft_model = None
prepare_model_for_kbit_training = None
Dataset = None
np = None
tqdm = None

def _ensure_training_libs_loaded():
    global AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling
    global TrainingArguments, BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    global Dataset, np, tqdm

    if AutoTokenizer is not None:
        return

    # Ensure env flag carried into current process imports
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    from transformers import (
        AutoTokenizer as _AutoTokenizer,
        AutoModelForCausalLM as _AutoModelForCausalLM,
        Trainer as _Trainer,
        DataCollatorForLanguageModeling as _DCFLM,
    )
    AutoTokenizer = _AutoTokenizer
    AutoModelForCausalLM = _AutoModelForCausalLM
    Trainer = _Trainer
    DataCollatorForLanguageModeling = _DCFLM

    # Guarded TrainingArguments import
    try:
        from transformers import TrainingArguments as _TrainingArguments
        TrainingArguments = _TrainingArguments
    except Exception:
        class _FallbackTrainingArguments:  # type: ignore
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        TrainingArguments = _FallbackTrainingArguments

    try:
        from transformers import BitsAndBytesConfig as _BitsAndBytesConfig
        BitsAndBytesConfig = _BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None

    from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model, prepare_model_for_kbit_training as _prepare
    LoraConfig = _LoraConfig
    get_peft_model = _get_peft_model
    prepare_model_for_kbit_training = _prepare

    from datasets import Dataset as _Dataset
    Dataset = _Dataset

    import numpy as _np
    np = _np

    from tqdm.auto import tqdm as _tqdm
    tqdm = _tqdm

@dataclass
class TrainingConfig:
    """Unified training configuration"""
    # Model settings - Choose your reasoning model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Best reasoning
    # Alternative models with reasoning:
    # "Qwen/Qwen2.5-7B-Instruct" - Excellent reasoning
    # "meta-llama/Meta-Llama-3.1-8B-Instruct" - Strong reasoning  
    # "mistralai/Mistral-7B-Instruct-v0.3" - Good reasoning
    dataset_path: str = "housebrain_dataset_r1_super_10k_aug"
    output_dir: str = "housebrain_trained"
    
    # Training parameters
    max_length: int = 768
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 100
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Monitoring
    save_steps: int = 2000
    eval_steps: int = 0
    logging_steps: int = 25
    save_total_limit: int = 1
    
    # Test mode for 10K
    test_mode: bool = False
    max_samples: Optional[int] = None
    
    # Memory/throughput options
    gradient_checkpointing: bool = False

class HouseBrainTrainer:
    """Unified HouseBrain training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        print("🎯 HouseBrain Trainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"📂 Dataset: {config.dataset_path}")
        print(f"🤖 Model: {config.model_name}")
        
    def _detect_device(self) -> str:
        """Detect the best available device"""
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name()})"
            print(f"✅ GPU detected: {device}")
            return "cuda"
        else:
            print("⚠️ No GPU detected, using CPU")
            return "cpu"
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with error handling"""
        _ensure_training_libs_loaded()
        print(f"🤖 Loading model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True
            }

            if self.device == "cuda" and BitsAndBytesConfig is not None:
                # Use 4-bit quantization to fit 7B models on T4
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    model_kwargs.update({
                        "quantization_config": bnb_config,
                        "device_map": "auto",
                    })
                except Exception:
                    # BitsAndBytes not available; fall back to fp16 without quant
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                    })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": None,
                })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Prepare for training
            if self.device == "cuda":
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self._get_target_modules(),
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            print("✅ Model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Enable TF32 on CUDA for speed without quality loss
        if self.device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    
    def _get_target_modules(self) -> List[str]:
        """Get LoRA target modules based on model architecture"""
        model_name = self.config.model_name.lower()
        
        if "deepseek" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "qwen" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "mistral" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gpt" in model_name:
            # GPT-2 style attention uses a fused projection module name
            return ["c_attn"]
        else:
            # Default for unknown models
            return ["q_proj", "v_proj"]
    
    def _load_dataset(self):
        """Load and prepare the dataset"""
        _ensure_training_libs_loaded()
        print(f"📊 Loading dataset from: {self.config.dataset_path}")
        
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Load training data
        train_data = []
        
        # Look for files directly in the dataset directory (v2 format)
        json_files = list(dataset_path.glob("*.json"))
        
        # If no files found directly, try train subdirectory (old format)
        if not json_files:
            train_path = dataset_path / "train"
            if train_path.exists():
                json_files = list(train_path.glob("*.json"))
                print(f"📄 Found {len(json_files)} training files in train/ subdirectory")
            else:
                print(f"📄 No JSON files found in {dataset_path} or {train_path}")
        else:
            print(f"📄 Found {len(json_files)} training files directly in dataset directory")
        
        # Limit for test mode
        if self.config.test_mode and self.config.max_samples:
            json_files = json_files[:self.config.max_samples]
            print(f"🧪 Test mode: using {len(json_files)} samples")
        
        for file_path in tqdm(json_files, desc="Loading data"):
            try:
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                    train_data.append(self._format_sample(sample))
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")
        
        print(f"✅ Loaded {len(train_data)} training samples")
        
        # Create dataset
        self.dataset = Dataset.from_list(train_data)
        self.dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        
        print(f"📊 Dataset prepared: {len(self.dataset)} samples")

    def _estimate_effective_ga(self) -> int:
        """Estimate a gradient accumulation value that guarantees progress."""
        total_samples = len(self.dataset) if self.dataset is not None else 0
        per_device = max(1, self.config.batch_size)
        max_updates = max(1, total_samples // per_device)
        if max_updates == 0:
            return 1
        return min(self.config.gradient_accumulation_steps, max_updates)

    def _warmup_kernels(self):
        """Optional one-step warmup to compile kernels before training."""
        if self.model is None or self.tokenizer is None:
            return
        try:
            input_ids = self.tokenizer(
                "Warmup HouseBrain.",
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=min(64, self.config.max_length),
            )["input_ids"].to(self.model.device)
            with torch.no_grad():
                _ = self.model(input_ids=input_ids)
            print("🔥 Kernels warmup complete")
        except Exception as _e:
            # Warmup is best-effort; ignore failures
            pass
    
    def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format a sample for training"""
        # Check if this is v2 format (has metadata, walls, spaces, etc.)
        if "metadata" in sample and "walls" in sample and "spaces" in sample:
            # This is v2 format - create a simple prompt for v2 data
            prompt = f"""You are HouseBrain v2. Generate a complete HouseBrain Plan v2 JSON based on the requirements.

Requirements: Create a professional architectural plan with proper metadata, walls, spaces, openings, electrical, and schedules.

Output: {json.dumps(sample, indent=2)}"""
        else:
            # This is old format with input/output
            input_data = sample.get("input", {})
            output_data = sample.get("output", {})
            
            # Create formatted prompt
            prompt = f"""You are HouseBrain, an expert architectural AI. Generate a detailed house design based on the requirements.

Input: {json.dumps(input_data, indent=2)}

Output: {json.dumps(output_data, indent=2)}"""
        
        return {"text": prompt}
    
    def _tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.config.max_length,
            return_tensors=None
        )
    
    def train(self):
        """Main training function"""
        _ensure_training_libs_loaded()
        print("🚀 Starting HouseBrain training...")
        
        try:
            # Load model and data
            self._load_model_and_tokenizer()
            self._load_dataset()

            # Optional warmup to avoid long first-step latency
            if os.environ.get("HB_WARMUP", "0") == "1":
                self._warmup_kernels()
            
            # Training arguments
            effective_ga = self._estimate_effective_ga()
            if effective_ga != self.config.gradient_accumulation_steps:
                print(f"ℹ️ Adjusting gradient_accumulation_steps {self.config.gradient_accumulation_steps} -> {effective_ga} to ensure progress with {len(self.dataset)} samples")

            # Prefer bf16 on modern GPUs (A100) for speed/stability
            use_bf16 = self.device == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)()

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=effective_ga,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                fp16=self.device == "cuda" and not use_bf16,
                bf16=use_bf16,
                dataloader_drop_last=True,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb
                save_safetensors=True,
                optim="adamw_torch",
                gradient_checkpointing=self.config.gradient_checkpointing,
                **({"evaluation_strategy": "steps", "eval_steps": self.config.eval_steps}
                   if (getattr(self.config, "eval_steps", 0) or 0) > 0 else {})
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator
            )
            
            # Start training
            print("🎯 Training started...")
            start_time = time.time()
            
            trainer.train()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Training complete
            elapsed_time = time.time() - start_time
            print(f"✅ Training completed in {elapsed_time/3600:.1f} hours")
            print(f"💾 Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HouseBrain Unified Training")
    parser.add_argument("--test", action="store_true", help="Test mode with 10K samples")
    parser.add_argument("--dataset", type=str, default="housebrain_dataset_r1_super_10k_aug", help="Dataset path")
    parser.add_argument("--output", type=str, default="housebrain_trained", help="Output directory")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Model to use")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to use")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--max-length", type=int, help="Max sequence length (tokens)")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--save-steps", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, help="Max checkpoints to keep")
    parser.add_argument("--logging-steps", type=int, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, help="Run evaluation every N steps (0 disables)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        test_mode=args.test,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )

    # Optional overrides
    if args.max_length:
        config.max_length = args.max_length
    if args.grad_accum_steps:
        config.gradient_accumulation_steps = args.grad_accum_steps
    if args.grad_checkpointing:
        config.gradient_checkpointing = True
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.save_total_limit:
        config.save_total_limit = args.save_total_limit
    if args.logging_steps:
        config.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        config.eval_steps = max(0, args.eval_steps)
    
    # Adjust for test mode
    if args.test:
        config.max_samples = config.max_samples or 1000  # 1K samples for quick test
        # Faster defaults for smoke tests unless explicitly overridden
        if not args.max_length:
            config.max_length = 512
        if not args.grad_accum_steps:
            config.gradient_accumulation_steps = 1
        config.save_steps = 100
        config.eval_steps = 0
        print("🧪 Test mode enabled - using reduced dataset")
    
    # Create and run trainer
    trainer = HouseBrainTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
