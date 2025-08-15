#!/usr/bin/env python3
"""
HouseBrain LLM Fine-tuning Module

Handles fine-tuning of the HouseBrain LLM using QLoRA on synthetic architectural data.
Supports both local (MPS) and cloud (CUDA) training environments.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, is_dataclass, asdict
import random

# Import with fallback for notebook environments
try:
    from housebrain.schema import HouseInput, HouseOutput
except ImportError:
    # Fallback schema for notebook environments
    @dataclass
    class HouseInput:
        basicDetails: Dict
        plot: Dict
        roomBreakdown: List
    
    @dataclass
    class HouseOutput:
        input: Dict
        levels: List
        total_area: int
        construction_cost: int
        materials: Dict
        render_paths: Dict

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from datasets import Dataset

# Disable wandb prompts
os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"
    dataset_path: str = "housebrain_dataset_v5_350k"
    output_dir: str = "models/housebrain-trained"
    
    # Training parameters
    num_epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 512
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Memory optimization
    use_4bit: bool = True
    use_nested_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    
    # Device detection
    device: str = "auto"
    # Sequence packing to improve utilization
    pack_sequences: bool = False
    
    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Disable 4-bit quantization for CPU
        if self.device == "cpu":
            self.use_4bit = False


def _to_serializable_dict(obj: Any) -> Dict[str, Any]:
    """Convert pydantic BaseModel or dataclass or dict-like to plain dict."""
    try:
        # Pydantic v2 BaseModel
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
    except Exception:
        pass
    try:
        # Dataclass instance
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    # Already a dict or unknown object
    if isinstance(obj, dict):
        return obj
    raise TypeError("Unsupported object type for serialization")


class HouseBrainDataset:
    """Dataset class for HouseBrain fine-tuning"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.dataset = None
        
    def _load_dataset(self, split: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON files"""
        data_path = Path(self.config.dataset_path)
        split_path = data_path / split
        
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        
        print(f"📂 Loading dataset from: {self.config.dataset_path}")
        print(f" {split} path: {split_path}")
        
        samples = []
        json_files = list(split_path.glob("*.json"))
        print(f"📄 Found {len(json_files)} {split} files")
        
        for i, json_file in enumerate(json_files):
            if (i + 1) % 1000 == 0:
                print(f" Loading file {i+1}/{len(json_files)}...")
            
            try:
                with open(json_file, 'r') as f:
                    sample = json.load(f)
                samples.append(sample)
            except Exception as e:
                print(f"⚠️  Skipping {json_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(samples)} {split} samples")
        return samples
    
    def _convert_to_house_input(self, sample: Dict[str, Any]) -> HouseInput:
        """Convert dataset sample to HouseInput"""
        try:
            # Handle both old and new dataset formats
            if "project" in sample:
                # Old format
                project = sample["project"]
                input_data = project.get("input", {})
                output_data = project.get("output", {})
            else:
                # New format (v5)
                input_data = sample.get("input", {})
                output_data = sample.get("output", {})
            
            # Extract basic details with fallbacks
            basic_details = input_data.get("basicDetails", {})
            plot_data = input_data.get("plot", {})
            room_breakdown = input_data.get("roomBreakdown", [])
            
            # Ensure required fields exist
            if "bathrooms" not in basic_details:
                basic_details["bathrooms"] = basic_details.get("bedrooms", 1)
            
            return HouseInput(
                basicDetails=basic_details,
                plot=plot_data,
                roomBreakdown=room_breakdown
            )
        except Exception as e:
            print(f"⚠️  Error converting sample: {e}")
            # Return minimal valid input
            return HouseInput(
                basicDetails={
                    "totalArea": 1000,
                    "unit": "sqft",
                    "bedrooms": 2,
                    "bathrooms": 1,
                    "floors": 1,
                    "budget": 500000,
                    "style": "Modern"
                },
                plot={
                    "length": 40,
                    "width": 25,
                    "unit": "ft",
                    "orientation": "N",
                    "setbacks_ft": {"front": 5, "rear": 5, "left": 3, "right": 3}
                },
                roomBreakdown=[]
            )
    
    def _create_prompt_and_response(self, input_data: HouseInput, output_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Create separated prompt and response strings for masked SFT."""
        try:
            input_json = json.dumps(_to_serializable_dict(input_data), indent=2)
            response_json = json.dumps(output_data, indent=2)

            system = (
                "You are HouseBrain, an expert architectural AI. "
                "Generate detailed house designs in JSON format."
            )

            prompt = (
                f"<|im_start|>system\n{system}\n<|im_end|>\n"
                f"<|im_start|>user\nDesign a house with these specifications:\n{input_json}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            response = f"{response_json}\n<|im_end|>"
            return {"prompt": prompt, "response": response}
        except Exception as e:
            print(f"⚠️  Error creating prompt/response: {e}")
            return None
    
    def prepare_dataset(self, max_samples: Optional[int] = None) -> Dataset:
        """Prepare dataset for training. Returns un-tokenized prompt/response pairs."""
        print("📊 Preparing dataset...")

        train_samples = self._load_dataset("train")
        if max_samples:
            train_samples = train_samples[:max_samples]
            print(f" Using first {max_samples} samples for memory efficiency")

        pairs: List[Dict[str, str]] = []
        for sample in train_samples:
            try:
                input_data = self._convert_to_house_input(sample)
                output_data = sample.get("output", {})
                pair = self._create_prompt_and_response(input_data, output_data)
                if pair and pair.get("prompt") and pair.get("response"):
                    pairs.append(pair)
            except Exception as e:
                print(f"⚠️  Skipping sample due to error: {e}")
                continue

        print(f"📊 Dataset prepared: {len(pairs)} samples")
        return Dataset.from_list(pairs)

    def prepare_eval_dataset(self, max_eval_samples: Optional[int] = 1000) -> Optional[Dataset]:
        """Prepare evaluation dataset from validation split if available."""
        try:
            val_samples = self._load_dataset("validation")
        except FileNotFoundError:
            print("ℹ️ No validation split found. Skipping eval dataset.")
            return None

        if max_eval_samples:
            val_samples = val_samples[:max_eval_samples]

        pairs: List[Dict[str, str]] = []
        for sample in val_samples:
            try:
                input_data = self._convert_to_house_input(sample)
                output_data = sample.get("output", {})
                pair = self._create_prompt_and_response(input_data, output_data)
                if pair:
                    pairs.append(pair)
            except Exception as e:
                continue
        print(f"📊 Eval dataset prepared: {len(pairs)} samples")
        return Dataset.from_list(pairs)


class PromptResponseCollator:
    """Tokenizes prompt/response and masks loss on prompt tokens (labels=-100)."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int, pack_sequences: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences

    def __call__(self, features: List[Dict[str, str]]):
        prompts = [f["prompt"] for f in features]
        responses = [f["response"] for f in features]

        # Encode separately to retain boundaries for masking
        prompt_enc = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        response_enc = self.tokenizer(
            responses,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        segments = []
        for i in range(len(prompts)):
            p_ids = prompt_enc.input_ids[i]
            r_ids = response_enc.input_ids[i]
            segments.append((p_ids, r_ids))

        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        if self.pack_sequences:
            current_ids = []
            current_labels = []
            for p_ids, r_ids in segments:
                combined = torch.cat([p_ids, r_ids], dim=0)
                labels = torch.full_like(combined, fill_value=-100)
                labels[len(p_ids):] = combined[len(p_ids):]
                # If current is empty, start new
                if not current_ids:
                    if combined.size(0) <= self.max_length:
                        current_ids = [combined]
                        current_labels = [labels]
                    else:
                        # Truncate oversized pair
                        current_ids = [combined[: self.max_length]]
                        current_labels = [labels[: self.max_length]]
                        # finalize immediately
                        input_ids_list.append(torch.cat(current_ids))
                        labels_list.append(torch.cat(current_labels))
                        attention_mask_list.append(torch.ones_like(input_ids_list[-1]))
                        current_ids, current_labels = [], []
                    continue
                # Try to append to current
                curr_len = sum(t.size(0) for t in current_ids)
                if curr_len + combined.size(0) <= self.max_length:
                    current_ids.append(combined)
                    current_labels.append(labels)
                else:
                    # finalize current
                    seq_ids = torch.cat(current_ids)
                    seq_labels = torch.cat(current_labels)
                    input_ids_list.append(seq_ids)
                    labels_list.append(seq_labels)
                    attention_mask_list.append(torch.ones_like(seq_ids))
                    # start new with this combined (truncate if needed)
                    if combined.size(0) <= self.max_length:
                        current_ids = [combined]
                        current_labels = [labels]
                    else:
                        current_ids = [combined[: self.max_length]]
                        current_labels = [labels[: self.max_length]]
                        seq_ids = torch.cat(current_ids)
                        seq_labels = torch.cat(current_labels)
                        input_ids_list.append(seq_ids)
                        labels_list.append(seq_labels)
                        attention_mask_list.append(torch.ones_like(seq_ids))
                        current_ids, current_labels = [], []
            # finalize remainder
            if current_ids:
                seq_ids = torch.cat(current_ids)
                seq_labels = torch.cat(current_labels)
                input_ids_list.append(seq_ids)
                labels_list.append(seq_labels)
                attention_mask_list.append(torch.ones_like(seq_ids))
        else:
            for p_ids, r_ids in segments:
                combined = torch.cat([p_ids, r_ids], dim=0)
                combined = combined[: self.max_length]
                labels = torch.full_like(combined, fill_value=-100)
                prompt_len = min(len(p_ids), self.max_length)
                labels[prompt_len:] = combined[prompt_len:]
                attn_mask = torch.ones_like(combined)
                input_ids_list.append(combined)
                labels_list.append(labels)
                attention_mask_list.append(attn_mask)

        # Pad to max length in batch if needed
        batch_max = max(x.size(0) for x in input_ids_list)
        def pad_tensor(t, pad_to):
            if t.size(0) == pad_to:
                return t
            pad_len = pad_to - t.size(0)
            return torch.cat([t, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=t.dtype)], dim=0)
        def pad_labels(t, pad_to):
            if t.size(0) == pad_to:
                return t
            pad_len = pad_to - t.size(0)
            return torch.cat([t, torch.full((pad_len,), -100, dtype=t.dtype)], dim=0)

        input_ids = torch.stack([pad_tensor(t, batch_max) for t in input_ids_list])
        labels = torch.stack([pad_labels(t, batch_max) for t in labels_list])
        attention_mask = torch.stack([pad_tensor(t, batch_max) for t in attention_mask_list])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class HouseBrainFineTuner:
    """Main fine-tuning class"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with proper quantization"""
        print(f"🔧 Setting up model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ Tokenizer loaded: {self.config.model_name}")
        
        # Load model with quantization
        if self.config.use_4bit and self.config.device != "cpu":
            print("✅ Loading model with aggressive 4-bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_4bit=True,
                quantization_config=bnb.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=self.config.use_nested_quant,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype)
                ),
                device_map="auto" if self.config.device == "cuda" else None,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            print("✅ Loading model without quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            )
        
        # Setup LoRA
        self._setup_lora()
        
        # Ensure model is in training mode and gradients are enabled
        self.model.train()
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad = True
        
        # Print parameter stats
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.2f}")
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        # Dynamic target modules based on model architecture
        if "deepseek" in self.config.model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in self.config.model_name.lower():
            target_modules = ["c_attn", "c_proj", "c_fc", "c_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"✅ LoRA configured with rank {self.config.lora_r}")
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup trainer with proper configuration"""
        print("🔧 Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type="linear",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # Temporarily disable gradient checkpointing to avoid conflicts
            gradient_checkpointing=False,
            fp16=self.config.device == "cuda",
            max_grad_norm=1.0,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=1000 if eval_dataset is not None else None,
            report_to=None,  # Disable wandb
            seed=42,
        )
        
        # Data collator with masked SFT
        data_collator = PromptResponseCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            pack_sequences=self.config.pack_sequences,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("✅ Trainer initialized successfully!")
    
    def train(self, max_samples: Optional[int] = None):
        """Main training function"""
        try:
            print("🚀 Starting training...")
            
            # Setup model and tokenizer
            self.setup_model_and_tokenizer()
            
            # Prepare dataset
            dataset_generator = HouseBrainDataset(self.config)
            train_dataset = dataset_generator.prepare_dataset(max_samples)
            eval_dataset = dataset_generator.prepare_eval_dataset(max_eval_samples=1000)
            
            # Setup trainer
            self.setup_trainer(train_dataset, eval_dataset)
            
            # Start training
            print("🚀 Starting training loop...")
            if max_samples:
                print(f"📊 Training on {max_samples}K samples (memory optimized)...")
            else:
                print("📊 Training on full dataset...")
            print(" Keep this notebook active and don't close the browser tab!")
            
            self.trainer.train()
            
            # Save model
            print("💾 Saving model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            print("✅ Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune HouseBrain LLM")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-base", help="Model to fine-tune")
    parser.add_argument("--dataset", default="housebrain_dataset_v5_350k", help="Dataset path")
    parser.add_argument("--output", default="models/housebrain-trained", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to use")
    parser.add_argument("--test", action="store_true", help="Test mode with small dataset")
    
    args = parser.parse_args()
    
    # Configuration
    config = FineTuningConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Test mode
    if args.test:
        config.num_epochs = 1
        max_samples = 100
    else:
        max_samples = args.max_samples
    
    # Run training
    trainer = HouseBrainFineTuner(config)
    success = trainer.train(max_samples)
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")
        exit(1)


if __name__ == "__main__":
    main()
