"""
HouseBrain LLM Fine-tuning Module

This module handles the fine-tuning of the HouseBrain LLM using QLoRA
and the synthetic architectural dataset.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
import pandas as pd
from tqdm import tqdm

from .schema import HouseInput, HouseOutput
from .layout import solve_house_layout


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning with DeepSeek R1"""
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"  # DeepSeek R1 equivalent
    dataset_path: str = "housebrain_dataset_v1"
    output_dir: str = "models/housebrain-finetuned"
    max_length: int = 2048
    batch_size: int = 2  # Reduced for memory efficiency
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_4bit: bool = False  # Disabled for MPS compatibility
    use_nested_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


class HouseBrainDataset(Dataset):
    """Custom dataset for HouseBrain fine-tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_dataset(data_path)
        
    def _load_dataset(self, data_path: str, split: str = "train") -> List[Dict]:
        """Load and preprocess the dataset"""
        data = []
        
        # Load data from specified split
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
            
        for filename in os.listdir(split_path):
            if filename.endswith('.json'):
                file_path = os.path.join(split_path, filename)
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                    data.append(self._preprocess_sample(sample))
        
        return data
    
    def _preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a single sample"""
        # Convert the dataset format to our schema format
        house_input = self._convert_to_house_input(sample)
        
        # Generate the expected output using our layout solver
        try:
            house_output = solve_house_layout(house_input)
            output_json = house_output.model_dump()
        except Exception as e:
            # If layout solver fails, create a basic output
            output_json = self._create_basic_output(house_input)
        
        # Create the training prompt
        prompt = self._create_training_prompt(house_input, output_json)
        
        return {
            "text": prompt,
            "input": house_input.model_dump(),
            "output": output_json
        }
    
    def _convert_to_house_input(self, sample: Dict) -> HouseInput:
        """Convert dataset format to HouseInput"""
        project = sample["project"]
        plot = sample["plot"]
        
        # Count rooms by type
        room_counts = {}
        for level in sample["levels"]:
            for room in level["rooms"]:
                room_type = room["name"].lower().replace(" ", "_")
                room_counts[room_type] = room_counts.get(room_type, 0) + 1
        
        # Estimate bathrooms (including half baths)
        bathrooms = room_counts.get("bathroom", 0) + room_counts.get("half_bath", 0) * 0.5
        
        return HouseInput(
            basicDetails={
                "totalArea": project["total_area_sqft"],
                "unit": "sqft",
                "floors": project["floors"],
                "bedrooms": room_counts.get("bedroom", 0) + room_counts.get("master_bedroom", 0),
                "bathrooms": bathrooms,
                "style": project["style"],
                "budget": project.get("budget_estimate", 500000)  # Use budget_estimate or default
            },
            plot={
                "length": plot.get("length", plot.get("length_ft", 50)),  # Handle different field names
                "width": plot.get("width", plot.get("width_ft", 40)),  # Handle different field names
                "unit": plot.get("unit", "ft"),
                "orientation": "NE",  # Default orientation
                "setbacks_ft": plot.get("setbacks_ft", {"front": 4, "rear": 4, "left": 2, "right": 2})
            },
            roomBreakdown=[]
        )
    
    def _create_basic_output(self, house_input: HouseInput) -> Dict:
        """Create a basic output when layout solver fails"""
        return {
            "input": house_input.model_dump(),
            "levels": [],
            "total_area": house_input.basicDetails["totalArea"],
            "construction_cost": house_input.basicDetails["budget"] * 0.6,
            "materials": {},
            "render_paths": {}
        }
    
    def _create_training_prompt(self, house_input: HouseInput, output_json: Dict) -> str:
        """Create the training prompt for the LLM"""
        prompt = f"""You are HouseBrain, an expert architectural AI that designs residential houses.

Given the following house requirements, generate a complete house design:

INPUT:
{json.dumps(house_input.model_dump(), indent=2)}

OUTPUT:
{json.dumps(output_json, indent=2)}

The design must be:
- Functional and practical
- Code compliant
- Cost-effective
- Aesthetically pleasing
- Optimized for the given plot and requirements

Think like an experienced architect who has designed hundreds of successful homes."""
        
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class HouseBrainFineTuner:
    """Main fine-tuning class for HouseBrain"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        # Set device (support MPS for Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings for the device
        if self.config.use_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype)
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        elif torch.backends.mps.is_available():
            # MPS (Apple Silicon) loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,  # MPS works better with float32
                device_map=None,  # MPS doesn't support device_map
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # CPU loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Setup LoRA for DeepSeek model
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self) -> Tuple[HouseBrainDataset, HouseBrainDataset]:
        """Prepare training and validation datasets"""
        self.logger.info("Preparing datasets...")
        
        # Create datasets
        train_dataset = HouseBrainDataset(
            self.config.dataset_path,
            self.tokenizer,
            self.config.max_length
        )
        train_dataset.data = train_dataset._load_dataset(self.config.dataset_path, "train")
        
        val_dataset = HouseBrainDataset(
            self.config.dataset_path,
            self.tokenizer,
            self.config.max_length
        )
        val_dataset.data = val_dataset._load_dataset(self.config.dataset_path, "validation")
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )
    
    def train(self):
        """Run the fine-tuning process"""
        self.logger.info("Starting HouseBrain fine-tuning...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_dataset()
        
        # Convert to HuggingFace datasets
        train_hf_dataset = HFDataset.from_list([{"text": item["text"]} for item in train_dataset])
        val_hf_dataset = HFDataset.from_list([{"text": item["text"]} for item in val_dataset])
        
        # Tokenize datasets
        train_tokenized = train_hf_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_hf_dataset.column_names
        )
        val_tokenized = val_hf_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_hf_dataset.column_names
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
            dataloader_pin_memory=False,  # Disable for CPU/MPS
            remove_unused_columns=False,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
        )
        
        # Start training
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        self.logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        self.logger.info("Fine-tuning completed successfully!")
        
        return trainer


def run_finetuning(config: Optional[FineTuningConfig] = None):
    """Run the fine-tuning process"""
    if config is None:
        config = FineTuningConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize and run fine-tuning
    finetuner = HouseBrainFineTuner(config)
    trainer = finetuner.train()
    
    return trainer


if __name__ == "__main__":
    # Example usage
    config = FineTuningConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        dataset_path="housebrain_dataset_v1",
        output_dir="models/housebrain-finetuned",
        num_epochs=1,  # Start with 1 epoch for testing
        batch_size=2,  # Smaller batch size for testing
    )
    
    run_finetuning(config)
