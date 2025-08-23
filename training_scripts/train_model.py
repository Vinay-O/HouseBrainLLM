#!/usr/bin/env python3
"""
DeepSeek R1 Training Script for HouseBrain Enhancement
"""

import torch
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import json

class HouseBrainDataset(Dataset):
    def __init__(self, tokenized_examples):
        self.examples = tokenized_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_housebrain_model():
    """Train DeepSeek R1 model for HouseBrain enhancement"""
    
    print("🚀 Starting HouseBrain DeepSeek R1 Training")
    print("=" * 50)
    
    # Model configuration
    base_model_path = "models/deepseek_r1_600k"
    output_dir = "models/housebrain_deepseek_enhanced"
    
    # Load model and tokenizer
    print("📥 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load preprocessed data
    print("📊 Loading training data...")
    from preprocess_data import HouseBrainDataPreprocessor
    preprocessor = HouseBrainDataPreprocessor(base_model_path)
    train_examples = preprocessor.preprocess_architectural_data("training_dataset")
    
    # Create dataset
    train_dataset = HouseBrainDataset(train_examples)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        eval_steps=250,
        save_total_limit=3,
        prediction_loss_only=True,
        fp16=True,
        dataloader_pin_memory=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Start training
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving enhanced model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Training completed successfully!")
    print(f"📁 Enhanced model saved to: {output_dir}")

if __name__ == "__main__":
    train_housebrain_model()\n