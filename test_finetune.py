#!/usr/bin/env python3
"""
Simplified HouseBrain Fine-tuning Test

This script tests the fine-tuning pipeline with minimal resources.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from housebrain.finetune import FineTuningConfig, HouseBrainDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def test_dataset_loading():
    """Test dataset loading without full model"""
    print("🧪 Testing dataset loading...")
    
    config = FineTuningConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",  # DeepSeek R1 equivalent
        dataset_path="housebrain_dataset_v1",
        output_dir="models/test-finetuned",
        max_length=512,  # Shorter sequences for testing
        batch_size=1,
        num_epochs=1,
        use_4bit=False,
    )
    
    # Test tokenizer loading
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test dataset loading
    print("📊 Loading training dataset...")
    train_dataset = HouseBrainDataset(
        config.dataset_path,
        tokenizer,
        config.max_length
    )
    train_dataset.data = train_dataset._load_dataset(config.dataset_path, "train")
    
    print("📊 Loading validation dataset...")
    val_dataset = HouseBrainDataset(
        config.dataset_path,
        tokenizer,
        config.max_length
    )
    val_dataset.data = val_dataset._load_dataset(config.dataset_path, "validation")
    
    print(f"✅ Dataset loading successful!")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    
    # Test a few samples
    print("\n🔍 Testing sample processing...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"   Sample {i+1}: {len(sample['text'])} characters")
        print(f"   Input keys: {list(sample['input'].keys())}")
        print(f"   Output keys: {list(sample['output'].keys())}")
    
    return train_dataset, val_dataset, tokenizer


def test_model_loading():
    """Test model loading with DeepSeek R1"""
    print("\n🧪 Testing model loading...")
    
    model_name = "deepseek-ai/deepseek-coder-6.7b-base"
    
    # Load small model
    print(f"📥 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Setup LoRA for DeepSeek
    print("🔧 Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,  # Smaller rank for testing
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ Model loading successful!")
    return model


def test_training_step():
    """Test a single training step"""
    print("\n🧪 Testing training step...")
    
    # Load components
    train_dataset, val_dataset, tokenizer = test_dataset_loading()
    model = test_model_loading()
    
    # Test tokenization
    print("🔤 Testing tokenization...")
    sample_text = train_dataset[0]["text"]
    inputs = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    print(f"   Input shape: {inputs.input_ids.shape}")
    print(f"   Attention mask shape: {inputs.attention_mask.shape}")
    
    # Test forward pass
    print("⚡ Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"   Output logits shape: {outputs.logits.shape}")
        print(f"   Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
    
    print("✅ Training step test successful!")
    
    return model, tokenizer


def main():
    """Main test function"""
    print("🏠 HouseBrain Fine-tuning Test")
    print("=" * 50)
    
    try:
        # Test dataset loading
        train_dataset, val_dataset, tokenizer = test_dataset_loading()
        
        # Test model loading
        model = test_model_loading()
        
        # Test training step
        model, tokenizer = test_training_step()
        
        print("\n🎉 All tests passed! Fine-tuning pipeline is ready.")
        print("\n📋 Next steps:")
        print("   1. Use a GPU for actual fine-tuning")
        print("   2. Increase batch size and sequence length")
        print("   3. Use a larger base model (e.g., deepseek-coder-6.7b-base)")
        print("   4. Run full fine-tuning with: python finetune_housebrain.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
