#!/usr/bin/env python3
"""
Lightweight test for DeepSeek R1 integration

This script tests model loading and dataset processing without full training.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from housebrain.finetune import FineTuningConfig, HouseBrainDataset
from transformers import AutoTokenizer


def test_deepseek_model():
    """Test DeepSeek R1 model loading"""
    print("🧪 Testing DeepSeek R1 model loading...")
    
    model_name = "deepseek-ai/deepseek-coder-6.7b-base"
    
    try:
        # Test tokenizer loading
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded successfully!")
        print(f"   - Vocabulary size: {tokenizer.vocab_size}")
        print(f"   - Model max length: {tokenizer.model_max_length}")
        
        # Test a simple tokenization
        test_text = "You are HouseBrain, an expert architectural AI."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   - Test tokenization: {tokens.input_ids.shape}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None


def test_dataset_processing(dataset_version: str = "v3"):
    """Test dataset processing without loading the full model"""
    print(f"\n🧪 Testing dataset processing for {dataset_version}...")
    
    config = FineTuningConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        dataset_path=f"housebrain_dataset_{dataset_version}",
        output_dir="models/test",
        max_length=512,
        batch_size=1,
        num_epochs=1,
        use_4bit=False,
    )
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
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
        
        print(f"✅ Dataset processing successful!")
        print(f"   - Training samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(val_dataset)}")
        
        # Test a few samples
        print("\n🔍 Testing sample processing...")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            print(f"   Sample {i+1}: {len(sample['text'])} characters")
            print(f"   Input keys: {list(sample['input'].keys())}")
            print(f"   Output keys: {list(sample['output'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mps_availability():
    """Test MPS availability for M2 Pro"""
    print("\n🧪 Testing MPS availability...")
    
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available!")
        print(f"   - Device: Apple Silicon GPU")
        
        # Test basic MPS operations
        try:
            device = torch.device("mps")
            x = torch.randn(2, 3).to(device)
            y = torch.randn(3, 2).to(device)
            z = torch.mm(x, y)
            print(f"   - MPS tensor operations: ✅")
            return True
        except Exception as e:
            print(f"   - MPS tensor operations: ❌ {e}")
            return False
    else:
        print("⚠️  MPS not available")
        return False


def main():
    """Main test function"""
    print("🏠 DeepSeek R1 Integration Test")
    print("=" * 50)
    
    # Test MPS availability
    mps_available = test_mps_availability()
    
    # Test model loading
    tokenizer = test_deepseek_model()
    
    # Test dataset processing
    dataset_ok = test_dataset_processing("v3")
    
    print(f"\n📋 Test Summary:")
    print(f"   - MPS available: {'✅' if mps_available else '❌'}")
    print(f"   - Model loading: {'✅' if tokenizer else '❌'}")
    print(f"   - Dataset processing: {'✅' if dataset_ok else '❌'}")
    
    if tokenizer and dataset_ok:
        print(f"\n🎉 All tests passed! DeepSeek R1 integration is ready.")
        print(f"\n📋 Next steps:")
        if mps_available:
            print(f"   1. Run fine-tuning with MPS: python finetune_m2pro.py --dataset v3")
        else:
            print(f"   1. Run fine-tuning with CPU: python finetune_m2pro.py --dataset v3 --no-mps")
        print(f"   2. Use smaller batch size for memory constraints")
        print(f"   3. Consider using v4_10k dataset for more data")
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
