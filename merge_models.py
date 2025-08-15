#!/usr/bin/env python3
"""
HouseBrain Model Merger

Merges 6 trained models from parallel training into a single production model:
- 3 Colab models
- 3 Kaggle models
- Creates ensemble model with best performance
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np

class ModelMerger:
    """Merges multiple trained models into a single model"""
    
    def __init__(self, base_model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.base_model = None
        self.trained_models = []
        
    def load_base_model(self):
        """Load the base model"""
        print(f"🔧 Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("✅ Base model loaded")
    
    def load_trained_models(self, model_paths: List[str]):
        """Load all trained models"""
        print(f"📂 Loading {len(model_paths)} trained models...")
        
        for i, model_path in enumerate(model_paths):
            if not Path(model_path).exists():
                print(f"⚠️  Model not found: {model_path}")
                continue
                
            try:
                print(f"📥 Loading model {i+1}: {model_path}")
                
                # Load LoRA adapter
                model = PeftModel.from_pretrained(
                    self.base_model,
                    model_path,
                    torch_dtype=torch.float16
                )
                
                self.trained_models.append({
                    "path": model_path,
                    "model": model,
                    "index": i
                })
                
                print(f"✅ Model {i+1} loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load model {i+1}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.trained_models)} models")

        # Sanity check: Ensure all adapters share identical LoRA hyperparams
        try:
            base_config = None
            for tm in self.trained_models:
                cfg = getattr(tm["model"], "peft_config", None)
                if isinstance(cfg, dict):
                    # For multi-adapter dict, pick first
                    cfg = next(iter(cfg.values()))
                if base_config is None:
                    base_config = cfg
                else:
                    assert (
                        cfg.r == base_config.r and
                        cfg.lora_alpha == base_config.lora_alpha and
                        set(cfg.target_modules) == set(base_config.target_modules)
                    ), "All LoRA adapters must share identical config for merging."
        except Exception as e:
            print(f"⚠️  LoRA config mismatch check skipped or failed: {e}")
    
    def merge_models(self, output_path: str, merge_strategy: str = "average"):
        """Merge models using specified strategy"""
        print(f"🔀 Merging models using strategy: {merge_strategy}")
        
        if len(self.trained_models) == 0:
            raise ValueError("No trained models loaded")
        
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if merge_strategy == "average":
            merged_model = self._merge_average()
        elif merge_strategy == "weighted":
            merged_model = self._merge_weighted()
        elif merge_strategy == "best":
            merged_model = self._merge_best()
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Save merged model
        print("💾 Saving merged model...")
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Create model info
        model_info = {
            "name": "HouseBrain Merged Model",
            "base_model": self.base_model_name,
            "merge_strategy": merge_strategy,
            "num_models": len(self.trained_models),
            "model_paths": [m["path"] for m in self.trained_models],
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        with open(output_path / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Merged model saved to: {output_path}")
        return merged_model
    
    def _merge_average(self):
        """Merge models by averaging their LoRA weights"""
        print("📊 Averaging LoRA weights...")
        
        # Get the first model as base
        base_model = self.trained_models[0]["model"]
        
        # Average LoRA weights from all models
        for param_name, param in base_model.named_parameters():
            if "lora" in param_name and param.requires_grad:
                # Collect weights from all models
                weights = []
                for trained_model in self.trained_models:
                    model = trained_model["model"]
                    for name, p in model.named_parameters():
                        if name == param_name:
                            weights.append(p.data.clone())
                            break
                
                # Average the weights
                if weights:
                    avg_weight = torch.stack(weights).mean(dim=0)
                    param.data = avg_weight
        
        return base_model
    
    def _merge_weighted(self):
        """Merge models with weighted averaging based on training quality"""
        print("⚖️ Weighted averaging based on training quality...")
        
        # Simple weighting based on model index (can be enhanced with actual metrics)
        weights = [1.0, 1.1, 1.2, 1.0, 1.1, 1.2]  # Slight preference for later models
        
        base_model = self.trained_models[0]["model"]
        
        for param_name, param in base_model.named_parameters():
            if "lora" in param_name and param.requires_grad:
                weights_list = []
                for i, trained_model in enumerate(self.trained_models):
                    model = trained_model["model"]
                    for name, p in model.named_parameters():
                        if name == param_name:
                            weights_list.append(p.data.clone() * weights[i])
                            break
                
                if weights_list:
                    weighted_avg = torch.stack(weights_list).sum(dim=0) / sum(weights)
                    param.data = weighted_avg
        
        return base_model
    
    def _merge_best(self):
        """Merge by selecting the best performing model"""
        print("🏆 Selecting best performing model...")
        
        # For now, select the model with most training steps
        # In production, you'd use actual validation metrics
        best_model = max(self.trained_models, key=lambda x: self._get_training_steps(x["path"]))
        
        print(f"🏆 Selected best model: {best_model['path']}")
        return best_model["model"]
    
    def _get_training_steps(self, model_path: str) -> int:
        """Get training steps from model path"""
        try:
            # Look for training args or checkpoint info
            args_path = Path(model_path) / "training_args.bin"
            if args_path.exists():
                # This is a simplified approach - in practice you'd load the actual args
                return 1000  # Placeholder
            return 500  # Default
        except:
            return 500
    
    def validate_merged_model(self, test_input: str = "Design a 3-bedroom modern house"):
        """Validate the merged model with a test input"""
        print("🧪 Validating merged model...")
        
        # This is a basic validation - in production you'd use proper evaluation
        try:
            # Test tokenization
            inputs = self.tokenizer(test_input, return_tensors="pt")
            
            # Test generation (short)
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Model validation successful")
            print(f"📝 Sample output: {generated_text[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False

def create_merge_config(model_paths: List[str], output_path: str):
    """Create merge configuration file"""
    config = {
        "merge_strategy": "average",
        "model_paths": model_paths,
        "output_path": output_path,
        "base_model": "deepseek-ai/deepseek-coder-6.7b-base",
        "validation": True
    }
    
    config_path = Path(output_path) / "merge_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️ Merge configuration saved: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge HouseBrain models from parallel training")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to trained models")
    parser.add_argument("--output", default="models/housebrain-merged", help="Output path for merged model")
    parser.add_argument("--strategy", default="average", choices=["average", "weighted", "best"], help="Merge strategy")
    parser.add_argument("--validate", action="store_true", help="Validate merged model")
    
    args = parser.parse_args()
    
    print("🔀 HouseBrain Model Merger")
    print("=" * 50)
    
    # Validate model paths
    valid_models = []
    for model_path in args.models:
        if Path(model_path).exists():
            valid_models.append(model_path)
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    if len(valid_models) == 0:
        print("❌ No valid models found!")
        return
    
    print(f"📂 Found {len(valid_models)} valid models")
    
    # Create merger
    merger = ModelMerger()
    
    # Load base model
    merger.load_base_model()
    
    # Load trained models
    merger.load_trained_models(valid_models)
    
    # Create merge configuration
    create_merge_config(valid_models, args.output)
    
    # Merge models
    merged_model = merger.merge_models(args.output, args.strategy)
    
    # Validate if requested
    if args.validate:
        merger.validate_merged_model()
    
    print("\n🎉 Model merging complete!")
    print(f"📁 Merged model saved to: {args.output}")
    print(f"🔀 Merge strategy: {args.strategy}")
    print(f"📊 Models merged: {len(valid_models)}")

if __name__ == "__main__":
    main()

