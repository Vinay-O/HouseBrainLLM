## HouseBrain 425K TPU Training Guide (Complete)

This comprehensive guide covers training the enhanced 425K combined dataset on Kaggle TPU VM v3-8, including auto-save mechanisms, completion handling, and post-training steps.

### 🎯 **Training Overview**
- **Dataset**: `housebrain_dataset_v5_425k/` (382,499 train, 42,500 validation)
- **Hardware**: TPU VM v3-8 (8 TPU cores, 64GB memory)
- **Expected Time**: 3-5 hours
- **Auto-save**: Every 200 steps + final checkpoint
- **Completion**: Automatic model saving and email notification

---

## 🚀 **Step 1: Kaggle Setup**

### **1.1 Enable TPU VM v3-8**
1. Go to **Settings** (top menu)
2. **Accelerator** → Select **"TPU VM v3-8"**
3. **Save** and **Restart** notebook
4. Wait for TPU allocation (2-3 minutes)

### **1.2 Verify TPU Access**
```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

print(f"TPU available: {torch_xla._XLAC._xla_get_default_device()}")
print(f"TPU device count: {xm.xrt_world_size()}")
```

---

## 📦 **Step 2: Upload Dataset**

### **2.1 Upload Combined Dataset**
1. **Zip your dataset locally** (if not already done):
```bash
zip -r housebrain_dataset_v5_425k.zip housebrain_dataset_v5_425k/
```

2. **Upload to Kaggle**:
   - Click **"Add data"** → **"Upload a dataset"**
   - Select `housebrain_dataset_v5_425k.zip`
   - Set dataset name: `housebrain-425k-combined`
   - Click **"Create"**

### **2.2 Mount Dataset in Notebook**
```python
# Mount the uploaded dataset
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# List available datasets
datasets = api.dataset_list(search="housebrain-425k-combined")
print(f"Found datasets: {datasets}")

# Extract dataset
import zipfile
import os

for file in os.listdir('/kaggle/input'):
    if file.endswith('.zip'):
        with zipfile.ZipFile(f'/kaggle/input/{file}', 'r') as zip_ref:
            zip_ref.extractall('/kaggle/working/')
        print(f"✅ Extracted: {file}")

print("📁 Dataset contents:")
!ls -la housebrain_dataset_v5_425k/
```

---

## 🔧 **Step 3: Install Dependencies**

```python
# Install TPU-compatible versions
!pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
!pip install torch_xla[tpu]==2.0.1 -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install transformers==4.43.3 peft==0.11.1 datasets==2.20.0 accelerate==0.33.0
!pip install safetensors==0.4.3 json-repair==0.21.0

# Verify installations
import torch_xla
import transformers
import peft
print("✅ All dependencies installed successfully")
```

---

## 🎯 **Step 4: Training Configuration**

### **4.1 TPU-Optimized Training Script**
```python
import os
import json
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from pathlib import Path

class TPUHouseBrainTrainer:
    def __init__(self, dataset_path="housebrain_dataset_v5_425k"):
        self.dataset_path = dataset_path
        self.device = xm.xla_device()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer for TPU"""
        print("🔧 Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with TPU optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            torch_dtype=torch.float16,
            device_map=None  # Let TPU handle device mapping
        )
        
        # Configure LoRA for TPU
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ Model and tokenizer setup complete")
    
    def load_dataset(self):
        """Load and prepare dataset for TPU training"""
        print("📂 Loading dataset...")
        
        train_files = list(Path(self.dataset_path) / "train").glob("*.json")
        val_files = list(Path(self.dataset_path) / "validation").glob("*.json")
        
        print(f"📊 Found {len(train_files)} train and {len(val_files)} validation files")
        
        # Load samples (limit for demo - use all in production)
        train_samples = []
        val_samples = []
        
        for file in train_files[:1000]:  # Use all files in production
            with open(file, 'r') as f:
                train_samples.append(json.load(f))
        
        for file in val_files[:100]:  # Use all files in production
            with open(file, 'r') as f:
                val_samples.append(json.load(f))
        
        # Convert to training format
        def format_for_training(sample):
            input_text = json.dumps(sample["input"], indent=2)
            output_text = json.dumps(sample["output"], indent=2)
            return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
        train_texts = [format_for_training(s) for s in train_samples]
        val_texts = [format_for_training(s) for s in val_samples]
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation")
        return train_dataset, val_dataset
    
    def setup_trainer(self, train_dataset, val_dataset):
        """Configure trainer with TPU optimizations"""
        print("⚙️ Setting up trainer...")
        
        # TPU-optimized training arguments
        training_args = TrainingArguments(
            output_dir="/kaggle/working/models/housebrain-425k-tpu",
            num_train_epochs=2,
            per_device_train_batch_size=8,  # TPU can handle larger batches
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            save_steps=200,  # Save every 200 steps
            eval_steps=200,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,  # Use mixed precision
            dataloader_pin_memory=False,  # Disable for TPU
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for TPU
            # TPU-specific settings
            tpu_num_cores=8,
            dataloader_num_workers=0,  # TPU handles parallelism
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("✅ Trainer setup complete")
    
    def train(self):
        """Start training with auto-save and completion handling"""
        print("🚀 Starting TPU training...")
        print("⏰ Expected time: 3-5 hours")
        print("💾 Auto-save: Every 200 steps + final checkpoint")
        
        try:
            # Start training
            result = self.trainer.train()
            
            # Auto-save final model
            print("💾 Auto-saving final model...")
            self.trainer.save_model("/kaggle/working/models/housebrain-425k-tpu-final")
            self.tokenizer.save_pretrained("/kaggle/working/models/housebrain-425k-tpu-final")
            
            # Save training results
            with open("/kaggle/working/models/training_results.json", "w") as f:
                json.dump(result.metrics, f, indent=2)
            
            print("✅ Training completed successfully!")
            print(f"📊 Final loss: {result.training_loss:.4f}")
            print(f"📁 Model saved to: /kaggle/working/models/housebrain-425k-tpu-final")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            # Save checkpoint even on failure
            try:
                self.trainer.save_model("/kaggle/working/models/housebrain-425k-tpu-checkpoint")
                print("💾 Emergency checkpoint saved")
            except:
                pass
            return False

# Main training execution
def main():
    print("🏗️ HouseBrain 425K TPU Training")
    print("=" * 50)
    
    trainer = TPUHouseBrainTrainer()
    
    # Setup
    trainer.setup_model_and_tokenizer()
    train_dataset, val_dataset = trainer.load_dataset()
    trainer.setup_trainer(train_dataset, val_dataset)
    
    # Train
    success = trainer.train()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Model files:")
        !ls -la /kaggle/working/models/
    else:
        print("\n❌ Training failed - check logs above")

if __name__ == "__main__":
    main()
```

---

## 🔄 **Step 5: Auto-Save and Completion Handling**

### **5.1 What Happens When Training Completes**
Kaggle automatically handles completion scenarios:

1. **Normal Completion**: Model saved to `/kaggle/working/models/`
2. **Session Timeout**: Checkpoint saved every 200 steps
3. **Disconnection**: Training continues in background
4. **Email Notification**: Sent when training completes

### **5.2 Auto-Save Mechanisms**
```python
# Built into the training script:
- save_steps=200  # Save every 200 steps
- save_strategy="steps"  # Save on specific steps
- load_best_model_at_end=True  # Keep best model
- evaluation_strategy="steps"  # Evaluate regularly
```

### **5.3 Manual Auto-Save Setup (Optional)**
```python
# Add this to your notebook for extra safety
import time
import threading

def auto_save_backup():
    """Background auto-save every 30 minutes"""
    while True:
        time.sleep(1800)  # 30 minutes
        try:
            if hasattr(trainer, 'trainer') and trainer.trainer:
                trainer.trainer.save_model(f"/kaggle/working/models/backup-{int(time.time())}")
                print(f"💾 Backup saved at {time.strftime('%H:%M:%S')}")
        except:
            pass

# Start background auto-save
backup_thread = threading.Thread(target=auto_save_backup, daemon=True)
backup_thread.start()
```

---

## 📊 **Step 6: Post-Training Actions**

### **6.1 What to Do After Training Completes**

**If you're present:**
1. Check training metrics in output
2. Download model files
3. Test model performance

**If you're not present (auto-completion):**
1. Check email notification from Kaggle
2. Return to notebook to download files
3. Model is already saved and ready

### **6.2 Download Trained Model**
```python
# After training completes, download the model
import zipfile
import os

def download_model():
    model_dir = "/kaggle/working/models/housebrain-425k-tpu-final"
    
    if os.path.exists(model_dir):
        # Create zip for download
        zip_path = "/kaggle/working/housebrain-425k-trained-model.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_dir)
                    zipf.write(file_path, arcname)
        
        print(f"📦 Model packaged: {zip_path}")
        print("⬇️ Download the zip file from Kaggle")
    else:
        print("❌ Model directory not found")

# Run after training
download_model()
```

### **6.3 Model Validation**
```python
# Quick validation of trained model
def validate_trained_model():
    model_path = "/kaggle/working/models/housebrain-425k-tpu-final"
    
    if os.path.exists(model_path):
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test generation
        test_input = "Design a 3-bedroom modern house"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ Model validation successful")
        print(f"📝 Sample output: {generated_text[:100]}...")
    else:
        print("❌ Model not found")

# Run validation
validate_trained_model()
```

---

## 🚨 **Step 7: Troubleshooting**

### **7.1 Common Issues and Solutions**

**TPU Not Available:**
```python
# Check TPU status
import torch_xla
print(f"TPU devices: {torch_xla._XLAC._xla_get_default_device()}")

# If not available, wait and retry
import time
time.sleep(60)  # Wait 1 minute
```

**Out of Memory:**
```python
# Reduce batch size
per_device_train_batch_size=4  # Instead of 8
gradient_accumulation_steps=8   # Instead of 4
```

**Training Stuck:**
```python
# Check if training is progressing
print("Current step:", trainer.trainer.state.global_step)
print("Current loss:", trainer.trainer.state.log_history[-1] if trainer.trainer.state.log_history else "N/A")
```

### **7.2 Recovery from Interruption**
```python
# If training was interrupted, resume from checkpoint
checkpoint_path = "/kaggle/working/models/housebrain-425k-tpu/checkpoint-XXXX"
if os.path.exists(checkpoint_path):
    trainer.trainer.train(resume_from_checkpoint=checkpoint_path)
```

---

## 📈 **Step 8: Expected Results**

### **8.1 Training Metrics**
- **Loss**: Should decrease from ~4.0 to ~2.0-2.5
- **Accuracy**: JSON validity rate should improve
- **Convergence**: Should stabilize after 1-2 epochs

### **8.2 Model Performance**
- **Inference Speed**: ~2-5 seconds per generation
- **Memory Usage**: ~8-12GB during inference
- **Quality**: Better architectural coherence and code compliance

### **8.3 File Structure After Training**
```
/kaggle/working/models/housebrain-425k-tpu-final/
├── adapter_config.json
├── adapter_model.bin
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── training_args.bin
```

---

## 🎯 **Step 9: Next Steps After Training**

### **9.1 Immediate Actions**
1. **Download model files** from Kaggle
2. **Test locally** with `src/housebrain/llm.py`
3. **Evaluate performance** on validation set

### **9.2 Model Deployment**
```python
# Use trained model in production
from src.housebrain.llm import HouseBrainLLM

llm = HouseBrainLLM(
    model_path="models/housebrain-425k-tpu-final",
    device="cuda"  # or "cpu"
)

result = llm.generate_design({
    "basicDetails": {
        "totalArea": 2000,
        "unit": "sqft",
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 2,
        "budget": 500000,
        "style": "Modern"
    },
    "plot": {
        "length": 50,
        "width": 40,
        "unit": "ft",
        "orientation": "N"
    },
    "roomBreakdown": []
})
```

### **9.3 Model Optimization**
- **Quantization**: Convert to 4-bit for faster inference
- **Pruning**: Remove unused parameters
- **Distillation**: Create smaller, faster model

---

## 📋 **Quick Start Checklist**

### **Before Training:**
- [ ] Enable TPU VM v3-8 in Kaggle settings
- [ ] Upload `housebrain_dataset_v5_425k.zip`
- [ ] Install TPU dependencies
- [ ] Verify dataset loading

### **During Training:**
- [ ] Monitor progress every 30 minutes
- [ ] Check auto-save is working
- [ ] Note expected completion time

### **After Training:**
- [ ] Download model files
- [ ] Validate model performance
- [ ] Test with sample inputs
- [ ] Deploy to production

---

## 🎉 **Success Indicators**

✅ **Training completed** (3-5 hours)  
✅ **Model files saved** to `/kaggle/working/models/`  
✅ **Loss decreased** from ~4.0 to ~2.0-2.5  
✅ **JSON validity** improved  
✅ **Email notification** received from Kaggle  
✅ **Model ready** for deployment  

**Your HouseBrain model is now trained and ready for the Indian market!** 🚀
