# HouseBrain LLM - Colab Pro+ Training Guide (R1 Super-Quality)

## 🎯 Overview

This guide covers training HouseBrain LLM on Google Colab Pro+ using the new **1M super-quality reasoning dataset** with **DeepSeek-R1-Distill-Qwen-7B** model.

### Key Features
- **Model**: DeepSeek-R1-Distill-Qwen-7B (7B parameters, ~14GB FP16)
- **Dataset**: 1M super-quality reasoning samples with 74% geometric construction focus
- **Hardware**: Colab Pro+ (A100 40GB/V100 16GB)
- **Training**: LoRA fine-tuning with chat-style formatting
- **2D/3D Ready**: Direct floor plan and model generation capabilities

## 📊 Dataset Details

### Super-Quality Dataset (`housebrain_dataset_r1_super_1M`)
- **Total Samples**: 1,000,000
- **Training**: 900,196 (90%)
- **Validation**: 99,804 (10%)
- **India Ratio**: 60%
- **Quality Threshold**: 85%
- **Dataset Size**: 43GB (2.9GB compressed)
- **Geometric Focus**: 74% Geometric_Construction samples

### Problem Types (Optimized for 2D/3D Generation)

#### **High Priority (74% - Geometric Focus)**
1. **Geometric_Construction** (74%) - Exact coordinates, construction geometry, 2D/3D generation
2. **Structural_Engineering** (45%) - Engineering design and analysis

#### **Medium Priority (30% - Technical Foundation)**
3. **Basic_Design** (30%) - Standard architectural design
4. **Code_Compliance** - Building code analysis and compliance
5. **Multi_Constraint** - Balancing multiple conflicting requirements
6. **Conflict_Resolution** - Stakeholder conflict resolution
7. **Advanced_Reasoning** - Complex multi-step reasoning
8. **Mathematical_Analysis** - Structural and financial calculations

#### **Lower Priority (15% - Reduced Focus)**
9. **Sustainability_Design** (16%) - Green building and LEED compliance
10. **Smart_Home_Integration** (15%) - IoT and automation systems

## 🚀 Quick Start

### 1. Generate Dataset (Local - COMPLETED ✅)

```bash
# ✅ 1M samples generated locally on M2 Pro
python generate_1m_super_quality.py --target 1000000 --quality 0.85 --india 0.60 --shard 50000

# Results:
# - 1,000,000 samples in 8 hours 36 minutes
# - 74% Geometric_Construction focus
# - 43GB dataset (2.9GB compressed)
# - 46.13% acceptance rate
```

### 2. Prepare for Colab (COMPLETED ✅)

```bash
# ✅ Dataset compressed and ready
tar -czf housebrain_dataset_r1_super_1M.tar.gz housebrain_dataset_r1_super_1M/

# ✅ Size: 2.9GB (compressed), 43GB (extracted)
# ✅ Ready for Colab Pro+ upload
```

### 3. Colab Pro+ Setup

#### Notebook Setup
1. **Runtime Type**: GPU (A100 or V100)
2. **Hardware Accelerator**: GPU
3. **Runtime Shape**: High-RAM (if available)

#### Upload Dataset
```python
# Upload the tar.gz file to Colab
from google.colab import files
uploaded = files.upload()  # Select housebrain_dataset_r1_super_1M.tar.gz

# Extract dataset
!tar -xzf housebrain_dataset_r1_super_1M.tar.gz
!ls -la housebrain_dataset_r1_super_1M/
```

#### Install Dependencies
```python
!pip install torch==2.1.0 transformers==4.36.0 peft==0.7.0 accelerate==0.25.0 datasets==2.15.0 tqdm
```

#### Upload Training Script
```python
# Upload the training script
from google.colab import files
uploaded = files.upload()  # Select colab_proplus_train_r1_super.py
```

## 🏋️ Training Configuration

### Model Settings
- **Base Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **Sequence Length**: 2048 (doubled for R1 reasoning)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05

### Training Settings (Optimized for 1M Dataset)
- **Batch Size**: 4 (optimized for A100/V100)
- **Gradient Accumulation**: 8 (effective batch size = 32)
- **Max Steps**: 100,000 (increased for 1M dataset)
- **Save Steps**: 2,000 (more frequent saves)
- **Eval Steps**: 1,000 (more frequent evaluation)
- **Warmup Steps**: 100
- **Learning Rate**: 2e-4
- **Eval Steps**: 1,000

### Memory Requirements
- **Model**: ~14GB (FP16)
- **Training**: ~18-22GB total
- **A100 40GB**: ✅ Perfect fit
- **V100 16GB**: ⚠️ May need 4-bit quantization

## 📈 Training Execution

### Start Training
```python
!python colab_proplus_train_r1_super.py
```

### Expected Output
```
🔧 Setting up environment...
✅ GPU: Tesla A100-SXM4-40GB (40.0GB)
🤖 Loading DeepSeek-R1-Distill-Qwen-7B model and tokenizer...
📊 Loading super-quality dataset...
📈 Training samples: 900,000
📉 Validation samples: 100,000
🔧 Tokenizing and applying chat formatting...
🚀 Starting training...
📊 Config: LR=2e-4, Batch=4, SeqLen=2048
🎯 Target: 50000 steps
```

### Training Timeline
- **Setup**: ~5-10 minutes
- **Training**: ~8-12 hours (50k steps)
- **Checkpoints**: Every 1k steps
- **Evaluation**: Every 500 steps

## 📊 Monitoring

### Built-in Logging
- **Log File**: `training_log_r1_super.txt`
- **Metrics**: `training_metrics_r1_super.json`
- **Checkpoints**: `housebrain-r1-super-trained/`

### Monitor Progress
```python
# Check training status
!tail -f training_log_r1_super.txt

# Plot metrics
import json
import matplotlib.pyplot as plt

with open('training_metrics_r1_super.json', 'r') as f:
    metrics = json.load(f)

steps = [m['step'] for m in metrics]
losses = [m.get('train_loss', 0) for m in metrics]
eval_losses = [m.get('eval_loss', 0) for m in metrics]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, losses)
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(steps, eval_losses)
plt.title('Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
```

## 🛠️ Troubleshooting

### Memory Issues
If you encounter OOM errors on V100:

```python
# Modify in colab_proplus_train_r1_super.py
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Dataset Issues
```python
# Check dataset structure
!find housebrain_dataset_r1_super_1M -name "*.json" | wc -l
!ls housebrain_dataset_r1_super_1M/train/
!ls housebrain_dataset_r1_super_1M/validation/
```

### Training Interruption
```python
# Resume from checkpoint
!python colab_proplus_train_r1_super.py --resume_from_checkpoint housebrain-r1-super-trained/checkpoint-10000
```

## 📦 Post-Training

### Download Model
```python
# Create archive for download
!tar -czf housebrain-r1-super-trained.tar.gz housebrain-r1-super-trained/

# Download
from google.colab import files
files.download('housebrain-r1-super-trained.tar.gz')
```

### Test Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load trained model
base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
model = PeftModel.from_pretrained(base_model, "housebrain-r1-super-trained/final")
tokenizer = AutoTokenizer.from_pretrained("housebrain-r1-super-trained/final")

# Test complex reasoning
test_input = {
    "problem_type": "Structural_Engineering",
    "context": {"indian_market": True, "region": "Mumbai"},
    "plot_details": {"area_sqft": 2500, "floors": 3},
    "requirements": {"budget_inr": 5000000}
}

# Generate response
input_text = json.dumps(test_input)
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🎯 Expected Performance

### Before R1 Training
- Basic architectural generation
- Limited reasoning capabilities
- Surface-level code compliance

### After R1 Super-Quality Training
- **Advanced reasoning**: Step-by-step problem solving
- **Mathematical analysis**: Structural and cost calculations
- **Code compliance**: Detailed NBC 2016 analysis
- **Multi-constraint optimization**: Balancing conflicting requirements
- **Sustainability focus**: Energy and green building expertise
- **Smart home integration**: IoT and automation knowledge

### Metrics
- **Training Loss**: Should decrease from ~3.0 to ~1.5
- **Validation Loss**: Should stabilize around ~1.8
- **Reasoning Quality**: 90%+ step-by-step reasoning
- **Code Compliance**: 95%+ NBC 2016 accuracy
- **Mathematical Accuracy**: 85%+ calculation correctness

## 🔄 Alternative Approaches

### Smaller Dataset Training
```bash
# Train with 500K samples for faster iteration
python generate_1m_super_quality.py --target 500000 --quality 0.92
```

### Lower Quality Threshold
```bash
# Faster generation with 85% quality
python generate_1m_super_quality.py --quality 0.85
```

### 4-bit Training (V100)
```python
# Enable 4-bit quantization for V100
load_in_4bit=True
bnb_4bit_compute_dtype=torch.bfloat16
```

## 📚 Resources

- **Dataset Generator**: `generate_1m_super_quality.py`
- **Training Script**: `colab_proplus_train_r1_super.py`
- **Monitoring**: `monitor_training.py`
- **Resume Training**: `resume_training.py`
- **Model Merging**: `merge_models.py`

## 🎉 Success Checklist

- [ ] Dataset generated (1M samples)
- [ ] Colab Pro+ GPU runtime active
- [ ] Dataset uploaded and extracted
- [ ] Dependencies installed
- [ ] Training script uploaded
- [ ] Training started successfully
- [ ] Checkpoints saving regularly
- [ ] Loss decreasing over time
- [ ] Model downloaded after completion
- [ ] Model tested with sample inputs

---

**Note**: This training will take 8-12 hours on Colab Pro+. Ensure you have stable internet connection and consider using Colab's "Keep alive" extensions for long training sessions.
