# HouseBrain LLM 🏠🧠

**Advanced AI for Architectural Design with DeepSeek-R1 Reasoning**

HouseBrain LLM is a specialized AI model for architectural design, building code compliance, and construction planning. Built on DeepSeek-R1-Distill-Qwen-7B with advanced reasoning capabilities.

## 🚀 Key Features

- **Advanced Reasoning**: Step-by-step architectural problem solving
- **Building Code Compliance**: NBC 2016 (India) and international standards
- **Mathematical Analysis**: Structural calculations and cost optimization
- **Multi-Constraint Optimization**: Balancing conflicting requirements
- **Sustainability Focus**: Energy efficiency and green building expertise
- **Smart Home Integration**: IoT and automation systems
- **India-Specific**: Regional climate, materials, and regulatory compliance

## 🎯 Problem Types Supported

### **High Priority (45% - 2D/3D Generation Focus)**
1. **Geometric_Construction** (25%) - Exact coordinates, construction geometry, 2D/3D generation
2. **Spatial_Floor_Planning** (20%) - Room placement, spatial relationships, multi-floor continuity

### **Medium Priority (35% - Technical Foundation)**
3. **Basic_Design** (10%) - Standard architectural design
4. **Code_Compliance** (8%) - Building code analysis and compliance
5. **Multi_Constraint** (8%) - Balancing multiple conflicting requirements
6. **Optimization** (8%) - Mathematical optimization and analysis
7. **Structural_Engineering** (12%) - Engineering design and analysis

### **Lower Priority (20% - Reduced Focus)**
8. **Conflict_Resolution** (5%) - Stakeholder conflict resolution
9. **Advanced_Reasoning** (5%) - Complex multi-step reasoning
10. **Mathematical_Analysis** (5%) - Structural and financial calculations
11. **Sustainability_Design** (3%) - Green building and LEED compliance
12. **Performance_Optimization** (2%) - Multi-metric performance optimization
13. **Smart_Home_Integration** (2%) - IoT and automation systems

## 📊 Dataset

### Super-Quality Dataset (1M samples)
- **Total Samples**: 1,000,000
- **Training**: 900,000 (90%)
- **Validation**: 100,000 (10%)
- **India Ratio**: 60%
- **Quality Threshold**: 85%
- **Problem Types**: 13 different reasoning categories
- **Geometric Focus**: 45% spatial and geometric intelligence
- **2D/3D Ready**: Direct floor plan and model generation

## 🏗️ Architecture

- **Base Model**: DeepSeek-R1-Distill-Qwen-7B
- **Parameters**: 7B (distilled from 671B R1)
- **Sequence Length**: 2048 (optimized for reasoning)
- **Training**: LoRA fine-tuning with chat-style formatting
- **Hardware**: Colab Pro+ (A100 40GB/V100 16GB)

## 🚀 Quick Start

### 1. Generate Dataset

```bash
# Generate 1M super-quality samples (optimized for 2D/3D generation)
python generate_1m_super_quality.py --target 1000000 --quality 0.85 --india 0.60 --shard 100000

# Or generate in batches (recommended for local generation)
python generate_1m_super_quality.py --target 200000 --quality 0.85 --india 0.60 --shard 50000
# Repeat 5 times to get 1M total

# Local generation on M2 Pro (tested)
python generate_1m_super_quality.py --target 50000 --quality 0.85 --india 0.60 --shard 5000
# ~440 samples/minute, 1 hour 53 minutes for 50K samples
```

### 2. Train on Colab Pro+

```python
# Install dependencies
!pip install torch==2.1.0 transformers==4.36.0 peft==0.7.0 accelerate==0.25.0 datasets==2.15.0 tqdm

# Upload dataset and training script
from google.colab import files
uploaded = files.upload()  # Select housebrain_dataset_r1_super_1M.tar.gz and colab_proplus_train_r1_super.py

# Extract and train
!tar -xzf housebrain_dataset_r1_super_1M.tar.gz
!python colab_proplus_train_r1_super.py
```

### 3. Use Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load trained model
base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
model = PeftModel.from_pretrained(base_model, "housebrain-r1-super-trained/final")
tokenizer = AutoTokenizer.from_pretrained("housebrain-r1-super-trained/final")

# Generate architectural design
test_input = {
    "problem_type": "Structural_Engineering",
    "context": {"indian_market": True, "region": "Mumbai"},
    "plot_details": {"area_sqft": 2500, "floors": 3},
    "requirements": {"budget_inr": 5000000}
}

input_text = json.dumps(test_input)
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📁 Project Structure

```
housebrain_v1_1/
├── generate_1m_super_quality.py          # Dataset generator
├── colab_proplus_train_r1_super.py       # Training script
├── monitor_training.py                   # Training monitor
├── resume_training.py                    # Resume training
├── merge_models.py                       # Model merging
├── requirements.txt                      # Dependencies
├── COLAB_PROPLUS_R1_SUPER_GUIDE.md      # Training guide
├── COMPLETE_TRAINING_GUIDE.md           # Complete guide
├── FUTURE_ROADMAP.md                    # Future development
└── src/                                 # Source code
    └── housebrain/
        ├── finetune.py                  # Core fine-tuning
        └── llm.py                       # LLM interface
```

## 🎯 Performance Expectations

### Before R1 Training
- Basic architectural generation
- Limited reasoning capabilities
- Surface-level code compliance

### After R1 Super-Quality Training
- **Advanced reasoning**: Step-by-step problem solving
- **Mathematical analysis**: Structural and cost calculations
- **Code compliance**: Detailed NBC 2016 analysis
- **Multi-constraint optimization**: Balancing conflicting requirements
- **Spatial intelligence**: Room placement and spatial relationships
- **Geometric construction**: Exact coordinates and construction geometry
- **2D/3D generation**: Direct floor plan and model generation
- **Material specifications**: Detailed BOQ and cost estimation

### Metrics
- **Training Loss**: Should decrease from ~3.0 to ~1.5
- **Validation Loss**: Should stabilize around ~1.8
- **Reasoning Quality**: 90%+ step-by-step reasoning
- **Code Compliance**: 95%+ NBC 2016 accuracy
- **Mathematical Accuracy**: 85%+ calculation correctness

## 🛠️ Training Configuration

### Model Settings
- **Base Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **Sequence Length**: 2048 (doubled for R1 reasoning)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05

### Training Settings
- **Batch Size**: 4 (optimized for A100/V100)
- **Gradient Accumulation**: 8 (effective batch size = 32)
- **Learning Rate**: 2e-4
- **Max Steps**: 50,000
- **Warmup Steps**: 100
- **Save Steps**: 1,000
- **Eval Steps**: 500

### Memory Requirements
- **Model**: ~14GB (FP16)
- **Training**: ~18-22GB total
- **A100 40GB**: ✅ Perfect fit
- **V100 16GB**: ⚠️ May need 4-bit quantization

## 📚 Documentation

- **[Colab Pro+ Training Guide](COLAB_PROPLUS_R1_SUPER_GUIDE.md)** - Complete training guide
- **[Complete Training Guide](COMPLETE_TRAINING_GUIDE.md)** - Detailed setup instructions
- **[Future Roadmap](FUTURE_ROADMAP.md)** - Development plans

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek AI** for the R1 model architecture
- **Hugging Face** for the transformers library
- **Google Colab** for the training platform

---

**Note**: This training will take 8-12 hours on Colab Pro+. Ensure you have stable internet connection and consider using Colab's "Keep alive" extensions for long training sessions.