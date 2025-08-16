# HouseBrain LLM - Architectural AI Model

## 🏗️ Overview

HouseBrain is a specialized Large Language Model (LLM) designed for architectural design and construction planning. Built on DeepSeek Coder 6.7B, it generates comprehensive house designs, floor plans, and construction documentation with a focus on Indian market requirements.

## 🚀 Features

- **Architectural Design Generation**: Complete house designs with room breakdowns
- **Indian Market Focus**: NBC 2016 compliance, regional variations, climate considerations
- **Construction Documentation**: BOM, cost estimates, timelines, permit requirements
- **Quality Gates**: Schema validation, code compliance, area consistency checks
- **2D/3D Pipeline**: Floor plan generation and 3D model integration (planned)

## 📊 Dataset

- **Current**: 425K high-quality samples
- **Target**: 1M samples with quality gates
- **Focus**: Indian residential and commercial projects
- **Quality**: ≥98% JSON parse rate, NBC 2016 compliance

## 🛠️ Technology Stack

- **Base Model**: DeepSeek Coder 6.7B
- **Training**: LoRA fine-tuning with BF16 precision
- **Hardware**: Optimized for Colab Pro+ (A100/V100)
- **Framework**: PyTorch, Transformers, PEFT

## 📁 Repository Structure

```
housebrain_v1_1/
├── src/housebrain/              # Core modules
│   ├── finetune.py             # Training script
│   ├── llm.py                  # Inference interface
│   ├── schema.py               # Data schemas
│   ├── validate.py             # Quality validation
│   └── layout.py               # Layout utilities
├── api/                        # FastAPI server
├── housebrain_dataset_v5_425k/ # Current dataset
├── generate_combined_dataset.py # Dataset combination
├── generate_india_dataset.py   # India-specific data
├── generate_enhanced_dataset.py # Enhanced generation (1M target)
├── merge_models.py             # LoRA merging
├── colab_proplus_train.py      # Colab Pro+ training script
├── COLAB_PROPLUS_TRAINING.md   # Training guide
├── FUTURE_ROADMAP.md           # Development roadmap
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### 1. Colab Pro+ Setup

1. **Subscribe to Google Colab Pro+**
2. **Create new notebook**: `HouseBrain_ProPlus_Training`
3. **Set runtime**: GPU A100 (or V100), High-RAM
4. **Mount Google Drive** for model storage

### 2. Install Dependencies

```python
!pip install -q transformers==4.36.0 peft==0.7.0 accelerate==0.25.0 datasets==2.15.0
!pip install -q wandb bitsandbytes==0.41.1
!pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Start Training

```python
# Upload dataset to Colab
# Run the training script
!python colab_proplus_train.py
```

## 🎯 Training Configuration

### A100 Optimized Settings
- **Sequence Length**: 1024-1536 tokens
- **LoRA Rank**: 8-16
- **Batch Size**: Effective 128 (via gradient accumulation)
- **Learning Rate**: 1.5e-4 with cosine scheduler
- **Precision**: BF16
- **Training Time**: 4-6 hours for 425K samples

### Quality Metrics
- **JSON Parse Rate**: ≥98%
- **Code Compliance**: ≥95%
- **India-Specific Tasks**: +15-20% improvement
- **Validation Loss**: Monitored with early stopping

## 📈 Dataset Expansion

### Current Status
- **425K samples** with quality validation
- **40% India-specific** data
- **NBC 2016 compliance** enforced

### Expansion Plan (1M Target)
- **Daily Generation**: 50K → 20K after quality gates
- **Weekly Shards**: 150-200K samples per training run
- **Quality Threshold**: 0.8 minimum score
- **Regional Focus**: 12 major Indian cities

## 🔧 Development

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd housebrain_v1_1

# Install dependencies
pip install -r requirements.txt

# Run validation
python -m src.housebrain.validate

# Generate dataset
python generate_enhanced_dataset.py
```

### API Development
```bash
# Start FastAPI server
cd api
uvicorn main:app --reload
```

## 📊 Performance

### Training Performance (Colab Pro+)
- **A100 40GB**: 4-6 hours for 425K samples
- **V100 32GB**: 6-8 hours for 425K samples
- **Memory Usage**: 30-35GB (comfortable headroom)
- **Throughput**: 2-4x faster than P100

### Model Quality
- **JSON Generation**: 98%+ parse rate
- **Code Compliance**: 95%+ NBC 2016 adherence
- **Regional Accuracy**: 90%+ for Indian markets
- **Cost Estimation**: ±10% accuracy

## 🗺️ Roadmap

### Phase 1: Baseline Training (Week 1)
- ✅ Complete 425K training on A100
- ✅ Establish baseline metrics
- ✅ Implement quality gates

### Phase 2: Data Expansion (Week 2-3)
- 🔄 Expand to 800K samples
- 🔄 Add hard cases and documentation tasks
- 🔄 Implement curriculum learning

### Phase 3: 2D/3D Integration (Week 4)
- 📋 2D floor plan generation
- 📋 3D model integration
- 📋 Construction documentation automation

### Future Goals
- 🤖 Multi-turn design refinement
- 🏢 Commercial building support
- 🌍 International market expansion
- 🔗 BIM integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and validation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the [training guide](COLAB_PROPLUS_TRAINING.md)
- Review the [future roadmap](FUTURE_ROADMAP.md)

---

**HouseBrain** - Building the future of architectural AI, one design at a time. 🏠✨