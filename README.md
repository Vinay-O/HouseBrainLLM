# 🏠 HouseBrain LLM

**An AI-powered architectural design system that generates engineering-grade house plans, 3D models, and construction estimates.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Development-orange.svg)]()

## 🎯 Overview

HouseBrain is a custom Large Language Model (LLM) that acts as an expert architect, civil engineer, and interior designer. It takes structured user input and generates comprehensive architectural designs including:

- **Engineering-grade 2D floor plans** (SVG)
- **Engineering-grade 3D floor plans** (OBJ/GLB)
- **Beautiful 3D elevations** (Blender render)
- **Construction-worthy 3D models**
- **Interior layouts**
- **Construction cost estimates**
- **Construction sequence/flow**

## 🏗️ Architecture

- **Strict JSON Schema**: All outputs follow `schema.py` specifications
- **Multi-floor Support**: Unlimited floors via `levels[]` array
- **Validation Engine**: Room areas, stair design, corridor widths, grid alignment, daylight, code compliance
- **Blender Integration**: Facade style kit for professional renders
- **Cost Estimation**: Realistic construction costs and material requirements

## 📁 Project Structure

```
housebrain_v1_1/
├── src/housebrain/          # Core modules
│   ├── schema.py           # Data structures and validation
│   ├── layout.py           # House layout generation
│   ├── validate.py         # Building code validation
│   ├── llm.py             # LLM interface and reasoning
│   └── finetune.py        # Model fine-tuning pipeline
├── api/                    # FastAPI server
│   └── main.py            # REST API endpoints
├── housebrain_dataset_v5_10k/  # Latest training dataset
├── outputs/               # Generated designs
├── data/                  # Sample data
├── generate_dataset.py    # Dataset generator
├── finetune_housebrain.py # Training script
├── finetune_m2pro.py     # M2 Pro training script
├── test_housebrain.py     # Demo script
├── colab_training.ipynb   # Colab training notebook
├── colab_dataset_generation.ipynb # Colab dataset generation
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Vinay-O/HouseBrainLLM.git
cd HouseBrainLLM

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Demo Mode

```bash
# Run demo with mock AI
python -m api.main --demo
```

This generates:
- `outputs/plan.json` → Full multi-floor JSON
- `outputs/level_0.svg`, `outputs/level_1.svg` → 2D plans
- `outputs/scene.obj` → Basic 3D model

### API Server

```bash
# Start FastAPI server
uvicorn api.main:app --reload

# API endpoints:
# POST /design - Generate house design
# POST /validate - Validate existing design
# GET /renders/{filename} - Serve generated renders
```

## 🎨 Usage Examples

### Generate House Design

```python
from src.housebrain.schema import HouseInput
from src.housebrain.llm import HouseBrainLLM

# Create input
input_data = HouseInput(
    basicDetails={
        "totalArea": 2000,
        "unit": "sqft",
        "bedrooms": 3,
        "floors": 2,
        "budget": 500000,
        "style": "Modern"
    },
    plot={
        "length": 50,
        "width": 40,
        "unit": "ft",
        "orientation": "N"
    },
    roomBreakdown=[
        {"type": "master_bedroom", "count": 1, "minArea": 200},
        {"type": "bedroom", "count": 2, "minArea": 150},
        {"type": "bathroom", "count": 2, "minArea": 60},
        {"type": "kitchen", "count": 1, "minArea": 180},
        {"type": "livingRoom", "count": 1, "minArea": 300}
    ]
)

# Generate design
llm = HouseBrainLLM()
result = llm.generate_design(input_data)
print(f"Construction Cost: ${result.construction_cost}")
```

### Validate Design

```python
from src.housebrain.validate import HouseValidator

validator = HouseValidator()
validation_result = validator.validate(result)
print(f"Compliance Score: {validation_result.compliance_score}%")
```

## 🧠 Model Training

### Generate Training Dataset

```bash
# Generate 50K samples (optimal for Colab)
python generate_dataset.py --samples 50000 --output housebrain_dataset_v5_50k --zip

# Fast mode (skip layout solving)
python generate_dataset.py --samples 50000 --output housebrain_dataset_v5_50k --fast --zip
```

### Train on Google Colab (Free GPU)

1. **Open Colab**: https://colab.research.google.com/
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Run Training**:

```python
# Install dependencies
!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm

# Clone repository
!git clone https://github.com/Vinay-O/HouseBrainLLM.git
%cd HouseBrainLLM

# Upload dataset
from google.colab import files
uploaded = files.upload()

# Extract and train
import zipfile
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')

# Start training
import sys
sys.path.append('src')
from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner

config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    dataset_path="housebrain_dataset_v5_50k",
    output_dir="models/housebrain-trained",
    max_length=1024,
    batch_size=2,
    num_epochs=3,
    use_4bit=True,
)

trainer = HouseBrainFineTuner(config)
trainer.train()
```

### Local Training (M2 Pro)

```bash
# Train on Apple Silicon
python finetune_housebrain.py --dataset housebrain_dataset_v5_50k --epochs 3
```

## 📊 Dataset Information

### Available Datasets

- **v5_10k**: Latest version with advanced features (10K samples)
- **Generate Custom**: Use `generate_dataset.py` for larger datasets

### Dataset Features

- **Realistic Parameters**: Plot sizes, room dimensions, budgets
- **Multiple Styles**: Modern, Traditional, Colonial, Mediterranean, etc.
- **Regional Variations**: US, EU, Asia, Australia
- **Climate Zones**: Tropical, Subtropical, Temperate, Cold
- **Material Specifications**: Exterior, roofing, flooring options

## 🔧 Configuration

### Model Settings

```python
# Fine-tuning configuration
config = FineTuningConfig(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",  # Base model
    dataset_path="housebrain_dataset_v5_50k",           # Dataset
    output_dir="models/housebrain-trained",             # Output
    max_length=1024,                                    # Context length
    batch_size=2,                                       # Batch size
    num_epochs=3,                                       # Training epochs
    learning_rate=2e-4,                                 # Learning rate
    use_4bit=True,                                      # 4-bit quantization
    fp16=True,                                          # Mixed precision
)
```

### Validation Rules

- **Room Sizes**: Minimum area requirements per room type
- **Stair Design**: Width, rise, run, headroom compliance
- **Floor Connectivity**: Proper stair placement between levels
- **Room Adjacency**: Logical room relationships
- **Circulation**: Corridor widths and flow
- **Daylight**: Window placement and natural light
- **Ventilation**: Air flow and mechanical requirements

## 🛠️ Development

### Running Tests

```bash
# Test end-to-end functionality
python test_housebrain.py

# Test specific modules
python -c "from src.housebrain.schema import HouseInput; print('Schema OK')"
python -c "from src.housebrain.layout import LayoutSolver; print('Layout OK')"
python -c "from src.housebrain.validate import HouseValidator; print('Validation OK')"
```

### Code Quality

```bash
# Check syntax
find . -name "*.py" -exec python -m py_compile {} \;

# Remove cache files
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

## 📈 Performance Metrics

### Expected Results

- **Training Time**: 2-4 hours on Colab T4 GPU
- **Model Performance**: 70-85% compliance score
- **Generation Speed**: 5-10 seconds per design
- **Memory Usage**: 8-16GB RAM for training

### Quality Metrics

- **Room Size Compliance**: 95%+
- **Stair Design Compliance**: 90%+
- **Cost Estimation Accuracy**: ±15%
- **Layout Logic**: 85%+

## 🗺️ Roadmap

### Phase 1: Residential Houses ✅
- Single-family homes
- Multi-story designs
- Cost estimation

### Phase 2: Mixed-Use Development 🚧
- Residential + Commercial
- Indian building codes
- Local regulations

### Phase 3: Commercial Buildings 📋
- Office buildings
- Retail spaces
- Industrial facilities

### Phase 4: High-Rise Projects 📋
- Apartment complexes
- Skyscrapers
- Urban planning

### Phase 5: Global Expansion 📋
- North America codes
- European standards
- Australian regulations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek AI** for the base model
- **Hugging Face** for the transformers library
- **FastAPI** for the web framework
- **Pydantic** for data validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Vinay-O/HouseBrainLLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Vinay-O/HouseBrainLLM/discussions)
- **Email**: [Your Email]

---

**Built with ❤️ for the future of architectural design**