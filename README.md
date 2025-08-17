# HouseBrain LLM v1.1 - AI-Powered Architectural Design System

## 🎯 Project Overview

HouseBrain LLM is a revolutionary AI system that generates comprehensive architectural designs with **spatial intelligence**, **geometric precision**, and **2D/3D generation capabilities**. Built specifically for the Indian market with NBC 2016 compliance.

## 🚀 Key Features

### 🧠 Advanced AI Capabilities
- **Spatial Intelligence**: Room placement, floor awareness, structural continuity
- **Geometric Construction**: Exact coordinates, wall assemblies, material specifications
- **2D/3D Generation**: Direct floor plan and 3D model generation
- **Technical Depth**: Structural, cost, and energy engineering
- **India-Specific**: NBC 2016 compliance, regional variations, local materials

### 📊 Dataset: 1M Super-Quality Reasoning Samples (Augmented v1.1)
- **Total Samples**: 1,000,000 (900K train, 100K validation)
- **India Ratio**: 60% | **Quality Threshold**: 85%
- **Dataset Size**: 43GB (2.9GB compressed)
- **Geometric Focus**: 74% Geometric_Construction samples
- **Enhanced Precision**: Units, datum, levels, IFC classes, DXF layers, GUIDs

### 🏗️ Problem Types Supported

**High Priority (74%)**
- **Geometric_Construction (25%)**: Exact coordinates, wall assemblies, material quantities
- **Spatial_Floor_Planning (20%)**: Room adjacencies, floor continuity, structural grid
- **Structural_Engineering (15%)**: Load calculations, foundation design, seismic analysis
- **Cost_Engineering (14%)**: BOQ, material costs, labor estimates

**Medium Priority (23%)**
- **Energy_Engineering (8%)**: Thermal analysis, HVAC design, sustainability
- **MEP_Design (7%)**: Electrical, plumbing, mechanical systems
- **Interior_Design (5%)**: Finishes, furniture, lighting design
- **Landscape_Design (3%)**: Site planning, hardscape, softscape

**Lower Priority (3%)**
- **Sustainability_Design (3%)**: Green building, renewable energy
- **Smart_Home_Integration (2%)**: IoT, automation, security systems

## 🔧 Technical Specifications

### Model Architecture
- **Base Model**: DeepSeek-Coder-6.7B-Base (stable, well-supported)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Sequence Length**: 2048 tokens
- **Precision**: FP16/FP32 (auto-detected)

### Dataset Features
- **India Ratio**: 60% India-specific content
- **Quality Threshold**: 85% minimum quality score
- **Geometric Focus**: 45% spatial and geometric intelligence
- **2D/3D Ready**: Direct floor plan and model generation
- **Enhanced Metadata**: Units, datum, levels, IFC classes, DXF layers

## 📋 Quick Start

### Generate Dataset
```bash
# Generate 1M super-quality dataset
python generate_1m_super_quality.py --target 1000000 --quality 0.85 --india 0.60

# Generate 10K test dataset
python generate_1m_super_quality.py --target 10000 --quality 0.85 --india 0.60
```

### Augment Dataset (v1.1)
```bash
# Add precision metadata for 2D/3D generation
python augment_dataset_v1_1.py --input housebrain_dataset_r1_super_1M \
    --output housebrain_dataset_r1_super_1M_aug_v1_1 --workers 8
```

### Train Model (Colab Pro+)
```bash
# Upload to Colab: housebrain_dataset_r1_super_10k_aug.tar.gz
# Run: python colab_proplus_train_r1_super.py
```

## 🎯 What's New in v1.1

### Enhanced Dataset Features
1. **Units & Datum**: mm units, project origin, north angle
2. **Floor Levels**: FFL/TOS elevations, floor-to-floor heights
3. **Element Metadata**: 
   - Doors: Sill/head heights, frame thickness, swing direction
   - Windows: Sill/head heights, frame specifications
   - Walls: Layer assemblies, U-values, material stacks
   - Stairs: Riser/tread dimensions, handrail heights
4. **IFC Integration**: Industry Foundation Classes mapping
5. **DXF Layers**: AutoCAD layer structure with colors/lineweights
6. **GUIDs**: Deterministic element identification
7. **2D Dimensions**: Scaffold for precise dimensioning

### Technical Improvements
- **Spatial Precision**: Exact coordinates for 2D/3D generation
- **Material Quantities**: Detailed BOQ for cost estimation
- **Structural Continuity**: Multi-floor awareness
- **Industry Standards**: IFC/DXF compatibility
- **Deterministic Output**: Consistent element identification

## 📊 Performance Expectations

### Before R1 Training
- Basic architectural generation
- Limited reasoning capabilities
- Surface-level code compliance

### After R1 Super-Quality Training
- **Spatial intelligence**: Advanced room placement and floor awareness
- **Geometric construction**: Exact coordinates and material specifications
- **2D/3D generation**: Direct floor plan and model creation
- **Material specifications**: Precise quantities and specifications
- **India compliance**: NBC 2016 and regional standards
- **Cost engineering**: Accurate BOQ and cost estimates

## 🛠️ Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Local Generation (M2 Pro)
- **Performance**: 50K samples generated locally
- **Ready for**: 1M generation
- **Optimization**: Local M2 Pro generation

### Colab Pro+ Training
- **Hardware**: A100 40GB or V100 16GB
- **Training Time**: 8-12 hours for 1M dataset
- **Memory**: Optimized for Colab Pro+ constraints

## 📁 Project Structure

```
housebrain_v1_1/
├── generate_1m_super_quality.py      # Main dataset generator
├── augment_dataset_v1_1.py           # Dataset augmentation (v1.1)
├── colab_proplus_train_r1_super.py   # Training script
├── colab_setup.py                    # Colab environment setup
├── COLAB_PROPLUS_R1_SUPER_GUIDE.md   # Detailed training guide
├── requirements.txt                  # Python dependencies
└── README.md                         # This file

Datasets:
├── housebrain_dataset_r1_super_1M/           # Original 1M dataset
├── housebrain_dataset_r1_super_1M_aug_v1_1/  # Augmented 1M dataset
└── housebrain_dataset_r1_super_10k_aug.tar.gz # 10K test dataset (32MB)
```

## 🎯 Use Cases

### Architectural Design
- Generate complete building designs from requirements
- Provide exact geometric data for CAD/BIM tools
- Ensure structural and code compliance
- Support multi-floor and complex layouts

### Construction Planning
- Calculate material quantities and costs
- Generate detailed BOQ (Bill of Quantities)
- Plan construction sequences
- Optimize resource allocation

### 2D/3D Generation
- Direct floor plan generation
- 3D model creation
- IFC/DXF export capabilities
- Industry-standard file formats

## 🔄 Development Workflow

1. **Dataset Generation**: Create high-quality training data
2. **Augmentation**: Add precision metadata for 2D/3D generation
3. **Training**: Fine-tune model on Colab Pro+
4. **Evaluation**: Test on architectural problems
5. **Integration**: Connect to CAD/BIM tools
6. **Deployment**: Deploy for real-world use

## 📚 Documentation

- **[Training Guide](COLAB_PROPLUS_R1_SUPER_GUIDE.md)**: Complete Colab Pro+ training instructions
- **[Dataset Generator](generate_1m_super_quality.py)**: Main dataset generation script
- **[Augmentation Script](augment_dataset_v1_1.py)**: Dataset enhancement for v1.1
- **[Training Script](colab_proplus_train_r1_super.py)**: Model training script

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the troubleshooting section in the training guide
- Review error logs carefully
- Ensure all dependencies are installed
- Verify dataset integrity

## 🎉 Acknowledgments

- **DeepSeek AI**: For the excellent R1-Distill-Qwen-7B base model
- **Hugging Face**: For the transformers and peft libraries
- **Google Colab**: For providing GPU resources for training

---

**Ready to revolutionize architectural design with AI! 🏠🧠**

*HouseBrain LLM v1.1 - Where AI meets Architecture*