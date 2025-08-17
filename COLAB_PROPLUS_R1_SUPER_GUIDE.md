# HouseBrain LLM - Colab Pro+ Training Guide v1.1

## 🎯 Project Overview
HouseBrain LLM is an AI-powered architectural design system that generates comprehensive building designs with:
- **Spatial Intelligence**: Room placement, floor awareness, structural continuity
- **Geometric Construction**: Exact coordinates, wall thicknesses, material specifications
- **2D/3D Generation**: Direct floor plan and 3D model generation capabilities
- **India-Specific**: NBC 2016 compliance, regional variations, local materials

## 🚀 Key Features

### Dataset: 1M Super-Quality Reasoning Samples (Augmented v1.1)
- **Training**: 900,196 samples | **Validation**: 99,804 samples
- **India Ratio**: 60% | **Quality Threshold**: 85%
- **Dataset Size**: 43GB (2.9GB compressed)
- **Geometric Focus**: 74% Geometric_Construction samples
- **2D/3D Ready**: Direct floor plan and model generation capabilities
- **Enhanced Precision**: Units, datum, levels, IFC classes, DXF layers, GUIDs

### Problem Types (Priority Distribution)
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

## 📋 Quick Start Guide

### Step 1: Upload Files to Colab
Upload these files to your Colab notebook:
1. `housebrain_dataset_r1_super_10k_aug.tar.gz` (32MB - for testing)
2. `colab_setup.py` (environment setup)
3. `colab_proplus_train_r1_super.py` (training script)

### Step 2: Environment Setup
```python
# Run setup script
!python colab_setup.py
```

This will:
- Check GPU availability
- Install dependencies
- Extract dataset
- Validate dataset quality

### Step 3: Training Configuration
The training script is pre-configured for optimal performance:

**Model Settings**
- **Base Model**: DeepSeek-R1-Distill-Qwen-7B
- **LoRA Rank**: 64 | **Alpha**: 128
- **Dropout**: 0.1 | **Target Modules**: q_proj, v_proj

**Training Settings**
- **Max Steps**: 100,000 (for 1M dataset)
- **Save Steps**: 2,000
- **Eval Steps**: 1,000
- **Learning Rate**: 2e-4
- **Batch Size**: 1 (gradient accumulation: 16)
- **Sequence Length**: 4096
- **Warmup Steps**: 1,000

**Memory Optimization**
- **Precision**: BF16
- **Gradient Checkpointing**: Enabled
- **CPU Offloading**: Enabled
- **Mixed Precision**: Enabled

### Step 4: Start Training
```python
# For 10K test run
!python colab_proplus_train_r1_super.py

# For full 1M training (after successful test)
# Upload housebrain_dataset_r1_super_1M_aug_v1_1.tar.gz
!python colab_proplus_train_r1_super.py
```

### Step 5: Monitor Training
- **Loss**: Should decrease from ~2.5 to ~1.2
- **Learning Rate**: Automatically scheduled
- **Checkpoints**: Saved every 2,000 steps
- **Evaluation**: Every 1,000 steps

## 🔧 What's New in v1.1

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

## 📊 Expected Performance

### After R1 Super-Quality Training
- **Architectural Reasoning**: Advanced spatial and geometric intelligence
- **Technical Depth**: Industry-standard engineering calculations
- **2D/3D Generation**: Direct floor plan and model creation
- **Material Specifications**: Precise quantities and specifications
- **India Compliance**: NBC 2016 and regional standards
- **Cost Engineering**: Accurate BOQ and cost estimates

### Model Capabilities
- Generate complete building designs from requirements
- Provide exact geometric data for CAD/BIM tools
- Calculate material quantities and costs
- Ensure structural and code compliance
- Support multi-floor and complex layouts

## 🛠️ Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size or sequence length
2. **Training Speed**: Increase gradient accumulation
3. **Loss Issues**: Check learning rate and warmup steps
4. **Dataset Loading**: Verify file paths and permissions

### Performance Tips
- Use Colab Pro+ for better GPU and longer sessions
- Monitor memory usage with `!nvidia-smi`
- Save checkpoints frequently
- Use mixed precision for speed

## 📁 File Structure

```
housebrain_dataset_r1_super_10k_aug/
├── dataset_info.json
├── train/
│   └── shard_01/ (9,000 files)
└── validation/
    └── shard_01/ (1,000 files)

housebrain_dataset_r1_super_1M_aug_v1_1/
├── train/
│   ├── shard_01/ (50,000 files)
│   ├── shard_02/ (50,000 files)
│   └── ... (18 shards total)
└── validation/
    ├── shard_01/ (5,000 files)
    ├── shard_02/ (5,000 files)
    └── ... (20 shards total)
```

## 🎯 Next Steps

1. **Test Run**: Use 10K dataset to validate setup
2. **Full Training**: Train on 1M augmented dataset
3. **Model Evaluation**: Test on architectural problems
4. **2D/3D Integration**: Connect to CAD/BIM tools
5. **Production Deployment**: Deploy for real-world use

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review error logs carefully
- Ensure all dependencies are installed
- Verify dataset integrity

---

**Ready to train your HouseBrain LLM! 🏠🧠**
