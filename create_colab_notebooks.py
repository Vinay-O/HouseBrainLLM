#!/usr/bin/env python3
"""
Create Colab notebooks for HouseBrain training and dataset generation
"""

import json
import os

def create_colab_training_notebook():
    """Create the Colab training notebook"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "header"},
                "source": [
                    "# 🏠 HouseBrain LLM Training\n\n",
                    "**Train your custom architectural AI on Google Colab (Free GPU)**\n\n",
                    "This notebook will help you train the HouseBrain LLM using QLoRA fine-tuning on the DeepSeek model.\n\n",
                    "---\n\n",
                    "## 📋 Prerequisites\n\n",
                    "1. **Enable GPU**: Runtime → Change runtime type → GPU (T4)\n",
                    "2. **Upload Dataset**: You'll need a HouseBrain dataset zip file\n",
                    "3. **Patience**: Training takes 2-4 hours\n\n",
                    "## 🎯 What You'll Get\n\n",
                    "- **Trained Model**: Ready-to-use HouseBrain LLM\n",
                    "- **Performance**: 70-85% architectural compliance\n",
                    "- **Cost**: Completely free (Google Colab)\n\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "setup"},
                "source": ["## 🚀 Step 1: Setup Environment"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install_deps"},
                "outputs": [],
                "source": [
                    "# Install required dependencies\n",
                    "!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic orjson svgwrite trimesh python-dotenv\n\n",
                    "print(\"✅ Dependencies installed successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "clone_repo"},
                "outputs": [],
                "source": [
                    "# Clone the HouseBrain repository\n",
                    "!git clone https://github.com/Vinay-O/HouseBrainLLM.git\n",
                    "%cd HouseBrainLLM\n\n",
                    "print(\"✅ Repository cloned successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload_dataset"},
                "source": [
                    "## 📁 Step 2: Upload Dataset\n\n",
                    "Upload your HouseBrain dataset zip file (generated locally with `generate_dataset.py`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload"},
                "outputs": [],
                "source": [
                    "# Upload your dataset zip file\n",
                    "from google.colab import files\n",
                    "uploaded = files.upload()\n\n",
                    "print(f\"📦 Uploaded files: {list(uploaded.keys())}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "extract_dataset"},
                "outputs": [],
                "source": [
                    "# Extract the dataset\n",
                    "import zipfile\n",
                    "import os\n\n",
                    "for filename in uploaded.keys():\n",
                    "    if filename.endswith('.zip'):\n",
                    "        print(f\"📂 Extracting {filename}...\")\n",
                    "        with zipfile.ZipFile(filename, 'r') as zip_ref:\n",
                    "            zip_ref.extractall('.')\n",
                    "        print(f\"✅ Extracted {filename}\")\n\n",
                    "# List available datasets\n",
                    "datasets = [d for d in os.listdir('.') if d.startswith('housebrain_dataset') and os.path.isdir(d)]\n",
                    "print(f\"\\n📊 Available datasets: {datasets}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "configure_training"},
                "source": ["## ⚙️ Step 3: Configure Training\n\n", "Set up your training configuration"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "import_modules"},
                "outputs": [],
                "source": [
                    "# Import training modules\n",
                    "import sys\n",
                    "sys.path.append('src')\n\n",
                    "from housebrain.finetune import FineTuningConfig, HouseBrainFineTuner\n",
                    "import torch\n\n",
                    "print(\"✅ Training modules imported successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "check_gpu"},
                "outputs": [],
                "source": [
                    "# Check GPU availability\n",
                    "if torch.cuda.is_available():\n",
                    "    gpu_name = torch.cuda.get_device_name(0)\n",
                    "    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
                    "    print(f\"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)\")\n",
                    "else:\n",
                    "    print(\"⚠️  No GPU detected. Training will be very slow on CPU.\")\n",
                    "    print(\"   Please enable GPU: Runtime → Change runtime type → GPU\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "training_config"},
                "outputs": [],
                "source": [
                    "# Training configuration\n",
                    "dataset_name = datasets[0] if datasets else \"housebrain_dataset_v5_50k\"  # Use first available dataset\n\n",
                    "config = FineTuningConfig(\n",
                    "    model_name=\"deepseek-ai/deepseek-coder-6.7b-base\",\n",
                    "    dataset_path=dataset_name,\n",
                    "    output_dir=\"models/housebrain-colab-trained\",\n",
                    "    max_length=1024,\n",
                    "    batch_size=2,  # Adjust based on GPU memory\n",
                    "    num_epochs=3,\n",
                    "    learning_rate=2e-4,\n",
                    "    use_4bit=True,  # Enable for CUDA\n",
                    "    fp16=True,      # Enable for CUDA\n",
                    "    warmup_steps=100,\n",
                    "    logging_steps=50,\n",
                    "    save_steps=500,\n",
                    "    eval_steps=500,\n",
                    "    gradient_accumulation_steps=4,\n",
                    "    lora_r=16,\n",
                    "    lora_alpha=32,\n",
                    "    lora_dropout=0.1,\n",
                    ")\n\n",
                    "print(f\"📋 Training Configuration:\")\n",
                    "print(f\"   Model: {config.model_name}\")\n",
                    "print(f\"   Dataset: {config.dataset_path}\")\n",
                    "print(f\"   Output: {config.output_dir}\")\n",
                    "print(f\"   Epochs: {config.num_epochs}\")\n",
                    "print(f\"   Batch Size: {config.batch_size}\")\n",
                    "print(f\"   Learning Rate: {config.learning_rate}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "start_training"},
                "source": ["## 🚀 Step 4: Start Training\n\n", "This will take 2-4 hours. Make sure to keep the notebook active!"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "initialize_trainer"},
                "outputs": [],
                "source": [
                    "# Initialize trainer\n",
                    "print(\"🔧 Setting up trainer...\")\n",
                    "trainer = HouseBrainFineTuner(config)\n",
                    "print(\"✅ Trainer initialized successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "train_model"},
                "outputs": [],
                "source": [
                    "# Start training\n",
                    "print(\"🎯 Starting training...\")\n",
                    "print(\"⏰ This will take 2-4 hours. Keep the notebook active!\")\n",
                    "print(\"📊 Monitor progress below:\")\n\n",
                    "try:\n",
                    "    trainer.train()\n",
                    "    print(\"\\n🎉 Training completed successfully!\")\n",
                    "except Exception as e:\n",
                    "    print(f\"\\n❌ Training failed: {e}\")\n",
                    "    print(\"💡 Try reducing batch_size or using a smaller model\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "save_model"},
                "source": ["## 💾 Step 5: Save Model\n\n", "Save your trained model for download"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "save_trained_model"},
                "outputs": [],
                "source": [
                    "# Save the trained model\n",
                    "print(\"💾 Saving model...\")\n",
                    "trainer.save_model()\n",
                    "print(\"✅ Model saved successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "create_zip"},
                "outputs": [],
                "source": [
                    "# Create zip archive for download\n",
                    "import zipfile\n",
                    "import os\n\n",
                    "model_dir = config.output_dir\n",
                    "zip_path = \"housebrain-model.zip\"\n\n",
                    "print(f\"📦 Creating zip archive: {zip_path}\")\n\n",
                    "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:\n",
                    "    for root, dirs, files in os.walk(model_dir):\n",
                    "        for file in files:\n",
                    "            file_path = os.path.join(root, file)\n",
                    "            arcname = os.path.relpath(file_path, model_dir)\n",
                    "            zipf.write(file_path, arcname)\n\n",
                    "print(f\"✅ Zip archive created: {zip_path}\")\n",
                    "print(f\"📁 Archive size: {os.path.getsize(zip_path) / 1e6:.1f} MB\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "download_model"},
                "outputs": [],
                "source": [
                    "# Download the trained model\n",
                    "from google.colab import files\n\n",
                    "print(\"⬇️  Downloading trained model...\")\n",
                    "files.download(zip_path)\n",
                    "print(\"✅ Model downloaded successfully!\")"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('colab_training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Created colab_training.ipynb")

def create_colab_dataset_notebook():
    """Create the Colab dataset generation notebook"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "header"},
                "source": [
                    "# 🏗️ HouseBrain Dataset Generation\n\n",
                    "**Generate massive training datasets on Google Colab (Free CPU/GPU)**\n\n",
                    "This notebook will help you generate large HouseBrain datasets for training your custom LLM.\n\n",
                    "---\n\n",
                    "## 📋 Strategy\n\n",
                    "1. **Generate Dataset on Colab** (this notebook)\n",
                    "2. **Download Dataset** to your computer\n",
                    "3. **Upload to Kaggle** for training\n",
                    "4. **Train Model on Kaggle** (separate notebook)\n\n",
                    "## 🎯 What You'll Get\n\n",
                    "- **Large Dataset**: 50K-100K+ samples\n",
                    "- **High Quality**: Realistic architectural parameters\n",
                    "- **Fast Generation**: Colab's powerful CPU\n",
                    "- **Free**: No cost involved\n\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "setup"},
                "source": ["## 🚀 Step 1: Setup Environment"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install_deps"},
                "outputs": [],
                "source": [
                    "# Install required dependencies\n",
                    "!pip install torch transformers datasets accelerate peft bitsandbytes wandb tqdm fastapi uvicorn pydantic orjson svgwrite trimesh python-dotenv\n\n",
                    "print(\"✅ Dependencies installed successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "clone_repo"},
                "outputs": [],
                "source": [
                    "# Clone the HouseBrain repository\n",
                    "!git clone https://github.com/Vinay-O/HouseBrainLLM.git\n",
                    "%cd HouseBrainLLM\n\n",
                    "print(\"✅ Repository cloned successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "configure_generation"},
                "source": ["## ⚙️ Step 2: Configure Dataset Generation\n\n", "Set up your dataset generation parameters"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "import_modules"},
                "outputs": [],
                "source": [
                    "# Import dataset generation modules\n",
                    "import sys\n",
                    "sys.path.append('.')\n\n",
                    "from generate_dataset import DatasetConfig, HouseBrainDatasetGenerator\n",
                    "import os\n\n",
                    "print(\"✅ Dataset generation modules imported successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "dataset_config"},
                "outputs": [],
                "source": [
                    "# Dataset generation configuration\n",
                    "# Adjust these parameters based on your needs\n\n",
                    "config = DatasetConfig(\n",
                    "    num_samples=50000,  # Number of samples to generate\n",
                    "    output_dir=\"housebrain_dataset_v5_50k_colab\",  # Output directory\n",
                    "    train_ratio=0.9,  # Train/validation split\n",
                    "    min_plot_size=1000,  # Minimum plot area (sqft)\n",
                    "    max_plot_size=10000,  # Maximum plot area (sqft)\n",
                    "    min_bedrooms=1,  # Minimum bedrooms\n",
                    "    max_bedrooms=6,  # Maximum bedrooms\n",
                    "    min_floors=1,  # Minimum floors\n",
                    "    max_floors=4,  # Maximum floors\n",
                    "    min_budget=100000,  # Minimum budget\n",
                    "    max_budget=2000000,  # Maximum budget\n",
                    "    fast_mode=True,  # Skip layout solving for speed\n",
                    ")\n\n",
                    "print(f\"📋 Dataset Configuration:\")\n",
                    "print(f\"   Samples: {config.num_samples:,}\")\n",
                    "print(f\"   Output: {config.output_dir}\")\n",
                    "print(f\"   Train Ratio: {config.train_ratio}\")\n",
                    "print(f\"   Plot Size: {config.min_plot_size:,} - {config.max_plot_size:,} sqft\")\n",
                    "print(f\"   Bedrooms: {config.min_bedrooms} - {config.max_bedrooms}\")\n",
                    "print(f\"   Floors: {config.min_floors} - {config.max_floors}\")\n",
                    "print(f\"   Budget: ${config.min_budget:,} - ${config.max_budget:,}\")\n",
                    "print(f\"   Fast Mode: {config.fast_mode}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "generate_dataset"},
                "source": ["## 🏗️ Step 3: Generate Dataset\n\n", "This will take 30-60 minutes depending on the number of samples."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "start_generation"},
                "outputs": [],
                "source": [
                    "# Initialize dataset generator\n",
                    "print(\"🔧 Setting up dataset generator...\")\n",
                    "generator = HouseBrainDatasetGenerator(config)\n",
                    "print(\"✅ Dataset generator initialized successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "generate_samples"},
                "outputs": [],
                "source": [
                    "# Generate the dataset\n",
                    "print(\"🎯 Starting dataset generation...\")\n",
                    "print(f\"⏰ This will take 30-60 minutes for {config.num_samples:,} samples.\")\n",
                    "print(\"📊 Monitor progress below:\")\n\n",
                    "try:\n",
                    "    output_dir = generator.generate_dataset()\n",
                    "    print(f\"\\n🎉 Dataset generation completed successfully!\")\n",
                    "    print(f\"📁 Output directory: {output_dir}\")\n",
                    "except Exception as e:\n",
                    "    print(f\"\\n❌ Dataset generation failed: {e}\")\n",
                    "    print(\"💡 Try reducing num_samples or using fast_mode=True\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "create_zip"},
                "source": ["## 📦 Step 4: Create Zip Archive\n\n", "Create a zip file for easy download and upload to Kaggle"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "zip_dataset"},
                "outputs": [],
                "source": [
                    "# Create zip archive\n",
                    "import zipfile\n",
                    "import os\n",
                    "from pathlib import Path\n\n",
                    "output_dir = Path(config.output_dir)\n",
                    "zip_path = f\"{config.output_dir}.zip\"\n\n",
                    "print(f\"📦 Creating zip archive: {zip_path}\")\n\n",
                    "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:\n",
                    "    for root, dirs, files in os.walk(output_dir):\n",
                    "        for file in files:\n",
                    "            file_path = os.path.join(root, file)\n",
                    "            arcname = os.path.relpath(file_path, output_dir.parent)\n",
                    "            zipf.write(file_path, arcname)\n\n",
                    "print(f\"✅ Zip archive created: {zip_path}\")\n",
                    "print(f\"📁 Archive size: {os.path.getsize(zip_path) / 1e6:.1f} MB\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "download_dataset"},
                "source": ["## ⬇️ Step 5: Download Dataset\n\n", "Download the dataset to your computer"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "download"},
                "outputs": [],
                "source": [
                    "# Download the dataset\n",
                    "from google.colab import files\n\n",
                    "print(\"⬇️  Downloading dataset...\")\n",
                    "files.download(zip_path)\n",
                    "print(\"✅ Dataset downloaded successfully!\")"
                ]
            }
        ],
        "metadata": {
            "accelerator": "CPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('colab_dataset_generation.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Created colab_dataset_generation.ipynb")

def main():
    """Create both notebooks"""
    print("🏗️ Creating Colab notebooks...")
    
    create_colab_training_notebook()
    create_colab_dataset_notebook()
    
    print("\n🎉 Both notebooks created successfully!")
    print("\n📋 Next steps:")
    print("1. Upload colab_dataset_generation.ipynb to Colab for dataset generation")
    print("2. Upload colab_training.ipynb to Colab for model training")
    print("3. Follow the COLAB_KAGGLE_WORKFLOW.md guide for complete instructions")

if __name__ == "__main__":
    main()
