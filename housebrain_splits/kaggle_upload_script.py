#!/usr/bin/env python3
# Kaggle Upload Script
# Upload this to each Kaggle account

import os
import zipfile

# Extract dataset (uploaded as zip)
for filename in os.listdir('.'):
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted: {filename}")

# Install dependencies
!pip install transformers peft bitsandbytes datasets

# Run training
!python colab_training_fixed.py --dataset split_XX --output models/housebrain-kaggle-X

print("🚀 Training started! Check email for completion notification.")
