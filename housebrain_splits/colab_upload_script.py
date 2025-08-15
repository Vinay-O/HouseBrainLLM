#!/usr/bin/env python3
# Colab Upload Script
# Upload this to each Colab account

import os
import zipfile
from google.colab import files

# Upload your split dataset
print("📤 Uploading dataset split...")
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted: {filename}")

# Install dependencies
!pip install transformers peft bitsandbytes datasets

# Run training
!python colab_training_fixed.py --dataset split_XX --output models/housebrain-colab-X

print("🚀 Training started! Keep this tab open.")
