#!/usr/bin/env python3
"""
Fix all dependency conflicts for HouseBrain training
Run this before training to ensure compatible versions
"""

import subprocess
import sys

def fix_dependencies():
    print("🔧 Fixing HouseBrain dependencies...")
    
    try:
        # Uninstall conflicting packages
        print("📦 Uninstalling conflicting packages...")
        packages_to_remove = [
            "torch", "torchvision", "torchaudio", "transformers", 
            "accelerate", "peft", "datasets", "torchao", "numpy"
        ]
        
        for package in packages_to_remove:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], check=False)
        
        # Install compatible versions
        print("📦 Installing compatible versions...")
        
        # PyTorch ecosystem
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0"], check=True)
        
        # Transformers ecosystem
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.36.0"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "accelerate==0.25.0"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "peft==0.7.0"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets==2.15.0"], check=True)
        
        # NumPy compatibility
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3"], check=True)
        
        # Other required packages
        subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "matplotlib"], check=True)
        
        print("✅ All dependencies fixed successfully!")
        print("🚀 Ready to run training!")
        
    except Exception as e:
        print(f"❌ Error fixing dependencies: {e}")
        print("🔄 Trying alternative approach...")
        
        # Alternative: Just install the essential packages
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "torch==2.1.0", "transformers==4.36.0", "accelerate==0.25.0", "peft==0.7.0", "datasets==2.15.0", "numpy==1.24.3"], check=True)
            print("✅ Alternative fix applied!")
        except Exception as e2:
            print(f"❌ Alternative fix also failed: {e2}")
            return False
    
    return True

if __name__ == "__main__":
    fix_dependencies()
