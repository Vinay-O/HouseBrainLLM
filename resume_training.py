#!/usr/bin/env python3
"""
Resume Training Script
Easily resume HouseBrain training from the latest checkpoint
"""

import os
from pathlib import Path

def find_latest_checkpoint():
    """Find the latest checkpoint directory"""
    checkpoint_dir = Path("housebrain-trained-model")
    
    if not checkpoint_dir.exists():
        print("❌ No training directory found. Start training first.")
        return None
    
    # Find all checkpoint directories
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    
    if not checkpoints:
        print("❌ No checkpoints found. Start training first.")
        return None
    
    # Sort by checkpoint number and get the latest
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split('-')[1]))[-1]
    
    return latest_checkpoint

def resume_training():
    """Resume training from the latest checkpoint"""
    print("🔄 Checking for available checkpoints...")
    
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint is None:
        return
    
    print(f"✅ Found latest checkpoint: {latest_checkpoint.name}")
    
    # Check if checkpoint is complete
    required_files = ["pytorch_model.bin", "config.json", "training_args.bin"]
    missing_files = [f for f in required_files if not (latest_checkpoint / f).exists()]
    
    if missing_files:
        print(f"⚠️ Warning: Checkpoint {latest_checkpoint.name} is incomplete.")
        print(f"   Missing files: {missing_files}")
        print("   This checkpoint may be corrupted.")
        
        # Try to find the previous checkpoint
        checkpoints = sorted(Path("housebrain-trained-model").glob("checkpoint-*"), 
                           key=lambda x: int(x.name.split('-')[1]))
        if len(checkpoints) > 1:
            previous_checkpoint = checkpoints[-2]
            print(f"   Trying previous checkpoint: {previous_checkpoint.name}")
            latest_checkpoint = previous_checkpoint
        else:
            print("❌ No valid checkpoints found. Start training from beginning.")
            return
    
    print(f"🚀 Resuming training from: {latest_checkpoint}")
    print(f"📊 Checkpoint step: {latest_checkpoint.name.split('-')[1]}")
    
    # Resume training
    os.system(f"python colab_proplus_train_simple.py --resume_from_checkpoint {latest_checkpoint}")

def list_checkpoints():
    """List all available checkpoints"""
    checkpoint_dir = Path("housebrain-trained-model")
    
    if not checkpoint_dir.exists():
        print("❌ No training directory found.")
        return
    
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    
    if not checkpoints:
        print("❌ No checkpoints found.")
        return
    
    print("📋 Available checkpoints:")
    for checkpoint in sorted(checkpoints, key=lambda x: int(x.name.split('-')[1])):
        step = checkpoint.name.split('-')[1]
        print(f"   {checkpoint.name} (Step {step})")

def check_training_status():
    """Check current training status"""
    print("📊 Training Status Check")
    print("=" * 40)
    
    # Check for training log
    log_file = Path("training_log.txt")
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print("📝 Latest log entries:")
                for line in lines[-3:]:  # Show last 3 entries
                    print(f"   {line.strip()}")
    
    # Check for metrics
    metrics_file = Path("training_metrics.json")
    if metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if metrics["train_loss"]:
            latest = metrics["train_loss"][-1]
            print(f"📈 Latest training loss: {latest[1]:.4f} (Step {latest[0]})")
    
    # List checkpoints
    print()
    list_checkpoints()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "resume":
            resume_training()
        elif command == "list":
            list_checkpoints()
        elif command == "status":
            check_training_status()
        else:
            print("Usage: python resume_training.py [resume|list|status]")
    else:
        print("🔄 Resume Training Helper")
        print("=" * 30)
        print("Commands:")
        print("  resume  - Resume training from latest checkpoint")
        print("  list    - List all available checkpoints")
        print("  status  - Check current training status")
        print()
        print("Example: python resume_training.py resume")\n