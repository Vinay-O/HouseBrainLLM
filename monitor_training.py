#!/usr/bin/env python3
"""
Simple Training Monitor
Run this in a separate Colab cell to monitor training progress
"""

import json
import time
import matplotlib.pyplot as plt
from pathlib import Path

def monitor_training():
    """Monitor training progress"""
    
    log_file = "training_log.txt"
    metrics_file = "training_metrics.json"
    
    print("📊 Training Monitor")
    print("=" * 50)
    
    # Check if files exist
    if not Path(log_file).exists():
        print("❌ No training log found. Training may not have started yet.")
        return
    
    # Read latest log entries
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if lines:
        print("📝 Latest log entries:")
        for line in lines[-5:]:  # Show last 5 entries
            print(f"   {line.strip()}")
    
    # Check metrics if available
    if Path(metrics_file).exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print("\n📈 Metrics Summary:")
        if metrics["train_loss"]:
            latest_train = metrics["train_loss"][-1]
            print(f"   Latest Train Loss: {latest_train[1]:.4f} (Step {latest_train[0]})")
        
        if metrics["eval_loss"]:
            latest_eval = metrics["eval_loss"][-1]
            print(f"   Latest Eval Loss: {latest_eval[1]:.4f} (Step {latest_eval[0]})")
        
        if metrics["gpu_memory"]:
            latest_gpu = metrics["gpu_memory"][-1]
            print(f"   GPU Memory: {latest_gpu[1]:.2f}GB (Step {latest_gpu[0]})")
    
    # Check for saved models
    model_dir = Path("housebrain-trained-model")
    if model_dir.exists():
        checkpoints = list(model_dir.glob("checkpoint-*"))
        if checkpoints:
            print("\n💾 Saved Checkpoints:")
            for checkpoint in sorted(checkpoints):
                print(f"   {checkpoint.name}")
    
    print(f"\n⏰ Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def plot_metrics():
    """Plot training metrics if available"""
    metrics_file = "training_metrics.json"
    
    if not Path(metrics_file).exists():
        print("❌ No metrics file found.")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics["train_loss"]:
        print("❌ No training data available yet.")
        return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if metrics["train_loss"]:
        steps, losses = zip(*metrics["train_loss"])
        ax1.plot(steps, losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
    
    # Evaluation loss
    if metrics["eval_loss"]:
        steps, losses = zip(*metrics["eval_loss"])
        ax2.plot(steps, losses, 'r-', label='Evaluation Loss')
        ax2.set_title('Evaluation Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
    
    # Learning rate
    if metrics["learning_rate"]:
        steps, lrs = zip(*metrics["learning_rate"])
        ax3.plot(steps, lrs, 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
    
    # GPU Memory
    if metrics["gpu_memory"]:
        steps, memory = zip(*metrics["gpu_memory"])
        ax4.plot(steps, memory, 'm-', label='GPU Memory')
        ax4.set_title('GPU Memory Usage')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Memory (GB)')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    monitor_training()
    print("\n" + "=" * 50)
    print("To plot metrics, run: plot_metrics()")
