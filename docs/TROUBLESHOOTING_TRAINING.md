# HouseBrain Training Troubleshooting Guide

## Common Issues and Solutions

### ❌ Issue: "num_samples should be a positive integer value, but got num_samples=0"

**Cause**: Empty dataset directory or incorrect dataset path.

**Solutions**:

1. **Check dataset directory exists and has files**:
   ```bash
   # In Colab, verify the dataset directory
   !ls -la /content/HouseBrainLLM/hb_v2_shard_01/
   !ls /content/HouseBrainLLM/hb_v2_shard_01/*.json | wc -l
   ```

2. **Regenerate data if directory is empty**:
   ```bash
   # Generate the missing data
   !python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_01 --n 100000
   ```

3. **Verify dataset path in training command**:
   ```bash
   # Ensure path matches exactly
   --dataset /content/HouseBrainLLM/hb_v2_shard_01
   ```

### ❌ Issue: "No module named 'torch'"

**Cause**: PyTorch not installed (local environment only).

**Solutions**:

1. **In Colab**: This should not happen as PyTorch is pre-installed
2. **Local testing**: Install PyTorch or skip trainer config test
3. **Verify Colab environment**: Ensure you're using GPU runtime

### ❌ Issue: Model download fails or is slow

**Cause**: Network issues or large model size.

**Solutions**:

1. **Use cached model** (if available):
   ```python
   # Check if model is already downloaded
   !ls -la /root/.cache/huggingface/hub/
   ```

2. **Verify internet connection**:
   ```bash
   !ping -c 3 huggingface.co
   ```

3. **Use smaller model for testing** (if needed):
   ```bash
   --model "microsoft/DialoGPT-small"  # For testing only
   ```

### ❌ Issue: GPU out of memory

**Cause**: Batch size too large for GPU memory.

**Solutions**:

1. **Reduce batch size**:
   ```bash
   --batch-size 2  # Instead of 4
   --grad-accum-steps 8  # Increase to maintain effective batch size
   ```

2. **Reduce max length**:
   ```bash
   --max-length 512  # Instead of 768
   ```

3. **Use gradient checkpointing**:
   ```python
   # Add to trainer config
   gradient_checkpointing=True
   ```

### ❌ Issue: Training is too slow

**Cause**: Suboptimal settings for your hardware.

**Solutions**:

1. **Use optimized settings** (from our guide):
   ```bash
   --batch-size 4 --grad-accum-steps 4 --epochs 1
   ```

2. **Disable validation during training**:
   ```bash
   --eval-steps 0
   ```

3. **Use mixed precision**:
   ```python
   # Should be enabled by default in Colab
   fp16=True
   ```

### ❌ Issue: Model saving fails

**Cause**: Disk space or permission issues.

**Solutions**:

1. **Check disk space**:
   ```bash
   !df -h
   ```

2. **Use Drive for saving**:
   ```bash
   --output /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s01
   ```

3. **Reduce save frequency**:
   ```bash
   --save-steps 2000  # Instead of 1000
   ```

## Pre-Training Checklist

### ✅ Before Starting Training:

1. **Verify GPU**:
   ```bash
   !nvidia-smi
   # Should show A100 or similar high-end GPU
   ```

2. **Check dataset**:
   ```bash
   !ls -la /content/HouseBrainLLM/hb_v2_shard_01/
   !ls /content/HouseBrainLLM/hb_v2_shard_01/*.json | wc -l
   # Should show 100000 files
   ```

3. **Test data generation**:
   ```bash
   !python generate_synthetic_v2.py --out_dir test_small --n 10
   !ls test_small/*.json | wc -l
   # Should show 10 files
   ```

4. **Verify model access**:
   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
   print("✅ Model accessible")
   ```

## Quick Fix Commands

### If dataset is missing:
```bash
# Generate all 3 shards
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_01 --n 100000
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_02 --n 100000
!python generate_synthetic_v2.py --out_dir /content/HouseBrainLLM/hb_v2_shard_03 --n 100000
```

### If training fails with memory issues:
```bash
# Use conservative settings
--batch-size 2 --grad-accum-steps 8 --max-length 512
```

### If model download is slow:
```bash
# Check if model is cached
!ls -la /root/.cache/huggingface/hub/
```

## Success Indicators

### ✅ Training is working correctly when you see:

1. **Model loading**:
   ```
   🤖 Loading model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
   ✅ Model and tokenizer loaded successfully
   ```

2. **Dataset loading**:
   ```
   📊 Loading dataset from: /content/HouseBrainLLM/hb_v2_shard_01
   ✅ Loaded 100000 training samples
   📊 Dataset prepared: 100000 samples
   ```

3. **Training progress**:
   ```
   🎯 Training started...
   [Training progress bars appear]
   ```

4. **Checkpoint saving**:
   ```
   💾 Saving checkpoint to /content/hb_v2_s01/checkpoint-1000
   ```

## Emergency Recovery

### If training crashes mid-way:

1. **Check for partial checkpoints**:
   ```bash
   !ls -la /content/hb_v2_s01/
   ```

2. **Resume from checkpoint** (if available):
   ```bash
   # Add to training command
   --resume_from_checkpoint /content/hb_v2_s01/checkpoint-1000
   ```

3. **Save partial progress**:
   ```bash
   !rsync -ah --progress /content/hb_v2_s01/ /content/drive/MyDrive/HouseBrainLLM/checkpoints/hb_v2_s01_partial/
   ```

## Performance Optimization

### For fastest training:

1. **Use A100 GPU** (40GB VRAM)
2. **Batch size 4** with **grad_accum_steps 4**
3. **Max length 768** (optimal for v2 data)
4. **No validation** during training (`eval_steps 0`)
5. **Mixed precision** (fp16=True)

### Expected performance:
- **A100 40GB**: ~2-2.5 hours per 100k shard
- **V100 16GB**: ~4-5 hours per 100k shard (with batch_size 2)
- **T4 16GB**: ~6-8 hours per 100k shard (with batch_size 1)

## Support

If you encounter issues not covered here:

1. **Check the logs** for specific error messages
2. **Verify all prerequisites** from the checklist
3. **Try with smaller dataset** first (10k samples)
4. **Use conservative settings** if memory is limited

Remember: The smoke test shows the core functionality works. Most issues are related to environment setup or resource constraints.
