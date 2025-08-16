#!/usr/bin/env python3
"""
HouseBrain Colab Pro+ Training Script with Notifications
Enhanced for mobile hotspot users with email notifications
"""

import os
import json
import torch
import time
import smtplib
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText

# Configuration
class TrainingConfig:
    # Model settings
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
    
    # A100 optimized settings
    SEQUENCE_LENGTH = 1024
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Training settings
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 8
    WARMUP_STEPS = 100
    MAX_STEPS = 50000
    SAVE_STEPS = 1000
    EVAL_STEPS = 500
    LOGGING_STEPS = 10
    
    # Dataset
    DATASET_PATH = "housebrain_dataset_v6_1M"
    
    # Output
    OUTPUT_DIR = "housebrain-trained-model"
    
    # Logging
    LOG_FILE = "training_log.txt"
    
    # Email notifications (optional)
    ENABLE_EMAIL = False  # Set to True and configure below
    EMAIL_SENDER = "your_email@gmail.com"
    EMAIL_PASSWORD = "your_app_password"  # Use app password, not regular password
    EMAIL_RECEIVER = "your_email@gmail.com"

class NotificationManager:
    """Handle email notifications for mobile hotspot users"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.ENABLE_EMAIL and config.EMAIL_SENDER != "your_email@gmail.com"
        
    def send_email(self, subject, message):
        """Send email notification"""
        if not self.enabled:
            return
            
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"HouseBrain Training: {subject}"
            msg['From'] = self.config.EMAIL_SENDER
            msg['To'] = self.config.EMAIL_RECEIVER
            
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.config.EMAIL_SENDER, self.config.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ Email sent: {subject}")
        except Exception as e:
            print(f"❌ Email failed: {e}")
    
    def notify_training_start(self, gpu_info):
        """Notify when training starts"""
        message = f"""
🏗️ HouseBrain Training Started!

📊 GPU: {gpu_info['name']}
💾 Memory: {gpu_info['memory']:.1f} GB
📈 Training samples: 517,243
📉 Validation samples: 57,773
⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 Expected completion: 8-12 hours
📧 You'll receive updates every 1000 steps.
        """
        self.send_email("Training Started", message)
    
    def notify_checkpoint(self, step, loss, eval_loss):
        """Notify when checkpoint is saved"""
        message = f"""
💾 Checkpoint Saved!

📊 Step: {step:,}
📉 Training Loss: {loss:.4f}
📈 Evaluation Loss: {eval_loss:.4f}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 Progress: {step/50000*100:.1f}% complete
        """
        self.send_email(f"Checkpoint {step}", message)
    
    def notify_training_complete(self, final_loss, best_eval_loss, total_steps):
        """Notify when training completes"""
        message = f"""
🎉 Training Complete!

📊 Total Steps: {total_steps:,}
📉 Final Loss: {final_loss:.4f}
📈 Best Eval Loss: {best_eval_loss:.4f}
⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💾 Model saved to: housebrain-trained-model/final/
        """
        self.send_email("Training Complete", message)
    
    def notify_error(self, error_message):
        """Notify if training encounters an error"""
        message = f"""
❌ Training Error!

🚨 Error: {error_message}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 Please check Colab and restart if needed.
        """
        self.send_email("Training Error", error_message)

class SimpleLogger:
    """Simple logging without external dependencies"""
    
    def __init__(self, log_file, notification_manager):
        self.log_file = log_file
        self.notification_manager = notification_manager
        self.start_time = time.time()
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "gpu_memory": []
        }
        
    def log(self, message, print_to_console=True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        
        if print_to_console:
            print(log_entry)
    
    def log_metrics(self, step, train_loss=None, eval_loss=None, lr=None):
        """Log training metrics"""
        if train_loss is not None:
            self.metrics["train_loss"].append((step, train_loss))
        if eval_loss is not None:
            self.metrics["eval_loss"].append((step, eval_loss))
        if lr is not None:
            self.metrics["learning_rate"].append((step, lr))
        
        # Get GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            self.metrics["gpu_memory"].append((step, gpu_memory))
        
        # Log summary
        self.log(f"Step {step}: Train Loss={train_loss:.4f}, Eval Loss={eval_loss:.4f}, LR={lr:.2e}, GPU={gpu_memory:.2f}GB")
        
        # Send checkpoint notification every 1000 steps
        if step % 1000 == 0 and step > 0:
            self.notification_manager.notify_checkpoint(step, train_loss, eval_loss)
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        with open("training_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        self.log(f"Metrics saved to training_metrics.json")

def setup_environment():
    """Setup training environment"""
    print("🔧 Setting up environment...")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! Please enable GPU in Colab.")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    return {
        "name": gpu_name,
        "memory": gpu_memory
    }

def load_dataset(config):
    """Load and prepare dataset"""
    print("📊 Loading dataset...")
    
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")
    
    # Load training files
    train_files = list((dataset_path / "train").glob("*.json"))
    validation_files = list((dataset_path / "validation").glob("*.json"))
    
    print(f"📈 Training samples: {len(train_files):,}")
    print(f"📉 Validation samples: {len(validation_files):,}")
    
    def load_samples(file_list):
        samples = []
        for file in tqdm(file_list, desc="Loading samples"):
            with open(file, 'r') as f:
                sample = json.load(f)
                # Format for training
                input_text = json.dumps(sample["input"], separators=(",", ":"))
                output_text = json.dumps(sample["output"], separators=(",", ":"))
                full_text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
                samples.append({"text": full_text})
        return samples
    
    train_samples = load_samples(train_files)
    validation_samples = load_samples(validation_files)
    
    return Dataset.from_list(train_samples), Dataset.from_list(validation_samples)

def create_model_and_tokenizer(config):
    """Create model and tokenizer"""
    print("🤖 Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    """Main training function"""
    config = TrainingConfig()
    notification_manager = NotificationManager(config)
    logger = SimpleLogger(config.LOG_FILE, notification_manager)
    
    try:
        # Setup
        gpu_info = setup_environment()
        logger.log(f"Starting HouseBrain training with {gpu_info['memory']:.1f}GB GPU")
        
        # Send start notification
        notification_manager.notify_training_start(gpu_info)
        
        # Load dataset
        train_dataset, eval_dataset = load_dataset(config)
        logger.log(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} validation samples")
        
        # Create model
        model, tokenizer = create_model_and_tokenizer(config)
        logger.log("Model and tokenizer loaded successfully")
        
        # Tokenize datasets
        def tokenize_train(examples):
            return tokenize_function(examples, tokenizer, config.SEQUENCE_LENGTH)
        
        def tokenize_eval(examples):
            return tokenize_function(examples, tokenizer, config.SEQUENCE_LENGTH)
        
        train_dataset = train_dataset.map(tokenize_train, batched=True)
        eval_dataset = eval_dataset.map(tokenize_eval, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=None,
            max_steps=config.MAX_STEPS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
            learning_rate=config.LEARNING_RATE,
            warmup_steps=config.WARMUP_STEPS,
            logging_steps=config.LOGGING_STEPS,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable WandB
            logging_dir="./logs",
            save_total_limit=3,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
        )
        
        # Custom callback for logging
        class LoggingCallback:
            def __init__(self, logger):
                self.logger = logger
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    train_loss = logs.get("loss")
                    eval_loss = logs.get("eval_loss")
                    lr = logs.get("learning_rate")
                    
                    if train_loss is not None or eval_loss is not None:
                        self.logger.log_metrics(
                            state.global_step,
                            train_loss=train_loss,
                            eval_loss=eval_loss,
                            lr=lr
                        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[LoggingCallback(logger)]
        )
        
        # Start training
        logger.log("🚀 Starting training...")
        logger.log(f"📊 Config: LR={config.LEARNING_RATE}, Batch={config.BATCH_SIZE}, SeqLen={config.SEQUENCE_LENGTH}")
        
        trainer.train()
        logger.log("✅ Training completed successfully!")
        
        # Save final model
        trainer.save_model(f"{config.OUTPUT_DIR}/final")
        tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final")
        logger.log(f"💾 Final model saved to {config.OUTPUT_DIR}/final")
        
        # Save metrics
        logger.save_metrics()
        
        # Print summary
        final_loss = trainer.state.log_history[-1].get('loss', 'N/A')
        best_eval_loss = trainer.state.best_metric
        total_steps = trainer.state.global_step
        
        logger.log("📈 Training Summary:")
        logger.log(f"   - Total steps: {total_steps}")
        logger.log(f"   - Final train loss: {final_loss}")
        logger.log(f"   - Best eval loss: {best_eval_loss}")
        
        # Send completion notification
        notification_manager.notify_training_complete(final_loss, best_eval_loss, total_steps)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.log(f"❌ {error_msg}")
        notification_manager.notify_error(error_msg)
        raise
    
    logger.log("🎉 All done! Check the logs and saved model.")

if __name__ == "__main__":
    main()
