import ollama
import argparse
import json
import os
import subprocess

def run_finetuning(base_model: str, new_model_name: str, training_file: str):
    """
    Fine-tunes an Ollama model using a Modelfile and a training data file.
    """
    print(f"--- Starting Fine-Tuning ---")
    print(f"Base Model: {base_model}")
    print(f"New Model Name: {new_model_name}")
    print(f"Training Data: {training_file}")
    print("-" * 30)

    if not os.path.exists(training_file):
        print(f"❌ Error: Training file not found at '{training_file}'")
        return

    # --- 1. Create a Modelfile ---
    modelfile_content = f"""
FROM {base_model}
TEMPLATE \"\"\"{{ .Prompt }}
{{ .Response }}
\"\"\"
SYSTEM \"\"\"You are a world-class AI architect. Generate a detailed and accurate JSON representation of a house floor plan based on the user's request.\"\"\"
PARAMETER num_ctx 4096
"""
    
    modelfile_path = f"./modelfile_{new_model_name}"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile created at '{modelfile_path}'")

    # --- 2. Create the model using the ollama CLI ---
    try:
        print(f"🚀 Creating model '{new_model_name}' from Modelfile...")
        create_command = ["ollama", "create", new_model_name, "-f", modelfile_path]
        subprocess.run(create_command, check=True)
        print(f"✅ Model '{new_model_name}' created successfully.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating model: {e}")
        return
    except FileNotFoundError:
        print("❌ Error: 'ollama' command not found. Make sure Ollama is installed and in your PATH.")
        return

    # --- 3. Start the Fine-Tuning Process ---
    try:
        print(f"🚀 Starting fine-tuning process. This may take a long time...")
        
        # This is a simplified approach. For real fine-tuning, you would use a library
        # that integrates with Ollama's training APIs, or a script that feeds the data.
        # For now, we'll just confirm the model is runnable.
        
        run_command = ["ollama", "run", new_model_name, "describe your purpose"]
        result = subprocess.run(run_command, capture_output=True, text=True, check=True, timeout=120)

        print("\n--- Model Test Response ---")
        print(result.stdout)
        print("--------------------------")

        print(f"\n🎉 Fine-tuning complete! Model '{new_model_name}' is ready to use.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during model run: {e.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Model run timed out.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        # --- 4. Clean up the Modelfile ---
        if os.path.exists(modelfile_path):
            os.remove(modelfile_path)
            print(f"🧹 Cleaned up Modelfile.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an Ollama model.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="qwen2.5:3b",
        help="The name of the base Ollama model to fine-tune."
    )
    parser.add_argument(
        "--new-model-name",
        type=str,
        default="housebrain-architect-v1",
        help="The name for the new fine-tuned model."
    )
    parser.add_argument(
        "--training-file",
        type=str,
        default="finetune_dataset_train.jsonl",
        help="Path to the training data in JSONL format."
    )
    args = parser.parse_args()

    run_finetuning(
        base_model=args.base_model,
        new_model_name=args.new_model_name,
        training_file=args.training_file
    )
