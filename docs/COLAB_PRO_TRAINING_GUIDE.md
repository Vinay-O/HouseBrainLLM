# HouseBrain: A100 Colab Pro+ Training Guide (Gold + Silver Standard)

This guide provides the definitive, up-to-date workflow for fine-tuning a production-quality HouseBrain model using a Google Colab Pro+ A100 environment.

This process leverages our highest-quality datasets:
-   **Gold Standard:** A small, handcrafted set of perfect, architect-grade examples.
-   **Silver Standard:** A larger, high-quality dataset generated via an automated critique-and-refine loop.

---

##  Prerequisites

1.  **Google Colab Pro+ Account:** With access to an A100 GPU.
2.  **GitHub Repository:** Your project code must be pushed to a GitHub repository.
3.  **Hugging Face Account:** You will need a Hugging Face access token with at least "read" permissions.

---

## The Workflow: A Single Notebook

The entire process—from data generation to final model training—is encapsulated in a single Jupyter Notebook: `train_on_colab.ipynb`.

### **Step 1: Open the Notebook in Colab**

1.  Navigate to [colab.research.google.com](https://colab.research.google.com).
2.  Click on the **"GitHub"** tab in the "Open notebook" dialog.
3.  Enter your GitHub repository URL (e.g., `https://github.com/Your-Username/HouseBrainLLM`) and press Enter.
4.  Select `train_on_colab.ipynb` from the list of files.

### **Step 2: Configure the Colab Environment**

1.  Once the notebook is open, connect to a powerful runtime.
2.  Go to **Runtime** -> **Change runtime type**.
3.  Under "Hardware accelerator," select **GPU**.
4.  From the "GPU type" dropdown, select **A100 GPU**.

### **Step 3: Execute the Notebook Cells**

Run each cell in the notebook from top to bottom. The notebook is divided into clear steps:

1.  **Setup the Environment:** This cell clones your GitHub repository and installs all required Python libraries. It handles all dependencies automatically.

2.  **Authenticate with Hugging Face:** This cell will prompt you to enter your Hugging Face access token. This is required to download the `deepseek-coder` base model.

3.  **Generate "Silver Standard" Data:** This is a crucial step. It runs the `generate_silver_standard_data.py` script, which uses the A100 GPU to rapidly generate 100 new, high-quality training examples. This process involves the LLM generating a design, critiquing its own work, and then refining it.

4.  **Prepare All Datasets:** This cell runs the `prepare_data_for_finetuning.py` script on both the "Gold" and "Silver" datasets to format them correctly for the trainer.

5.  **Run the Fine-Tuning Script:** This is the final training step. It launches the `run_finetuning.py` script with settings optimized for the A100 GPU:
    -   It trains on the **combined Gold and Silver datasets**.
    -   It uses a larger batch size for faster training.
    -   It runs for 15 epochs to ensure the model learns deeply from our high-quality data.
    -   The final model adapter will be saved to `models/housebrain-v1.0-silver`.

6.  **(Optional) Download the Trained Model:** The final cell provides the code to zip up the trained model adapter and download it directly to your local machine.

---

## Summary

By following the steps in the `train_on_colab.ipynb` notebook, you will execute a complete, professional-grade training pipeline. This will produce a highly specialized and capable model, fine-tuned on the best data we have created.
