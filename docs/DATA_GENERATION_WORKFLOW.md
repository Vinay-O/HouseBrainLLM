# HouseBrain Data Generation Workflow: The "Generate, Analyze, Repair" Cycle

This document outlines the standard operating procedure for creating high-quality training data for the HouseBrain AI. The goal is to produce "Gold Standard" `HouseOutput` JSON files that are both schema-compliant and architecturally sound.

We follow a three-step, iterative process we call the **"Generate, Analyze, Repair"** cycle.

---

### **Core Tools**

*   **Generation Model:** `llama3` (run via Ollama in Google Colab). It has proven to have the best balance of reasoning, instruction-following, and creativity for this architectural task.
*   **Generation Environment:** The `train_on_colab_v3_llama.ipynb` notebook. This provides the necessary GPU environment and the interactive "Generate, Analyze, Repair" cells.
*   **Analysis & Validation Tool:** The `run_pipeline.py` script. Running this script on a JSON file is the ultimate test. It performs a strict Pydantic validation and then attempts to render the 2D plan. If it runs without errors, the data is valid.

---

### **Step 1: Generate (The Raw Draft)**

The first step is to create a raw draft from a text prompt.

1.  **Open the Notebook:** Launch `train_on_colab_v3_llama.ipynb` in Google Colab with a GPU runtime.
2.  **Run Setup Cells:** Execute the cells for environment setup, HF login, and starting the Ollama server. Ensure `llama3` is pulled.
3.  **Define a Scenario:** In the "Interactive Testing & Curation" section, write a clear, detailed architectural prompt in the `TEST_SCENARIO` variable.
4.  **Execute Generation:** Run the cell. This will call the `llama3` model and generate a raw JSON draft, saving it to `data/training/curation_test_draft.json` and printing the raw output to the screen.

**Expected Outcome:** A raw JSON object that is likely to have one or more of the following flaws:
*   Pydantic validation errors (missing fields, incorrect types).
*   Geometric errors (overlapping rooms, incorrect coordinates).
*   Architectural flaws (missing doors, illogical window placements).
*   Chatty, non-JSON text surrounding the JSON object.

---

### **Step 2: Analyze (The Automated Critique)**

Now, we use our tools to find the flaws.

1.  **Visual Inspection:** Copy the raw JSON output from the notebook cell. Use an online JSON validator or your local editor to format it and perform a quick visual check. Does it look roughly correct?
2.  **Schema Validation:** The most definitive test is to run the pipeline. Locally, you can run:
    ```bash
    python run_pipeline.py --input /path/to/your/draft.json
    ```
    The script will immediately exit with clear Pydantic validation errors if the schema is not met.

---

### **Step 3: Repair (The AI Architect)**

This is the human-in-the-loop step where we correct the AI's mistakes and elevate the data to Gold Standard.

1.  **Fix Schema Errors:** Address all validation errors reported by the pipeline. This often involves adding missing fields (`room1`, `room2` for doors), correcting data types, or fixing structural issues.
2.  **Fix Geometric Errors:** Open the corrected JSON and carefully review all coordinates and dimensions. Ensure rooms do not overlap. Ensure the total plot dimensions are respected. Use a piece of paper or a simple drawing tool if necessary to visualize the layout.
3.  **Fix Architectural Errors:**
    *   Does every room that needs a door have one?
    *   Do the doors connect the correct rooms (`room1` and `room2`)?
    *   Are windows placed logically on exterior walls?
    *   Is the staircase connecting the correct levels and located within a `stairwell` room?
4.  **Final Validation:** Rerun `python run_pipeline.py` on your repaired file. **The file is not "Gold Standard" until this command runs successfully and produces a clean, logical SVG.**
5.  **Save and Commit:** Once the file is perfect, save it to `data/training/gold_standard/` with a descriptive name (e.g., `gold_standard_22_4bhk_g+1_vastu.json`) and commit it to the repository.

By repeating this cycle, we build a small but flawless dataset that will teach the fine-tuned model to avoid these common errors, significantly improving its performance.
