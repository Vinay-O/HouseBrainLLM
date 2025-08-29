# HouseBrain Project Roadmap

This document outlines the strategic vision and phased implementation plan for the HouseBrain project. Our goal is to create a state-of-the-art, multi-agent AI system for generating high-quality, architecturally sound house plans.

---

### Phase 0: Foundation & Curation (Status: COMPLETE)

This foundational phase focuses on creating the initial tools and data required for all subsequent steps. This is the bedrock of the project.

**Key Achievements:**
- ✅ **High-Volume Data Generation Pipeline:** Developed a resilient, parallelized Google Colab notebook (`data_generation_parallel_v2.ipynb`) capable of generating thousands of raw house plan drafts using powerful reasoning models like `phi4-reasoning`.
- ✅ **Robust Curation Pipeline:** Built a sophisticated, 4-stage local curation pipeline to process the raw model outputs.
    - **Stage 0: Schema Normalizer:** Heals and adapts non-standard model outputs (e.g., converting `name` to `id`, `dimensions` to `bounds`).
    - **Stage 1: Quality Sieve:** Validates the file for JSON integrity and basic schema compliance.
    - **Stage 2: Auto-Architect:** Programmatically enriches the plan by adding logical connections (e.g., doors between adjacent rooms).
    - **Stage 3: AI Sanity Check:** Uses a lightweight "judge" LLM (`llama3`) to ensure the plan is semantically consistent with the original prompt (e.g., correct number of bedrooms).

**Immediate Goal:**
- **Generate & Curate the "Gold" Dataset:** Utilize the established pipelines to generate **15,000 raw samples** and process them through the curation pipeline to create our foundational `gold_tier_curated` dataset.

---

### Phase 1: The Junior Architect (Initial Model Fine-Tuning)

This phase involves training our first custom model, the "Junior Architect," which will learn to generate schema-compliant plans directly.

**Tasks:**
1.  **Fine-tune `qwen2.5:3b`:** Use the `gold_tier_curated` dataset to fine-tune the `qwen2.5:3b` model.
2.  **Evaluate Performance:** Rigorously test the fine-tuned model.
    - How consistently does it produce schema-compliant JSON?
    - How well does it follow complex prompts compared to the original `phi4-reasoning`?
    - What are its most common failure modes?

**Outcome:**
- A highly efficient, specialized "Junior Architect" model capable of producing solid first drafts of house plans that adhere to our data schema.

---

### Phase 2: The Senior Architect (Building the AI Refiner)

This phase focuses on creating a second AI agent whose sole purpose is to correct and improve the output of the Junior Architect.

**Tasks:**
1.  **Create a "Correction Dataset":**
    - Log all the outputs from the Junior Architect that fail validation or are architecturally flawed.
    - Programmatically and manually correct these flawed plans.
    - The result is a dataset of `(flawed_plan, corrected_plan)` pairs.
2.  **Fine-tune the Refiner Model:** Fine-tune a second lightweight model (e.g., `llama3:instruct`) on this "Correction Dataset." This trains it to be an expert at spotting and fixing the Junior's specific mistakes.

**Outcome:**
- A "Senior Architect" AI that acts as an automated quality control step, significantly increasing the quality and reliability of the plans before they are seen by the final judge.

---

### Phase 3: The Big Boss (Implementing the AI Judge & Feedback Loop)

This is the most advanced stage, where we create a hierarchical, self-improving system.

**Tasks:**
1.  **Assemble the AI Team Pipeline:** Create a workflow where `Prompt -> Junior Architect -> Senior Architect`.
2.  **Implement the "Big Boss" Judge:** Use our most powerful reasoning model (`phi4-reasoning` or a commercial API like GPT-4/Claude 3.5 Sonnet) to perform a final, high-level qualitative review of the Senior Architect's output.
    - The "Big Boss" will not just check facts; it will judge architectural merit, flow, creativity, and aesthetic appeal, providing its reasoning in natural language.
3.  **Create the Feedback Loop:** This is the critical step for long-term improvement.
    - The `(Prompt, Senior_Architect_Output, Big_Boss_Review)` tuple is saved as a new "Platinum" or "Diamond" tier training example.
    - This new, ultra-high-quality data is periodically used to re-train and improve both the Junior and Senior Architect models.

**Outcome:**
- A dynamic, learning system that gets progressively better with every plan it generates. The "Junior" learns to make fewer mistakes, and the "Senior" learns to make more sophisticated corrections, guided by the "Big Boss."

---

### Future Projections & Advanced Features

Once the core multi-agent system is in place, it becomes a platform for numerous advanced features:

- **Interactive Planning:** Users can chat with the AI team in real-time ("*Can you make the kitchen 10% larger and add an island?*"). The "Big Boss" interprets the request, and the "Junior/Senior" team executes the change.
- **Multi-modal Capabilities:** Allow users to upload a rough sketch, a mood board of images, or a photo of a site to influence the generated plan.
- **Integrated 2D/3D Rendering:** The final, validated JSON becomes the direct input for an automated rendering engine, producing professional CAD drawings and 3D visualizations.
- **Hyper-Localization:** Integrate local building codes, material costs, and environmental factors (e.g., sun path analysis) for specific regions (Bangalore, Mumbai, etc.).
- **Vastu & Architectural Style Specialization:** Train specialized "consultant" agents that can refine a plan according to strict Vastu rules or specific architectural styles (e.g., Colonial, Brutalist, etc.).
