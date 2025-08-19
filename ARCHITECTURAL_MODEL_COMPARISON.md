## HouseBrain Architectural Reasoning Model Guide

This guide compares candidate LLMs specifically for architectural design: spatial reasoning, multi‑floor layout, building codes, BOQ/costing, climate/material choices, and strict JSON outputs.

### Comparison summary

| Model | Params | Context | Strength for architecture | JSON/schema control | Training stability | VRAM need (LoRA+4bit) | Best device |
|---|---:|---:|---|---|---|---:|---|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 16k | Outstanding multi‑step reasoning, spatial planning, code‑aware narratives | Excellent | Good | 18–28 GB | A100/T4 |
| Qwen/Qwen2.5-7B-Instruct | 7B | 32k | Strong reasoning, concise outputs, great tokenizer | Excellent | Very good | 18–28 GB | A100/T4 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | 8k–16k | Reliable, disciplined outputs; slightly weaker geometric detail | Very good | Excellent | 20–30 GB | A100/T4 |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B | 8k | Fastest to train; good but less precise geometry | Good | Excellent | 16–26 GB | A100/T4 |
| deepseek-ai/deepseek-coder-6.7b-base | 6.7B | 16k | Great JSON/code; weaker general reasoning than R1 | Very good | Good | 16–24 GB | A100/T4 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 (local) | 1.1B | 4k | Basic planning; for pipeline sanity checks only | Good | Excellent | <4 GB | CPU/MPS |

Notes: VRAM estimates assume LoRA + BitsAndBytes 4‑bit (NF4) + bf16/fp16 activations.

---

### What “architectural reasoning” needs

- Spatial floor planning: room adjacencies, vertical circulation, structural grid alignment.
- Geometric construction: coordinates, dimensions, multi‑floor continuity, stair logic.
- Codes/compliance: minimum room sizes, egress, accessibility hints.
- Cost engineering: BOQ quantities, materials, phase/timeline, climate impacts.
- Deterministic JSON: strict schema for `input` and `output` with long reasoning sections.

---

## Model deep‑dives

### DeepSeek R1 Distill Qwen 7B
- Why pick: best chain‑of‑thought quality for architectural workflows; detailed spatial narratives with strong JSON compliance.
- Risks: slightly heavier first‑step latency; strict dependency pinning recommended.
- HouseBrain settings:
  - LoRA targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
  - Context: `--max-length 2048` (use 4096 only if needed)
  - A100: `bf16` auto; `--batch-size 2–4` with `--grad-accum-steps 4–8`; `--grad-checkpointing` if memory bound

### Qwen/Qwen2.5‑7B‑Instruct
- Why pick: comparable or better reasoning on some tasks; faster and very stable with Transformers; great tokenizer for JSON.
- HouseBrain settings: same LoRA targets and batch strategy as R1; 32k context available if prompts grow.

### meta‑llama/Meta‑Llama‑3.1‑8B‑Instruct
- Why pick: stability and disciplined outputs; great for production inference; slightly less geometric depth.
- When to use: if training reliability and reproducibility outweigh peak reasoning.

### mistralai/Mistral‑7B‑Instruct‑v0.3
- Why pick: highest throughput per GPU hour; good baseline; efficient for large‑scale data sweeps.
- Caveat: slightly weaker on fine‑grained geometry; compensate with data emphasis on geometric tasks.

### deepseek‑ai/deepseek‑coder‑6.7b‑base
- Why pick: strong structured output and code generation; decent layouts when instructed; good as a JSON teacher.
- Use case: bootstrap schema‑strict datasets or ensembles.

### Local‑only option: TinyLlama/TinyLlama‑1.1B‑Chat‑v1.0
- Why pick: runs on CPU/MPS for pipeline smoke tests; not a production reasoning model.
- Settings (local): `--max-length 512`, `--batch-size 1`, no 4‑bit needed.

---

## Training guidance (Colab/A100)

- Precision: A100 -> bf16 (auto in trainer). Keep 4‑bit (NF4) for base weights; LoRA on top.
- Throughput: increase per‑device batch until VRAM hits ~38 GB, then balance with `--grad-accum-steps`.
- Checkpointing: enable `--grad-checkpointing` if you need larger batch/context; disable if you have headroom for speed.
- Optimizer: consider `paged_adamw_8bit` to offload optimizer states to CPU RAM (no quality loss).
- Padding: length grouping (toggle in trainer when added) reduces waste on mixed‑length prompts.

---

## Which model for which stage

| Stage | Goal | Pick |
|---|---|---|
| Data synthesis (JSON‑strict) | Generate schema‑compliant samples | DeepSeek Coder 6.7B or Qwen2.5 7B |
| 10K pilot | Validate training + prompts | DeepSeek R1 7B or Qwen2.5 7B |
| 100K shards (A100) | Max reasoning per GPU hour | Qwen2.5 7B or DeepSeek R1 7B |
| Production inference | Stable deterministic outputs | Llama 3.1 8B or Qwen2.5 7B |
| Local smoke tests | Verify pipeline on Mac | TinyLlama 1.1B |

---

## Trainer presets (copy/paste)

```
# DeepSeek R1 7B
--model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
--max-length 2048 --batch-size 2–4 --grad-accum-steps 4–8 --grad-checkpointing

# Qwen2.5 7B
--model "Qwen/Qwen2.5-7B-Instruct" \
--max-length 2048 --batch-size 2–4 --grad-accum-steps 4–8 --grad-checkpointing

# Llama 3.1 8B
--model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--max-length 2048 --batch-size 2–3 --grad-accum-steps 6–8 --grad-checkpointing

# Mistral 7B
--model "mistralai/Mistral-7B-Instruct-v0.3" \
--max-length 2048 --batch-size 3–4 --grad-accum-steps 4–6 --grad-checkpointing

# TinyLlama 1.1B (local CPU/MPS)
--model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--max-length 512 --batch-size 1 --grad-accum-steps 1
```

---

## Final recommendations

- Primary (cloud, production): DeepSeek R1 7B or Qwen2.5 7B — both excel at multi‑floor reasoning, code hints, BOQ, and long structured JSON outputs.
- Stable alternative: Llama 3.1 8B — slightly less geometric depth but excellent reliability.
- Fast baseline: Mistral 7B — best throughput, good for large sharded runs.
- Local smoke: TinyLlama 1.1B — validate pipeline on Mac; not for production reasoning.


