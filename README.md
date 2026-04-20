# tiny-tutor

**Goal:** Distill "Explain Like I'm 5" capability from a frontier model (Claude) into a
360M-parameter SmolLM2, using a fully synthetic dataset with self-critique filtering.

**Status:** 🟡 Week 1 — dataset design & seed prompt iteration

---

## Why

Small language models are cheap to run and privacy-friendly, but they struggle with
the kind of clear, analogy-rich explanations that make a good tutor. This project
tests whether careful synthetic data generation + LoRA fine-tuning can close that
gap for a specific, measurable task.

It's also a personal experiment in the full synthetic-data-to-deployed-model loop:
data generation, quality filtering, training, eval, and shipping a demo.

---

## Approach

1. **Seed corpus** — ~2,000 STEM concepts sourced from Wikipedia glossary pages
2. **Synthetic generation** — Claude generates ELI5 explanations with varied analogy styles
3. **Self-critique filter** — each output scored 1–5 by the same model; only ≥4 kept
4. **Fine-tune** — SmolLM2-360M-Instruct with LoRA (rank 16) via Unsloth
5. **Evaluate** — LLM-as-judge + readability scores + held-out test set
6. **Ship** — HuggingFace model card + Gradio demo on Spaces

---

## Reproducing This Project

### Prerequisites

- Python 3.10+
- An Anthropic API key (set as `ANTHROPIC_API_KEY` in your environment)
- A free Google Colab account (for training on T4 GPU) or RunPod
- A free Weights & Biases account (for training logs)
- A HuggingFace account (for pushing model weights and hosting the demo)

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/<you>/tiny-tutor
cd tiny-tutor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic training data (~$15–25 in API costs)
python src/generate.py \
  --concepts data/seed_concepts.jsonl \
  --output data/raw/generated.jsonl \
  --model claude-haiku-4-5-20251001 \
  --max-concepts 2000

# 4. Filter to high-quality examples
python src/filter.py \
  --input data/raw/generated.jsonl \
  --output data/filtered.jsonl \
  --min-score 4

# 5. Fine-tune (run this in Colab — see notebooks/train.ipynb)
python src/train.py \
  --dataset data/filtered.jsonl \
  --output-dir checkpoints/ \
  --wandb-project tiny-tutor

# 6. Evaluate base model vs. fine-tuned model
python src/evaluate.py \
  --base-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --finetuned-model checkpoints/final \
  --test-set data/test_concepts.jsonl \
  --output results/eval_report.json
```

---

## Data Format

Each training example is a JSONL record:

```json
{
  "concept": "photosynthesis",
  "domain": "biology",
  "difficulty": "easy",
  "analogy_style": "food",
  "explanation": "Plants are like tiny kitchens. They use sunlight as a stove, water from the ground as an ingredient, and air as another ingredient to cook their own food — a kind of sugar they use for energy.",
  "score": 5,
  "model": "claude-haiku-4-5-20251001",
  "generated_at": "2025-01-01T00:00:00Z"
}
```

Fields:
- `concept` — the STEM term being explained
- `domain` — physics / biology / cs / math
- `difficulty` — easy / medium / hard (relative to a 5–10 year old)
- `analogy_style` — animals / food / everyday_objects / sports / building
- `explanation` — the ELI5 explanation (target: 3–6 sentences)
- `score` — self-critique score (1–5); only ≥4 kept in filtered dataset
- `model` — which Claude model generated this example
- `generated_at` — ISO 8601 timestamp

---

## Prompt Design

The generation prompt lives in [`prompts/eli5_generator.md`](prompts/eli5_generator.md).
It is versioned there — iterate on the prompt file, not inline in the script.

Key design choices:
- **Analogy style is injected per-call** — we cycle through styles to ensure diversity without
  human curation. Three styles per concept would triple costs; cycling achieves similar diversity
  at the same cost.
- **Structured output via tool use** — the script uses Claude's tool_use API to get back a typed
  JSON object rather than parsing free text. This eliminates post-hoc extraction errors.
- **Difficulty is inferred, not prescribed** — the model rates its own output's difficulty after
  writing it, which is more accurate than asking it to target a difficulty upfront.

---

## Self-Critique Filter

After generating each example, a second Claude call asks the model to rate the output:

| Criterion | Weight |
|---|---|
| Clarity (would a 5-year-old follow this?) | 40% |
| Accuracy (is it factually correct?) | 40% |
| Age-appropriateness (no jargon, no condescension) | 20% |

A weighted composite score is computed; examples scoring < 4.0 are discarded.
Typical pass rate: ~65–70% (meaning ~30–35% of generated examples are filtered out).

---

## Training

Fine-tuning is done with [Unsloth](https://github.com/unslothai/unsloth) on top of
`HuggingFaceTB/SmolLM2-360M-Instruct`.

| Hyperparameter | Value |
|---|---|
| Base model | SmolLM2-360M-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Batch size | 4 (grad accum 4 → effective 16) |
| Epochs | 3 |
| Max sequence length | 512 |
| Compute | Colab T4 (free tier) |
| Training time | ~2 hours |

Training is logged to Weights & Biases. See [W&B project]() for loss curves.

---

## Evaluation

Three tracks:

**1. LLM-as-judge** — Claude rates outputs from the base model and fine-tuned model
side-by-side (blind, randomized order) on a 1–5 scale for clarity, accuracy, and
age-appropriateness. Reported as win/tie/loss rates.

**2. Readability** — [Flesch Reading Ease](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
scores computed with `textstat`. Higher = easier to read. Target: ≥60 (plain English).

**3. Held-out test set** — 200 concepts never seen during training. Used only once for
final numbers; not used for any prompt or hyperparameter decisions.

Results will be posted here once the fine-tuned model is done.

---

## Cost Breakdown

Target: under $50 total.

| Task | Model | Est. calls | Est. cost |
|---|---|---|---|
| Generation | claude-haiku-4-5-20251001 | 2,000 | ~$8 |
| Self-critique filter | claude-haiku-4-5-20251001 | 2,000 | ~$4 |
| LLM-as-judge eval | claude-haiku-4-5-20251001 | 400 | ~$2 |
| Training | Colab T4 | — | $0 |
| **Total** | | | **~$14** |

Actual costs will be tracked in real-time by `src/generate.py` and logged to
`data/cost_log.jsonl`.

---

## Design Decisions

- **SmolLM2-360M over Qwen2.5-0.5B** — better instruction-tuned baseline; Apache 2.0 license
  makes it unambiguously usable in public demos.
- **Synthetic only, no human labels** — the core hypothesis is that filtering > labeling at
  this scale. Human labels would cost more and take longer than the API calls.
- **Self-critique over a separate judge model** — cheaper, and prior work on constitutional AI
  suggests comparable quality for this kind of scalar rating task.
- **Analogy style diversity enforced in prompt** — increases output diversity without human
  curation or post-hoc clustering. Ablation TBD.
- **Tool use for structured output** — avoids regex parsing of free text, which fails ~5% of
  the time on long outputs.

---

## Roadmap

- [x] Project scoping, dataset schema, initial seed prompt
- [ ] Full seed concept list + generation pipeline
- [ ] 3K-example synthetic dataset with quality filter
- [ ] LoRA fine-tuning run + W&B logging
- [ ] Eval harness (LLM-as-judge + readability + head-to-head)
- [ ] HuggingFace model release + Gradio demo
- [ ] Writeup with cost breakdown and honest limitations

---

## Repo Structure

```
tiny-tutor/
├── README.md                     ← you are here
├── requirements.txt
├── .gitignore
├── data/
│   ├── seed_concepts.md          ← notes on Wikipedia source pages
│   └── raw/                      ← generated JSONL (gitignored if large)
├── prompts/
│   └── eli5_generator.md         ← generation prompt, versioned
├── notebooks/
│   └── compare_outputs.ipynb     ← visual comparison: base vs fine-tuned
└── src/
    ├── generate.py               ← synthetic data generation + cost tracking
    ├── filter.py                 ← self-critique filtering
    ├── train.py                  ← LoRA fine-tuning
    └── evaluate.py               ← eval harness
```

---

## Stack

Unsloth · SmolLM2 · HuggingFace Transformers · Weights & Biases · Gradio · Anthropic Claude API

---

## License

MIT
