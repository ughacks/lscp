# Learn by Surprise, Commit by Proof

A self-gated post-training framework for autonomous knowledge acquisition in language models.

LSCP enables language models to *learn from what they read*—selectively, verifiably, and without external supervision. It detects what a model does not already know (via surprisal), verifies new content against existing knowledge (via self-generated Q&A chains), and commits verified knowledge to parametric weights at a strength proportional to conviction depth (via per-item β₂ adjustment of AdamW).

**Paper**: [Learn by Surprise, Commit by Proof](https://arxiv.org/abs/XXXX.XXXXX)

## How It Works

LSCP operates in three stages:

**Stage 1 — Detect and Ground.** Compute per-token surprisal for each passage. Flag passages exceeding a threshold (μ + λσ) as surprising. This is a single forward pass with zero additional cost.

**Stage 2 — Verify, Grade and Annotate.** For each flagged passage, generate a Q&A chain tagged by epistemic origin (`[existing]`, `[mechanism]`, `[implication]`). Check each answer against the model's existing knowledge. The number of verified steps becomes the conviction depth *k*.

**Stage 3 — Gated Weight Update.** Adjust AdamW's β₂ proportionally to *k* via β₂ = 0.999 · rᵏ. When *k* = 0, β₂ = 0.999 (standard AdamW, no learning). As *k* increases, β₂ decreases, progressively opening the Variance Lock. A single parameter *r* governs the entire learning intensity.

## Key Results

- Standard fine-tuning produces rote memorization (perturbation gap 11.6× baseline); all LSCP conditions learn semantically (2.7–3.0×).
- The r=1.0 condition (identical optimizer, Q&A format only) confirms that the training data format, not β₂ gating, is the primary mechanism preventing memorization.
- β₂ gating provides adjacent knowledge protection: 93% at r=0.98 vs. 90% baseline.
- Results replicate across six models (8B–32B) from four model families.

## Requirements

- Python 3.10+
- [MLX](https://github.com/ml-explore/mlx) (Apple Silicon) or equivalent inference framework
- A quantized language model (tested with 4-bit models on M3 Pro 36GB)

Install dependencies:

```bash
pip install mlx mlx-lm numpy matplotlib
```

## Repository Structure

```
lscp1.py          # Stage 1: Detect and Ground (surprisal computation)
lscp2_1.py        # Stage 2-1: Q&A Chain Generation
lscp2_2.py        # Stage 2-2: Consistency Checking
lscp3.py          # Stage 3: Gated Weight Update (training + evaluation)
config.py          # Model configuration, paths, hyperparameters
passages.py        # Test corpus (60 passages: 20 known, 20 novel, 20 corrupt)
test_qa.json       # 70 test questions for five-way evaluation
paraphrases.json   # Semantically equivalent paraphrases for perturbation gap
```

## Quick Start

### 1. Configure

Edit `config.py` to set your model path and results directory:

```python
MODEL_NAME = "mlx-community/Qwen3-14B-4bit"
RESULTS_DIR = Path("lscp_results")
```

### 2. Run the Pipeline

```bash
# Stage 1: Detect surprising passages
python3 lscp1.py

# Stage 2-1: Generate Q&A chains
python3 lscp2_1.py

# Stage 2-2: Consistency checking
python3 lscp2_2.py

# Stage 3: Gated weight update (default r=0.98)
python3 lscp3.py 0.98
```

### 3. No-Passage Ablation

To train on Q&A pairs only (without source window text):

```bash
python3 lscp3.py --no-passage 0.95
```

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 0.98 | Decay factor for β₂ schedule |
| `LAMBDA` | 2.0 | Surprisal threshold (μ + λσ) |
| `QA_SCALE` | 3.0 | Scaling constant *c* for N = ⌈S_i · c⌉ |
| `LEARNING_RATE` | 1e-5 | AdamW learning rate |
| `EPOCHS` | 3 | Training epochs per passage |
| `LORA_RANK` | 8 | LoRA adapter rank |

## Output

Stage 3 produces `stage3_results_r{R}.json` containing:

- Baseline, Normal and LSCP perplexity (target, known, retain)
- Perturbation gap (paraphrase PPL / original PPL)
- Five-way Q&A accuracy (novel-direct, novel-adjacent, corrupt-direct, corrupt-adjacent, unrelated)
- Training metadata (items, time, loss curves)

## Terminology

The code uses `"aligned"` internally for passage labels (historical convention). The paper uses **"known"** for this category. Display output shows "Known".

## Citation

```bibtex
@article{choi2025lscp,
  title={Learn by Surprise, Commit by Proof},
  author={Choi, Kang-Sin},
  year={2025}
}
```

## License

MIT
