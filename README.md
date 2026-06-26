# Repository for Push Puppet Networks: A structured Bayesian pruning algorithm for language-model compression.

Push puppet networks learn a smooth hyper-function during training (a Bayesian
prior over a computational-cost parameter λ) that allows sub-network sampling to a given sparsity level. Unlike comparable methods (sparseGPT, Wanda), the push puppet sub-networks are structurally-pruned (entire gates, etc.) so that speed-ups can be realized on conventional hardware and kernels.

This repository holds the code that produces the results and figures in the
working paper *Push Puppet Networks: A Structured Bayesian Pruning Algorithm for
Language Model Compression* (`paper/thresholding_draft1.qmd`).

## Layout

```
python/    Model training, evaluation, and benchmarking scripts
scripts/   SLURM submission wrappers that run the python/ scripts on a cluster
```

The pipeline involves training a fitted OLMO-3 deep neural network with a $\lambda$ function for sparsity, then run the three
evaluation scripts to obtain paper plots and results. Each evaluation script writes a JSON or CSV
results file that the paper's R/Quarto code reads to render a figure.

## Environment

All Python scripts use [uv](https://docs.astral.sh/uv/) and declare their
dependencies inline (PEP 723), so they can be run directly:

```bash
uv run python/olmo3_mini_train.py --help
```

The core dependencies are `torch>=2.0`, `transformers>=5.0` (for the
`Olmo3ForCausalLM` architecture), `numpy`, and `torchcurves` (the monotone
B-spline layers used for the λ→τ threshold function).

### Data

Training and evaluation read OLMo-3 tokens stored as `uint32` binary files
(`train.bin`, `validation.bin`) plus a tokenizer directory. These are prepared
with `prepare_olmo3_mini.py` (downloads the OLMo-3 tokenizer and tokenizes the
dataset; a HuggingFace login is required for the tokenizer). That preparation
script is not part of this public distribution — point the scripts at an
existing `--data_dir` containing the tokenized files.

## Training

Training and evaluation of these models was done on Texas A&M's ACES cluster with Intel Data Center Max (PVC) GPUs with 48GB RAM. For model training, 6 GPUs were used with `pytorch` and DDP using the `xpu` framework. For model evaluation and testing of measured speed-ups, a single PVC GPU was used. CUDA is also programmed into the scripts and they should also run on CUDA but I have not verified all of them on CUDA hardware.

## `python/` scripts

### `olmo3_mini_train.py` — train the push puppet model

This is the main script that trains a push puppet network with Olmo-3 data and base model architecture. Specifically, it creates an `Olmo3ForCausalLM` object with what is classified in the script as the `hyper_joint` pruning architecture: the
B-spline λ→τ stochastic gating with per-unit amplitude rescaling, applied
jointly to FFN neurons and attention heads. This is the model defined in the
*Algorithm Definition*, *The τ function*, and *Amplitude and Input Scaling*
sections of the paper.

Prior to running this script, the `prepare_olmo3_mini.py` script can be run to download the relevant OLMO-3 tokens.

Key pieces that implement the paper's math:

- **λ sampling** (paper @eq-lambda, *Training*): λ = 0 with probability ⅓ (dense
  pass, no penalty), otherwise λ ~ Exp(rate) with probability ⅔.
- **B-spline τ(λ)** (@eq-spline-coef, @eq-tau): a monotone third-order B-spline
  per layer, with cumsum+softplus control points and a fixed −1/T intercept so
  all units stay active at λ=0.
- **Amplitude correction** (@eq-scale, @eq-weff): per-neuron / per-head gate and
  output scales `f_gate`, `f_scale`, plus the per-input-dimension `f_input`
  giving the separable rank-1 correction.
- **Stochastic mask + straight-through estimator** (@eq-mask, @eq-ste): the
  Bernoulli gate sampling and gradient pass.
- **FLOP-weighted penalty** (@eq-penalty): FFN weight `6·d·d_ff`, attention
  weight `8·d²`.

The ~1.3B configuration used in the paper's evaluation (`d=2048`, `d_ff=8192`,
`L=16`, `H=16`, `n_knots=10`) is selected with the flags shown in
`scripts/submit_olmo3_mini_1_3b.sh`. Supports DDP / FSDP, gradient
checkpointing, and CUDA / XPU accelerators.

**Output:** a checkpoint directory (`model.pt` plus config) that every
evaluation script below consumes. This checkpoint is the foundation for **all**
figures in the paper.

### `make_stochastic_results.py` — λ-sweep, stability, and size evaluation

Evaluates a trained Olmo-3 push puppet checkpoint (as defined above) and writes `stochastic_results.json` in the
format the paper's R code expects. Sweeps λ from 0 upward, pruning the model at
each value and measuring perplexity (mean/std over mask draws), active FFN/
attention fractions, the learned τ values, and compressed sizes.

**Output:** `stochastic_results.json`, containing `test_ppl.dense`,
`lambda_sweep.<variant>`, `resample_stability`, and `size_comparison`.

**Figures produced (from `stochastic_results.json`):**

| Paper figure | Label | What it shows |
|---|---|---|
| Lambda Response Curve | `fig-lambda-response` | PPL vs λ |
| Active Unit Fractions | `fig-active-frac` | Surviving FFN neurons / attention heads vs λ |
| Sampling Error | `fig-stability` | PPL distribution across mask draws at λ≈3 |
| Learned Threshold Behaviour | `fig-tau` | τ(λ) for FFN and attention |
| Compression–Quality Frontier | `fig-frontier` | PPL vs compressed size (MB) |

### `compression_comparison.py` — baseline comparison & measured speedup

Loads a trained Olmo-3 push puppet checkpoint and compares the push puppet τ-thresholding against
unstructured baselines — **SparseGPT** (Hessian-based) and **Wanda**
(|W|×‖X‖ importance) — plus int8/int4 quantization and post-hoc magnitude
pruning, all at matched memory budgets and evaluated on perplexity. Also times
the compiled sub-network against the compiled dense baseline to report the
real (not theoretical) inference speedup.

**Output:** `compression_comparison.json`.

**Figures produced (from `compression_comparison.json`):**

| Paper figure | Label | What it shows |
|---|---|---|
| Inference Speedup | `fig-speedup` | Compiled speedup vs λ (with physical MB) |
| Comparison with Wanda/SparseGPT (size) | `fig-comparison-size` | PPL vs model size across methods |
| Comparison with Wanda/SparseGPT (FLOPs) | `fig-comparison-flops` | PPL vs FLOPs across methods |

> **Dependency note:** this script imports prototype model classes from
> `stochastic_weight_test.py` (and, if present, `rosa_comparison.py` for the
> optional LoRA/RoSA rows). Those modules are not part of this public
> distribution; the OLMo-3 evaluation path is the supported one.

### `benchmark_speculative.py` — speculative-decoding benchmarks

Uses a single push puppet checkpoint as both the cheap draft (high λ) and the
accurate target (low λ) for speculative decoding (Leviathan et al., 2023). Has
several modes (selected with `--mode`):

- **`sweep`** — full generative timing sweep over draft λ, measuring end-to-end
  throughput and token acceptance rate. Produces the *measured* speculative
  results.
- **`ram_budget`** — analytical multi-draft sweep under a fixed GPU memory
  budget. For each draft λ it computes how many shared-weight candidates `K`
  fit, then compares a single draft against self-speculative tree decoding.
  Acceptance rates are *measured* from real forward passes (exact TV overlap);
  only the per-pass cost is modelled.
- **`optimize`** — fast analytical optimization of (λ_draft, γ) via Theorems
  3.8 / 3.11 of Leviathan et al.
- **`context_sweep`** — acceptance α(λ, T) across context lengths.

**Figures produced:**

| Paper figure | Label | Mode / output |
|---|---|---|
| Measured speculative speedup | `fig-speculative-measured` | `--mode sweep` → `speculative_decode.csv` |
| Self-speculative projection | `fig-speculative` | `--mode ram_budget` → `speculative_ram_budget.csv` |
| Self-speculative heatmap | `fig-speculative-heat` | `--mode ram_budget` (same CSV) |

> The `sweep` mode writes `speculative_sweep.csv`; the paper reads it as
> `speculative_decode.csv` (rename/copy when collecting results).

### `olmo3_speculative_decode.py` — standalone speculative generation

A self-contained implementation of the speculative-decoding loop (draft γ
steps, verify once with the target, accept/reject) for a single OLMo-3
stochastic-pruning checkpoint. Useful for generating text from a prompt or
benchmarking speculative vs. target-only vs. draft-only decoding directly. It
is imported by `benchmark_speculative.py` for its draft/target forward passes;
it does not itself produce a paper figure.

## `scripts/` — SLURM submission wrappers

Each wrapper sets up the cluster environment (Intel PVC / XPU partition, module
loads, virtualenv) and invokes the corresponding `python/` script with the
paper's parameters. Override paths and parameters via environment variables
where noted in each script (e.g. `CKPT_DIR`, `DATA_DIR`, `MODE`, `BUDGET_GB`).

| Script | Runs | Produces | Feeds figures |
|---|---|---|---|
| `submit_olmo3_mini_1_3b.sh` | `olmo3_mini_train.py` (6× PVC, 1.3B config, 3 epochs) | trained checkpoint | all (via the checkpoint) |
| `submit_make_stochastic.sh` | `make_stochastic_results.py` | `stochastic_results.json` | `fig-lambda-response`, `fig-active-frac`, `fig-stability`, `fig-tau`, `fig-frontier` |
| `submit_stochastic_compression.sh` | `compression_comparison.py` | `compression_comparison.json` | `fig-speedup`, `fig-comparison-size`, `fig-comparison-flops` |
| `submit_benchmark_speculative.sh` | `benchmark_speculative.py` (`MODE=sweep` default) | `speculative_sweep.csv` (→ `speculative_decode.csv`) | `fig-speculative-measured` |
| `submit_ram_budget.sh` | `benchmark_speculative.py --mode ram_budget` | `speculative_ram_budget.csv` | `fig-speculative`, `fig-speculative-heat` |

All evaluation jobs default to `CKPT_DIR=$SCRATCH/demo_ckpt/olmo3_mini_1b_all`,
matching the `results/olmo3_mini1b_all/` directory the paper reads from.

## Reproducing the paper figures

1. Prepare tokenized OLMo-3 data (`prepare_olmo3_mini.py`, external) into a
   `--data_dir`.
2. Train: `sbatch scripts/submit_olmo3_mini_1_3b.sh`.
3. Evaluate against the resulting checkpoint:
   - `sbatch scripts/submit_make_stochastic.sh` → `stochastic_results.json`
   - `sbatch scripts/submit_stochastic_compression.sh` → `compression_comparison.json`
   - `sbatch scripts/submit_benchmark_speculative.sh` → speculative sweep CSV
   - `sbatch scripts/submit_ram_budget.sh` → RAM-budget CSV
4. Place the JSON/CSV outputs under `results/olmo3_mini1b_all/` and render
   `thresholding_draft1.qmd`.
