# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "transformers>=5.0",
#   "numpy>=1.24",
#   "torchcurves",
# ]
# ///
"""
make_stochastic_results.py

Evaluates a trained olmo3_mini_train.py checkpoint and writes
stochastic_results.json in the format expected by the paper's R code
(thresholding_draft1.qmd).

Produces (for the given checkpoint):
  - test_ppl.dense           : PPL at lambda=0 (all units active)
  - lambda_sweep.<variant>   : per-lambda PPL mean/std + active fractions + tau
  - resample_stability       : PPL variance at training-mean lambda
  - size_comparison          : dense and pruned model sizes in MB

Usage:
    uv run python/make_stochastic_results.py \\
        --ckpt_dir results/olmo3_mini_1b \\
        --data_dir /path/to/olmo3_data

    # faster smoke test (fewer batches/draws):
    uv run python/make_stochastic_results.py \\
        --ckpt_dir results/olmo3_mini_1b \\
        --data_dir /path/to/olmo3_data \\
        --max_batches 20 --n_draws 5 --n_stability_draws 10
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Olmo3Config
from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))

from olmo3_mini_train import (
    BSplineScaledOlmo3MLP,
    BSplineScaledOlmo3Attn,
    TokenDataset,
    inject_stochastic_mlp,
    inject_stochastic_attn,
    inject_per_head_qk_norm,
    set_lam,
    set_hard_mask,
    set_sample_mask,
    structural_prune,
)

LAM_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def load_model(ckpt_dir: Path, device: torch.device):
    train_args = json.loads((ckpt_dir / "train_args.json").read_text())
    config = Olmo3Config.from_pretrained(ckpt_dir)
    model = Olmo3ForCausalLM(config)

    n_knots = train_args.get("n_knots", 8)
    degree  = train_args.get("degree", 3)

    inject_stochastic_mlp(model, n_knots=n_knots, degree=degree)
    inject_stochastic_attn(model, n_knots=n_knots, degree=degree)
    if not train_args.get("no_per_head_norm", False):
        inject_per_head_qk_norm(model)

    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_args


def model_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


@torch.no_grad()
def eval_ppl(model, loader, device, lam_val: float, max_batches: int) -> float:
    lam_t = torch.tensor(float(lam_val), device=device)
    set_lam(model, lam_t)
    total = n = 0
    for i, (inp, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(input_ids=inp)
        total += F.cross_entropy(
            out.logits.float().view(-1, out.logits.size(-1)),
            tgt.view(-1)).item() * inp.numel()
        n += inp.numel()
    return math.exp(min(total / n, 20))


@torch.no_grad()
def eval_ppl_multi(model, loader, device, lam_val: float,
                   n_draws: int, max_batches: int) -> tuple[float, float, float, float]:
    """PPL statistics over n_draws independent Bernoulli mask samples."""
    lam_t = torch.tensor(float(lam_val), device=device)
    set_lam(model, lam_t)
    set_sample_mask(model, True)
    set_hard_mask(model, False)
    ppls = []
    for _ in range(n_draws):
        total = n = 0
        for i, (inp, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(input_ids=inp)
            total += F.cross_entropy(
                out.logits.float().view(-1, out.logits.size(-1)),
                tgt.view(-1)).item() * inp.numel()
            n += inp.numel()
        ppls.append(math.exp(min(total / n, 20)))
    set_sample_mask(model, False)
    arr = np.array(ppls)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


@torch.no_grad()
def gate_info(model, lam_val: float, device: torch.device) -> dict:
    lam_t = torch.tensor(float(lam_val), device=device)
    set_lam(model, lam_t)

    # Hard fraction (rho > 0.5): matches what structural_prune actually keeps.
    # Soft fraction (mean rho): reported separately for the paper plots.
    ffn_rhos   = [m.inclusion_probs(lam_t)
                  for m in model.modules() if isinstance(m, BSplineScaledOlmo3MLP)]
    attn_rhos  = [m.inclusion_probs(lam_t)
                  for m in model.modules() if isinstance(m, BSplineScaledOlmo3Attn)]
    tau_ffn    = [m._tau(lam_t).item()
                  for m in model.modules() if isinstance(m, BSplineScaledOlmo3MLP)]
    tau_attn   = [m._tau(lam_t).item()
                  for m in model.modules() if isinstance(m, BSplineScaledOlmo3Attn)]

    def _soft(rhos): return float(np.mean([r.mean().item() for r in rhos])) if rhos else 1.0
    def _hard(rhos): return float(np.mean([(r > 0.5).float().mean().item() for r in rhos])) if rhos else 1.0

    return {
        "ffn_frac":       _soft(ffn_rhos),   # mean rho — used in paper plots
        "attn_frac":      _soft(attn_rhos),
        "ffn_frac_hard":  _hard(ffn_rhos),   # fraction with rho > 0.5 — matches structural_prune
        "attn_frac_hard": _hard(attn_rhos),
        "tau_ffn":   float(np.mean(tau_ffn))  if tau_ffn  else None,
        "tau_attn":  float(np.mean(tau_attn)) if tau_attn else None,
    }


def _cpu_copy(model):
    """Deep-copy model to CPU with B-spline cache cleared.

    After deepcopy + .cpu() the model weights are on CPU but cached spline
    tensors (_sp_tau, _sp_gate, _sp_scale, _sp_input) may still reference the
    original device (e.g. XPU). structural_prune calls inclusion_probs, which
    short-circuits to the cached tensor — causing a device mismatch. Clearing
    the cache forces a fresh CPU recompute inside structural_prune.
    """
    m = copy.deepcopy(model).cpu()
    for mod in m.modules():
        for attr in ("_sp_tau", "_sp_gate", "_sp_scale", "_sp_input"):
            if hasattr(mod, attr):
                setattr(mod, attr, None)
    return m


@torch.no_grad()
def compressed_mb(model, lam_val: float,
                  stochastic: bool = False, n_draws: int = 1) -> float:
    """Compressed model size in MB after structural pruning.

    stochastic=False (default): keep neurons with rho > 0.5 (deterministic).
    stochastic=True: draw the keep mask from Bernoulli(rho), matching the
        stochastic masking used during training and PPL evaluation.
        n_draws > 1 averages the size over multiple independent draws.
    """
    if not stochastic or n_draws == 1:
        m = _cpu_copy(model)
        structural_prune(m, lam_val, prune_attn=True, stochastic=stochastic)
        return model_mb(m)
    sizes = []
    for _ in range(n_draws):
        m = _cpu_copy(model)
        structural_prune(m, lam_val, prune_attn=True, stochastic=True)
        sizes.append(model_mb(m))
    return float(np.mean(sizes))


@torch.no_grad()
def time_forward_ms(model, seq_len: int, batch: int,
                    n_warmup: int = 3, n_runs: int = 10) -> float:
    device = next(model.parameters()).device
    inp = torch.randint(0, 100_000, (batch, seq_len), device=device)
    for _ in range(n_warmup):
        model(input_ids=inp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(input_ids=inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Produce stochastic_results.json from an olmo3_mini checkpoint")
    p.add_argument("--ckpt_dir",   required=True,
                   help="Checkpoint dir (model.pt, config.json, train_args.json)")
    p.add_argument("--data_dir",   required=True,
                   help="Dir with validation.bin (uint32 tokens from prepare_olmo3_mini.py)")
    p.add_argument("--out",        default=None,
                   help="Output path (default: {ckpt_dir}/stochastic_results.json)")
    p.add_argument("--variant",    default="rank1_hyper_joint",
                   help="Variant name written into the JSON")
    p.add_argument("--lam_grid",   nargs="+", type=float, default=LAM_GRID)
    p.add_argument("--n_draws",    type=int, default=10,
                   help="Bernoulli mask draws per lambda point")
    p.add_argument("--n_stability_draws", type=int, default=30,
                   help="Draws for resample_stability section")
    p.add_argument("--stability_lam", type=float, default=3.0,
                   help="Lambda used for stability test (should be near training mean)")
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cpu", "cuda", "xpu"])
    p.add_argument("--no_comp_time", action="store_true",
                   help="Skip inference timing (saves ~30s per lambda)")
    p.add_argument("--prune_mode",  default="hard",
                   choices=["hard", "stochastic"],
                   help="hard: keep neurons with rho > 0.5 (default); "
                        "stochastic: draw keep mask from Bernoulli(rho), "
                        "matching the masking used during training")
    p.add_argument("--n_prune_draws", type=int, default=10,
                   help="Draws to average comp_mb over when --prune_mode stochastic "
                        "(default 10; ignored for hard)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev_str = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dev_str = "xpu"
        else:
            dev_str = "cpu"
    else:
        dev_str = args.device
    device = torch.device(f"{dev_str}:0") if dev_str != "cpu" else torch.device("cpu")

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out) if args.out else ckpt_dir / "stochastic_results.json"

    stochastic_prune = (args.prune_mode == "stochastic")

    print("=" * 65)
    print(f"  Checkpoint : {ckpt_dir}")
    print(f"  Data       : {args.data_dir}")
    print(f"  Device     : {device}")
    print(f"  Variant    : {args.variant}")
    print(f"  Prune mode : {args.prune_mode}"
          + (f" ({args.n_prune_draws} draws)" if stochastic_prune else ""))
    print(f"  Output     : {out_path}")
    print("=" * 65, flush=True)

    print("  Loading model…", flush=True)
    model, train_args = load_model(ckpt_dir, device)
    seq_len  = train_args.get("seq_len", 512)
    dense_size_mb = model_mb(model)
    print(f"  Dense size : {dense_size_mb:.1f} MB  seq_len={seq_len}", flush=True)

    loader = DataLoader(
        TokenDataset(Path(args.data_dir) / "validation.bin", seq_len),
        batch_size=args.batch, shuffle=False, num_workers=0)

    # Dense PPL — model at lambda=0 (all units active)
    print("\n  Dense PPL (lambda=0)…", flush=True)
    set_hard_mask(model, False)
    set_sample_mask(model, False)
    dense_ppl = eval_ppl(model, loader, device, 0.0, args.max_batches)
    print(f"  Dense PPL = {dense_ppl:.2f}", flush=True)

    # Lambda sweep
    sweep_rows = []
    print(f"\n  Lambda sweep ({len(args.lam_grid)} pts × {args.n_draws} draws)…")
    print(f"  {'lam':>8}  {'ppl':>9}  {'std':>7}  "
          f"{'ffn%(soft)':>10}  {'ffn%(hard)':>10}  "
          f"{'attn%(soft)':>11}  {'attn%(hard)':>11}  "
          f"{'comp_mb':>8}  {'sec':>5}", flush=True)
    print("  " + "-" * 95)

    for lam in args.lam_grid:
        t0 = time.perf_counter()

        ppl_mean, ppl_std, _, _ = eval_ppl_multi(
            model, loader, device, lam, args.n_draws, args.max_batches)
        g = gate_info(model, lam, device)
        cmb = compressed_mb(model, lam,
                             stochastic=stochastic_prune,
                             n_draws=args.n_prune_draws)

        comp_ms = None
        if not args.no_comp_time:
            m_slim = _cpu_copy(model)
            structural_prune(m_slim, lam, prune_attn=True,
                             stochastic=stochastic_prune)
            m_slim.eval()
            comp_ms = time_forward_ms(m_slim, seq_len, args.batch)
            del m_slim

        sweep_rows.append({
            "lam":       lam,
            "ppl":       ppl_mean,
            "ppl_std":   ppl_std,
            "ffn_frac":  g["ffn_frac"],
            "attn_frac": g["attn_frac"],
            "token_frac": 1.0,
            "tau_ffn":   g["tau_ffn"],
            "tau_attn":  g["tau_attn"],
            "comp_ms":   comp_ms,
            "comp_mb":   cmb,
        })

        print(f"  {lam:>8.4f}  {ppl_mean:>9.2f}  {ppl_std:>7.3f}"
              f"  {g['ffn_frac']*100:>9.1f}%"
              f"  {g['ffn_frac_hard']*100:>9.1f}%"
              f"  {g['attn_frac']*100:>10.1f}%"
              f"  {g['attn_frac_hard']*100:>10.1f}%"
              f"  {cmb:>8.1f}"
              f"  {time.perf_counter()-t0:>5.0f}s", flush=True)

    # Resample stability at training-mean lambda
    print(f"\n  Resample stability (lambda={args.stability_lam}, "
          f"{args.n_stability_draws} draws)…", flush=True)
    st_mean, st_std, st_min, st_max = eval_ppl_multi(
        model, loader, device, args.stability_lam,
        args.n_stability_draws, args.max_batches)
    print(f"  PPL mean={st_mean:.2f}  std={st_std:.4f}  "
          f"range=[{st_min:.2f}, {st_max:.2f}]", flush=True)

    # Build JSON
    variant = args.variant
    results = {
        "config": {
            "d_model":  train_args.get("hidden"),
            "n_heads":  train_args.get("n_heads"),
            "n_layers": train_args.get("n_layers"),
            "seq_len":  seq_len,
            "vocab":    train_args.get("vocab"),
            "lam_rate": train_args.get("lam_rate"),
            "n_knots":  train_args.get("n_knots"),
            "degree":   train_args.get("degree"),
        },
        "test_ppl": {
            "dense":  dense_ppl,
            variant:  sweep_rows[0]["ppl"] if sweep_rows else None,
        },
        "lambda_sweep": {
            variant: sweep_rows,
        },
        "resample_stability": {
            "dense": {
                "ppl_mean":      dense_ppl,
                "deterministic": True,
            },
            variant: {
                "ppl_mean": st_mean,
                "ppl_std":  st_std,
                "ppl_min":  st_min,
                "ppl_max":  st_max,
            },
        },
        "size_comparison": [
            {"variant": "dense",  "comp_mb": dense_size_mb},
            {"variant": variant,  "comp_mb": sweep_rows[-1]["comp_mb"] if sweep_rows else None},
        ],
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Written → {out_path}", flush=True)


if __name__ == "__main__":
    main()
