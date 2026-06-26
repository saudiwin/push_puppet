"""
compression_comparison.py

Loads the three trained model variants from stochastic_results_ckpt.pt and
compares compression strategies at matched memory budgets:

  dense          — full fp32 baseline (trained without pruning)
  quant-int8     — RTN int8 per-row, dequantised for eval   (applied to dense)
  quant-int4     — same, 4-bit                               (applied to dense)
  sparsegpt-N    — Hessian-based unstructured pruning        (applied to dense)
  wanda-N        — |W|×‖X‖ importance pruning (Sun 2023)     (applied to dense)
  struct-joint   — post-hoc magnitude pruning FFN + heads    (applied to dense)
  tau-ffn λ=X   — τ-threshold FFN-only (ffn_poly model)
  tau-joint λ=X — τ-threshold joint FFN + attn (joint_poly model)
  tau-token λ=X — τ-threshold joint + token embedding (token_poly model)
  lora           — LoRA fine-tune (from rosa_comparison_results.pt if present)
  rosa           — RoSA fine-tune (LoRA + sparse delta, same checkpoint)

Evaluation: WikiText-2 PPL.

Usage:
    uv run python/compression_comparison.py \\
        --ckpt stochastic_results_ckpt.pt \\
        [--device cpu] [--n_calib 64] [--max_batches 100]
        [--rosa_ckpt rosa_comparison_results.pt]
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from stochastic_weight_test import (
    LM, PolyLM, ScaledPolyLM,
    StochasticRowLinear, StochasticHeadAttention,
    PolyStochasticRowLinear, PolyStochasticHeadAttention,
    PolyStochasticEmbedding,
    PolyFFN,
    build_tensor, build_vocab,
    _TokenDataset, _HAS_MEMMAP,
    _RemappedTokenDataset, _build_vocab_remap, OLMO3_VOCAB,
    evaluate, build_compressed,
)
try:
    from rosa_comparison import RoSALM, restore_masks
    _HAS_ROSA = True
except ImportError:
    _HAS_ROSA = False

try:
    from rank1_weight_test import Rank1ScaledPolyLM, build_compressed_rank1
    _HAS_RANK1 = True
except ImportError:
    _HAS_RANK1 = False

# ── constants ──────────────────────────────────────────────────────────────────

TAU_LAMS   = list(np.concatenate([np.arange(0.0, 2, 0.1),np.arange(2.25,5.25,0.25)]))                          # 0.0 … 5.0, step 0.1
SPARSITIES = list(np.arange(0.05, 0.96, 0.05)) + [0.99]              # 5% … 99%


def resolve_sparsities(args) -> list[float]:
    """Sparsity levels for the SparseGPT/Wanda sweeps.

    Uses the comma-separated ``--sparsities`` flag if given, else the default
    20-level grid (SPARSITIES). Fewer levels ≈ proportionally less prune time.
    """
    raw = getattr(args, "sparsities", None)
    if not raw:
        return list(SPARSITIES)
    return [float(s) for s in str(raw).split(",") if s.strip()]


# ── checkpoint loading ─────────────────────────────────────────────────────────

def load_models(ckpt_path: str, device: torch.device,
                temperature: float = 0.5,
                penalty_mode: str = "flops") -> tuple[dict, dict]:
    """
    Returns (models, config) where models includes dense, ffn_poly, joint_poly,
    and optionally token_poly (PolyLM with token_stoch=True).

    temperature: injected into old checkpoints that predate _temperature_buf.
    penalty_mode: used when not saved in the checkpoint config.
    """
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt["config"]
    states = ckpt["model_states"]
    token_stoch_tags = set(cfg.get("token_stoch_tags", []))
    pm  = cfg.get("penalty_mode", penalty_mode)
    nk  = cfg.get("n_knots", 8)   # old checkpoints default to 8
    deg = cfg.get("degree", 3)     # old checkpoints default to cubic

    models = {}
    for tag, variant, use_poly in [
        ("dense",             "dense", False),
        ("ffn_poly",          "ffn",   True),
        ("joint_poly",        "joint", True),
        ("token_poly",        "joint", True),
        ("hyper_ffn",         "ffn",   "hyper"),
        ("hyper_joint",       "joint", "hyper"),
        ("rank1_hyper_ffn",   "ffn",   "rank1_hyper"),
        ("rank1_hyper_joint", "joint", "rank1_hyper"),
    ]:
        if tag not in states:
            continue
        ts = tag in token_stoch_tags
        kw = dict(variant=variant, token_stoch=ts, penalty_mode=pm,
                  n_knots=nk, degree=deg)
        if use_poly == "rank1_hyper":
            if _HAS_RANK1:
                m = Rank1ScaledPolyLM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                                      cfg["n_layers"], cfg["seq_len"], **kw)
            else:
                m = ScaledPolyLM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                                 cfg["n_layers"], cfg["seq_len"], **kw)
        elif use_poly == "hyper":
            m = ScaledPolyLM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                             cfg["n_layers"], cfg["seq_len"], **kw)
        elif use_poly:
            m = PolyLM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                       cfg["n_layers"], cfg["seq_len"], **kw)
        else:
            m = LM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                   cfg["n_layers"], cfg["seq_len"], variant=variant,
                   penalty_mode=pm)
        missing, _ = m.load_state_dict(states[tag], strict=False)
        if any("_temperature_buf" in k for k in missing):
            for mod in m.modules():
                if hasattr(mod, "_temperature_buf"):
                    mod._temperature_buf.fill_(temperature)
        m.eval()   # stay on CPU; caller moves to device only for eval
        models[tag] = m

    return models, cfg


# ── size utilities ─────────────────────────────────────────────────────────────

def dense_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6


def sparse_mb(model: nn.Module) -> float:
    total = sum(int((p != 0).sum().item()) for p in model.parameters())
    return total * 4 / 1e6


def quant_mb(model: nn.Module, bits: int) -> float:
    return sum(p.numel() for p in model.parameters()) * bits / 8 / 1e6


def quant_sparse_mb(model: nn.Module, bits: int) -> float:
    """Non-zero params × bits/8 — for sparse+quantised models."""
    nz = sum(int((p != 0).sum().item()) for p in model.parameters())
    return nz * bits / 8 / 1e6


def expected_mb_poly_quant(model: PolyLM, lam_val: float, bits: int) -> float:
    """Expected size of τ-threshold model stored at `bits` bits per param."""
    return expected_mb_poly(model, lam_val) * bits / 32


@torch.no_grad()
def expected_mb_poly(model: PolyLM, lam_val: float) -> float:
    """Expected compressed size in MB for a PolyLM at the given λ."""
    lam      = torch.tensor(lam_val)
    stoch_id = set()
    stoch_p  = 0.0

    # ── Token embedding (shared with head via weight tying) ───────────────────
    if isinstance(model.embed, PolyStochasticEmbedding):
        stoch_id.add(id(model.embed.weight))
        rho = model.embed.inclusion_probs(lam)
        # Each active token row contributes embedding_dim params (embed + head share it)
        stoch_p += rho.sum().item() * model.embed.embedding_dim

    for block in model.blocks:
        # ── FFN stochastic neurons ────────────────────────────────────────────
        expand = block.ffn.expand
        if isinstance(expand, PolyStochasticRowLinear):
            contract = block.ffn.contract
            stoch_id.add(id(expand.weight))
            if expand.bias_param is not None:
                stoch_id.add(id(expand.bias_param))
            stoch_id.add(id(contract.weight))
            if contract.bias is not None:
                stoch_id.add(id(contract.bias))
            rho        = expand.inclusion_probs(lam)
            per_neuron = expand.weight.shape[1] + contract.weight.shape[0]
            if expand.bias_param is not None: per_neuron += 1
            if contract.bias  is not None:    per_neuron += 1
            stoch_p += rho.sum().item() * per_neuron

        # ── Attention stochastic heads ────────────────────────────────────────
        attn = block.attn
        if isinstance(attn, PolyStochasticHeadAttention):
            stoch_id.add(id(attn.qkv.weight))
            stoch_id.add(id(attn.proj.weight))
            rho      = attn.inclusion_probs(lam)
            per_head = 4 * attn.d_head * attn.d_model   # QKV + proj per head
            stoch_p += rho.sum().item() * per_head

    fixed = sum(p.numel() for p in model.parameters() if id(p) not in stoch_id)
    return (fixed + stoch_p) * 4 / 1e6


def solve_lam_poly(model: PolyLM, target_mb: float,
                   lam_max: float = 20.0, n_grid: int = 400) -> float | None:
    from scipy.optimize import brentq
    grid  = np.linspace(0.0, lam_max, n_grid)
    vals  = np.array([expected_mb_poly(model, float(lv)) for lv in grid])
    diffs = vals - target_mb
    if diffs[0] <= 0:
        return 0.0
    cross = np.where((diffs[:-1] > 0) & (diffs[1:] <= 0))[0]
    if len(cross) == 0:
        return None
    i = int(cross[0])
    return float(brentq(
        lambda lv: expected_mb_poly(model, float(lv)) - target_mb,
        float(grid[i]), float(grid[i + 1]), xtol=1e-6))


# ── mask helpers ──────────────────────────────────────────────────────────────

def set_dense_masks(model: nn.Module) -> None:
    """Set all stochastic masks to all-ones (= dense evaluation mode)."""
    for m in model.modules():
        if isinstance(m, (PolyStochasticRowLinear, StochasticRowLinear,
                          PolyStochasticHeadAttention, StochasticHeadAttention,
                          PolyStochasticEmbedding)):
            m.mask.fill_(1.0)


# ── quantization (RTN per output-row symmetric) ────────────────────────────────

def _quantize_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    maxval = 2 ** (bits - 1) - 1
    orig_shape = w.shape
    if w.dim() == 1:
        w = w.unsqueeze(0)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxval
    w_dq  = (w / scale).round_().clamp_(-maxval - 1, maxval) * scale
    return w_dq.reshape(orig_shape)


@torch.no_grad()
def _apply_quantization(model: nn.Module, bits: int) -> None:
    """Quantize all weight tensors in-place (masks and sparsity patterns preserved)."""
    seen_ids: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, (PolyStochasticRowLinear, StochasticRowLinear)):
            mod.weight.data.copy_(_quantize_tensor(mod.weight.data, bits))
            if mod.bias_param is not None:
                mod.bias_param.data.copy_(_quantize_tensor(mod.bias_param.data, bits))
        elif isinstance(mod, PolyStochasticEmbedding):
            if id(mod.weight) not in seen_ids:
                mod.weight.data.copy_(_quantize_tensor(mod.weight.data, bits))
                seen_ids.add(id(mod.weight))
        elif isinstance(mod, nn.Linear):
            if id(mod.weight) not in seen_ids:   # skip weight-tied head
                mod.weight.data.copy_(_quantize_tensor(mod.weight.data, bits))
                seen_ids.add(id(mod.weight))
            if mod.bias is not None:
                mod.bias.data.copy_(_quantize_tensor(mod.bias.data, bits))
        elif isinstance(mod, nn.Embedding):
            if id(mod.weight) not in seen_ids:
                mod.weight.data.copy_(_quantize_tensor(mod.weight.data, bits))
                seen_ids.add(id(mod.weight))
        elif isinstance(mod, nn.MultiheadAttention):
            if mod.in_proj_weight is not None:
                mod.in_proj_weight.data.copy_(
                    _quantize_tensor(mod.in_proj_weight.data, bits))


@torch.no_grad()
def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    m = copy.deepcopy(model)
    _apply_quantization(m, bits)
    set_dense_masks(m)
    return m


# ── SparseGPT ──────────────────────────────────────────────────────────────────

_SKIP_NAMES = {"head"}   # weight-tied to embed


@torch.no_grad()
def collect_inputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_calib: int,
) -> dict[str, torch.Tensor]:
    """Collect per-layer calibration inputs via named forward hooks."""
    storage: dict[str, list[torch.Tensor]] = {}

    def make_hook(name: str):
        def hook(mod, inp, out):
            # Move to CPU immediately: keeping every layer's activations on the
            # accelerator across all calib batches saturates GPU memory (OOM
            # mid-forward on larger models). Downstream uses them on CPU.
            x = inp[0].detach().float().cpu()
            storage.setdefault(name, []).append(x.reshape(-1, x.shape[-1]))
        return hook

    handles = []
    for name, mod in model.named_modules():
        if name in _SKIP_NAMES:
            continue
        if isinstance(mod, (nn.Linear, PolyStochasticRowLinear, StochasticRowLinear)):
            handles.append(mod.register_forward_hook(make_hook(name)))
        elif isinstance(mod, nn.MultiheadAttention):
            # Hook the out_proj; in_proj_weight gets same input as qkv
            handles.append(mod.register_forward_hook(make_hook(name + "._mha_input")))

    set_dense_masks(model)
    model.eval()
    with torch.no_grad():
        for i, (inp, _) in enumerate(loader):
            if i >= n_calib:
                break
            model(inp.to(device), None)

    for h in handles:
        h.remove()

    # For MHA, the stored input is the full (B,T,d) pre-attention tensor;
    # use it as calibration for in_proj_weight
    mha_inputs = {}
    for k in list(storage.keys()):
        if k.endswith("._mha_input"):
            base = k[: -len("._mha_input")]
            mha_inputs[base] = torch.cat(storage.pop(k), dim=0)

    return {name: torch.cat(xs, dim=0) for name, xs in storage.items()} | mha_inputs


@torch.no_grad()
def _wanda_weight(W: torch.Tensor, X: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Wanda (Sun et al. 2023): importance[i,j] = |W[i,j]| × ‖X[:,j]‖₂
    Prune the n_prune lowest-importance weights per output row.
    No Hessian computation, no weight correction.
    """
    out_f, in_f = W.shape
    n_prune = max(1, int(sparsity * in_f))

    col_norms  = X.float().norm(dim=0)                        # (in_f,)
    importance = W.float().abs() * col_norms.unsqueeze(0)     # (out_f, in_f)

    prune_cols = importance.argsort(dim=1)[:, :n_prune]       # (out_f, n_prune)
    W_out = W.float().clone()
    W_out.scatter_(1, prune_cols, 0.0)
    return W_out.to(W.dtype)


@torch.no_grad()
def wanda_model(model: LM, calib: dict[str, torch.Tensor],
                sparsity: float) -> LM:
    m = copy.deepcopy(model)
    for name, mod in m.named_modules():
        if name not in calib:
            continue
        X = calib[name]
        if isinstance(mod, (PolyStochasticRowLinear, StochasticRowLinear, nn.Linear)):
            mod.weight.data.copy_(
                _wanda_weight(mod.weight.data, X, sparsity))
        elif isinstance(mod, nn.MultiheadAttention) and mod.in_proj_weight is not None:
            mod.in_proj_weight.data.copy_(
                _wanda_weight(mod.in_proj_weight.data, X, sparsity))
    set_dense_masks(m)
    return m


@torch.no_grad()
def _sparsegpt_hinv(X: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Inverse Hessian for SparseGPT — depends only on calibration X, not on
    the sparsity level, so it can be computed once per layer and reused across
    every sparsity in a sweep."""
    H = (X.T @ X).float() / max(len(X), 1)
    dead = torch.diag(H) == 0
    H[dead, :] = 0; H[:, dead] = 0; H[dead, dead] = 1.0
    H.diagonal().add_(damp * H.diagonal()[~dead].mean().clamp(min=1e-6))

    try:
        return torch.linalg.inv(H)
    except torch.linalg.LinAlgError:
        return torch.linalg.pinv(H)


@torch.no_grad()
def _sparsegpt_weight_from_hinv(W: torch.Tensor, H_inv: torch.Tensor,
                                sparsity: float,
                                n_workers: int | None = None) -> torch.Tensor:
    """SparseGPT per-row pruning given a precomputed inverse Hessian."""
    from concurrent.futures import ThreadPoolExecutor

    out_f, in_f = W.shape
    n_prune = max(1, int(sparsity * in_f))

    diag_inv  = H_inv.diagonal().clamp(min=1e-8)
    H_inv_diag = H_inv.diagonal()   # alias for scalar lookups in inner loop
    W_out = W.float().clone()

    # Rows are independent — parallelize with threads.
    # The vector multiply-adds (scalar * H_inv[q], W_out[i] -= ...) release the
    # GIL so threads genuinely run concurrently on CPU.  Each thread writes to a
    # unique row index so there are no data races.
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    def _prune_row(i: int) -> None:
        w = W_out[i].clone()
        prune_idx = (w ** 2 / diag_inv).argsort()[:n_prune]
        for q in prune_idx.tolist():
            if H_inv_diag[q].abs() < 1e-10:
                continue
            w -= (w[q] / H_inv_diag[q]) * H_inv[q]
        w[prune_idx] = 0.0
        W_out[i] = w

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        list(exe.map(_prune_row, range(out_f)))

    return W_out.to(W.dtype)


@torch.no_grad()
def _sparsegpt_weight(W: torch.Tensor, X: torch.Tensor,
                      sparsity: float, damp: float = 0.01,
                      n_workers: int | None = None) -> torch.Tensor:
    """Convenience wrapper: build the inverse Hessian from X then prune.

    Prefer ``_sparsegpt_hinv`` + ``_sparsegpt_weight_from_hinv`` when pruning
    the same layer at multiple sparsity levels, to avoid rebuilding H each time.
    """
    H_inv = _sparsegpt_hinv(X, damp)
    return _sparsegpt_weight_from_hinv(W, H_inv, sparsity, n_workers)


@torch.no_grad()
def sparsegpt_model(model: LM, calib: dict[str, torch.Tensor],
                    sparsity: float, n_row_workers: int | None = None) -> LM:
    # _sparsegpt_weight contains an O(out_features) Python loop with per-row
    # .tolist() device→host syncs.  Running it on XPU/CUDA stalls the device
    # on every iteration and is orders of magnitude slower than CPU.  Force CPU
    # here regardless of where the model/calib live; the result is copied back.
    m = copy.deepcopy(model)
    for name, mod in m.named_modules():
        if name not in calib and (name + "._mha_input") not in calib:
            if name not in calib:
                continue
        X = calib.get(name, calib.get(name + "._mha_input")).cpu()

        if isinstance(mod, (PolyStochasticRowLinear, StochasticRowLinear, nn.Linear)):
            dev = mod.weight.device
            mod.weight.data.copy_(
                _sparsegpt_weight(mod.weight.data.cpu(), X, sparsity,
                                  n_workers=n_row_workers).to(dev))
        elif isinstance(mod, nn.MultiheadAttention) and name in calib:
            if mod.in_proj_weight is not None:
                dev = mod.in_proj_weight.device
                mod.in_proj_weight.data.copy_(
                    _sparsegpt_weight(mod.in_proj_weight.data.cpu(), X, sparsity,
                                      n_workers=n_row_workers).to(dev))
    set_dense_masks(m)
    return m


# ── structured joint pruning (post-hoc, magnitude-based) ──────────────────────

@torch.no_grad()
def struct_joint_prune(model: LM, ffn_keep: float, head_keep: float) -> LM:
    """
    Post-hoc magnitude pruning of FFN neurons and attention heads.
    Works on the dense LM (uses nn.MultiheadAttention + plain nn.Linear FFN).
    """
    m = copy.deepcopy(model)

    for block in m.blocks:
        d_model = block.ln1.normalized_shape[0]
        ffn_exp = block.ffn.expand
        ffn_con = block.ffn.contract

        # ── FFN neuron pruning ────────────────────────────────────────────────
        if isinstance(ffn_exp, nn.Linear):
            d_ff      = ffn_exp.weight.shape[0]
            n_keep    = max(1, int(ffn_keep * d_ff))
            norms     = ffn_exp.weight.data.norm(dim=1)   # (d_ff,)
            keep_idx  = norms.topk(n_keep).indices
            mask      = torch.zeros(d_ff, device=norms.device)
            mask[keep_idx] = 1.0
            ffn_exp.weight.data *= mask.unsqueeze(1)
            if ffn_exp.bias is not None:
                ffn_exp.bias.data *= mask
            ffn_con.weight.data *= mask.unsqueeze(0)

        # ── Attention head pruning (nn.MultiheadAttention) ───────────────────
        if isinstance(block.attn, nn.MultiheadAttention):
            attn   = block.attn
            n_head = attn.num_heads
            d_head = d_model // n_head
            n_keep_h = max(1, int(head_keep * n_head))

            # Head importance = Frobenius norm of out_proj columns per head
            W_out = attn.out_proj.weight.data   # (d_model, d_model)
            head_norms = torch.stack([
                W_out[:, h * d_head:(h + 1) * d_head].norm()
                for h in range(n_head)
            ])
            keep_heads  = head_norms.topk(n_keep_h).indices.tolist()
            prune_heads = [h for h in range(n_head) if h not in keep_heads]

            W_in = attn.in_proj_weight.data   # (3*d_model, d_model)
            for h in prune_heads:
                for offset in [0, d_model, 2 * d_model]:   # Q, K, V
                    W_in[offset + h * d_head: offset + (h + 1) * d_head, :] = 0.0
                W_out[:, h * d_head:(h + 1) * d_head] = 0.0

    set_dense_masks(m)
    return m


# ── evaluation ────────────────────────────────────────────────────────────────


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default="stochastic_results_ckpt.pt")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--n_calib",      type=int, default=64)
    p.add_argument("--max_calib_rows", type=int, default=32768,
                   help="Max calibration rows retained per layer (host-RAM cap "
                        "for Wanda/SparseGPT stats). <=0 disables subsampling.")
    p.add_argument("--sparsities",   type=str, default=None,
                   help="Comma-separated SparseGPT/Wanda sparsity levels, e.g. "
                        "'0.1,0.3,0.5,0.7,0.8,0.9,0.99'. Default: 20-level grid.")
    p.add_argument("--max_batches",  type=int, default=100)
    p.add_argument("--rosa_ckpt",    default=None,
                   help="path to rosa_comparison_results.pt; "
                        "auto-detected if omitted and file exists")
    p.add_argument("--out",          default="compression_comparison.json",
                   help="path to write JSON results")
    p.add_argument("--temperature",  type=float, default=0.2,
                   help="temperature to inject into old checkpoints missing _temperature_buf")
    p.add_argument("--penalty_mode", default="flops",
                   choices=["flops", "params", "uniform"],
                   help="penalty aggregation mode (not saved in old checkpoints)")
    p.add_argument("--data_dir",     default=None,
                   help="OLMo3 data directory with validation.bin; "
                        "if set, uses OLMo3 eval instead of WikiText")
    p.add_argument("--max_train_tokens", type=int, default=200_000_000,
                   help="token cap passed to _build_memmap_datasets (default: 200M)")
    p.add_argument("--prune_workers", type=int, default=None,
                   help="parallel worker processes for SparseGPT/Wanda sparsity sweep "
                        "(default: min(n_sparsity_levels, cpu_count))")
    p.add_argument("--model", default="hyper_joint",
                   help="model tag from checkpoint to use as compression baseline "
                        "(default: hyper_joint; use rank1_hyper_joint for rank-1 checkpoints)")
    p.add_argument("--skip_compile", action="store_true",
                   help="skip the torch.compile speedup sweep (auto-skipped on CPU)")
    p.add_argument("--compile_stride", type=int, default=5,
                   help="evaluate every Nth lambda in the compile sweep "
                        "(default: 5 ≈ 11 points; set 1 for all 51)")
    p.add_argument("--n_resample", type=int, default=10,
                   help="number of stochastic mask samples per lambda value "
                        "in the tau sweep (default: 10)")
    p.add_argument("--calib_split", default="train",
                   choices=["train", "validation"],
                   help="dataset split used to collect SparseGPT/Wanda calibration "
                        "activations (default: train, to avoid contaminating val PPL). "
                        "For OLMo3 checkpoints, train uses train_sample.bin.")
    return p.parse_args()


# ── FLOP fraction for structural pruning ──────────────────────────────────────

def flops_frac_poly(model: PolyLM, ffn_frac: float, attn_frac: float) -> float:
    fw = model._ffn_flops_w
    aw = model._attn_flops_w
    total = fw + aw
    if total == 0:
        return 1.0
    return (ffn_frac * fw + attn_frac * aw) / total


# ── parallel pruning helpers (module-level so they're picklable) ──────────────
#
# _PRUNE_STATE is populated in the parent process before pool creation.
# On Linux (fork), workers inherit it via copy-on-write — zero serialization.
# On other platforms (spawn), _init_prune_worker receives it as initargs.

_PRUNE_STATE: dict = {}


def _init_prune_worker(state_dict: dict, calib: dict,
                       cfg: dict, n_row_workers: int,
                       model_tag: str = "hyper_joint",
                       n_resample: int = 10) -> None:
    _PRUNE_STATE["state_dict"]    = state_dict
    _PRUNE_STATE["calib"]         = calib
    _PRUNE_STATE["cfg"]           = cfg
    _PRUNE_STATE["n_row_workers"] = n_row_workers
    _PRUNE_STATE["model_tag"]     = model_tag
    _PRUNE_STATE["n_resample"]    = n_resample


def _make_model(cfg: dict, model_tag: str) -> ScaledPolyLM:
    """Instantiate the right model class for a given checkpoint tag."""
    kw = dict(variant="joint",
              penalty_mode=cfg.get("penalty_mode", "flops"),
              n_knots=cfg.get("n_knots", 8),
              degree=cfg.get("degree", 3))
    if model_tag.startswith("rank1_") and _HAS_RANK1:
        return Rank1ScaledPolyLM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                                  cfg["n_layers"], cfg["seq_len"], **kw)
    return ScaledPolyLM(cfg["vocab"], cfg["d_model"], cfg["n_heads"],
                        cfg["n_layers"], cfg["seq_len"], **kw)


def _sparsegpt_sp_worker(sp: float) -> tuple[float, float, dict]:
    cfg = _PRUNE_STATE["cfg"]
    m   = _make_model(cfg, _PRUNE_STATE.get("model_tag", "hyper_joint"))
    m.load_state_dict(_PRUNE_STATE["state_dict"])
    set_dense_masks(m)
    pruned = sparsegpt_model(m, _PRUNE_STATE["calib"], sp,
                             n_row_workers=_PRUNE_STATE["n_row_workers"])
    return sp, sparse_mb(pruned), {k: v.cpu() for k, v in pruned.state_dict().items()}


def _wanda_sp_worker(sp: float) -> tuple[float, float, dict]:
    cfg = _PRUNE_STATE["cfg"]
    m   = _make_model(cfg, _PRUNE_STATE.get("model_tag", "hyper_joint"))
    m.load_state_dict(_PRUNE_STATE["state_dict"])
    set_dense_masks(m)
    pruned = wanda_model(m, _PRUNE_STATE["calib"], sp)
    return sp, sparse_mb(pruned), {k: v.cpu() for k, v in pruned.state_dict().items()}


def _tau_lam_worker(lv: float) -> tuple[float, float, float, float, dict]:
    """
    Resample the tau-threshold model N times at lambda=lv, record per-resample
    active stats, and return the state dict whose active fraction is closest to
    the mean — so the single GPU eval in the main process is representative.
    No evaluation happens here; all eval is done on-device by the caller.
    """
    cfg        = _PRUNE_STATE["cfg"]
    n_resample = _PRUNE_STATE.get("n_resample", 10)

    m = _make_model(cfg, _PRUNE_STATE.get("model_tag", "hyper_joint"))
    m.load_state_dict(_PRUNE_STATE["state_dict"])
    m.eval()

    lam_t = torch.tensor(lv)
    sz    = expected_mb_poly(m, lv)   # deterministic from poly weights

    ffn_fracs, attn_fracs, states = [], [], []
    for _ in range(n_resample):
        m.resample(lam_t)
        s = m.active_stats()
        ffn_fracs.append(s["ffn_frac"])
        attn_fracs.append(s["attn_frac"])
        states.append({k: v.cpu().clone() for k, v in m.state_dict().items()})

    mean_ff = float(np.mean(ffn_fracs))
    # pick the resample whose ffn_frac is closest to the mean
    best_i  = int(np.argmin(np.abs(np.array(ffn_fracs) - mean_ff)))
    return (lv,
            float(np.mean(ffn_fracs)), float(np.mean(attn_fracs)),
            sz, states[best_i])


# ── compiled speedup helpers ──────────────────────────────────────────────────

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


@torch.no_grad()
def _time_model(model: nn.Module, batch: torch.Tensor,
                device: torch.device,
                n_warmup: int = 8, n_time: int = 20) -> float:
    """Median wall-clock ms per forward pass after n_warmup warmup calls."""
    model.eval()
    # First call may trigger torch.compile JIT; absorb it in warmup.
    for _ in range(n_warmup):
        model(batch, None)
    _sync(device)
    durations = []
    for _ in range(n_time):
        _sync(device)
        t0 = time.perf_counter()
        model(batch, None)
        _sync(device)
        durations.append(time.perf_counter() - t0)
    return float(np.median(durations)) * 1e3   # ms


def _compress(model: nn.Module, lam_t: torch.Tensor, model_tag: str) -> nn.Module:
    """Dispatch to build_compressed_rank1 or build_compressed based on model_tag."""
    if model_tag.startswith("rank1_") and _HAS_RANK1:
        return build_compressed_rank1(model, lam_t)
    return build_compressed(model, lam_t)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import json

    args   = parse_args()

    # Dispatch to OLMo3 backend when the checkpoint is an olmo3_mini_train directory.
    if _HAS_OLMO3:
        ckpt_dir = _olmo3_ckpt_dir(args.ckpt)
        if ckpt_dir is not None:
            main_olmo3(args, ckpt_dir)
            return

    device = torch.device(args.device)

    model_tag = args.model

    print("=" * 72, flush=True)
    print(f"  Compression Method Comparison  (SparseGPT / Wanda / {model_tag})", flush=True)
    print("=" * 72, flush=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"  Loading {args.ckpt} …", flush=True)
    models, cfg = load_models(args.ckpt, device,
                               temperature=args.temperature,
                               penalty_mode=args.penalty_mode)
    if model_tag not in models:
        available = list(models.keys())
        raise KeyError(f"--model {model_tag!r} not in checkpoint. "
                       f"Available: {available}")
    print(f"  Config: vocab={cfg['vocab']}  d={cfg['d_model']}  "
          f"L={cfg['n_layers']}  H={cfg['n_heads']}  seq={cfg['seq_len']}", flush=True)
    d_mb = dense_mb(models[model_tag])
    print(f"  {model_tag} size: {d_mb:.3f} MB  "
          f"({sum(p.numel() for p in models[model_tag].parameters())/1e6:.2f}M params)",
          flush=True)

    # ── Validation data ───────────────────────────────────────────────────────
    if args.data_dir:
        if not _HAS_MEMMAP:
            raise RuntimeError("olmo3_mini_train.py not found on sys.path; "
                               "cannot use --data_dir without it.")
        data_dir = Path(args.data_dir)
        print(f"  Loading OLMo3 validation data from {data_dir} …")
        val_ds = _TokenDataset(data_dir / "validation.bin", cfg["seq_len"])
        if cfg["vocab"] < OLMO3_VOCAB:
            remap  = _build_vocab_remap(data_dir / "train_sample.bin", cfg["vocab"])
            val_ds = _RemappedTokenDataset(val_ds, remap)

        # Calibration: prefer train_sample.bin (in-distribution); fall back to val.
        train_bin = data_dir / "train_sample.bin"
        if args.calib_split == "train" and train_bin.exists():
            print(f"  Loading OLMo3 calibration data from {train_bin} …")
            calib_ds: object = _TokenDataset(train_bin, cfg["seq_len"])
            if cfg["vocab"] < OLMO3_VOCAB:
                calib_ds = _RemappedTokenDataset(calib_ds, remap)
        else:
            if args.calib_split == "train":
                print("  Warning: train_sample.bin not found; "
                      "calibrating on validation data instead.", flush=True)
            calib_ds = val_ds
    else:
        if cfg["vocab"] >= OLMO3_VOCAB:
            raise RuntimeError(
                f"Checkpoint has vocab={cfg['vocab']} (OLMo3 BPE tokenizer). "
                "Pass --data_dir <path/to/olmo3_data> to evaluate on the correct data. "
                "WikiText word-level tokens are incompatible with this model."
            )
        print("  Building vocabulary …")
        vocab_map = build_vocab(cfg["vocab"])
        print("  Loading validation set …")
        val_ds = build_tensor("validation", vocab_map, cfg["seq_len"])
        print(f"  Loading calibration set ({args.calib_split}) …")
        calib_ds = build_tensor(args.calib_split, vocab_map, cfg["seq_len"])

    calib_loader = DataLoader(calib_ds, batch_size=4, shuffle=False, num_workers=0)

    # MPS has a deepcopy bug with GPU tensors; everything else (XPU, CUDA) can
    # prune in-place on device.
    prune_on_device = device.type not in ("cpu", "mps")

    print(f"\n  Collecting calibration activations ({args.n_calib} batches) …")
    set_dense_masks(models[model_tag])
    models[model_tag].to(device)
    calib = collect_inputs(models[model_tag], calib_loader, device, args.n_calib)
    if not prune_on_device:
        models[model_tag].cpu()
        calib = {k: v.cpu() for k, v in calib.items()}
    print(f"  Captured {len(calib)} weight matrices.")

    # ── Run variants ──────────────────────────────────────────────────────────
    results: list[dict] = []

    def record(label: str, method: str, size_mb: float, flops_frac: float,
               ppl: float, sparsity: float | None = None,
               lam: float | None = None) -> None:
        results.append(dict(label=label, method=method, size_mb=size_mb,
                            flops_frac=flops_frac, ppl=ppl,
                            sparsity=sparsity, lam=lam))
        print(f"    {label:<40}  {size_mb:>6.3f} MB  "
              f"FLOPs={flops_frac*100:>5.1f}%  PPL={ppl:.2f}")

    # λ=0 baseline
    print(f"\n  ── {model_tag} λ=0 baseline ──")
    set_dense_masks(models[model_tag])
    models[model_tag].to(device)
    base_ppl = evaluate(models[model_tag], val_ds, None, device=device)
    if not prune_on_device:
        models[model_tag].cpu()
    record(f"{model_tag} λ=0", model_tag, d_mb, 1.0, base_ppl)

    # ── parallel pruning setup ────────────────────────────────────────────────
    sparsities = resolve_sparsities(args)
    n_prune_workers = args.prune_workers or min(len(sparsities), os.cpu_count() or 1)
    n_row_workers = max(1, (os.cpu_count() or 1) // n_prune_workers)

    base_state_cpu = {k: v.cpu() for k, v in models[model_tag].state_dict().items()}
    calib_cpu      = {k: v.cpu() for k, v in calib.items()}

    if sys.platform.startswith("linux"):
        _PRUNE_STATE.update(state_dict=base_state_cpu, calib=calib_cpu,
                            cfg=cfg, n_row_workers=n_row_workers,
                            model_tag=model_tag)
        pool_kw: dict = dict(max_workers=n_prune_workers,
                             mp_context=mp.get_context("fork"))
    else:
        pool_kw = dict(max_workers=n_prune_workers,
                       mp_context=mp.get_context("spawn"),
                       initializer=_init_prune_worker,
                       initargs=(base_state_cpu, calib_cpu, cfg, n_row_workers,
                                 model_tag))

    def _eval_pruned(tag: str, method: str, pruned_results: list) -> None:
        for sp, mb, state in sorted(pruned_results, key=lambda x: x[0]):
            m = _make_model(cfg, model_tag)
            m.load_state_dict(state)
            set_dense_masks(m)
            ppl = evaluate(m.to(device), val_ds, None, device=device)
            record(f"{tag} sp={sp:.2f}", method, mb, 1.0 - sp, ppl, sparsity=sp)
            del m

    # SparseGPT — all sparsity levels pruned in parallel, eval sequentially on device
    print(f"\n  ── SparseGPT ({len(sparsities)} levels, {n_prune_workers} workers) ──",
          flush=True)
    with ProcessPoolExecutor(**pool_kw) as exe:
        sgpt_results = list(exe.map(_sparsegpt_sp_worker, sparsities))
    _eval_pruned("sparsegpt", "sparsegpt", sgpt_results)

    # Wanda — same pattern
    print(f"\n  ── Wanda ({len(sparsities)} levels, {n_prune_workers} workers) ──",
          flush=True)
    with ProcessPoolExecutor(**pool_kw) as exe:
        wanda_results = list(exe.map(_wanda_sp_worker, sparsities))
    _eval_pruned("wanda", "wanda", wanda_results)

    # τ-threshold sweep — parallelized across lambda values (workers run on CPU)
    N_RESAMPLE_SWEEP = args.n_resample
    print(f"\n  ── {model_tag} ({len(TAU_LAMS)} λ values, "
          f"{N_RESAMPLE_SWEEP} resamples each, {n_prune_workers} workers) ──",
          flush=True)

    if sys.platform.startswith("linux"):
        _PRUNE_STATE["n_resample"] = N_RESAMPLE_SWEEP
        tau_pool_kw: dict = dict(max_workers=n_prune_workers,
                                 mp_context=mp.get_context("fork"))
    else:
        tau_pool_kw = dict(
            max_workers=n_prune_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_prune_worker,
            initargs=(base_state_cpu, calib_cpu, cfg, n_row_workers,
                      model_tag, N_RESAMPLE_SWEEP))

    with ProcessPoolExecutor(**tau_pool_kw) as exe:
        tau_raw = list(exe.map(_tau_lam_worker, TAU_LAMS))

    # Eval on device sequentially — mirrors the _eval_pruned pattern for SparseGPT/Wanda.
    print(f"\n  ── {model_tag} τ eval on device ──", flush=True)
    for lv, ff, af, sz, state in sorted(tau_raw, key=lambda x: x[0]):
        m = _make_model(cfg, model_tag)
        m.load_state_dict(state)
        ppl     = evaluate(m.to(device), val_ds, torch.tensor(lv, device=device),
                           device=device)
        ff_frac = flops_frac_poly(m, ff, af)
        record(f"tau-{model_tag} λ={lv:.1f}", f"tau-{model_tag}",
               sz, ff_frac, ppl, lam=lv)
        del m

    # ── Compiled speedup sweep (GPU/XPU only) ─────────────────────────────────
    do_compile = (device.type in ("cuda", "xpu")) and not args.skip_compile
    if do_compile:
        compile_lams = TAU_LAMS[:: args.compile_stride]
        print(f"\n  ── {model_tag} compiled speedup sweep "
              f"({len(compile_lams)} λ values, stride={args.compile_stride}) ──",
              flush=True)

        timing_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
        timing_batch  = next(iter(timing_loader))[0].to(device)

        # Compile and time the dense baseline for a fair comparison.
        set_dense_masks(models[model_tag])
        m_base = models[model_tag].to(device).eval()
        try:
            m_base_c = torch.compile(m_base)
        except Exception:
            m_base_c = m_base
        base_ms = _time_model(m_base_c, timing_batch, device)
        print(f"    {'dense baseline (compiled)':<44}  {base_ms:>7.2f} ms/batch", flush=True)

        # Build a lam→result-dict lookup keyed by rounded value to avoid fp drift.
        tau_result_map = {round(r["lam"], 6): r
                         for r in results if r["method"] == f"tau-{model_tag}"}

        for lv in compile_lams:
            lam_t = torch.tensor(lv, device=device)
            models[model_tag].resample(lam_t)
            compressed = _compress(models[model_tag], lam_t, model_tag).to(device).eval()
            actual_mb  = dense_mb(compressed)
            try:
                compiled  = torch.compile(compressed)
                comp_ms   = _time_model(compiled, timing_batch, device)
                speedup   = base_ms / comp_ms
                del compiled
            except Exception as exc:
                print(f"    torch.compile failed at λ={lv:.1f}: {exc}", flush=True)
                comp_ms, speedup = float("nan"), float("nan")
            del compressed

            key = round(lv, 6)
            if key in tau_result_map:
                tau_result_map[key]["actual_mb"] = actual_mb
                tau_result_map[key]["speedup"]   = speedup
                tau_result_map[key]["ms_batch"]  = comp_ms
            print(f"    tau-{model_tag} λ={lv:<4.1f}  "
                  f"actual={actual_mb:>5.1f} MB  "
                  f"speedup={speedup:>5.2f}x  ({comp_ms:.2f} ms/batch)", flush=True)

        models[model_tag].cpu()

    # ── Summary ───────────────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: r["size_mb"])
    has_speedup    = any("speedup" in r for r in results_sorted)
    print()
    print("=" * 72)
    hdr = f"  {'method':<40}  {'MB':>6}  {'FLOPs%':>7}  {'PPL':>9}"
    if has_speedup:
        hdr += f"  {'actMB':>6}  {'speedup':>8}"
    print(hdr)
    print("  " + "-" * (68 + (18 if has_speedup else 0)))
    for r in results_sorted:
        dppl = r["ppl"] - base_ppl
        row = (f"  {r['label']:<40}  {r['size_mb']:>6.3f}  "
               f"{r['flops_frac']*100:>6.1f}%  {r['ppl']:>9.2f}  ({dppl:+.2f})")
        if has_speedup:
            amb = r.get("actual_mb")
            spd = r.get("speedup")
            row += (f"  {amb:>6.1f}" if amb is not None else "      —  ")
            row += (f"  {spd:>6.2f}x" if spd is not None and not np.isnan(spd) else "       —")
        print(row)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    if out_path.is_dir():
        out_path = out_path / "compression_comparison.json"
    out_data = {
        "dense_mb":  d_mb,
        "base_ppl": base_ppl,
        "results":   results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results saved to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# OLMo3 (olmo3_mini_train.py) backend
# ══════════════════════════════════════════════════════════════════════════════

try:
    from olmo3_mini_train import (
        BSplineScaledOlmo3MLP, BSplineScaledOlmo3Attn,
        TokenDataset as _Olmo3TokenDataset,
        inject_stochastic_mlp, inject_stochastic_attn, inject_per_head_qk_norm,
        set_lam as _olmo3_set_lam,
        set_sample_mask as _olmo3_set_sample_mask,
        set_hard_mask as _olmo3_set_hard_mask,
        structural_prune as _olmo3_structural_prune,
        bake_pruning as _olmo3_bake_pruning,
    )
    from transformers import Olmo3Config
    from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM
    _HAS_OLMO3 = True
except ImportError:
    _HAS_OLMO3 = False


# ── OLMo3 parallel pruning helpers (module-level so they're picklable) ────────

_OLMO3_PRUNE_STATE: dict = {}


def _init_olmo3_prune_worker(state_dict: dict, calib: dict,
                              config_dict: dict, n_row_workers: int,
                              hinv: dict | None = None) -> None:
    _OLMO3_PRUNE_STATE["state_dict"]    = state_dict
    _OLMO3_PRUNE_STATE["calib"]         = calib
    _OLMO3_PRUNE_STATE["config"]        = config_dict
    _OLMO3_PRUNE_STATE["n_row_workers"] = n_row_workers
    _OLMO3_PRUNE_STATE["hinv"]          = hinv or {}


def _rebuild_olmo3_baked(state_dict: dict, config_dict: dict):
    """Reconstruct a plain (baked) Olmo3ForCausalLM on CPU from serialisable parts.

    bake_pruning leaves PerHeadRMSNorm in place (it only replaces MLP/o_proj),
    so q_norm/k_norm weights have shape [head_dim] not [n_heads*head_dim].
    Auto-detect this from the state dict and inject the matching norm before
    loading, otherwise load_state_dict raises a shape mismatch.
    """
    from transformers import Olmo3Config
    from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM
    from olmo3_mini_train import inject_per_head_qk_norm

    config = Olmo3Config(**config_dict)
    m = Olmo3ForCausalLM(config)

    head_dim   = config.hidden_size // config.num_attention_heads
    probe_key  = "model.layers.0.self_attn.q_norm.weight"
    if probe_key in state_dict and state_dict[probe_key].shape[0] == head_dim:
        inject_per_head_qk_norm(m)

    m.load_state_dict(state_dict)
    return m


def _olmo3_sparsegpt_worker(sp: float):
    m     = _rebuild_olmo3_baked(_OLMO3_PRUNE_STATE["state_dict"],
                                  _OLMO3_PRUNE_STATE["config"])
    calib = _OLMO3_PRUNE_STATE["calib"]
    hinv  = _OLMO3_PRUNE_STATE.get("hinv") or {}
    nrw   = _OLMO3_PRUNE_STATE.get("n_row_workers")
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and name in calib:
            # Reuse the per-layer inverse Hessian precomputed once in the parent
            # (shared copy-on-write across fork workers) instead of rebuilding it
            # at every sparsity level.  Fall back to building it if absent.
            if name in hinv:
                mod.weight.data.copy_(
                    _sparsegpt_weight_from_hinv(mod.weight.data, hinv[name], sp,
                                                n_workers=nrw))
            else:
                mod.weight.data.copy_(
                    _sparsegpt_weight(mod.weight.data, calib[name], sp,
                                      n_workers=nrw))
    return sp, sparse_mb(m), {k: v for k, v in m.state_dict().items()}


def _olmo3_wanda_worker(sp: float):
    m     = _rebuild_olmo3_baked(_OLMO3_PRUNE_STATE["state_dict"],
                                  _OLMO3_PRUNE_STATE["config"])
    calib = _OLMO3_PRUNE_STATE["calib"]
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and name in calib:
            mod.weight.data.copy_(_wanda_weight(mod.weight.data, calib[name], sp))
    return sp, sparse_mb(m), {k: v for k, v in m.state_dict().items()}


def _olmo3_ckpt_dir(ckpt_path: str) -> "Path | None":
    """Return the checkpoint directory if this looks like an olmo3_mini_train checkpoint."""
    p = Path(ckpt_path)
    if p.is_dir() and (p / "model.pt").exists() and (p / "config.json").exists():
        return p
    # also accept a direct path to model.pt whose parent has config.json + train_args.json
    if (p.suffix == ".pt"
            and (p.parent / "config.json").exists()
            and (p.parent / "train_args.json").exists()):
        return p.parent
    return None


@torch.no_grad()
def _eval_olmo3(model, val_ds, device: torch.device,
                lam_val: float = 0.0, max_batches: int = 100,
                batch_size: int = 4) -> float:
    import math
    import torch.nn.functional as F
    lam_t = torch.tensor(float(lam_val), device=device)
    _olmo3_set_lam(model, lam_t)
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
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
def _collect_inputs_olmo3(model, val_ds, device: torch.device,
                          n_calib: int = 64, batch_size: int = 4,
                          max_rows: int = 32768) -> dict:
    """Hook-based calibration input collection from a plain Olmo3ForCausalLM.

    Wanda needs only per-column activation norms and SparseGPT only the Gram
    matrix Xᵀ X — both are row reductions, so storing *every* token's
    activation for *every* layer is wasteful and OOMs host RAM on larger
    models. We randomly subsample each batch so at most ``max_rows`` rows are
    retained per layer, which leaves the statistics essentially unbiased while
    bounding memory. ``max_rows <= 0`` disables the cap.
    """
    storage: dict = {}
    handles = []
    seen_weights: set = set()

    # Rows kept per batch so the per-layer total stays under max_rows.
    rows_per_batch = (max(1, max_rows // max(1, n_calib))
                      if max_rows and max_rows > 0 else None)

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if id(mod.weight) in seen_weights:      # skip weight-tied lm_head
            continue
        if "embed_tokens" in name or "lm_head" in name:
            continue
        seen_weights.add(id(mod.weight))

        def make_hook(n):
            def hook(m, inp, out):
                # Move to CPU immediately: keeping every layer's activations on
                # the accelerator across all calib batches saturates GPU memory
                # (OOM mid-forward on larger models). Downstream uses them on CPU.
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                if rows_per_batch is not None and x.shape[0] > rows_per_batch:
                    idx = torch.randperm(x.shape[0], device=x.device)[:rows_per_batch]
                    x = x[idx]
                storage.setdefault(n, []).append(x.float().cpu())
            return hook
        handles.append(mod.register_forward_hook(make_hook(name)))

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    for i, (inp, _) in enumerate(loader):
        if i >= n_calib:
            break
        model(input_ids=inp.to(device))

    for h in handles:
        h.remove()

    return {k: torch.cat(v, 0) for k, v in storage.items()}


@torch.no_grad()
def _prune_linear_olmo3(model, calib: dict, sparsity: float, method: str):
    """Deep-copy model to CPU and apply Wanda or SparseGPT.

    Always operates on CPU: bake_pruning may leave newly-created Olmo3MLP
    modules on CPU even when the parent model is on an accelerator, and
    _sparsegpt_weight's per-row Python loop is orders of magnitude faster on
    CPU than on XPU/CUDA anyway.  Caller moves the result to device for eval.
    """
    m = copy.deepcopy(model).cpu()
    for name, mod in m.named_modules():
        if not isinstance(mod, nn.Linear) or name not in calib:
            continue
        X = calib[name].cpu()
        if method == "wanda":
            mod.weight.data.copy_(_wanda_weight(mod.weight.data, X, sparsity))
        else:  # sparsegpt
            mod.weight.data.copy_(_sparsegpt_weight(mod.weight.data, X, sparsity))
    return m


def _olmo3_copy_clear_cache(model, device=None):
    """Deep-copy model (optionally to a new device) with B-spline cache cleared.

    After deepcopy the cached spline tensors (_sp_tau, _sp_gate, _sp_scale,
    _sp_input) still reference the original device and the last lambda value.
    structural_prune short-circuits to those cached tensors in inclusion_probs,
    causing either a device mismatch (cpu copy) or wrong-lambda pruning (same
    device copy). Clearing the cache forces a fresh recompute at the correct
    lambda on the correct device.
    """
    m = copy.deepcopy(model)
    if device is not None:
        m = m.to(device)
    for mod in m.modules():
        for attr in ("_sp_tau", "_sp_gate", "_sp_scale", "_sp_input"):
            if hasattr(mod, attr):
                setattr(mod, attr, None)
    return m


def _structural_mb_olmo3(model, lam_val: float) -> float:
    """Structural (row-removal) compressed size in MB at the given lambda."""
    m = _olmo3_copy_clear_cache(model, device=torch.device("cpu"))
    _olmo3_structural_prune(m, lam_val, prune_attn=True)
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 ** 2


@torch.no_grad()
def _flops_frac_olmo3(model, lam_val: float, device: torch.device) -> float:
    """FLOP-weighted active fraction at the given lambda."""
    lam_t = torch.tensor(float(lam_val), device=device)
    _olmo3_set_lam(model, lam_t)
    cfg = model.config
    ffn_fracs  = [m.inclusion_probs(lam_t).mean().item()
                  for m in model.modules() if isinstance(m, BSplineScaledOlmo3MLP)]
    attn_fracs = [m.inclusion_probs(lam_t).mean().item()
                  for m in model.modules() if isinstance(m, BSplineScaledOlmo3Attn)]
    ffn_frac  = float(np.mean(ffn_fracs))  if ffn_fracs  else 1.0
    attn_frac = float(np.mean(attn_fracs)) if attn_fracs else 1.0
    ffn_w  = 6 * cfg.hidden_size * cfg.intermediate_size
    attn_w = 8 * cfg.hidden_size ** 2
    total_w = ffn_w + attn_w
    return (ffn_frac * ffn_w + attn_frac * attn_w) / total_w


class _Olmo3ForwardWrapper(nn.Module):
    """Wraps Olmo3ForCausalLM so _time_model(model, batch, None) works."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, batch, *_):
        return self.m(input_ids=batch)


def main_olmo3(args, ckpt_dir: "Path") -> None:
    import json

    device = torch.device(args.device)
    print("=" * 72, flush=True)
    print("  Compression Comparison  (OLMo3-mini / olmo3_mini_train checkpoint)",
          flush=True)
    print(f"  ckpt_dir : {ckpt_dir}  device : {device}", flush=True)
    print("=" * 72, flush=True)

    # ── load model ────────────────────────────────────────────────────────────
    train_args = json.loads((ckpt_dir / "train_args.json").read_text())
    config     = Olmo3Config.from_pretrained(ckpt_dir)
    raw        = Olmo3ForCausalLM(config)

    n_knots = train_args.get("n_knots", 8)
    degree  = train_args.get("degree",  3)
    inject_stochastic_mlp(raw, n_knots=n_knots, degree=degree)
    inject_stochastic_attn(raw, n_knots=n_knots, degree=degree)
    if not train_args.get("no_per_head_norm", False):
        inject_per_head_qk_norm(raw)

    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    raw.load_state_dict(state)
    model = raw.to(device).eval()

    seq_len = train_args.get("seq_len", 512)
    d_mb    = dense_mb(model)
    print(f"  dense size : {d_mb:.1f} MB  seq_len={seq_len}", flush=True)

    if not args.data_dir:
        raise ValueError("--data_dir is required for OLMo3 checkpoints")
    val_ds = _Olmo3TokenDataset(Path(args.data_dir) / "validation.bin", seq_len)

    # ── dense baseline ────────────────────────────────────────────────────────
    _olmo3_set_lam(model, torch.tensor(0.0, device=device))
    _olmo3_set_hard_mask(model, False)
    _olmo3_set_sample_mask(model, False)
    dense_ppl = _eval_olmo3(model, val_ds, device, 0.0, args.max_batches)
    print(f"  dense PPL (lambda=0) : {dense_ppl:.2f}", flush=True)

    results: list[dict] = []

    def record(method, size_mb, flops_frac, ppl,
               sparsity=None, lam=None, **kw) -> None:
        r = dict(method=method, size_mb=size_mb, flops_frac=flops_frac,
                 ppl=ppl, sparsity=sparsity, lam=lam)
        r.update(kw)
        results.append(r)
        delta = ppl - dense_ppl
        print(f"    {method:<46}  {size_mb:>6.1f} MB  "
              f"FLOPs={flops_frac*100:>5.1f}%  PPL={ppl:.2f} ({delta:+.2f})",
              flush=True)

    record("dense", d_mb, 1.0, dense_ppl)

    # ── bake gates at lambda=0 (identity op) for SparseGPT/Wanda ─────────────
    print("\n  Preparing dense baseline (baking lambda=0 gates)…", flush=True)
    dense_baked = copy.deepcopy(model)
    _olmo3_bake_pruning(dense_baked, 0.0)
    # bake_pruning replaces BSplineScaledOlmo3MLP with plain Olmo3MLP(config),
    # which is created on CPU regardless of the parent model's device — re-move.
    dense_baked = dense_baked.to(device)
    dense_baked.eval()

    # ── calibration activations ───────────────────────────────────────────────
    print(f"  Collecting calibration activations ({args.n_calib} batches)…",
          flush=True)
    calib_ds = _Olmo3TokenDataset(Path(args.data_dir) / "validation.bin", seq_len)
    calib    = _collect_inputs_olmo3(dense_baked, calib_ds, device,
                                     n_calib=args.n_calib,
                                     max_rows=args.max_calib_rows)
    calib_cpu = {k: v.cpu() for k, v in calib.items()}
    print(f"  Captured {len(calib_cpu)} weight matrices.", flush=True)
    del calib

    sparsities = resolve_sparsities(args)

    # ── precompute SparseGPT inverse Hessians (once per layer) ────────────────
    # H_inv depends only on the calibration activations, not the sparsity level,
    # so building it once here (BLAS-multithreaded) and sharing it across all
    # sparsity workers avoids rebuilding it len(sparsities)× inside the pool.
    print(f"  Precomputing {len(calib_cpu)} SparseGPT inverse Hessians…",
          flush=True)
    hinv_cpu = {name: _sparsegpt_hinv(X) for name, X in calib_cpu.items()}

    # ── parallel pruning setup (mirrors old-path pattern) ─────────────────────
    n_prune_workers = args.prune_workers or min(len(sparsities), os.cpu_count() or 1)
    n_row_workers   = max(1, (os.cpu_count() or 1) // n_prune_workers)
    baked_state_cpu = {k: v.cpu() for k, v in dense_baked.state_dict().items()}
    config_dict     = config.to_dict()

    if sys.platform.startswith("linux"):
        _OLMO3_PRUNE_STATE.update(state_dict=baked_state_cpu, calib=calib_cpu,
                                   config=config_dict, n_row_workers=n_row_workers,
                                   hinv=hinv_cpu)
        olmo3_pool_kw: dict = dict(max_workers=n_prune_workers,
                                    mp_context=mp.get_context("fork"))
    else:
        olmo3_pool_kw = dict(
            max_workers=n_prune_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_olmo3_prune_worker,
            initargs=(baked_state_cpu, calib_cpu, config_dict, n_row_workers,
                      hinv_cpu),
        )

    def _eval_olmo3_pruned(tag: str, method: str, pruned_results: list) -> None:
        for sp, mb, state in sorted(pruned_results, key=lambda x: x[0]):
            m = _rebuild_olmo3_baked(state, config_dict)
            ppl = _eval_olmo3(m.to(device), val_ds, device, 0.0, args.max_batches)
            record(method, mb, 1.0 - sp, ppl, sparsity=sp)
            del m

    # ── SparseGPT ─────────────────────────────────────────────────────────────
    print(f"\n  ── SparseGPT ({len(sparsities)} levels, {n_prune_workers} workers) ──",
          flush=True)
    with ProcessPoolExecutor(**olmo3_pool_kw) as exe:
        sgpt_raw = list(exe.map(_olmo3_sparsegpt_worker, sparsities))
    _eval_olmo3_pruned("sparsegpt", "sparsegpt", sgpt_raw)

    # ── Wanda ─────────────────────────────────────────────────────────────────
    print(f"\n  ── Wanda ({len(sparsities)} levels, {n_prune_workers} workers) ──",
          flush=True)
    with ProcessPoolExecutor(**olmo3_pool_kw) as exe:
        wanda_raw = list(exe.map(_olmo3_wanda_worker, sparsities))
    _eval_olmo3_pruned("wanda", "wanda", wanda_raw)

    del dense_baked, baked_state_cpu, calib_cpu

    # ── tau sweep ─────────────────────────────────────────────────────────────
    method_name = f"tau-{args.model}"
    print(f"\n  ── {method_name} tau sweep ({len(TAU_LAMS)} lambda values, "
          f"{args.n_resample} draws each) ──", flush=True)

    _olmo3_set_sample_mask(model, True)   # stochastic Bernoulli draws per batch
    _olmo3_set_hard_mask(model, False)

    for lv in TAU_LAMS:
        smb = _structural_mb_olmo3(model, lv)
        ff  = _flops_frac_olmo3(model, lv, device)
        # average over n_resample independent evaluations (each uses fresh masks)
        ppls = [_eval_olmo3(model, val_ds, device, lv, args.max_batches)
                for _ in range(args.n_resample)]
        ppl  = float(np.mean(ppls))
        record(method_name, smb, ff, ppl, lam=lv)

    _olmo3_set_sample_mask(model, False)

    # ── compiled speedup sweep (GPU/XPU only) ─────────────────────────────────
    do_compile = (device.type in ("cuda", "xpu")) and not args.skip_compile
    if do_compile:
        compile_lams = TAU_LAMS[:: args.compile_stride]
        print(f"\n  ── compiled speedup sweep ({len(compile_lams)} lambda values, "
              f"stride={args.compile_stride}) ──", flush=True)

        timing_ds    = _Olmo3TokenDataset(Path(args.data_dir) / "validation.bin",
                                          seq_len)
        timing_batch = next(iter(
            DataLoader(timing_ds, batch_size=4, shuffle=False, num_workers=0)
        ))[0].to(device)

        # Dense baseline: baked model compiled
        dense_base = copy.deepcopy(model)
        _olmo3_bake_pruning(dense_base, 0.0)
        dense_base = dense_base.to(device).eval()
        try:
            base_c  = torch.compile(_Olmo3ForwardWrapper(dense_base))
            base_ms = _time_model(base_c, timing_batch, device)
        except Exception:
            base_ms = _time_model(_Olmo3ForwardWrapper(dense_base), timing_batch, device)
        print(f"    dense baseline (compiled): {base_ms:.2f} ms/batch", flush=True)
        del dense_base

        tau_result_map = {round(r["lam"], 6): r
                          for r in results if r["method"] == method_name}

        for lv in compile_lams:
            m_slim = _olmo3_copy_clear_cache(model, device=device)
            _olmo3_structural_prune(m_slim, lv, prune_attn=True, stochastic=True)
            actual_mb = dense_mb(m_slim)
            m_slim.eval()
            try:
                compiled  = torch.compile(_Olmo3ForwardWrapper(m_slim))
                comp_ms   = _time_model(compiled, timing_batch, device)
                speedup   = base_ms / comp_ms
                del compiled
            except Exception as exc:
                print(f"    torch.compile failed at lambda={lv:.1f}: {exc}", flush=True)
                comp_ms = speedup = float("nan")
            del m_slim

            key = round(lv, 6)
            if key in tau_result_map:
                tau_result_map[key]["actual_mb"] = actual_mb
                tau_result_map[key]["speedup"]   = speedup
                tau_result_map[key]["ms_batch"]  = comp_ms
            print(f"    {method_name} lambda={lv:<4.1f}  "
                  f"actual={actual_mb:.1f} MB  speedup={speedup:.2f}x  "
                  f"({comp_ms:.2f} ms/batch)", flush=True)

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    if out_path.is_dir():
        out_path = out_path / "compression_comparison.json"
    out_data = {
        "dense_mb":  d_mb,
        "dense_ppl": dense_ppl,
        "results":   results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
