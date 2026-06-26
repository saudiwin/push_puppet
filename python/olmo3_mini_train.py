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
olmo3_mini_train.py

Trains an OLMo-3 language model with the hyper_joint pruning architecture:
B-spline λ→τ stochastic gating with per-unit amplitude rescaling, applied
jointly to FFN neurons and attention heads.

This matches the ScaledPolyLM / hyper_joint variant from
stochastic_weight_test.py, adapted to the real Olmo3ForCausalLM.

Default config (~90M, smoke-test friendly):
  hidden=512  intermediate=1376  layers=12  heads=8

~1.3B config (matches OLMo-3 1B dense architecture) — pass these flags:
  --hidden 2048 --intermediate 8192 --n_layers 16 --n_heads 16
  Unique params ≈ 1.27B (205M embed + 16 × 67M layers, weight-tied).
  Two A30s (24 GB each): add --fsdp --grad_ckpt --batch 2 --seq_len 1024

Pruning (hyper_joint):
  Gates: FFN neurons + attention heads (no token embedding gating).
  τ(λ): monotone B-spline (8 knots, degree 3, rational map λ/(λ+1)).
        cumsum+softplus control points guarantee τ non-decreasing in λ.
        Fixed −1/T intercept keeps all units active at λ=0.
  f_gate_ℓ(λ)  = exp(gate_scale_raw  · u_gate(λ)):  amplitude inside gate ρ
  f_scale_ℓ(λ) = exp(output_scale_raw · u_scale(λ)): amplitude on unit outputs
        Per-neuron/per-head, independent B-spline u(λ) for each role.
        All scale_raw params init 0 → f≡1 at step 0 (identical to PolyLM start).

Parallelism:
  --fsdp        Use FSDP (ZeRO-3 by default) instead of DDP.
                Required for large models that exceed per-GPU memory.
  --sharding    full (ZeRO-3, default) or grad_op (ZeRO-2).
  --grad_ckpt   Enable activation checkpointing (trades compute for memory).

λ sampling (matches stochastic_weight_test.py):
  λ = 0          with prob 1/3  → dense pass, no penalty
  λ ~ Exp(rate)  with prob 2/3  (mean = 1/lam_rate)

Data:
  Prepare with prepare_olmo3_mini.py first (downloads OLMo-3 tokenizer and
  tokenizes the dataset into uint32 binary files).

  HF login required to download the tokenizer:
      huggingface-cli login        # run once; token stored in ~/.huggingface/token
      ! huggingface-cli login      # or run from this session prompt

Single-GPU smoke test:
    uv run python olmo3_mini_train.py --no_ddp --max_steps 200 --data_dir ./olmo3_data --ckpt_dir ./olmo3_mini_ckpt

Cluster (2× H100, DDP, ~90M):
    torchrun --nproc_per_node=2 olmo3_mini_train.py --data_dir /scratch/$USER/olmo3_data --ckpt_dir /scratch/$USER/olmo3_mini_ckpt

Cluster (2× A30, FSDP, ~1B):
    torchrun --nproc_per_node=2 olmo3_mini_train.py --fsdp --grad_ckpt --sharding full --hidden 2048 --intermediate 8192 --n_layers 16 --n_heads 16 --batch 2 --seq_len 1024 --data_dir /scratch/$USER/olmo3_data --ckpt_dir /scratch/$USER/olmo3_1b_ckpt

Cluster (4× Intel PVC, XPU, ~90M):
    torchrun --nproc_per_node=4 olmo3_mini_train.py --accelerator xpu --data_dir /scratch/$USER/olmo3_data --ckpt_dir /scratch/$USER/olmo3_mini_ckpt_pvc

Outputs:
    {ckpt_dir}/model.pt           model state dict
    {ckpt_dir}/config.json        Olmo3Config
    {ckpt_dir}/train_args.json    training hyperparameters
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import Olmo3Config, Olmo3ForCausalLM
from transformers.models.olmo3.modeling_olmo3 import Olmo3MLP, Olmo3RMSNorm

try:
    import torchcurves as tc
    _HAS_TORCHCURVES = True
except ImportError:
    _HAS_TORCHCURVES = False

# ── defaults (7B architecture scaled to ~90M) ─────────────────────────────────

VOCAB        = 100_278    # OLMo-3 tokenizer vocab
HIDDEN       = 512        # 4096 / 8
INTERMEDIATE = 1_376      # 11008 / 8  (ratio 2.69 preserved)
N_LAYERS     = 12
N_HEADS      = 8
SEQ_LEN      = 512
ROPE_THETA   = 500_000    # matches OLMo-3 7B
TEMPERATURE  = 0.5
BATCH        = 16
LR           = 3e-4
WARMUP       = 2_000
LAM_RATE     = 0.3        # Exp(rate) sampler; mean λ = 1/LAM_RATE ≈ 3.3
GRAD_CLIP    = 1.0
LOG_EVERY    = 100
EVAL_EVERY   = 1_000
SAVE_EVERY   = 1_000


# ── accelerator detection ─────────────────────────────────────────────────────

def _detect_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


# ── B-spline τ mixin ──────────────────────────────────────────────────────────

class _BSplineTauMixin:
    """Monotone B-spline τ(λ) and independent amplitude splines for gate/scale.

    Call _init_bspline() inside __init__ after setting self.temperature.
    Provides _monotone_coeffs(), _tau(lam), _u_gate(lam), _u_scale(lam).

    The two amplitude splines map λ ∈ [0,∞) → u ∈ [0,1] independently:
        _u_gate(λ)  — used by _f_gate, multiplied against row/head norms in ρ
        _u_scale(λ) — used by _f_scale, multiplied against output amplitudes
    Both are initialized to linspace(0,1,n_knots) ≈ the rational map λ/(λ+1),
    so training starts identical to the single-f(λ) baseline.

    BSplineScaledOlmo3MLP additionally defines f_input_u_raw / _u_input(lam)
    for the rank-1 input-direction correction; that parameter lives on the MLP
    subclass only so DDP can account for it in every forward pass.
    """

    def _precompute_splines(self, lam: torch.Tensor) -> None:
        """Pre-compute and cache spline outputs for a given λ.

        Call via set_lam() before each forward pass.  The cached tensors are
        still part of the autograd graph so gradients flow to tau_raw /
        f_gate_u_raw / f_scale_u_raw as normal.

        Cache must be cleared first so that the _tau/_u_gate/_u_scale methods
        don't return the previous step's stale tensors (whose saved autograd
        intermediates were freed by the last backward call).
        """
        self._sp_tau = self._sp_gate = self._sp_scale = None
        self._sp_tau   = self._tau(lam)
        self._sp_gate  = self._u_gate(lam)
        self._sp_scale = self._u_scale(lam)
        if hasattr(self, "f_input_u_raw"):
            self._sp_input = None
            self._sp_input = self._u_input(lam)

    def _init_bspline(self, n_knots: int = 8, degree: int = 3) -> None:
        if not _HAS_TORCHCURVES:
            raise ImportError(
                "torchcurves is required. Install with: uv add torchcurves")
        # tau_raw holds n_knots-1 increments; first coefficient is fixed to -1/T
        # so τ(0) = -1/T exactly (clamped B-spline evaluates to first coeff at x=0)
        self.tau_raw = nn.Parameter(torch.zeros(n_knots - 1))
        # Force CPU build for the BSplineBasis: torchcurves' __init__ calls
        # .item() on a knot tensor, which crashes if the surrounding device
        # context is torch.device("meta") (used by the rank-0-only FSDP load
        # path in olmo3_31_32b_think_midtrain.py). The basis is internal
        # book-keeping, not a parameter — keeping it on CPU is fine and it
        # rides .to(device) like any other module attribute.
        with torch.device("cpu"):
            self._basis  = tc.BSplineBasis(
                degree          = degree,
                knots_config    = n_knots,
                parameter_range = (0.0, 1.0),
                input_map       = tc.maps.Nonneg.rational(),
            )
        # Independent B-spline u-values for the two shared amplitude roles.
        # linspace(0,1) init ≈ λ/(λ+1) under the rational input_map.
        self.f_gate_u_raw  = nn.Parameter(torch.linspace(0.0, 1.0, n_knots))
        self.f_scale_u_raw = nn.Parameter(torch.linspace(0.0, 1.0, n_knots))
        # Spline output cache — populated by set_lam() before each forward pass.
        self._sp_tau: torch.Tensor | None   = None
        self._sp_gate: torch.Tensor | None  = None
        self._sp_scale: torch.Tensor | None = None
        # _sp_input and f_input_u_raw are MLP-only (rank-1 input correction);
        # they are added in BSplineScaledOlmo3MLP.__init__ to keep the Attn
        # subclass free of unused parameters (which would break DDP).

    def _monotone_coeffs(self) -> torch.Tensor:
        first = torch.full((1,), -1.0 / self.temperature,
                           device=self.tau_raw.device, dtype=self.tau_raw.dtype)
        increments = F.softplus(self.tau_raw)
        return torch.cumsum(torch.cat([first, increments], dim=-1), dim=-1)

    def _tau(self, lam) -> torch.Tensor:
        if self._sp_tau is not None:
            return self._sp_tau
        # Force fp32: torchcurves _basis uses matmul internally, which autocast
        # would convert to bf16, making grad_control_points bf16 in the backward
        # while `updates` stays fp32 — causing index_add_ to fail.
        with torch.amp.autocast(device_type=self.tau_raw.device.type, enabled=False):
            coeffs = self._monotone_coeffs()
            dev = coeffs.device
            x = torch.as_tensor(lam, dtype=torch.float32, device=dev).reshape(1, 1)
            c = coeffs.unsqueeze(0).unsqueeze(-1)
            return self._basis(x, c).squeeze()

    def _u_gate(self, lam) -> torch.Tensor:
        """B-spline u(λ) for the gate amplitude role (multiplied against row norms)."""
        if self._sp_gate is not None:
            return self._sp_gate
        with torch.amp.autocast(device_type=self.f_gate_u_raw.device.type, enabled=False):
            dev = self.f_gate_u_raw.device
            x = torch.as_tensor(lam, dtype=torch.float32, device=dev).reshape(1, 1)
            c = self.f_gate_u_raw.float().unsqueeze(0).unsqueeze(-1)
            return self._basis(x, c).squeeze()

    def _u_scale(self, lam) -> torch.Tensor:
        """B-spline u(λ) for the output amplitude role (multiplied against unit outputs)."""
        if self._sp_scale is not None:
            return self._sp_scale
        with torch.amp.autocast(device_type=self.f_scale_u_raw.device.type, enabled=False):
            dev = self.f_scale_u_raw.device
            x = torch.as_tensor(lam, dtype=torch.float32, device=dev).reshape(1, 1)
            c = self.f_scale_u_raw.float().unsqueeze(0).unsqueeze(-1)
            return self._basis(x, c).squeeze()

    def _u_input(self, lam) -> torch.Tensor:
        """B-spline u(λ) for the input-direction amplitude role (rank-1 f_V correction)."""
        if self._sp_input is not None:
            return self._sp_input
        with torch.amp.autocast(device_type=self.f_input_u_raw.device.type, enabled=False):
            dev = self.f_input_u_raw.device
            x = torch.as_tensor(lam, dtype=torch.float32, device=dev).reshape(1, 1)
            c = self.f_input_u_raw.float().unsqueeze(0).unsqueeze(-1)
            return self._basis(x, c).squeeze()


# ── FFN neuron gate ────────────────────────────────────────────────────────────

class BSplineScaledOlmo3MLP(_BSplineTauMixin, Olmo3MLP):
    """
    Drop-in replacement for Olmo3MLP: B-spline τ(λ) + rank-1 amplitude correction.

    Gate:  ρ_i(λ) = σ((‖gate_proj[i,:] ⊙ f_input(λ)‖ · f_gate_i(λ) − τ(λ)) / T)
    f_gate_i(λ)  = exp(gate_scale_raw[i]  · u_gate(λ))  — output-amplitude inside gate
    f_scale_i(λ) = exp(output_scale_raw[i] · u_scale(λ)) — amplitude on unit outputs
    f_input_j(λ) = exp(input_scale_raw[j]  · u_input(λ)) — per-input-dim scale (rank-1)

    u_gate, u_scale, u_input are independent B-splines (shared across units, learned
    per layer), initialized to linspace(0,1) ≈ λ/(λ+1).
    All scale_raw params init 0 → f_gate = f_scale = f_input = 1 at step 0.

    The rank-1 input correction f_input scales all input dimensions before gate_proj
    and up_proj, equivalent to W_eff = diag(m · f_gate) @ W @ diag(f_input).

    Usage: set module.current_lam via set_lam() before each forward pass.

    Gating modes (training):
        ste=True  (default): hard Bernoulli mask forward, gradient via ρ
                  (straight-through estimator, matches stochastic_weight_test.py).
                  Inactive-neuron safeguard: if all draws are 0, force argmax→1.
        ste=False           : soft gating uses ρ directly (legacy path).
    Inference uses (ρ > 0.5) when self.hard is set (via set_hard_mask).
    """

    def __init__(self, config: Olmo3Config, temperature: float = TEMPERATURE,
                 scale_output: bool = True, n_knots: int = 8, degree: int = 3):
        Olmo3MLP.__init__(self, config)
        self.temperature = temperature
        self._init_bspline(n_knots=n_knots, degree=degree)
        # Rank-1 input-direction B-spline — MLP only (not on Attn) so DDP
        # sees it used in every forward pass and doesn't error.
        self.f_input_u_raw = nn.Parameter(torch.linspace(0.0, 1.0, n_knots))
        self._sp_input: torch.Tensor | None = None
        self.gate_scale_raw   = nn.Parameter(torch.zeros(config.intermediate_size))
        self.output_scale_raw = nn.Parameter(torch.zeros(config.intermediate_size))
        self.input_scale_raw  = nn.Parameter(torch.zeros(config.hidden_size))
        self.current_lam: torch.Tensor | None = None
        self.last_rho_mean: torch.Tensor | None = None
        self.hard:   bool = False
        self.sample: bool = False   # Bernoulli sampling in eval (no STE)
        self.ste:    bool = True
        self.scale_output: bool = scale_output

    def row_norms(self, lam: torch.Tensor | None = None) -> torch.Tensor:
        if lam is not None:
            f_v = self._f_input(lam)                              # (hidden_size,)
            return (self.gate_proj.weight.float() * f_v.unsqueeze(0)).norm(dim=1)
        return self.gate_proj.weight.norm(dim=1)      # (intermediate_size,)

    def _f_gate(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.gate_scale_raw.float() * self._u_gate(lam))

    def _f_scale(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.output_scale_raw.float() * self._u_scale(lam))

    def _f_input(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.input_scale_raw.float() * self._u_input(lam))

    def inclusion_probs(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            (self.row_norms(lam).float() * self._f_gate(lam) - self._tau(lam)) / self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = (self.current_lam
               if self.current_lam is not None
               else torch.zeros((), device=x.device))
        # Always recompute splines fresh — required so activation-checkpoint
        # recompute observes the same number of saved tensors as the original
        # forward. _precompute_splines clears self._sp_tau first so calling it
        # unconditionally is safe whether or not set_lam already cached.
        self._precompute_splines(lam)

        rho = self.inclusion_probs(lam)
        self.last_rho_mean = rho.mean()

        if self.hard:
            combined = (rho > 0.5).to(rho)
        elif self.sample:
            hard = (torch.rand_like(rho) < rho).to(rho)
            if hard.sum() == 0:
                hard[rho.argmax()] = 1.0
            combined = hard
        elif self.training and self.ste:
            with torch.no_grad():
                hard = (torch.rand_like(rho) < rho).to(rho)
                if hard.sum() == 0:
                    hard[rho.argmax()] = 1.0
            combined = hard + (rho - rho.detach())     # STE
        else:
            combined = rho

        if self.scale_output:
            combined = combined * self._f_scale(lam)

        f_v = self._f_input(lam).to(x.dtype)         # (hidden_size,) rank-1 input scale
        x_scaled = x * f_v
        combined = combined.to(x.dtype)
        gate = self.act_fn(self.gate_proj(x_scaled))
        up   = self.up_proj(x_scaled)
        return self.down_proj(gate * up * combined)


# ── attention head gate ────────────────────────────────────────────────────────

class BSplineScaledOlmo3Attn(_BSplineTauMixin, nn.Module):
    """
    Replaces o_proj to add per-head B-spline gating + amplitude scaling.

    Gate:  ρ_h(λ) = σ((‖o_proj[:,h,:]‖_F · f_gate_h(λ) − τ(λ)) / T)
    f_gate_h(λ)  = exp(gate_scale_raw[h]   · u_gate(λ))  — amplitude inside gate
    f_scale_h(λ) = exp(output_scale_raw[h] · u_scale(λ)) — amplitude on head outputs

    u_gate(λ) and u_scale(λ) are independent B-splines (shared across heads,
    learned per attention layer), initialized to linspace(0,1) ≈ λ/(λ+1).

    Training uses straight-through Bernoulli sampling (ste=True, default);
    inference uses (ρ > 0.5) when self.hard is set.
    """

    def __init__(self, o_proj: nn.Linear, n_heads: int,
                 temperature: float = TEMPERATURE, scale_output: bool = True,
                 n_knots: int = 8, degree: int = 3):
        nn.Module.__init__(self)
        self.o_proj      = o_proj
        self.n_heads     = n_heads
        self.head_dim    = o_proj.in_features // n_heads
        self.temperature = temperature
        self._init_bspline(n_knots=n_knots, degree=degree)
        self.gate_scale_raw   = nn.Parameter(torch.zeros(n_heads))
        self.output_scale_raw = nn.Parameter(torch.zeros(n_heads))
        self.current_lam: torch.Tensor | None = None
        self.last_rho_mean: torch.Tensor | None = None
        self.hard:   bool = False
        self.sample: bool = False   # Bernoulli sampling in eval (no STE)
        self.ste:    bool = True
        self.scale_output: bool = scale_output

    def head_norms(self) -> torch.Tensor:
        w = self.o_proj.weight.view(
            self.o_proj.out_features, self.n_heads, self.head_dim)
        return w.norm(dim=(0, 2))      # Frobenius per head → (n_heads,)

    def _f_gate(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.gate_scale_raw.float() * self._u_gate(lam))

    def _f_scale(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.output_scale_raw.float() * self._u_scale(lam))

    def inclusion_probs(self, lam: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            (self.head_norms().float() * self._f_gate(lam) - self._tau(lam)) / self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = (self.current_lam
               if self.current_lam is not None
               else torch.zeros((), device=x.device))
        # Always recompute splines fresh — required for checkpoint recompute
        # consistency. See BSplineScaledOlmo3MLP.forward for the rationale.
        self._precompute_splines(lam)
        rho = self.inclusion_probs(lam)
        self.last_rho_mean = rho.mean()

        if self.hard:
            head_mask = (rho > 0.5).to(rho)
        elif self.sample:
            hard = (torch.rand_like(rho) < rho).to(rho)
            if hard.sum() == 0:
                hard[rho.argmax()] = 1.0
            head_mask = hard
        elif self.training and self.ste:
            with torch.no_grad():
                hard = (torch.rand_like(rho) < rho).to(rho)
                if hard.sum() == 0:
                    hard[rho.argmax()] = 1.0
            head_mask = hard + (rho - rho.detach())     # STE
        else:
            head_mask = rho

        if self.scale_output:
            head_mask = head_mask * self._f_scale(lam)

        x_h = x.view(*x.shape[:-1], self.n_heads, self.head_dim)
        head_mask = head_mask.to(x_h.dtype)
        x_h = (x_h * head_mask.view(1, 1, self.n_heads, 1)).reshape(
            *x_h.shape[:-2], self.n_heads * self.head_dim)
        return self.o_proj(x_h)


_STOCH_TYPES = (BSplineScaledOlmo3MLP, BSplineScaledOlmo3Attn)


# ── λ-conditional embedding amplitude ─────────────────────────────────────────

class EmbeddingAmplitude(nn.Module):
    """λ-conditional per-dimension amplitude scale on the embedding output.

    Wraps nn.Embedding; exposes .weight for LM-head weight tying.

    When many FFN neurons are pruned (high λ) the residual stream carries less
    energy from fewer active paths.  This module learns a per-dimension
    correction scale(λ) ∈ R^hidden applied right at the token→continuous
    boundary:

        h = embed(input_ids) * scale(λ)

    scale(λ) = exp(B-spline(λ; ctrl))   — always positive, exactly 1 at init
    ctrl ∈ R^{n_knots × hidden}          — zero-init → scale ≡ 1

    The correction is structurally distinct from the FFN/attention gates: those
    control *which paths* carry information; this controls the *amplitude* of
    the token signal entering the residual stream.

    Set module.current_lam via set_lam() before each forward pass.
    """

    def __init__(self, embed: nn.Embedding, n_knots: int = 8, degree: int = 3):
        super().__init__()
        if not _HAS_TORCHCURVES:
            raise ImportError("torchcurves is required for EmbeddingAmplitude")
        self.embed  = embed
        # Expose .weight so HuggingFace weight-tying logic still works.
        # Same Python object as embed.weight — set() deduplication avoids
        # double-counting in the optimizer.
        self.weight = embed.weight
        hidden      = embed.embedding_dim
        self.ctrl   = nn.Parameter(torch.zeros(n_knots, hidden))
        self._basis = tc.BSplineBasis(
            degree=degree,
            knots_config=n_knots,
            parameter_range=(0.0, 1.0),
            input_map=tc.maps.Nonneg.rational(),
        )
        self.current_lam = None

    def _scale(self, lam) -> torch.Tensor:
        dev = self.ctrl.device
        x   = torch.as_tensor(lam, dtype=torch.float32, device=dev).reshape(1, 1)
        c   = self.ctrl.float().unsqueeze(0)      # [1, n_knots, hidden]
        raw = self._basis(x, c).squeeze(0)         # [hidden]
        return torch.exp(raw)                      # positive; = 1 when ctrl = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lam   = self.current_lam if self.current_lam is not None else 0.0
        h     = self.embed(input_ids)
        scale = self._scale(lam).to(h.dtype)
        return h * scale


# ── per-head q/k normalization ────────────────────────────────────────────────

class PerHeadRMSNorm(nn.Module):
    """RMSNorm applied independently per attention head.

    Replaces the default OLMo-3 q_norm/k_norm which normalize over the flat
    (n_heads * head_dim) vector.  Because each head is normalized in isolation,
    structural pruning of heads is numerically exact: the weight vector stays
    (head_dim,) regardless of which heads survive — only self.n_heads changes.

    Initialise from a flat (n_heads * head_dim,) weight by averaging each
    head's block:  w_per_head = w.view(n_heads, head_dim).mean(0)
    """

    def __init__(self, head_dim: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads  = n_heads
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, n_heads * head_dim) — standard OLMo-3 q/k shape
        B, T, _ = hidden_states.shape
        x = hidden_states.view(B, T, self.n_heads, self.head_dim)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (self.weight * x).to(input_dtype)
        return x.reshape(B, T, self.n_heads * self.head_dim)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, n_heads={self.n_heads}"


def inject_per_head_qk_norm(model: Olmo3ForCausalLM) -> None:
    """Replace each attention's q_norm/k_norm with PerHeadRMSNorm in-place.

    Converts existing flat (n_heads * head_dim) weights by averaging each
    head's block into a single (head_dim,) weight.  This is lossless when the
    original per-head weights were identical (freshly initialised) and a
    reasonable approximation when they diverged during training.

    Call this before inject_stochastic_attn (or after — order doesn't matter
    since BSplineScaledOlmo3Attn does not touch q_norm/k_norm).
    """
    n_q_heads  = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_q_heads)
    head_dim   = model.config.hidden_size // n_q_heads
    eps        = model.config.rms_norm_eps
    for layer in model.model.layers:
        attn = layer.self_attn
        for attr, h in (("q_norm", n_q_heads), ("k_norm", n_kv_heads)):
            old = getattr(attn, attr)
            new = PerHeadRMSNorm(head_dim, h, eps).to(old.weight.device)
            # average the h blocks of the flat weight → (head_dim,)
            new.weight.data.copy_(
                old.weight.data.view(h, head_dim).mean(0)
            )
            setattr(attn, attr, new)


# ── model wrapper ─────────────────────────────────────────────────────────────

def make_olmo3_config(args) -> Olmo3Config:
    """Build Olmo3Config with 7B proportions scaled to ~90M params."""
    n = args.n_layers
    return Olmo3Config(
        vocab_size            = args.vocab,
        hidden_size           = args.hidden,
        intermediate_size     = args.intermediate,
        num_hidden_layers     = n,
        num_attention_heads   = args.n_heads,
        num_key_value_heads   = args.n_heads,
        max_position_embeddings = args.seq_len,
        sliding_window        = args.seq_len,
        # OLMo-3 7B pattern: 3 sliding + 1 full, tiled to n_layers
        layer_types           = (["sliding_attention"] * 3 + ["full_attention"])
                                  * (n // 4) + ["sliding_attention"] * (n % 4),
        rope_parameters       = {"rope_theta": args.rope_theta,
                                 "rope_type":  "default"},
        rms_norm_eps          = 1e-5,
        tie_word_embeddings   = True,   # weight tying → ~90M unique params
        hidden_act            = "silu",
        use_cache             = False,
        attention_bias        = False,
        attention_dropout     = 0.0,
    )


def inject_stochastic_mlp(model: Olmo3ForCausalLM,
                           temperature: float = TEMPERATURE,
                           scale_output: bool = True,
                           n_knots: int = 8, degree: int = 3) -> None:
    """Replace every Olmo3MLP with BSplineScaledOlmo3MLP in-place."""
    for layer in model.model.layers:
        old = layer.mlp
        new = BSplineScaledOlmo3MLP(model.config, temperature, scale_output,
                                    n_knots=n_knots, degree=degree)
        new.gate_proj.weight.data.copy_(old.gate_proj.weight.data)
        new.up_proj.weight.data.copy_(old.up_proj.weight.data)
        new.down_proj.weight.data.copy_(old.down_proj.weight.data)
        layer.mlp = new


def inject_stochastic_attn(model: Olmo3ForCausalLM,
                            temperature: float = TEMPERATURE,
                            scale_output: bool = True,
                            n_knots: int = 8, degree: int = 3) -> None:
    """Replace o_proj in every attention layer with BSplineScaledOlmo3Attn."""
    n_heads = model.config.num_attention_heads
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.o_proj = BSplineScaledOlmo3Attn(attn.o_proj, n_heads, temperature,
                                              scale_output, n_knots=n_knots, degree=degree)


def inject_embedding_amplitude(model: Olmo3ForCausalLM,
                                n_knots: int = 8, degree: int = 3) -> None:
    """Wrap embed_tokens with EmbeddingAmplitude in-place.

    model.model.embed_tokens becomes an EmbeddingAmplitude that still exposes
    .weight for LM-head weight tying and accepts (input_ids) as before.
    """
    model.model.embed_tokens = EmbeddingAmplitude(
        model.model.embed_tokens, n_knots=n_knots, degree=degree)


def set_lam(model: nn.Module, lam: torch.Tensor) -> None:
    """Broadcast current λ to all stochastic gate modules before forward.

    Non-FSDP (DDP / single-GPU): splines are precomputed here once per step
    so forward() can use the cached values — torchcurves kernel calls happen
    exactly once regardless of how many layers share the same λ.

    FSDP (ZeRO-3/ZeRO-2): parameters are sharded views of a flat tensor that
    FSDP modifies inplace during reshard after backward.  Computing splines on
    those views outside the FSDP forward context triggers a ViewBackward
    inplace-modification error.  Instead we just invalidate the cache here and
    let each module's forward() recompute inside the parameter-gather context.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        _fsdp = isinstance(model, FSDP)
    except ImportError:
        _fsdp = False

    for m in model.modules():
        if isinstance(m, (*_STOCH_TYPES, EmbeddingAmplitude)):
            m.current_lam = lam
            if isinstance(m, _BSplineTauMixin):
                if _fsdp:
                    m._sp_tau = m._sp_gate = m._sp_scale = None
                    if hasattr(m, "_sp_input"):
                        m._sp_input = None
                else:
                    m._precompute_splines(lam)


def invalidate_spline_cache(model: nn.Module) -> None:
    """Clear cached spline outputs (_sp_tau / _sp_gate / _sp_scale / _sp_input)
    on every gate module so the next inclusion_probs()/_tau() call recomputes
    fresh for whatever λ is passed.

    Use this before reading inclusion_probs(lam) directly (active_fraction,
    structural_prune) on a model that may have been committed or forward-passed
    at a different λ — otherwise _tau() short-circuits to the stale cached value
    and ignores the requested λ.

    Unlike set_lam(), this only sets the caches to None; it never stores a freshly
    computed (non-leaf, autograd-tracked) tensor on the modules, so it cannot
    break copy.deepcopy() of the model in build_compressed_olmo3().
    """
    for m in model.modules():
        if isinstance(m, _BSplineTauMixin):
            m._sp_tau = m._sp_gate = m._sp_scale = None
            if hasattr(m, "_sp_input"):
                m._sp_input = None


def set_hard_mask(model: nn.Module, hard: bool = True) -> None:
    """Switch all gate modules between soft (hard=False) and hard (hard=True) gating."""
    for m in model.modules():
        if isinstance(m, _STOCH_TYPES):
            m.hard = hard


def set_sample_mask(model: nn.Module, sample: bool = True) -> None:
    """Enable Bernoulli sampling on all gate modules (eval-mode stochastic gates).

    When sample=True, each forward pass draws a fresh independent subnet from
    Bernoulli(ρ).  Use this to estimate E_mask[α] by averaging several forward
    passes.  Mutually exclusive with hard=True — call set_hard_mask(False) first.
    """
    for m in model.modules():
        if isinstance(m, _STOCH_TYPES):
            m.sample = sample


def set_gate_mode(model: nn.Module, gate_mode: str) -> None:
    """Set STE vs soft gating on all stochastic gate modules.

    gate_mode="ste":  hard Bernoulli forward + straight-through gradient (default).
    gate_mode="soft": multiply each unit by ρ directly; the forward pass
                      estimates E[NLL] under the Bernoulli gate distribution.
    """
    use_ste = (gate_mode == "ste")
    for m in model.modules():
        if isinstance(m, _STOCH_TYPES):
            m.ste = use_ste


@torch.no_grad()
def bake_pruning(model: Olmo3ForCausalLM, lam: float) -> None:
    """Apply the hard mask at a fixed λ by folding mask·f(λ) into the weights.

    Bakes the per-unit scale into the projection weights then replaces every
    stochastic module with its plain equivalent (Olmo3MLP / nn.Linear),
    removing all B-spline and scale_raw parameters from memory.

    For FFN:   down_proj.weight[:, i] *= mask[i] * f(λ)[i]   (scale_output=True)
                              or  *= mask[i]                  (scale_output=False)
    For attn:  o_proj.weight[:, h*d:(h+1)*d] *= mask[h] * f(λ)[h]  (scale_output=True)
                                            or  *= mask[h]           (scale_output=False)

    This is equivalent to running forward with hard=True at the given λ,
    but the computation is paid once at load time rather than every step.
    Note: zeros are retained in weight matrices (structural pruning — actually
    removing rows/cols — requires rebuilding the model with a smaller config).
    """
    dev   = next(model.parameters()).device
    lam_t = torch.tensor(float(lam), device=dev)

    for layer in model.model.layers:
        # ── FFN ──────────────────────────────────────────────────────────────
        mlp = layer.mlp
        if isinstance(mlp, BSplineScaledOlmo3MLP):
            rho  = mlp.inclusion_probs(lam_t)
            mask = (rho > 0.5).to(rho)                     # (intermediate,)
            scale = (mask * mlp._f_scale(lam_t)) if mlp.scale_output else mask
            # fold scale into down_proj input columns (avoids nonlinear act_fn issue)
            mlp.down_proj.weight.data *= scale.to(mlp.down_proj.weight.dtype)
            # fold rank-1 input-direction scale f_V into gate_proj and up_proj columns
            f_v = mlp._f_input(lam_t).to(mlp.gate_proj.weight.dtype)
            mlp.gate_proj.weight.data *= f_v.unsqueeze(0)
            mlp.up_proj.weight.data   *= f_v.unsqueeze(0)
            # replace with plain Olmo3MLP, copying the (now-baked) weights
            plain = Olmo3MLP(model.config)
            plain.gate_proj.weight.data.copy_(mlp.gate_proj.weight.data)
            plain.up_proj.weight.data.copy_(mlp.up_proj.weight.data)
            plain.down_proj.weight.data.copy_(mlp.down_proj.weight.data)
            layer.mlp = plain

        # ── attention o_proj ─────────────────────────────────────────────────
        attn   = layer.self_attn
        o_gate = attn.o_proj
        if isinstance(o_gate, BSplineScaledOlmo3Attn):
            rho    = o_gate.inclusion_probs(lam_t)
            mask   = (rho > 0.5).to(rho)                         # (n_heads,)
            scale  = (mask * o_gate._f_scale(lam_t)) if o_gate.scale_output else mask
            # fold per-head scale into the input columns of o_proj
            # o_proj.weight shape: (hidden, n_heads * head_dim)
            col_scale = scale.repeat_interleave(o_gate.head_dim)  # (n_heads*head_dim,)
            o_gate.o_proj.weight.data *= col_scale.to(o_gate.o_proj.weight.dtype)
            attn.o_proj = o_gate.o_proj   # unwrap: replace gated module with plain Linear


class _SlimMLP(nn.Module):
    """SwiGLU FFN with a per-layer intermediate size determined by structural pruning."""

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear,
                 down_proj: nn.Linear, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj   = up_proj
        self.down_proj = down_proj
        self.act_fn    = act_fn   # stateless callable; no parameters to register

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@torch.no_grad()
def structural_prune(model: Olmo3ForCausalLM, lam: float,
                     prune_attn: bool = True,
                     stochastic: bool = False) -> dict:
    """Physically slice pruned rows/columns from weight matrices at a given λ.

    Unlike bake_pruning() which zeros weights in-place, this creates new smaller
    nn.Linear layers containing only surviving units — giving real memory and
    FLOP savings measurable with parameter counts and forward-pass profiling.

    FFN  (BSplineScaledOlmo3MLP → _SlimMLP):
        gate_proj / up_proj : slice rows  (keep surviving neurons)
        down_proj           : slice cols  (keep surviving neurons)
        This is NUMERICALLY EXACT: the FFN is element-wise, no cross-coupling.

    Attention (BSplineScaledOlmo3Attn unwrapped) — prune_attn=True only:
        q_proj / k_proj / v_proj : slice rows  (keep head_dim blocks for surviving heads)
        o_proj                   : slice cols  (same head_dim blocks)
        q_norm / k_norm          : slice weight vector to n_keep_h * head_dim
        CAVEAT: OLMo-3 applies q_norm/k_norm to the flat (n_heads*head_dim)
        q_proj/k_proj output, normalising over ALL heads jointly. Slicing to
        n_keep_h heads changes the RMS denominator, so the pruned model will
        NOT be numerically identical to the full model with a hard mask at the
        same λ. For speculative-decoding draft use this approximation is fine;
        set prune_attn=False if you need bit-exact FFN-only correctness.

    The OLMo-3 attention forward uses -1 in both internal reshapes (hidden_shape
    and the final attn_output.reshape), so no attribute patching is needed — the
    forward pass adapts automatically to whatever projection sizes are present.

    Assumes MHA (num_attention_heads == num_key_value_heads). At least one unit
    per layer is always kept (falls back to argmax if all rho <= 0.5 / all draws 0).

    stochastic: if True, draw the keep mask from Bernoulli(rho) instead of using
        the hard threshold rho > 0.5. Matches the stochastic masking used during
        training and PPL evaluation. Call multiple times and average model_mb for
        a stable size estimate.

    Returns {"ffn": [n_survive_layer0, ...], "attn": [n_heads_layer0, ...]}
    """
    dev   = next(model.parameters()).device
    lam_t = torch.tensor(float(lam), device=dev)
    stats: dict[str, list[int]] = {"ffn": [], "attn": []}

    # Invalidate the per-module spline cache (_sp_tau / _sp_gate / ...) so the
    # inclusion_probs(lam_t) / _f_input(lam_t) / _f_scale(lam_t) reads below
    # recompute fresh for THIS λ. _tau()/_u_gate() short-circuit to a cached
    # value if one is present, ignoring the lam argument — so a model previously
    # run/committed at a different λ (e.g. a λ=0 baseline pass before a draft
    # sweep) would otherwise yield τ(0) here and keep 100% of units at every λ.
    invalidate_spline_cache(model)

    for layer in model.model.layers:
        # ── FFN ──────────────────────────────────────────────────────────────
        mlp = layer.mlp
        if isinstance(mlp, BSplineScaledOlmo3MLP):
            rho  = mlp.inclusion_probs(lam_t)
            if stochastic:
                keep = torch.bernoulli(rho).bool().nonzero(as_tuple=True)[0]
            else:
                keep = (rho > 0.5).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                keep = rho.argmax().unsqueeze(0)
            n_keep = keep.numel()
            stats["ffn"].append(n_keep)

            dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
            new_gate = nn.Linear(mlp.gate_proj.in_features, n_keep, bias=False, device=dev, dtype=dt)
            new_up   = nn.Linear(mlp.up_proj.in_features,   n_keep, bias=False, device=dev, dtype=dt)
            new_down = nn.Linear(n_keep, mlp.down_proj.out_features, bias=False, device=dev, dtype=dt)
            # fold rank-1 input-direction scale f_V into gate_proj / up_proj columns
            f_v = mlp._f_input(lam_t).to(dt)
            new_gate.weight.data.copy_(mlp.gate_proj.weight.data[keep] * f_v.unsqueeze(0))
            new_up.weight.data.copy_(mlp.up_proj.weight.data[keep]   * f_v.unsqueeze(0))
            # fold amplitude scale f(λ) into down_proj columns so the slimmed
            # model produces identical logits to the full model with hard mask
            down_cols = mlp.down_proj.weight.data[:, keep]
            if mlp.scale_output:
                down_cols = down_cols * mlp._f_scale(lam_t)[keep].to(dt)
            new_down.weight.data.copy_(down_cols)
            layer.mlp = _SlimMLP(new_gate, new_up, new_down, mlp.act_fn)

        # ── attention ────────────────────────────────────────────────────────
        attn   = layer.self_attn
        o_gate = attn.o_proj
        if prune_attn and isinstance(o_gate, BSplineScaledOlmo3Attn):
            rho    = o_gate.inclusion_probs(lam_t)
            if stochastic:
                keep_h = torch.bernoulli(rho).bool().nonzero(as_tuple=True)[0]
            else:
                keep_h = (rho > 0.5).nonzero(as_tuple=True)[0]
            if keep_h.numel() == 0:
                keep_h = rho.argmax().unsqueeze(0)   # always keep at least 1 head
            n_keep_h = keep_h.numel()
            stats["attn"].append(n_keep_h)
            head_dim = o_gate.head_dim

            # contiguous column indices for surviving heads
            col_idx = torch.cat([
                torch.arange(h * head_dim, (h + 1) * head_dim, device=keep_h.device)
                for h in keep_h
            ])

            # q_proj / k_proj / v_proj — slice rows (output features)
            for attr in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, attr)
                dev, dt = proj.weight.device, proj.weight.dtype
                new_proj = nn.Linear(proj.in_features, n_keep_h * head_dim,
                                     bias=False, device=dev, dtype=dt)
                new_proj.weight.data.copy_(proj.weight.data[col_idx])
                setattr(attn, attr, new_proj)

            # o_proj — unwrap BSplineScaledOlmo3Attn, slice columns (input features)
            # fold per-head amplitude scale f(λ) into the surviving columns
            inner_o = o_gate.o_proj
            dev, dt = inner_o.weight.device, inner_o.weight.dtype
            new_o = nn.Linear(n_keep_h * head_dim, inner_o.out_features,
                              bias=False, device=dev, dtype=dt)
            o_cols = inner_o.weight.data[:, col_idx]
            if o_gate.scale_output:
                head_scale = o_gate._f_scale(lam_t)[keep_h].repeat_interleave(head_dim).to(dt)
                o_cols = o_cols * head_scale
            new_o.weight.data.copy_(o_cols)
            attn.o_proj = new_o

            # q_norm / k_norm:
            #   PerHeadRMSNorm  — weight stays (head_dim,); just update n_heads
            #   Olmo3RMSNorm    — flat (n_heads*head_dim,); slice to col_idx (approximate)
            for attr in ("q_norm", "k_norm"):
                norm = getattr(attn, attr)
                if isinstance(norm, PerHeadRMSNorm):
                    norm.n_heads = n_keep_h   # weight unchanged — exact
                else:
                    new_norm = Olmo3RMSNorm(
                        n_keep_h * head_dim, norm.variance_epsilon).to(dev)
                    new_norm.weight.data.copy_(norm.weight.data[col_idx])
                    setattr(attn, attr, new_norm)

    return stats


def collect_penalty(model: nn.Module, penalty_mode: str = "flops",
                    ffn_w: float = 1.0, attn_w: float = 1.0) -> torch.Tensor:
    """Weighted mean inclusion probability across FFN and attention gates.

    penalty_mode:
        "flops"   — weight by projection FLOPs (SwiGLU: 6·hidden·intermediate
                    for FFN; 8·hidden² for attention)
        "params"  — weight by parameter count (3·hidden·intermediate FFN;
                    4·hidden² attention)
        "uniform" — unweighted mean across all gate modules
    """
    ffn_rhos  = [m.last_rho_mean for m in model.modules()
                 if isinstance(m, BSplineScaledOlmo3MLP)
                 and m.last_rho_mean is not None]
    attn_rhos = [m.last_rho_mean for m in model.modules()
                 if isinstance(m, BSplineScaledOlmo3Attn)
                 and m.last_rho_mean is not None]

    if penalty_mode == "uniform":
        all_rhos = ffn_rhos + attn_rhos
        if not all_rhos:
            return torch.tensor(0.0)
        return torch.stack(all_rhos).mean()

    weighted, total_w = [], 0.0
    for r in ffn_rhos:
        weighted.append(ffn_w * r)
        total_w += ffn_w
    for r in attn_rhos:
        weighted.append(attn_w * r)
        total_w += attn_w
    if not weighted:
        return torch.tensor(0.0)
    return torch.stack(weighted).sum() / total_w


def active_stats(model: nn.Module, lam: torch.Tensor) -> dict:
    def _frac(cls):
        vals = [m.inclusion_probs(lam).mean().item()
                for m in model.modules() if isinstance(m, cls)]
        return float(np.mean(vals)) if vals else None

    return {
        "ffn_frac":  _frac(BSplineScaledOlmo3MLP),
        "attn_frac": _frac(BSplineScaledOlmo3Attn),
    }


# ── data ──────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """Memory-mapped uint32 token file (OLMo-3 vocab > uint16 capacity)."""

    def __init__(self, path: str | Path, seq_len: int):
        self.data    = np.memmap(path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def build_loader(data_dir: str | Path, split: str, seq_len: int,
                 batch: int, rank: int = 0, world_size: int = 1,
                 **kwargs) -> DataLoader:
    ds      = TokenDataset(Path(data_dir) / f"{split}.bin", seq_len)
    sampler = (DistributedSampler(ds, world_size, rank, shuffle=(split == "train"))
               if world_size > 1 else None)
    return DataLoader(ds, batch, sampler=sampler,
                      shuffle=(sampler is None and split == "train"),
                      drop_last=True, **kwargs)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, data_dir: str | Path, split: str,
             device: torch.device, lam_val: float = 0.0,
             max_batches: int = 100, seq_len: int = SEQ_LEN,
             batch: int = 4) -> float:
    model.eval()
    ds     = TokenDataset(Path(data_dir) / f"{split}.bin", seq_len)
    loader = DataLoader(ds, batch, shuffle=False, num_workers=0)
    lam_t  = torch.tensor(lam_val, device=device)
    total = n = 0
    for i, (inp, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        set_lam(model, lam_t)
        out    = model(input_ids=inp)
        total += F.cross_entropy(
            out.logits.float().view(-1, out.logits.size(-1)),
            tgt.view(-1)).item() * inp.numel()
        n += inp.numel()
    model.train()
    return math.exp(min(total / n, 20))


# ── training ──────────────────────────────────────────────────────────────────

def train(args, rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # ── build & wrap model ───────────────────────────────────────────────────
    config    = make_olmo3_config(args)
    raw_model = Olmo3ForCausalLM(config)
    # Inject gates on CPU before moving to device — new modules are created on CPU.
    inject_stochastic_mlp(raw_model, scale_output=args.scale_output,
                          n_knots=args.n_knots, degree=args.degree)
    inject_stochastic_attn(raw_model, scale_output=args.scale_output,
                           n_knots=args.n_knots, degree=args.degree)
    if not args.no_per_head_norm:
        inject_per_head_qk_norm(raw_model)
    set_gate_mode(raw_model, args.gate_mode)
    if args.embed_amp:
        inject_embedding_amplitude(raw_model, n_knots=args.n_knots, degree=args.degree)

    # When using FSDP, keep the model on CPU here — FSDP will move and shard
    # parameters rank-by-rank via device_id, avoiding a full-model copy on every
    # GPU simultaneously.  For DDP / single-device, move to device immediately.
    use_fsdp_init = args.fsdp and not args.no_ddp and device.type in ("cuda", "xpu")
    if not use_fsdp_init:
        raw_model = raw_model.to(device)



    if rank == 0:
        n_par   = sum(p.numel() for p in set(raw_model.parameters()))
        n_tau   = sum(m.tau_raw.numel() for m in raw_model.modules()
                      if isinstance(m, _STOCH_TYPES))
        n_scale = sum(
            m.gate_scale_raw.numel() + m.output_scale_raw.numel()
            + (m.input_scale_raw.numel() if hasattr(m, "input_scale_raw") else 0)
            for m in raw_model.modules() if isinstance(m, _STOCH_TYPES))
        print(f"  Parameters (unique) : {n_par:,}  ({n_par/1e6:.1f}M)")
        print("  Gates               : FFN neurons + attention heads")
        print(f"  τ params (B-spline) : {n_tau}  |  scale params (gate+output+input): {n_scale}")
        if args.embed_amp:
            amp_params = sum(m.ctrl.numel() for m in raw_model.modules()
                             if isinstance(m, EmbeddingAmplitude))
            print(f"  Embed amplitude     : {amp_params} params (n_knots×hidden)")
        if args.lam_warmup > 0:
            print(f"  λ warmup            : steps 0–{args.lam_warmup}"
                  f"  (scale 0 → 1, then full Exp(rate={args.lam_rate}) sampling)")

    # FSDP supported on CUDA and XPU (requires xccl process group for XPU).
    use_fsdp = args.fsdp and not args.no_ddp and device.type in ("cuda", "xpu")
    # Manual grad sync is no longer needed: xccl process group supports DDP
    # all-reduce on XPU tensors directly.  Kept False to preserve the code path
    # in case of future fallback needs.
    _manual_grad_sync = False

    if use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer

        _strat = {"full": ShardingStrategy.FULL_SHARD,
                  "grad_op": ShardingStrategy.SHARD_GRAD_OP}[args.sharding]

        model = FSDP(
            raw_model,
            sharding_strategy=_strat,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Olmo3DecoderLayer},
            ),
            mixed_precision=MixedPrecision(
                # param_dtype intentionally omitted: parameters stay in fp32.
                # torch.autocast in the training loop handles bf16 compute.
                # Removing param_dtype avoids dtype conflicts in custom autograd
                # functions (e.g. torchcurves B-spline backward) that compute
                # gradients in fp32 and index_add_ into parameter grad tensors.
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=device,
        )
        if args.grad_ckpt:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda m: isinstance(m, Olmo3DecoderLayer),
            )
    elif not args.no_ddp and not _manual_grad_sync:
        if args.grad_ckpt:
            raw_model.gradient_checkpointing_enable()
        ddp_kwargs: dict = {}
        if device.type in ("cuda", "xpu"):
            ddp_kwargs["device_ids"] = [int(os.environ["LOCAL_RANK"])]
        model = DDP(
            raw_model,
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=args.ddp_static_graph,
            gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            **ddp_kwargs,
        )
    else:
        model = raw_model
        if args.grad_ckpt:
            raw_model.gradient_checkpointing_enable()
        if _manual_grad_sync and rank == 0:
            print("  XPU: manual gradient all_reduce via gloo (no DDP)", flush=True)

    # ── penalty weights (computed once from config) ──────────────────────────
    if args.penalty_mode == "flops":
        # SwiGLU FFN: gate+up+down = 3 projections × 2 for MACs
        _ffn_w  = float(6 * config.hidden_size * config.intermediate_size)
        # Attention: Q+K+V+O projections ≈ 8·h² MACs
        _attn_w = float(8 * config.hidden_size * config.hidden_size)
    elif args.penalty_mode == "params":
        _ffn_w  = float(3 * config.hidden_size * config.intermediate_size)
        _attn_w = float(4 * config.hidden_size * config.hidden_size)
    else:  # uniform
        _ffn_w, _attn_w = 1.0, 1.0

    if rank == 0:
        print(f"  penalty_mode={args.penalty_mode}"
              f"  ffn_w={_ffn_w:.2e}  attn_w={_attn_w:.2e}", flush=True)

    # ── data & schedule ──────────────────────────────────────────────────────
    train_loader = build_loader(
        args.data_dir, "train", args.seq_len, args.batch,
        rank, world_size, num_workers=4,
        pin_memory=(device.type in ("cuda", "xpu")))

    grad_accum = args.grad_accum
    n_steps = args.max_steps or (len(train_loader) * args.epochs // grad_accum)
    warmup  = min(WARMUP, n_steps // 10)

    # optimizer AFTER wrapping — FSDP needs to see the sharded parameter views
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / max(1, warmup) if s < warmup
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * (s - warmup) / max(1, n_steps - warmup)))))

    # ── training loop ────────────────────────────────────────────────────────
    step = 0
    if not args.restart and rank == 0 and (Path(args.ckpt_dir) / "model.pt").exists():
        step = _load_checkpoint(args.ckpt_dir, model, opt, sched, device)
    if not args.no_ddp:
        step_t = torch.tensor(step, dtype=torch.long, device=device)
        dist.broadcast(step_t, src=0)
        step = int(step_t.item())

    t0          = time.perf_counter()
    accum_count = 0
    nll_accum   = 0.0
    pen_accum   = 0.0
    lam_last    = 0.0
    lam_scale   = 1.0

    while step < n_steps:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(step)

        for inp, tgt in train_loader:
            if step >= n_steps:
                break

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            # λ=0 (dense pass) with prob 1/3; λ~Exp(rate) otherwise
            if random.random() > 1/3:
                lam_val = random.expovariate(args.lam_rate)
            else:
                lam_val = 0.0
            # λ annealing: linearly scale from 0 → full over lam_warmup steps
            if args.lam_warmup > 0:
                lam_scale = min(1.0, step / args.lam_warmup)
                lam_val  *= lam_scale
            lam_last = lam_val
            lam_t = torch.tensor(lam_val, device=device)
            set_lam(model, lam_t)

            is_sync_step = (accum_count + 1 == grad_accum)
            sync_ctx = (contextlib.nullcontext()
                        if (use_fsdp or args.no_ddp or is_sync_step)
                        else model.no_sync())
            with sync_ctx, torch.autocast(device.type, dtype=torch.bfloat16,
                                          enabled=(device.type in ("cuda", "xpu"))):
                out  = model(input_ids=inp)
                nll  = F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)),
                    tgt.view(-1))
                pen  = collect_penalty(model, args.penalty_mode, _ffn_w, _attn_w)
                nll_eff = pen.detach() * nll if args.weighted_nll else nll
                loss = (nll_eff + lam_val * pen) / grad_accum

            loss.backward()
            nll_accum   += nll.item() / grad_accum
            pen_accum   += pen.item() / grad_accum
            accum_count += 1

            if accum_count < grad_accum:
                continue
            accum_count = 0

            if _manual_grad_sync:
                grads = [p.grad for p in raw_model.parameters()
                         if p.grad is not None]
                if grads:
                    flat_cpu = torch.cat([g.reshape(-1).cpu() for g in grads])
                    dist.all_reduce(flat_cpu, op=dist.ReduceOp.AVG)
                    flat_xpu = flat_cpu.to(device, non_blocking=True)
                    offset = 0
                    for g in grads:
                        n = g.numel()
                        g.copy_(flat_xpu[offset:offset + n].reshape(g.shape))
                        offset += n
            if use_fsdp:
                model.clip_grad_norm_(GRAD_CLIP)
            else:
                nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if rank == 0 and args.lam_warmup > 0 and step == args.lam_warmup:
                print(f"  step {step}: λ warmup complete — full Exp(rate={args.lam_rate}) sampling now active", flush=True)

            if rank == 0 and step % LOG_EVERY == 0:
                tok_s = (LOG_EVERY * args.batch * grad_accum * args.seq_len
                         * world_size / (time.perf_counter() - t0))
                t0 = time.perf_counter()
                _ws = f" [warmup {100 * lam_scale:.0f}%]" if lam_scale < 1.0 else ""
                print(f"  step {step:6d}/{n_steps}"
                      f"  nll={nll_accum:.3f}"
                      f"  pen={pen_accum:.3f}"
                      f"  λ={lam_last:.3f}{_ws}"
                      f"  lr={sched.get_last_lr()[0]:.2e}"
                      f"  {tok_s/1e3:.1f}K tok/s"
                      + (f" ({world_size}×GPU)" if world_size > 1 else ""),
                      flush=True)
            nll_accum = 0.0
            pen_accum = 0.0

            # FSDP requires all ranks to participate in forward; DDP/single: rank 0 only
            if step % EVAL_EVERY == 0:
                if use_fsdp or rank == 0:
                    val_ppl = evaluate(model, args.data_dir, "validation",
                                       device, seq_len=args.seq_len, batch=args.batch)
                if rank == 0:
                    print(f"  step {step}  val_ppl={val_ppl:.1f}", flush=True)

            if step % SAVE_EVERY == 0:
                if use_fsdp or rank == 0:
                    _save(model, config, args, rank, use_fsdp, opt, sched, step)

    # ── final save + λ sweep ─────────────────────────────────────────────────
    if use_fsdp or rank == 0:
        _save(model, config, args, rank, use_fsdp, opt, sched, step)
        _lambda_sweep(model, args, device, rank)


def _save(model: nn.Module, config: Olmo3Config, args,
          rank: int = 0, use_fsdp: bool = False,
          opt: torch.optim.Optimizer | None = None,
          sched=None, step: int = 0) -> None:
    out = Path(args.ckpt_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    if use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, out / "model.pt")
    elif rank == 0:
        raw = model.module if hasattr(model, "module") else model
        torch.save(raw.state_dict(), out / "model.pt")

    if rank == 0:
        config.save_pretrained(out)
        with open(out / "train_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        if opt is not None and not use_fsdp:
            torch.save({
                "step":      step,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None,
            }, out / "train_state.pt")
        print(f"  Saved → {out / 'model.pt'}  (step {step})", flush=True)


def _load_checkpoint(ckpt_dir: str | Path, model: nn.Module,
                     opt: torch.optim.Optimizer, sched,
                     device: torch.device) -> int:
    """Load model weights + optimizer/scheduler state. Returns step to resume from."""
    out = Path(ckpt_dir)
    model_pt = out / "model.pt"
    state_pt  = out / "train_state.pt"
    if not model_pt.exists():
        return 0

    # Verify the saved checkpoint is compatible before loading.
    args_file = out / "train_args.json"
    if args_file.exists():
        saved = json.load(open(args_file))
        raw = model.module if hasattr(model, "module") else model
        base_cfg = getattr(raw, "config", None)
        if base_cfg is not None:
            actual = {
                "hidden":       base_cfg.hidden_size,
                "intermediate": base_cfg.intermediate_size,
                "n_layers":     base_cfg.num_hidden_layers,
            }
            saved_shape = {
                "hidden":       saved.get("hidden"),
                "intermediate": saved.get("intermediate"),
                "n_layers":     saved.get("n_layers"),
            }
            if actual != saved_shape:
                print(f"  WARNING: checkpoint architecture {saved_shape} != current {actual}."
                      f"  Skipping resume — starting from scratch.", flush=True)
                return 0

    raw = model.module if hasattr(model, "module") else model
    # Always load weights to CPU first; load_state_dict copies them to wherever
    # the model parameters currently live.  Loading directly to device fails when
    # the checkpoint was saved from a CPU run (tensors tagged 'cpu:0' rather than
    # a valid XPU/CUDA location).
    raw.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
    step = 0
    if state_pt.exists():
        state = torch.load(state_pt, map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        if sched is not None and state["scheduler"] is not None:
            sched.load_state_dict(state["scheduler"])
        step = state["step"]
        print(f"  Resumed from step {step}", flush=True)
    else:
        print(f"  Loaded weights from {model_pt} (no optimizer state)", flush=True)
    return step


def _lambda_sweep(model: nn.Module, args, device: torch.device,
                  rank: int = 0) -> None:
    # eval uses soft ρ (deterministic): model.eval() makes self.training=False
    # so the forward hits `combined = rho` regardless of gate_mode.
    # Use set_hard_mask(model, True) before this call for hard rho>0.5 sweeps.
    grid = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
    if rank == 0:
        print("\n  λ sweep (soft ρ, deterministic)")
        print(f"  {'lam':>8}  {'val_ppl':>10}  {'ffn%':>7}  {'attn%':>7}")
        print("  " + "-" * 42)
    for lam in grid:
        ppl  = evaluate(model, args.data_dir, "validation", device,
                        lam_val=lam, max_batches=50, seq_len=args.seq_len, batch=args.batch)
        def _mean_rho(cls):
            vals = [m.last_rho_mean.item() for m in model.modules()
                    if isinstance(m, cls) and m.last_rho_mean is not None]
            return float(np.mean(vals)) * 100 if vals else float("nan")

        if rank == 0:
            print(f"  {lam:>8.3f}  {ppl:>10.1f}"
                  f"  {_mean_rho(BSplineScaledOlmo3MLP):>6.1f}%"
                  f"  {_mean_rho(BSplineScaledOlmo3Attn):>6.1f}%", flush=True)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train OLMo-3 mini (~90M) with hyper_joint pruning "
                    "(B-spline τ + per-unit amplitude scaling, FFN + attention)")
    p.add_argument("--data_dir",    required=True,
                   help="Directory with train.bin/validation.bin (uint32, "
                        "from prepare_olmo3_mini.py)")
    p.add_argument("--ckpt_dir",    default="./olmo3_mini_ckpt")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--max_steps",   type=int,   default=None)
    p.add_argument("--batch",       type=int,   default=BATCH)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--lam_rate",    type=float, default=LAM_RATE,
                   help="rate of Exp(rate) λ sampler (mean=1/rate)")
    p.add_argument("--vocab",       type=int,   default=VOCAB)
    p.add_argument("--hidden",      type=int,   default=HIDDEN)
    p.add_argument("--intermediate",type=int,   default=INTERMEDIATE)
    p.add_argument("--n_layers",    type=int,   default=N_LAYERS)
    p.add_argument("--n_heads",     type=int,   default=N_HEADS)
    p.add_argument("--seq_len",     type=int,   default=SEQ_LEN)
    p.add_argument("--rope_theta",  type=float, default=ROPE_THETA)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_ddp",      action="store_true")
    p.add_argument("--fsdp",        action="store_true",
                   help="use FSDP instead of DDP (required for large models)")
    p.add_argument("--sharding",    type=str,   default="full",
                   choices=["full", "grad_op"],
                   help="full=ZeRO-3 (default), grad_op=ZeRO-2")
    p.add_argument("--grad_ckpt",   action="store_true",
                   help="enable activation checkpointing (trades compute for memory)")
    p.add_argument("--grad_accum",  type=int, default=1,
                   help="gradient accumulation steps (amortizes grad sync cost on XPU)")
    p.add_argument("--accelerator", type=str, default="auto",
                   choices=["auto", "cuda", "xpu", "cpu", "mps"])
    p.add_argument("--weighted_nll", action="store_true",
                   help="scale NLL by mean inclusion probability (rho_bar * nll) "
                        "so the CE gradient is discounted when the model is sparse")
    p.add_argument("--penalty_mode", type=str, default="flops",
                   choices=["flops", "params", "uniform"],
                   help="how to weight FFN vs attention in the sparsity penalty "
                        "(flops=SwiGLU·6·h·d vs 8·h², params=3·h·d vs 4·h², "
                        "uniform=unweighted mean)")
    p.add_argument("--gate_mode", type=str, default="ste",
                   choices=["ste", "soft"],
                   help="ste (default): hard Bernoulli mask forward + straight-through "
                        "gradient; soft: multiply each unit by ρ, estimates E[NLL]")
    p.add_argument("--lam_warmup", type=int, default=0,
                   help="steps over which λ is linearly annealed from 0 to its full "
                        "sampled value (0 = no annealing, full λ from step 0)")
    p.add_argument("--no_scale_output", dest="scale_output", action="store_false",
                   default=True,
                   help="disable output amplitude rescaling (f_i(λ) applied to surviving "
                        "neuron/head outputs); on by default")
    p.add_argument("--restart", action="store_true",
                   help="ignore any existing checkpoint in ckpt_dir and train from scratch")
    p.add_argument("--embed_amp", action="store_true",
                   help="add a λ-conditional per-dimension amplitude scale on the "
                        "embedding output (n_knots×hidden params, init → scale≡1)")
    p.add_argument("--ddp_static_graph", action="store_true",
                   help="DDP static_graph=True: pre-schedule bucket comms when the "
                        "parameter graph is fixed every step. Disable if any module "
                        "is conditionally skipped (e.g. Bernoulli gate drops).")
    p.add_argument("--ddp_gradient_as_bucket_view", action="store_true",
                   help="DDP gradient_as_bucket_view=True: gradients alias bucket "
                        "memory directly, eliminating one copy per all-reduce.")
    p.add_argument("--ddp_bucket_cap_mb", type=int, default=25,
                   help="DDP bucket_cap_mb: max size of each gradient all-reduce "
                        "bucket in MB (default PyTorch=25; try 100–200 on PCIe to "
                        "reduce the number of round-trips).")
    p.add_argument("--no_per_head_norm", action="store_true",
                   help="skip inject_per_head_qk_norm; use the flat Olmo3RMSNorm "
                        "for q/k (structural head pruning becomes approximate rather "
                        "than exact)")
    p.add_argument("--n_knots", type=int, default=8,
                   help="number of B-spline control points for τ(λ) and amplitude "
                        "splines (default 8; must be > degree)")
    p.add_argument("--degree",  type=int, default=3,
                   help="polynomial degree for B-spline τ(λ) and amplitude splines "
                        "(default 3, cubic); must be < n_knots")
    args = p.parse_args()
    if args.degree >= args.n_knots:
        p.error(f"--degree ({args.degree}) must be < --n_knots ({args.n_knots})")
    return args


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    accel = args.accelerator if args.accelerator != "auto" else _detect_accelerator()

    if args.no_ddp:
        rank, world_size = 0, 1
        device = torch.device("cpu") if accel == "cpu" else torch.device(f"{accel}:0")
        if accel == "cuda":
            torch.cuda.set_device(device)
        elif accel == "xpu":
            torch.xpu.set_device(device)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = torch.device(f"{accel}:{local_rank}")
        if accel == "cuda":
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
        elif accel == "xpu":
            torch.xpu.set_device(device)
            dist.init_process_group("xccl", device_id=device)
        else:
            dist.init_process_group("gloo")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()

    outdir = Path(args.ckpt_dir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print("  OLMo-3 Mini — B-spline λ→τ + amplitude scaling (hyper_joint)")
        print(f"  hidden={args.hidden}  intermediate={args.intermediate}"
              f"  layers={args.n_layers}  heads={args.n_heads}  seq={args.seq_len}")
        print(f"  vocab={args.vocab}  rope_theta={args.rope_theta}")
        print(f"  device={device}  world_size={world_size}"
              f"  {'FSDP/' + args.sharding if args.fsdp else 'DDP'}")
        print(f"  grad_ckpt={args.grad_ckpt}")
        print(f"  λ=0 (prob 1/3) or Exp(rate={args.lam_rate})  mean_λ={1/args.lam_rate:.1f}"
              + (f"  lam_warmup={args.lam_warmup}" if args.lam_warmup > 0 else ""))
        _gate_desc = {
            "ste":  "hard Bernoulli + STE (training)  |  soft ρ deterministic (eval)",
            "soft": "soft ρ deterministic (training + eval)",
        }
        print(f"  gate_mode={args.gate_mode}: {_gate_desc[args.gate_mode]}")
        print(f"  penalty_mode={args.penalty_mode}")
        print(f"  scale_output={'on (f_i(λ) applied to surviving units)' if args.scale_output else 'off'}")
        print("=" * 70, flush=True)

    train(args, rank, world_size, device)

    if not args.no_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
