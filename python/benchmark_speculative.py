# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "transformers>=5.0",
#   "numpy>=1.24",
#   "matplotlib>=3.7",
#   "torchcurves",
# ]
# ///
"""
benchmark_speculative.py

Two modes:

  sweep  (default) — generative timing sweep over draft λ values.
  optimize         — analytical optimization of (λ_draft, γ) using
                     Theorems 3.8 and 3.11 of Leviathan et al. (2023).

Sweep mode produces a plot with:
  - Token throughput (tok/s) vs draft λ
  - Accept rate (%) vs draft λ
  - Expected tokens per target call vs draft λ

Optimize mode produces a 4-panel plot:
  - α(λ_draft)  — theoretical acceptance rate 1 − TV(p,q)
  - c(λ_draft)  — relative draft cost (FFN-weighted param fraction)
  - S*(λ_draft) — optimal wall-clock speedup at best γ* (Th. 3.8)
  - γ*(λ_draft) — optimal draft length (Th. 3.11)
  plus a 2-D heatmap of S(γ, λ_draft).

Usage:
    uv run python python/benchmark_speculative.py --ckpt_dir ./results/olmo3 --tokenizer_dir ./results/olmo3/tokenizer --lam_target 0.1 --gamma 4 --device cpu

    # optimize mode (fast, analytical):
    uv run python python/benchmark_speculative.py --ckpt_dir ./results/olmo3 --mode optimize --lam_target 0.1 --n_lam 60 --device mps --out opt.png
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Model-type is resolved early from argv so the right module is imported
# before any function references are bound.
_MODEL_TYPE = "olmo3"
for _i, _a in enumerate(sys.argv):
    if _a == "--model_type" and _i + 1 < len(sys.argv):
        _MODEL_TYPE = sys.argv[_i + 1]

if _MODEL_TYPE == "stoch":
    from stoch_speculative_decode import (
        active_fraction,
        build_compressed_olmo3,
        commit_masks,
        expected_acceptance_rate,
        generate_greedy,
        generate_speculative,
        load_checkpoint,
    )
else:
    from olmo3_speculative_decode import (
        _logprobs_at_all,
        active_fraction,
        build_compressed_olmo3,
        commit_masks,
        expected_acceptance_rate,
        generate_greedy,
        generate_speculative,
        load_checkpoint,
        set_hard_mask,
        set_lam,
        set_sample_mask,
        stochastic_expected_acceptance_rate,
    )


# ── draft-model factory (optional torch.compile) ─────────────────────────────

_compile_draft: bool = False   # set to True in main() via --compile

def _build_draft(model: "nn.Module", lam_val: float,
                 device: "torch.device") -> "nn.Module":
    """Structurally pruned draft model, optionally wrapped with torch.compile.

    structural_prune() (called inside build_compressed_olmo3) physically slices
    inactive rows/columns from weight matrices, giving real FLOP and memory
    savings.  torch.compile then fuses kernels on top of the smaller matrices.
    Only applied on CUDA/XPU — the warmup cost dominates on CPU.
    """
    m = build_compressed_olmo3(model, lam_val, device)
    if _compile_draft:
        if device.type in ("cuda", "xpu"):
            try:
                m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"  torch.compile failed ({e}); running in eager mode", flush=True)
    return m


# ── H(f_s): norm-entropy diagnostic ──────────────────────────────────────────

def compute_norm_entropy(model, n_bins: int = 100) -> dict:
    """Differential entropy of the gate-projection row-norm distribution.

    The λ-sparsity curve satisfies dρ̄/dτ = −f_s(τ), so the smoothness of the
    curve is set by the spread of f_s.  The differential entropy

        H(f_s) = −∫ f_s(t) log f_s(t) dt

    measures that spread:  high H → norms spread out → smooth sparsity curve;
    low / negative H → norms concentrated → cliff behavior.

    Also reports the coefficient of variation CV = σ/μ as a scale-free proxy.
    """
    norms_list = []
    for name, param in model.named_parameters():
        if "gate_proj" in name and param.dim() == 2:
            norms_list.append(param.detach().float().norm(dim=1).cpu().numpy())

    # stoch model fallback: ffn.expand.weight drives the gate
    if not norms_list:
        for name, param in model.named_parameters():
            if "ffn.expand.weight" in name and param.dim() == 2:
                norms_list.append(param.detach().float().norm(dim=1).cpu().numpy())

    if not norms_list:
        return {}

    all_norms = np.concatenate(norms_list)

    # histogram-based differential entropy estimate
    counts, bin_edges = np.histogram(all_norms, bins=n_bins, density=True)
    bin_width = float(bin_edges[1] - bin_edges[0])
    nz = counts > 0
    H = float(-np.sum(counts[nz] * np.log(counts[nz]) * bin_width))

    cv = float(all_norms.std() / all_norms.mean())

    return {
        "H_fs":      H,
        "cv":        cv,
        "mean":      float(all_norms.mean()),
        "std":       float(all_norms.std()),
        "n_neurons": int(len(all_norms)),
        "norms":     all_norms,
        "counts":    counts,
        "bin_edges": bin_edges,
    }


# ── Theorems 3.8 / 3.11: speedup and optimal γ ───────────────────────────────

def speedup_th38(alpha: float, gamma: float, c: float) -> float:
    """Wall-clock speedup from Theorem 3.8 of Leviathan et al. (2023).

    S(γ, α, c) = [(1 − α^{γ+1}) / (1 − α)] / (γc + 1)

    where α is the per-token acceptance rate and c = t_draft / t_target.
    """
    if alpha >= 1.0 - 1e-9:
        # L'Hôpital limit as α→1
        return (gamma + 1.0) / (gamma * c + 1.0)
    return (1.0 - alpha ** (gamma + 1)) / ((1.0 - alpha) * (gamma * c + 1.0))


def optimal_gamma_th311(alpha: float, c: float, gamma_max: int = 500) -> tuple[int, float]:
    """γ* that maximises wall-clock speedup (Theorem 3.11).

    The optimal γ satisfies the transcendental equation derived from
    dS/dγ = 0:

        −α^{γ*+1} ln α · (γ*c + 1) = c (1 − α^{γ*+1})

    S is unimodal in γ so a simple integer grid search is exact.

    Returns (gamma_star, S_star).
    """
    best_g, best_s = 1, speedup_th38(alpha, 1, c)
    for g in range(2, gamma_max + 1):
        s = speedup_th38(alpha, g, c)
        if s > best_s:
            best_s = s
            best_g = g
        else:
            break  # unimodal: first decrease → global max found
    return best_g, best_s


def _ffn_param_fraction(model) -> float:
    """Fraction of layer compute parameters that belong to FFN layers.

    Used to convert active-neuron fraction to a compute-cost ratio c.
    Gate, up, and down projection weights are counted as FFN; attention
    projections (q/k/v/o) are counted as fixed layer overhead.

    Embedding and LM head parameters are excluded from the denominator:
    the embedding lookup is free (no multiply), and the LM head projection
    costs the same for both draft and target per token, so including either
    in the total would artificially inflate the 'fixed' fraction and make c
    spuriously close to 1.
    """
    _embed_lm = {"embed_tokens", "lm_head"}
    ffn_names  = {"gate_proj", "up_proj", "down_proj"}

    ffn_params   = sum(p.numel() for name, p in model.named_parameters()
                       if any(n in name for n in ffn_names))
    # stoch model fallback: ffn.expand + ffn.contract
    if ffn_params == 0:
        ffn_params = sum(p.numel() for name, p in model.named_parameters()
                         if "ffn.expand" in name or "ffn.contract" in name)

    # exclude embedding / lm_head from the denominator
    layer_total = sum(p.numel() for name, p in model.named_parameters()
                      if not any(n in name for n in _embed_lm))
    return ffn_params / layer_total if layer_total > 0 else 0.0


def draft_cost_ratio(ffn_active_draft: float, ffn_active_target: float,
                     ffn_frac: float) -> float:
    """Relative cost of one draft forward pass vs one target forward pass.

    Compute is split between FFN (prunable) and the rest (fixed).
    c = (fixed + ffn_active_draft * ffn_frac) /
        (fixed + ffn_active_target * ffn_frac)
    """
    fixed = 1.0 - ffn_frac
    return (fixed + ffn_active_draft * ffn_frac) / (fixed + ffn_active_target * ffn_frac)


# ── RAM-budget multi-draft model ──────────────────────────────────────────────

def model_bytes(m) -> int:
    """Total weight bytes of a model (deduplicated over shared/tied params)."""
    return sum(p.numel() * p.element_size() for p in set(m.parameters()))


def kv_cache_bytes(config, seq_len: int, bytes_per_elem: int,
                   n_kv_heads: int | None = None) -> int:
    """Approximate KV-cache bytes for one sequence at a given context length.

    KV = 2 (k,v) × L layers × n_kv_heads × head_dim × seq_len × bytes/elem.
    """
    L        = config.num_hidden_layers
    n_kv     = n_kv_heads or getattr(config, "num_key_value_heads",
                                     config.num_attention_heads)
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    return 2 * L * int(n_kv) * head_dim * seq_len * bytes_per_elem


def tree_tokens_per_round(alpha_eff: float, gamma: int) -> float:
    """Expected accepted tokens per verification round = (1−α^{γ+1})/(1−α)."""
    if alpha_eff >= 1.0 - 1e-9:
        return float(gamma + 1)
    return (1.0 - alpha_eff ** (gamma + 1)) / (1.0 - alpha_eff)


def best_gamma_tree(alpha_eff: float, c: float, draft_passes_per_token: float,
                    gamma_max: int = 64) -> tuple[int, float]:
    """γ* and speedup* for a multi-draft round.

    cost per round (in target-pass units) = draft_passes_per_token · γ · c + 1,
    where draft_passes_per_token is K for K distinct subnets that must each be
    run (and their weights streamed) every step, or 1 for a shared-weight masked
    batch that scores all K candidates in a single forward pass. The +1 is the
    single dense target verification pass (memory-bound, ≈ constant in width).
    """
    best_g, best_s = 1, -1.0
    for g in range(1, gamma_max + 1):
        cost = draft_passes_per_token * g * c + 1.0
        s = tree_tokens_per_round(alpha_eff, g) / cost
        if s > best_s:
            best_g, best_s = g, s
    return best_g, best_s


def sweep_ram_budget(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    budget_gb: float,
    kv_seq_len: int,
    temperature: float,
    device: torch.device,
    gamma_max: int = 64,
    k_max: int = 256,
    target_model=None,
    consensus: bool = True,
) -> list[dict]:
    """Analytical multi-draft speedup under a fixed GPU memory budget.

    For each draft λ we hold one dense target plus as many drafts as fit in the
    budget, and compare five operating points at the budget-derived K(λ):

      single    — one draft (K=1), today's baseline.
      distinct  — K(λ) physically pruned copies under the budget; each copy is a
                  separate weight set, so the draft phase streams K weight sets
                  per token (draft_passes = K).
      shared    — same K candidates but produced by masking ONE dense weight set
                  in a batched pass (draft_passes = 1). This is the counterfactual
                  that shows where the parallel-candidate benefit actually comes
                  from; it uses 1× weight RAM, not K×.
      selfspec  — self-speculative decoding: the draft is a *subnet of the dense
                  target*, sharing its weights AND KV cache with the verifier.
                  Because weights are not replicated, the marginal cost of each
                  extra candidate is just its KV slice (no S_draft term), so many
                  more candidates fit (K_self ≫ K) for the same budget. Verify is
                  exact (standard accept/reject), draft_passes = 1 (shared
                  weights). This is the "free" exact speedup: at K_self = 1 it is
                  ordinary single-draft self-speculation, and it is the only exact
                  point that still works when a separate draft copy would not fit.
      consensus — the K distinct copies vote: if all K greedily agree on the next
                  token it is emitted directly, otherwise the full model is run
                  for that token. Unlike the other four (exact spec-decoding),
                  this is *approximate* — agreed tokens are committed unverified,
                  so speedup_consensus comes with a measured token_error.

    single/distinct/shared are analytical: K diverse drafts use the effective
    acceptance α_K = 1−(1−α)^K (independent-draft idealisation, an optimistic
    upper bound). The consensus point is *empirical* — it samples K stochastic
    subnets and measures their true agreement, so it does not assume
    independence. Set consensus=False to skip the sampling and keep the sweep
    fully analytical / fast.
    """
    ffn_frac       = _ffn_param_fraction(model)
    tgt            = target_model if target_model is not None else model
    bytes_per_elem = next(model.parameters()).element_size()
    S_target       = model_bytes(tgt)
    KV_target      = kv_cache_bytes(tgt.config, kv_seq_len, bytes_per_elem)
    budget_bytes   = int(budget_gb * (1024 ** 3))
    n_kv_full      = getattr(tgt.config, "num_key_value_heads",
                             tgt.config.num_attention_heads)

    if target_model is None:
        frac_target = active_fraction(model, lam_target, device)["ffn"]
        tgt_lam     = lam_target
    else:
        frac_target = 1.0
        tgt_lam     = None

    # Consensus point needs a stochastic draft sampler and the target's greedy
    # tokens (computed once — independent of λ_draft).
    draft_sampler = None
    target_tok    = None
    if consensus:
        if target_model is None:
            commit_masks(model, lam_target, device)
        target_tok = _logprobs_at_all(tgt, prompt, temperature).argmax(dim=-1)
        draft_sampler, n_stoch = _make_stoch_sampler(model, device)
        if n_stoch == 0:
            print("  WARNING: no stochastic gate modules — consensus point will be "
                  "degenerate (p_agree≡1).")

    print(f"  budget={budget_gb:.0f} GB  target={S_target/1e9:.2f} GB  "
          f"KV/seq(@{kv_seq_len})={KV_target/1e9:.3f} GB  "
          f"avail for drafts={(budget_bytes-S_target-KV_target)/1e9:.2f} GB")

    results = []
    for lam_draft in lam_draft_values:
        fr         = active_fraction(model, lam_draft, device)
        frac_draft = fr["ffn"]
        attn_draft = fr.get("attn", 1.0)
        attn_draft = 1.0 if attn_draft != attn_draft else attn_draft  # NaN→1
        c          = draft_cost_ratio(frac_draft, frac_target, ffn_frac)

        draft_model = _build_draft(model, lam_draft, device)
        if target_model is None:
            commit_masks(model, lam_target, device)
        alpha, _ = expected_acceptance_rate(
            tgt, draft_model, prompt, tgt_lam, temperature, device)

        S_draft  = model_bytes(draft_model)
        KV_draft = kv_cache_bytes(tgt.config, kv_seq_len, bytes_per_elem,
                                  n_kv_heads=max(1, round(n_kv_full * attn_draft)))
        avail    = budget_bytes - S_target - KV_target
        per_copy = S_draft + KV_draft
        K        = int(avail // per_copy) if per_copy > 0 else 1
        K        = max(1, min(K, k_max))

        alpha_K = 1.0 - (1.0 - alpha) ** K

        # self-speculative: the draft is a subnet of the dense target (shared
        # weights + shared KV), so a candidate costs only its KV slice — no
        # S_draft term. Hence many more candidates fit than the weight-limited K.
        per_self    = max(1, KV_draft)
        K_self      = int(avail // per_self) if avail > 0 else 1
        K_self      = max(1, min(K_self, k_max))
        alpha_Kself = 1.0 - (1.0 - alpha) ** K_self

        g1,  s1  = best_gamma_tree(alpha,       c, 1.0,      gamma_max)   # single
        gKd, sKd = best_gamma_tree(alpha_K,     c, float(K), gamma_max)   # distinct
        gKs, sKs = best_gamma_tree(alpha_K,     c, 1.0,      gamma_max)   # shared
        gSe, sSe = best_gamma_tree(alpha_Kself, c, 1.0,      gamma_max)   # self-spec

        row = {
            "lam_draft":        lam_draft,
            "alpha":            alpha,
            "alpha_K":          alpha_K,
            "alpha_Kself":      alpha_Kself,
            "c":                c,
            "k_budget":         K,
            "k_selfspec":       K_self,
            "draft_mb":         S_draft / 1e6,
            "target_mb":        S_target / 1e6,
            "ffn_active":       frac_draft,
            "attn_active":      attn_draft,
            "gamma_single":     g1,
            "speedup_single":   s1,
            "gamma_distinct":   gKd,
            "speedup_distinct": sKd,
            "gamma_shared":     gKs,
            "speedup_shared":   sKs,
            "gamma_selfspec":   gSe,
            "speedup_selfspec": sSe,
        }

        if consensus:
            set_lam(draft_sampler,
                    torch.tensor(lam_draft, dtype=torch.float32, device=device))
            draft_tok = _sample_draft_tokens(draft_sampler, prompt, temperature,
                                              K, device)
            p_agree, p_corr_ag, token_err = _consensus_metrics(draft_tok, target_tok, K)
            cost_cons = K * c + (1.0 - p_agree)
            s_cons    = 1.0 / cost_cons if cost_cons > 0 else float("inf")
            row.update({
                "p_agree":          p_agree,
                "p_correct_agree":  p_corr_ag,
                "token_error":      token_err,
                "speedup_consensus": s_cons,
            })

        results.append(row)
        cons_msg = (f" consensus={row['speedup_consensus']:.2f}"
                    f"(err={row['token_error']:.3f})" if consensus else "")
        print(f"  λ={lam_draft:.3g}  draft={S_draft/1e9:.2f}GB  K={K:>3d}  "
              f"α={alpha:.3f}→α_K={alpha_K:.3f}  c={c:.3f}  "
              f"S: single={s1:.2f} distinct={sKd:.2f} shared={sKs:.2f} "
              f"selfspec={sSe:.2f}(K_self={K_self}){cons_msg}")

    return results


def make_ram_budget_plot(results: list[dict], budget_gb: float,
                         kv_seq_len: int, out_path: Path) -> None:
    """Two-panel plot: speedup (single/distinct/shared) and K(λ) vs draft λ."""
    lam      = np.array([r["lam_draft"]        for r in results])
    s_single = np.array([r["speedup_single"]   for r in results])
    s_dist   = np.array([r["speedup_distinct"] for r in results])
    s_shared = np.array([r["speedup_shared"]   for r in results])
    K        = np.array([r["k_budget"]         for r in results])

    fig, (ax_s, ax_k) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    fig.suptitle(f"Multi-draft speculative decoding under a "
                 f"{budget_gb:.0f} GB budget  (KV @ {kv_seq_len} tok)",
                 fontsize=12, y=0.95)

    ax_s.axhline(1.0, color="grey", lw=1, ls=":")
    ax_s.plot(lam, s_single, "-o", color="#555555", lw=1.8, ms=4,
              label="single draft (K=1)")
    ax_s.plot(lam, s_dist, "-o", color="#e08214", lw=1.8, ms=4,
              label="budget / distinct pruned copies (K weight loads)")
    ax_s.plot(lam, s_shared, "--o", color="#2166ac", lw=1.8, ms=4,
              label="shared-weights reference (1 weight load, K masks)")
    if all("speedup_selfspec" in r for r in results):
        s_self = np.array([r["speedup_selfspec"] for r in results])
        ax_s.plot(lam, s_self, "-^", color="#1a9850", lw=1.8, ms=4,
                  label="self-speculative (subnet draft, shared weights+KV, exact)")
    if all("speedup_consensus" in r for r in results):
        s_cons = np.array([r["speedup_consensus"] for r in results])
        ax_s.plot(lam, s_cons, "-s", color="#b2182b", lw=1.8, ms=4,
                  label="consensus (K agree → emit, else full; approximate)")
    ax_s.set_ylabel("Analytical speedup over baseline")
    ax_s.legend(loc="best", fontsize=9, frameon=False)
    ax_s.set_xscale("log")

    ax_k.plot(lam, K, "-o", color="#4dac26", lw=1.8, ms=4,
              label="distinct copies (weight-limited)")
    if all("k_selfspec" in r for r in results):
        ax_k.plot(lam, [r["k_selfspec"] for r in results], "-^",
                  color="#1a9850", lw=1.5, ms=3,
                  label="self-spec candidates (KV-limited)")
        ax_k.legend(loc="best", fontsize=8, frameon=False)
    ax_k.set_ylabel(f"candidates that fit, K(λ)")
    ax_k.set_xlabel(r"$\lambda_{draft}$  (larger = smaller draft, more copies fit)")
    ax_k.set_yscale("log")
    ax_k.set_xscale("log")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote plot → {out_path}")


# ── K × λ grid sweep ──────────────────────────────────────────────────────────

def sweep_k_lambda_grid(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    k_values: list[int],
    temperature: float,
    device: torch.device,
    gamma_max: int = 64,
    target_model=None,
) -> list[dict]:
    """Analytical speedup on a full (λ_draft × K) grid.

    Unlike ``ram_budget`` (which derives a single K(λ) from a memory budget),
    this sweeps K explicitly as an independent axis. α and c depend only on the
    draft λ, so each draft model is built once per λ and the K loop is a cheap
    closed-form evaluation. For each cell we report the same three operating
    points as the budget sweep:

      single    — one draft (K=1) baseline (constant across K, kept per row).
      distinct  — K physically pruned copies; draft phase streams K weight sets
                  per token (draft_passes = K).
      shared    — same K candidates from ONE masked dense copy in a batched pass
                  (draft_passes = 1).

    K diverse drafts use the independent-draft idealisation α_K = 1−(1−α)^K (an
    optimistic upper bound — real subnets of one checkpoint share weights and so
    their errors correlate).
    """
    ffn_frac = _ffn_param_fraction(model)
    tgt      = target_model if target_model is not None else model

    if target_model is None:
        frac_target = active_fraction(model, lam_target, device)["ffn"]
        tgt_lam     = lam_target
    else:
        frac_target = 1.0
        tgt_lam     = None

    print(f"  grid: {len(lam_draft_values)} λ × {len(k_values)} K "
          f"= {len(lam_draft_values) * len(k_values)} cells  (γ_max={gamma_max})")

    results = []
    for lam_draft in lam_draft_values:
        fr         = active_fraction(model, lam_draft, device)
        frac_draft = fr["ffn"]
        attn_draft = fr.get("attn", 1.0)
        attn_draft = 1.0 if attn_draft != attn_draft else attn_draft  # NaN→1
        c          = draft_cost_ratio(frac_draft, frac_target, ffn_frac)

        draft_model = _build_draft(model, lam_draft, device)
        if target_model is None:
            commit_masks(model, lam_target, device)
        alpha, _ = expected_acceptance_rate(
            tgt, draft_model, prompt, tgt_lam, temperature, device)
        S_draft  = model_bytes(draft_model)

        g1, s1 = best_gamma_tree(alpha, c, 1.0, gamma_max)   # single, K-independent

        for K in k_values:
            alpha_K  = 1.0 - (1.0 - alpha) ** K
            gKd, sKd = best_gamma_tree(alpha_K, c, float(K), gamma_max)  # distinct
            gKs, sKs = best_gamma_tree(alpha_K, c, 1.0,      gamma_max)  # shared
            results.append({
                "lam_draft":        lam_draft,
                "K":                K,
                "alpha":            alpha,
                "alpha_K":          alpha_K,
                "c":                c,
                "draft_mb":         S_draft / 1e6,
                "ffn_active":       frac_draft,
                "attn_active":      attn_draft,
                "gamma_single":     g1,
                "speedup_single":   s1,
                "gamma_distinct":   gKd,
                "speedup_distinct": sKd,
                "gamma_shared":     gKs,
                "speedup_shared":   sKs,
            })
        print(f"  λ={lam_draft:.3g}  draft={S_draft/1e9:.2f}GB  α={alpha:.3f}  "
              f"c={c:.3f}  shared S: K={k_values[0]}→{results[-len(k_values)]['speedup_shared']:.2f}  "
              f"K={k_values[-1]}→{results[-1]['speedup_shared']:.2f}")

    return results


def make_k_lambda_grid_plot(results: list[dict], out_path: Path) -> None:
    """Two heatmaps (distinct, shared) of analytical speedup over (λ_draft, K)."""
    lam_vals = sorted({r["lam_draft"] for r in results})
    k_vals   = sorted({r["K"] for r in results})
    li = {v: i for i, v in enumerate(lam_vals)}
    ki = {v: i for i, v in enumerate(k_vals)}

    Z_dist   = np.full((len(k_vals), len(lam_vals)), np.nan)
    Z_shared = np.full((len(k_vals), len(lam_vals)), np.nan)
    for r in results:
        Z_dist[ki[r["K"]], li[r["lam_draft"]]]   = r["speedup_distinct"]
        Z_shared[ki[r["K"]], li[r["lam_draft"]]] = r["speedup_shared"]

    lam = np.array(lam_vals)
    K   = np.array(k_vals)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Analytical speculative-decoding speedup over the (λ, K) grid",
                 fontsize=12, y=0.98)
    for ax, Z, title in ((axes[0], Z_dist, "distinct (K weight loads)"),
                         (axes[1], Z_shared, "shared (1 weight load, K masks)")):
        vmax = max(1.01, float(np.nanmax(Z)))
        im = ax.pcolormesh(lam, K, Z, cmap="RdYlGn", shading="nearest",
                           vmin=2.0 - vmax, vmax=vmax)
        ax.contour(lam, K, Z, levels=[1.0], colors="black", linewidths=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\lambda_{draft}$")
        fig.colorbar(im, ax=ax, label="speedup")
    axes[0].set_ylabel("K (number of drafts)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote plot → {out_path}")


# ── consensus (K-agreement) grid sweep ────────────────────────────────────────

def _device_sync(device: torch.device) -> None:
    """Block until queued device work finishes — needed for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def _make_stoch_sampler(model, device):
    """Deep-copy of ``model`` in stochastic-sampling mode (fresh subnet/forward).

    Returns (sampler, n_stoch) where n_stoch is the number of gate modules that
    actually sample — 0 means the copies will be identical (not a valid draft
    ensemble for the consensus method).
    """
    import copy
    sampler = copy.deepcopy(model)
    set_hard_mask(sampler, hard=False)
    set_sample_mask(sampler, sample=True)
    n_stoch = sum(1 for m in sampler.modules()
                  if hasattr(m, "sample") and getattr(m, "sample"))
    return sampler, n_stoch


@torch.no_grad()
def _sample_draft_tokens(sampler, prompt, temperature, n_subnets, device):
    """Greedy next-token row from each of ``n_subnets`` freshly-sampled subnets.

    Returns an (n_subnets, T) int64 tensor — row k is subnet k's argmax token at
    every position of ``prompt`` (teacher-forced, one forward pass per subnet).
    """
    toks = torch.empty((n_subnets, prompt.numel()), dtype=torch.long, device=device)
    for k in range(n_subnets):
        toks[k] = _logprobs_at_all(sampler, prompt, temperature).argmax(dim=-1)
    return toks


def _consensus_metrics(draft_tok, target_tok, K):
    """Agreement statistics among the first ``K`` sampled subnets.

    Returns (p_agree, p_correct_given_agree, token_error):
      p_agree     — fraction of positions where subnets 0..K-1 all agree.
      p_corr_ag   — P(agreed token == target greedy token | agreed).
      token_error — fraction of positions emitted wrong ( = p_agree·(1−p_corr) ).
    """
    agree   = (draft_tok[:K] == draft_tok[0]).all(dim=0)        # (T,)
    n_agree = int(agree.sum().item())
    T       = target_tok.numel()
    p_agree = n_agree / T
    agreed_ok = agree & (draft_tok[0] == target_tok)
    p_corr_ag = (int(agreed_ok.sum().item()) / n_agree) if n_agree else float("nan")
    token_err = (agree & (draft_tok[0] != target_tok)).float().mean().item()
    return p_agree, p_corr_ag, token_err


def sweep_consensus_grid(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    k_values: list[int],
    temperature: float,
    device: torch.device,
    target_model=None,
) -> list[dict]:
    """Consensus decoding over a (λ_draft × K) grid using stochastic drafts.

    Decoding rule, evaluated per token position:

      1. Draw K *distinct* subnets by Bernoulli-sampling the stochastic gates at
         λ_draft — each forward pass yields one independent subnet.
      2. If all K subnets greedily agree on the next token, emit it directly
         WITHOUT running the full model: the K-way self-consistency is the
         confidence gate that replaces verification.
      3. If they disagree, fall back to one full target forward pass for that
         token (which we treat as the ground-truth greedy decode).

    This is NOT exact / distribution-preserving speculative decoding: agreed
    tokens are committed unverified, so the K drafts can agree on a *wrong*
    token. We therefore report, per (λ, K) cell:

        p_agree            fraction of positions where all K subnets agree
        p_correct_agree    P(agreed token == target greedy token | agreed)
        token_error        fraction of emitted tokens that differ from target
                           greedy  ( = p_agree · (1 − p_correct_agree) )

    Cost model in target-pass units (K distinct copies ⇒ K weight loads/token):

        E[cost/token] = K · c + (1 − p_agree)            (drafts always; full
                                                          model only on miss)
        speedup       = 1 / (K · c + 1 − p_agree)        baseline = 1 (always full)

    where c is the per-pass draft/target cost ratio. The drafts are sampled
    stochastically, so increasing K both *raises* the cost (K·c) and *tightens*
    the agreement gate (fewer false commits, lower p_agree) — the grid exposes
    where that trade-off pays off.
    """
    ffn_frac = _ffn_param_fraction(model)
    tgt      = target_model if target_model is not None else model
    K_max    = max(k_values)

    if target_model is None:
        commit_masks(model, lam_target, device)
        frac_target = active_fraction(model, lam_target, device)["ffn"]
    else:
        frac_target = 1.0

    # Target greedy tokens — independent of λ_draft, so computed once.
    target_tok = _logprobs_at_all(tgt, prompt, temperature).argmax(dim=-1)   # (T,)

    # One stochastic draft sampler reused across λ (set_lam each iteration).
    draft_m, n_stoch = _make_stoch_sampler(model, device)
    if n_stoch == 0:
        print("  WARNING: no stochastic gate modules found — the K copies will be "
              "identical (p_agree≡1). This mode needs a stochastically-gated model.")

    print(f"  grid: {len(lam_draft_values)} λ × {len(k_values)} K  "
          f"(sampling {K_max} subnets per λ)")

    results = []
    for lam_draft in lam_draft_values:
        set_lam(draft_m, torch.tensor(lam_draft, dtype=torch.float32, device=device))
        frac_draft = active_fraction(model, lam_draft, device)["ffn"]
        c          = draft_cost_ratio(frac_draft, frac_target, ffn_frac)

        draft_tok = _sample_draft_tokens(draft_m, prompt, temperature, K_max, device)
        top1_acc  = (draft_tok[0] == target_tok).float().mean().item()

        for K in k_values:
            p_agree, p_corr_ag, token_err = _consensus_metrics(draft_tok, target_tok, K)
            cost = K * c + (1.0 - p_agree)
            results.append({
                "lam_draft":       lam_draft,
                "K":               K,
                "c":               c,
                "ffn_active":      frac_draft,
                "draft_top1_acc":  top1_acc,
                "p_agree":         p_agree,
                "p_correct_agree": p_corr_ag,
                "token_error":     token_err,
                "cost_per_token":  cost,
                "speedup":         1.0 / cost if cost > 0 else float("inf"),
            })
        r_lo, r_hi = results[-len(k_values)], results[-1]
        print(f"  λ={lam_draft:.3g}  c={c:.3f}  top1={top1_acc:.3f}  "
              f"K={r_lo['K']}: agree={r_lo['p_agree']:.2f} err={r_lo['token_error']:.3f} "
              f"S={r_lo['speedup']:.2f}  |  "
              f"K={r_hi['K']}: agree={r_hi['p_agree']:.2f} err={r_hi['token_error']:.3f} "
              f"S={r_hi['speedup']:.2f}")

    return results


def make_consensus_grid_plot(results: list[dict], out_path: Path) -> None:
    """Two heatmaps over (λ_draft, K): consensus speedup and token error rate."""
    lam_vals = sorted({r["lam_draft"] for r in results})
    k_vals   = sorted({r["K"] for r in results})
    li = {v: i for i, v in enumerate(lam_vals)}
    ki = {v: i for i, v in enumerate(k_vals)}

    Z_spd = np.full((len(k_vals), len(lam_vals)), np.nan)
    Z_err = np.full((len(k_vals), len(lam_vals)), np.nan)
    for r in results:
        Z_spd[ki[r["K"]], li[r["lam_draft"]]] = r["speedup"]
        Z_err[ki[r["K"]], li[r["lam_draft"]]] = r["token_error"]

    lam = np.array(lam_vals)
    K   = np.array(k_vals)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Consensus decoding (K stochastic drafts) over the (λ, K) grid",
                 fontsize=12, y=0.98)

    vmax = max(1.01, float(np.nanmax(Z_spd)))
    im0 = axes[0].pcolormesh(lam, K, Z_spd, cmap="RdYlGn", shading="nearest",
                             vmin=2.0 - vmax, vmax=vmax)
    axes[0].contour(lam, K, Z_spd, levels=[1.0], colors="black", linewidths=1.0)
    axes[0].set_title("speedup over full model", fontsize=10)
    fig.colorbar(im0, ax=axes[0], label="speedup")

    im1 = axes[1].pcolormesh(lam, K, Z_err, cmap="Reds", shading="nearest",
                             vmin=0.0, vmax=max(1e-3, float(np.nanmax(Z_err))))
    axes[1].set_title("token error rate vs target greedy", fontsize=10)
    fig.colorbar(im1, ax=axes[1], label="token error")

    for ax in axes:
        ax.set_xlabel(r"$\lambda_{draft}$")
    axes[0].set_ylabel("K (number of drafts)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote plot → {out_path}")


# ── verification: real timed consensus generation ─────────────────────────────

@torch.no_grad()
def verify_consensus_points(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    points: list[tuple[float, int]],
    n_gen: int,
    temperature: float,
    device: torch.device,
    target_model=None,
) -> list[dict]:
    """Validate the consensus grid by *actually generating* and timing it.

    For each (λ_draft, K) point this runs two real autoregressive decodes from
    the same prompt and compares wall-clock time:

      baseline   — full-model greedy generation of ``n_gen`` tokens.
      consensus  — at each step, draw K stochastic subnets; if all K greedily
                   agree, emit that token (no full-model pass); otherwise run the
                   full model for that step.

    Both decodes use the same uncached recompute stepping so the ratio is a fair
    apples-to-apples comparison of *per-step* cost (the regime the cost model
    targets); it is not a tuned cached-throughput benchmark. Returns, per point:

      speedup_real      = t_full / t_consensus           (measured)
      speedup_pred      = 1 / (K·c + 1 − p_agree_real)   (cost-model prediction
                          using the *measured* agreement — validates the formula)
      p_agree_real      fraction of steps where the K subnets agreed
      token_error_real  fraction of generated tokens that differ from the full
                        model's greedy token on the same prefix
      full_calls        number of steps that fell back to the full model
    """
    ffn_frac = _ffn_param_fraction(model)
    tgt      = target_model if target_model is not None else model

    if target_model is None:
        commit_masks(model, lam_target, device)
        frac_target = active_fraction(model, lam_target, device)["ffn"]
    else:
        frac_target = 1.0

    sampler, n_stoch = _make_stoch_sampler(model, device)
    if n_stoch == 0:
        print("  WARNING: no stochastic gate modules — drafts will be identical.")

    def _next_token(m, seq):
        return int(_logprobs_at_all(m, seq, temperature)[-1].argmax().item())

    results = []
    for lam_draft, K in points:
        set_lam(sampler, torch.tensor(lam_draft, dtype=torch.float32, device=device))
        frac_draft = active_fraction(model, lam_draft, device)["ffn"]
        c          = draft_cost_ratio(frac_draft, frac_target, ffn_frac)

        # ── baseline: full-model greedy, timed ──
        seq = prompt.clone()
        _device_sync(device); t0 = time.perf_counter()
        for _ in range(n_gen):
            nxt = _next_token(tgt, seq)
            seq = torch.cat([seq, torch.tensor([nxt], device=device)])
        _device_sync(device); t_full = time.perf_counter() - t0

        # ── consensus: K subnets vote, full model only on disagreement, timed ──
        seq_c      = prompt.clone()
        full_calls = 0
        _device_sync(device); t0 = time.perf_counter()
        for _ in range(n_gen):
            cand = [_next_token(sampler, seq_c) for _ in range(K)]
            if all(t == cand[0] for t in cand):
                nxt = cand[0]
            else:
                nxt = _next_token(tgt, seq_c)
                full_calls += 1
            seq_c = torch.cat([seq_c, torch.tensor([nxt], device=device)])
        _device_sync(device); t_cons = time.perf_counter() - t0

        # ── token error along the realised trajectory (untimed, one tgt pass) ──
        pl        = prompt.numel()
        tgt_arg   = _logprobs_at_all(tgt, seq_c, temperature).argmax(dim=-1)
        emitted   = seq_c[pl:]                    # (n_gen,)  tokens we committed
        predicted = tgt_arg[pl - 1: pl - 1 + n_gen]   # full model's greedy choice
        token_err = (emitted != predicted).float().mean().item()

        p_agree_real = 1.0 - full_calls / n_gen
        speed_real   = t_full / t_cons if t_cons > 0 else float("inf")
        cost_pred    = K * c + (1.0 - p_agree_real)
        speed_pred   = 1.0 / cost_pred if cost_pred > 0 else float("inf")

        results.append({
            "lam_draft":        lam_draft,
            "K":                K,
            "c":                c,
            "n_gen":            n_gen,
            "p_agree_real":     p_agree_real,
            "token_error_real": token_err,
            "full_calls":       full_calls,
            "t_full_s":         t_full,
            "t_consensus_s":    t_cons,
            "speedup_real":     speed_real,
            "speedup_pred":     speed_pred,
        })
        print(f"  λ={lam_draft:.3g} K={K:<3d}  agree={p_agree_real:.2f}  "
              f"err={token_err:.3f}  "
              f"S_real={speed_real:.2f}  S_pred={speed_pred:.2f}  "
              f"(t_full={t_full:.2f}s t_cons={t_cons:.2f}s)")

    return results


# ── optimization sweep ────────────────────────────────────────────────────────

def sweep_lambda_optimization(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    temperature: float,
    device: torch.device,
    gamma_max: int = 64,
    target_model=None,
) -> list[dict]:
    """For each draft λ, compute α analytically, derive c from active fractions,
    then find the (γ*, S*) pair that maximises wall-clock speedup.

    target_model: if provided, use as a fixed target (e.g. dense model) instead
                  of model@lam_target.  Its masks are not modified.

    Returns a list of dicts with keys:
        lam_draft, alpha, tv, ffn_active_draft, ffn_active_target, c,
        gamma_star, speedup_star, speedup_gamma4
    """
    ffn_frac = _ffn_param_fraction(model)

    if target_model is None:
        frac_target = active_fraction(model, lam_target, device)["ffn"]
        tgt = model
        tgt_lam = lam_target
    else:
        frac_target = 1.0   # dense / external model treated as fully active
        tgt = target_model
        tgt_lam = None      # skip commit — target model is already in eval state

    results = []
    for lam_draft in lam_draft_values:
        frac_draft = active_fraction(model, lam_draft, device)["ffn"]
        c = draft_cost_ratio(frac_draft, frac_target, ffn_frac)

        draft_model = _build_draft(model, lam_draft, device)
        if target_model is None:
            commit_masks(model, lam_target, device)

        mean_alpha, _ = expected_acceptance_rate(
            tgt, draft_model, prompt, tgt_lam, temperature, device)

        g_star, s_star = optimal_gamma_th311(mean_alpha, c, gamma_max)
        s_gamma4 = speedup_th38(mean_alpha, 4, c)

        results.append({
            "lam_draft":          lam_draft,
            "alpha":              mean_alpha,
            "tv":                 1.0 - mean_alpha,
            "ffn_active_draft":   frac_draft,
            "ffn_active_target":  frac_target,
            "c":                  c,
            "gamma_star":         g_star,
            "speedup_star":       s_star,
            "speedup_gamma4":     s_gamma4,
        })
        print(f"  λ={lam_draft:.3g}  FFN={frac_draft:.0%}  c={c:.3f}  "
              f"α={mean_alpha:.3f}  γ*={g_star}  S*={s_star:.3f}")

    return results


# ── optimization plots ────────────────────────────────────────────────────────

def make_optimization_plot(
    opt_results: list[dict],
    lam_target: float,
    gamma_max: int,
    out_path: Path,
    norm_diag: dict | None = None,
) -> None:
    """4-panel curve plot + 2-D S(γ, λ) heatmap + H(f_s) norm histogram."""
    lams  = np.array([r["lam_draft"]        for r in opt_results])
    alpha = np.array([r["alpha"]             for r in opt_results])
    c_arr = np.array([r["c"]                 for r in opt_results])
    g_star = np.array([r["gamma_star"]       for r in opt_results])
    s_star = np.array([r["speedup_star"]     for r in opt_results])
    s_g4   = np.array([r["speedup_gamma4"]   for r in opt_results])
    ffn    = np.array([r["ffn_active_draft"] for r in opt_results]) * 100

    n_rows = 4 if norm_diag else 3
    fig = plt.figure(figsize=(13, 4 * n_rows))
    fig.suptitle(
        f"Speculative decoding optimisation  (λ_target={lam_target})\n"
        f"Theorems 3.8 & 3.11, Leviathan et al. 2023",
        fontsize=13, y=1.01)

    gs = fig.add_gridspec(n_rows, 2, hspace=0.5, wspace=0.35)

    ax_alpha  = fig.add_subplot(gs[0, 0])
    ax_c      = fig.add_subplot(gs[0, 1])
    ax_s      = fig.add_subplot(gs[1, 0])
    ax_g      = fig.add_subplot(gs[1, 1])
    ax_heat   = fig.add_subplot(gs[2, 0])
    ax_heat2  = fig.add_subplot(gs[2, 1])   # second heatmap panel (norm hist placeholder)

    # ── α(λ) ──────────────────────────────────────────────────────────────
    ax_alpha.plot(lams, alpha, color="steelblue", lw=2)
    ax_alpha.fill_between(lams, 0, alpha, alpha=0.15, color="steelblue")
    ax_alpha.axhline(1.0, color="grey", lw=1, ls=":")
    ax_alpha.set_xlabel("λ_draft")
    ax_alpha.set_ylabel("α = 1 − TV(p,q)")
    ax_alpha.set_ylim(0, 1.05)
    ax_alpha.set_title("Acceptance rate α  (Sec. 3.2)")

    ax2 = ax_alpha.twinx()
    ax2.plot(lams, ffn, color="darkorange", lw=1.2, ls="--", alpha=0.7)
    ax2.set_ylabel("FFN active %", color="darkorange", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=7)
    ax2.set_ylim(0, 110)

    # ── c(λ) ──────────────────────────────────────────────────────────────
    ax_c.plot(lams, c_arr, color="purple", lw=2)
    ax_c.fill_between(lams, 0, c_arr, alpha=0.15, color="purple")
    ax_c.axhline(1.0, color="grey", lw=1, ls=":", label="c=1 (no savings)")
    ax_c.set_xlabel("λ_draft")
    ax_c.set_ylabel("c = t_draft / t_target")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_title("Relative draft cost c  (FFN-weighted)")
    ax_c.legend(fontsize=8)

    # ── S*(λ) ─────────────────────────────────────────────────────────────
    ax_s.plot(lams, s_star, color="seagreen", lw=2, label="S* = max_γ S(γ,α,c)  Th.3.8")
    ax_s.plot(lams, s_g4,   color="steelblue", lw=1.5, ls="--", label="S(γ=4, α, c)")
    ax_s.axhline(1.0, color="tomato", lw=1.2, ls="--", label="S=1 (no gain)")
    ax_s.fill_between(lams, 1, np.maximum(s_star, 1), alpha=0.12, color="seagreen",
                       label="speedup region")
    ax_s.set_xlabel("λ_draft")
    ax_s.set_ylabel("Wall-clock speedup S*")
    ax_s.set_title("Optimal speedup  (Th. 3.8/3.11)")
    ax_s.legend(fontsize=8)
    best_idx = int(np.argmax(s_star))
    ax_s.annotate(
        f"max S*={s_star[best_idx]:.3f}\nλ={lams[best_idx]:.3g}",
        xy=(lams[best_idx], s_star[best_idx]),
        xytext=(lams[best_idx] + (lams[-1]-lams[0])*0.1,
                s_star[best_idx] * 0.95),
        arrowprops=dict(arrowstyle="->", color="seagreen"),
        fontsize=8, color="seagreen",
    )

    # ── γ*(λ) ─────────────────────────────────────────────────────────────
    ax_g.plot(lams, g_star, color="darkorange", lw=2, drawstyle="steps-post")
    ax_g.set_xlabel("λ_draft")
    ax_g.set_ylabel("γ*  (optimal draft length)")
    ax_g.set_title("Optimal γ  (Th. 3.11)")
    ax_g.set_ylim(bottom=0)

    # ── 2-D heatmap S(γ, λ) ───────────────────────────────────────────────
    gammas = np.arange(1, min(gamma_max, 30) + 1)
    S_grid = np.array([
        [speedup_th38(a, float(g), c)
         for a, c in zip(alpha, c_arr)]
        for g in gammas
    ])   # (n_gamma, n_lam)

    im = ax_heat.pcolormesh(lams, gammas, S_grid, cmap="RdYlGn",
                             vmin=0.5, vmax=max(2.0, s_star.max() + 0.1),
                             shading="auto")
    fig.colorbar(im, ax=ax_heat, label="S(γ, λ_draft)  Th. 3.8")
    ax_heat.plot(lams, g_star, color="white", lw=2, ls="--", label="γ*  (Th. 3.11)")
    ax_heat.contour(lams, gammas, S_grid, levels=[1.0],
                    colors="black", linewidths=1.2)
    ax_heat.set_xlabel("λ_draft")
    ax_heat.set_ylabel("γ  (draft tokens per round)")
    ax_heat.set_title("S(γ, λ)  —  dashes: γ*(λ),  black: S=1")
    ax_heat.legend(fontsize=8)

    # ── H(f_s) norm histogram ─────────────────────────────────────────────
    if norm_diag:
        ax = ax_heat2
        counts = norm_diag["counts"]
        edges  = norm_diag["bin_edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=edges[1]-edges[0],
               color="steelblue", alpha=0.7, label="empirical $f_s$")
        ax.set_xlabel("Row-norm $s_i = \\|w_i^{\\mathrm{gate}}\\|_2$")
        ax.set_ylabel("Density")
        ax.set_title("Gate-proj row-norm distribution  $f_s$")
        txt = (f"$H(f_s)$ = {norm_diag['H_fs']:.3f}\n"
               f"CV = {norm_diag['cv']:.3f}\n"
               f"$n$ = {norm_diag['n_neurons']:,}")
        ax.text(0.97, 0.95, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.legend(fontsize=8)
    else:
        ax_heat2.set_visible(False)

    # ── H(f_s) norm histogram (row 3, if present) ─────────────────────────
    if norm_diag and n_rows == 4:
        ax_norm = fig.add_subplot(gs[3, :])
        norms = norm_diag["norms"]
        ax_norm.hist(norms, bins=150, density=True,
                     color="steelblue", alpha=0.7, label="empirical $f_s$")
        ax_norm.set_xlabel("Row-norm $s_i$")
        ax_norm.set_ylabel("Density")
        ax_norm.set_title(
            f"Full norm distribution  —  $H(f_s)={norm_diag['H_fs']:.3f}$,  "
            f"CV={norm_diag['cv']:.3f},  "
            f"$n$={norm_diag['n_neurons']:,}")
        ax_norm.legend(fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nOptimisation plot saved → {out_path}")


def make_comparison_plot(
    results_by_tag: dict[str, list[dict]],
    lam_target: float,
    gamma_max: int,
    out_path,
    target_label: str | None = None,
) -> None:
    """Overlay α(λ), c(λ), S*(λ), γ*(λ) curves for multiple model variants."""
    import matplotlib.cm as cm
    colors = [cm.tab10(i) for i in range(10)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    target_desc = f"target={target_label}" if target_label else f"λ_target={lam_target}"
    fig.suptitle(
        f"Speculative decoding comparison  ({target_desc})\n"
        "Theorems 3.8 & 3.11, Leviathan et al. 2023",
        fontsize=12)

    for i, (tag, results) in enumerate(results_by_tag.items()):
        color = colors[i % len(colors)]
        lams   = np.array([r["lam_draft"]        for r in results])
        alpha  = np.array([r["alpha"]             for r in results])
        c_arr  = np.array([r["c"]                 for r in results])
        g_star = np.array([r["gamma_star"]        for r in results])
        s_star = np.array([r["speedup_star"]      for r in results])
        ffn    = np.array([r["ffn_active_draft"]  for r in results]) * 100
        kw = dict(color=color, lw=2, marker="o", ms=4, label=tag)

        axes[0, 0].plot(lams, alpha, **kw)
        axes[0, 1].plot(lams, c_arr, **kw)
        axes[1, 0].plot(lams, s_star, **kw)
        axes[1, 1].plot(lams, g_star, color=color, lw=2,
                        drawstyle="steps-post", label=tag)

    axes[0, 0].set_title(f"Accept rate α  ({target_desc})")
    axes[0, 0].set_xlabel("λ_draft"); axes[0, 0].set_ylabel("α = 1 − TV(p,q)")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].axhline(1.0, color="grey", lw=1, ls=":")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].set_title("Relative draft cost c  (FFN-weighted)")
    axes[0, 1].set_xlabel("λ_draft"); axes[0, 1].set_ylabel("c = t_draft / t_target")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].axhline(1.0, color="grey", lw=1, ls=":", label="c=1 (no savings)")
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].set_title("Optimal speedup S*  (Th. 3.8/3.11)")
    axes[1, 0].set_xlabel("λ_draft"); axes[1, 0].set_ylabel("Wall-clock speedup S*")
    axes[1, 0].axhline(1.0, color="tomato", lw=1.2, ls="--", label="S=1 (no gain)")
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].set_title("Optimal draft length γ*  (Th. 3.11)")
    axes[1, 1].set_xlabel("λ_draft"); axes[1, 1].set_ylabel("γ*")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved → {out_path}")


def print_optimization_table(opt_results: list[dict], lam_target: float) -> None:
    best = max(opt_results, key=lambda r: r["speedup_star"])
    print(f"\n{'─'*78}")
    print(f"  Optimisation sweep  (λ_target={lam_target})")
    print(f"  FFN param fraction estimated from model weights")
    print(f"{'─'*78}")
    print(f"  {'λ_draft':>8}  {'FFN%':>6}  {'c':>6}  {'α':>6}  "
          f"{'γ*':>5}  {'S*':>7}  {'S(γ=4)':>8}")
    print(f"{'─'*78}")
    for r in opt_results:
        flag = "  ← best" if r is best else ""
        print(f"  {r['lam_draft']:>8.3g}  "
              f"{r['ffn_active_draft']:>5.0%}  "
              f"{r['c']:>6.3f}  "
              f"{r['alpha']:>6.3f}  "
              f"{r['gamma_star']:>5d}  "
              f"{r['speedup_star']:>7.3f}  "
              f"{r['speedup_gamma4']:>8.3f}{flag}")
    print(f"{'─'*78}\n")


# ── analytical α sweep (fast, no sampling) ───────────────────────────────────

def sweep_alpha(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    temperature: float,
    device: torch.device,
    gamma: int,
    n_stoch_samples: int = 0,
) -> list[dict]:
    """Compute theoretical α = 1 − TV(p, q) for each draft λ via two forward
    passes per λ value.  Also reports E[tok/call] = γα + 1.

    If n_stoch_samples > 0 and model_type is olmo3, also estimates E_mask[α]
    with Bernoulli-sampled gates (stochastic masks) averaged over that many
    forward passes, stored in alpha_stoch / alpha_stoch_std.

    Much faster than the generative sweep — use this to find a good lam_draft
    before running the full timing benchmark.
    """
    results = []
    for lam_draft in lam_draft_values:
        frac = active_fraction(model, lam_draft, device)
        draft_model = _build_draft(model, lam_draft, device)
        commit_masks(model, lam_target, device)

        mean_alpha, _ = expected_acceptance_rate(
            model, draft_model, prompt, lam_target, temperature, device)

        row = {
            "lam_draft":              lam_draft,
            "ffn_active":             frac["ffn"],
            "alpha":                  mean_alpha,
            "tv_distance":            1.0 - mean_alpha,
            "expected_toks_per_call": gamma * mean_alpha + 1,
        }
        print(f"  λ_draft={lam_draft:.3g}  FFN={frac['ffn']:.0%}  "
              f"α_hard={mean_alpha:.3f}  TV={1-mean_alpha:.3f}  "
              f"E[tok/call]={gamma * mean_alpha + 1:.2f}", end="")

        if n_stoch_samples > 0 and _MODEL_TYPE != "stoch":
            s_mu, s_sd = stochastic_expected_acceptance_rate(
                model, prompt, lam_target, lam_draft,
                temperature, device, n_stoch_samples)
            row["alpha_stoch"]     = s_mu
            row["alpha_stoch_std"] = s_sd
            print(f"  α_stoch={s_mu:.3f}±{s_sd:.3f}", end="")

        print()
        results.append(row)

    return results


# ── α(λ, T): context-length sweep ────────────────────────────────────────────

def _make_prompt_at_length(base_prompt: torch.Tensor, T: int) -> torch.Tensor:
    """Return a prompt of exactly T tokens by truncating or tiling base_prompt."""
    if len(base_prompt) >= T:
        return base_prompt[:T]
    reps = (T + len(base_prompt) - 1) // len(base_prompt)
    return base_prompt.repeat(reps)[:T]


def sweep_alpha_contexts(
    model,
    base_prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    context_lengths: list[int],
    temperature: float,
    device: torch.device,
    n_stoch_samples: int = 0,
) -> dict[int, list[dict]]:
    """Measure α(λ, T) — acceptance rate vs draft λ for each context length T.

    Builds each draft model once per λ value, then evaluates it at every T.
    If n_stoch_samples > 0, also estimates E_mask[α] with Bernoulli-sampled
    gates (n_stoch_samples draws per point) stored in alpha_stoch/alpha_stoch_std.
    Returns {T: [{"lam_draft", "alpha", "ffn_active", "context_len", ...}, ...]}.
    """
    results_by_ctx: dict[int, list] = {T: [] for T in context_lengths}
    prompts = {T: _make_prompt_at_length(base_prompt, T) for T in context_lengths}

    for lam_draft in lam_draft_values:
        frac = active_fraction(model, lam_draft, device)
        draft_model = _build_draft(model, lam_draft, device)
        commit_masks(model, lam_target, device)

        print(f"  λ={lam_draft:.3g}  FFN={frac['ffn']:.0%}", end="")
        for T in context_lengths:
            mean_alpha, _ = expected_acceptance_rate(
                model, draft_model, prompts[T], lam_target, temperature, device)
            row: dict = {
                "lam_draft":   lam_draft,
                "alpha":       mean_alpha,
                "ffn_active":  frac["ffn"],
                "context_len": T,
            }
            print(f"  T={T}: α_hard={mean_alpha:.3f}", end="")
            if n_stoch_samples > 0:
                s_mu, s_sd = stochastic_expected_acceptance_rate(
                    model, prompts[T], lam_target, lam_draft,
                    temperature, device, n_stoch_samples)
                row["alpha_stoch"]     = s_mu
                row["alpha_stoch_std"] = s_sd
                print(f" α_stoch={s_mu:.3f}±{s_sd:.3f}", end="")
            results_by_ctx[T].append(row)
        print()

    return results_by_ctx


def make_context_sweep_plot(
    results_by_ctx: dict[int, list[dict]],
    lam_target: float,
    out_path: "Path",
) -> None:
    """Two-panel plot: α(λ) curves per context length + α(λ, T) heatmap."""
    import matplotlib.cm as cm

    ctx_lengths = sorted(results_by_ctx.keys())
    lam_values  = np.array([r["lam_draft"] for r in results_by_ctx[ctx_lengths[0]]])
    alpha_matrix = np.array([
        [r["alpha"] for r in results_by_ctx[T]]
        for T in ctx_lengths
    ])  # (n_ctx, n_lam)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"α(λ, T)  —  acceptance rate vs draft λ and context length\n"
        f"(λ_target={lam_target})",
        fontsize=13)

    # ── curves panel ──────────────────────────────────────────────────────────
    ax = axes[0]
    colors = cm.viridis(np.linspace(0.1, 0.9, len(ctx_lengths)))
    for i, T in enumerate(ctx_lengths):
        alphas = [r["alpha"] for r in results_by_ctx[T]]
        ax.plot(lam_values, alphas, color=colors[i], lw=2, marker="o", ms=4,
                label=f"T={T}")
    ax.set_xlabel("λ_draft")
    ax.set_ylabel("α = 1 − TV(p, q)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=1, ls=":")
    ax.set_title("α(λ) by context length")
    ax.legend(fontsize=9)

    # ── heatmap panel ─────────────────────────────────────────────────────────
    ax = axes[1]
    # Use integer indices on y so pcolormesh stays rectangular
    y_idx = np.arange(len(ctx_lengths) + 1) - 0.5
    x_edges = np.concatenate([[lam_values[0] - (lam_values[1] - lam_values[0]) * 0.5],
                               (lam_values[:-1] + lam_values[1:]) / 2,
                               [lam_values[-1] + (lam_values[-1] - lam_values[-2]) * 0.5]])
    im = ax.pcolormesh(x_edges, y_idx, alpha_matrix,
                       cmap="RdYlGn", vmin=0, vmax=1, shading="flat")
    fig.colorbar(im, ax=ax, label="α = 1 − TV(p, q)")
    ax.set_yticks(range(len(ctx_lengths)))
    ax.set_yticklabels([str(T) for T in ctx_lengths])
    ax.set_xlabel("λ_draft")
    ax.set_ylabel("Context length T  (tokens)")
    ax.set_title("α(λ, T)  heatmap")

    # Annotate cells
    for i, T in enumerate(ctx_lengths):
        for j, lam in enumerate(lam_values):
            ax.text(lam, i, f"{alpha_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if 0.2 < alpha_matrix[i, j] < 0.8 else "white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nContext sweep plot saved → {out_path}")


def print_context_sweep_table(
    results_by_ctx: dict[int, list[dict]],
    lam_target: float,
) -> None:
    ctx_lengths = sorted(results_by_ctx.keys())
    lam_values  = [r["lam_draft"] for r in results_by_ctx[ctx_lengths[0]]]
    has_stoch   = "alpha_stoch" in results_by_ctx[ctx_lengths[0]][0]

    lam_hdr = "  ".join(f"{lam:>7.3g}" for lam in lam_values)
    print(f"\n{'─'*72}")
    print(f"  α_hard(λ, T)  —  λ_target={lam_target}")
    print("  Rows = context length T,  Cols = λ_draft")
    print(f"{'─'*72}")
    print(f"  {'T':>6}  {lam_hdr}")
    print(f"{'─'*72}")
    for T in ctx_lengths:
        row = f"  {T:>6}  " + "  ".join(f"{r['alpha']:>7.3f}" for r in results_by_ctx[T])
        print(row)
    print(f"{'─'*72}\n")

    if has_stoch:
        print(f"  α_stoch(λ, T)  —  λ_target={lam_target}  (mean ± std)")
        print(f"{'─'*72}")
        print(f"  {'T':>6}  {lam_hdr}")
        print(f"{'─'*72}")
        for T in ctx_lengths:
            row = f"  {T:>6}  " + "  ".join(
                f"{r['alpha_stoch']:>5.3f}±{r['alpha_stoch_std']:.2f}"
                for r in results_by_ctx[T])
            print(row)
        print(f"{'─'*72}\n")


# ── α(λ, N): decode-length sweep ─────────────────────────────────────────────

def sweep_alpha_decode_lengths(
    model,
    base_prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    decode_lengths: list[int],
    temperature: float,
    device: torch.device,
    n_stoch_samples: int = 0,
) -> dict[int, list[dict]]:
    """Measure α(λ, N) by generating N tokens from base_prompt with the target
    model, then evaluating α on the full extended context (T_base + N tokens).

    Each draft model is built once per λ, then tested against all N values.
    If n_stoch_samples > 0, also estimates E_mask[α] with stochastic gates.
    Returns {N: [{"lam_draft", "alpha", "ffn_active", "decode_len",
                  "total_len", ...}, ...]}.
    """
    commit_masks(model, lam_target, device)
    print("Generating extended contexts with target model …")
    extended_prompts: dict[int, torch.Tensor] = {}
    for N in sorted(decode_lengths):
        out_ids, _ = generate_greedy(model, base_prompt, N, lam_target,
                                     temperature, device, verbose=False)
        extended_prompts[N] = out_ids.to(device)
        print(f"  N={N}: context now {len(extended_prompts[N])} tokens")

    results_by_N: dict[int, list] = {N: [] for N in decode_lengths}

    print()
    for lam_draft in lam_draft_values:
        frac = active_fraction(model, lam_draft, device)
        draft_model = _build_draft(model, lam_draft, device)
        commit_masks(model, lam_target, device)

        print(f"  λ={lam_draft:.3g}  FFN={frac['ffn']:.0%}", end="")
        for N in decode_lengths:
            prompt = extended_prompts[N]
            mean_alpha, _ = expected_acceptance_rate(
                model, draft_model, prompt, lam_target, temperature, device)
            row: dict = {
                "lam_draft":  lam_draft,
                "alpha":      mean_alpha,
                "ffn_active": frac["ffn"],
                "decode_len": N,
                "total_len":  len(prompt),
            }
            print(f"  N={N}: α_hard={mean_alpha:.3f}", end="")
            if n_stoch_samples > 0:
                s_mu, s_sd = stochastic_expected_acceptance_rate(
                    model, prompt, lam_target, lam_draft,
                    temperature, device, n_stoch_samples)
                row["alpha_stoch"]     = s_mu
                row["alpha_stoch_std"] = s_sd
                print(f" α_stoch={s_mu:.3f}±{s_sd:.3f}", end="")
            results_by_N[N].append(row)
        print()

    return results_by_N


def make_decode_sweep_plot(
    results_by_N: dict[int, list[dict]],
    lam_target: float,
    base_prompt_len: int,
    out_path: "Path",
) -> None:
    """Two-panel: α(λ) curves per decode length + α(λ, N) heatmap."""
    import matplotlib.cm as cm

    decode_lengths = sorted(results_by_N.keys())
    lam_values     = np.array([r["lam_draft"] for r in results_by_N[decode_lengths[0]]])
    alpha_matrix   = np.array([
        [r["alpha"] for r in results_by_N[N]]
        for N in decode_lengths
    ])  # (n_N, n_lam)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    total_lens = [results_by_N[N][0]["total_len"] for N in decode_lengths]
    fig.suptitle(
        f"α(λ, N)  —  acceptance rate vs draft λ and decode length\n"
        f"(λ_target={lam_target},  base prompt={base_prompt_len} tokens)",
        fontsize=13)

    colors = cm.plasma(np.linspace(0.1, 0.9, len(decode_lengths)))
    ax = axes[0]
    for i, N in enumerate(decode_lengths):
        alphas = [r["alpha"] for r in results_by_N[N]]
        lbl = f"N={N}  (ctx={total_lens[i]})"
        ax.plot(lam_values, alphas, color=colors[i], lw=2, marker="o", ms=4,
                label=lbl)
    ax.set_xlabel("λ_draft")
    ax.set_ylabel("α = 1 − TV(p, q)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=1, ls=":")
    ax.set_title("α(λ) by decode length")
    ax.legend(fontsize=9)

    ax = axes[1]
    y_idx  = np.arange(len(decode_lengths) + 1) - 0.5
    x_edges = np.concatenate([
        [lam_values[0] - (lam_values[1] - lam_values[0]) * 0.5],
        (lam_values[:-1] + lam_values[1:]) / 2,
        [lam_values[-1] + (lam_values[-1] - lam_values[-2]) * 0.5],
    ])
    im = ax.pcolormesh(x_edges, y_idx, alpha_matrix,
                       cmap="RdYlGn", vmin=0, vmax=1, shading="flat")
    fig.colorbar(im, ax=ax, label="α = 1 − TV(p, q)")
    ax.set_yticks(range(len(decode_lengths)))
    ax.set_yticklabels([f"N={N}\n(ctx={t})" for N, t in zip(decode_lengths, total_lens)],
                       fontsize=8)
    ax.set_xlabel("λ_draft")
    ax.set_ylabel("Tokens decoded (N)")
    ax.set_title("α(λ, N)  heatmap")

    for i, N in enumerate(decode_lengths):
        for j in range(len(lam_values)):
            ax.text(lam_values[j], i, f"{alpha_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if 0.2 < alpha_matrix[i, j] < 0.8 else "white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nDecode sweep plot saved → {out_path}")


def print_decode_sweep_table(
    results_by_N: dict[int, list[dict]],
    lam_target: float,
) -> None:
    decode_lengths = sorted(results_by_N.keys())
    lam_values     = [r["lam_draft"] for r in results_by_N[decode_lengths[0]]]
    has_stoch      = "alpha_stoch" in results_by_N[decode_lengths[0]][0]

    lam_hdr = "  ".join(f"{lam:>7.3g}" for lam in lam_values)
    print(f"\n{'─'*72}")
    print(f"  α_hard(λ, N)  —  λ_target={lam_target}")
    print("  Rows = tokens decoded N,  Cols = λ_draft")
    print(f"{'─'*72}")
    print(f"  {'N':>6}  {lam_hdr}")
    print(f"{'─'*72}")
    for N in decode_lengths:
        row = f"  {N:>6}  " + "  ".join(f"{r['alpha']:>7.3f}" for r in results_by_N[N])
        print(row)
    print(f"{'─'*72}\n")

    if has_stoch:
        print(f"  α_stoch(λ, N)  —  λ_target={lam_target}  (mean ± std)")
        print(f"{'─'*72}")
        print(f"  {'N':>6}  {lam_hdr}")
        print(f"{'─'*72}")
        for N in decode_lengths:
            row = f"  {N:>6}  " + "  ".join(
                f"{r['alpha_stoch']:>5.3f}±{r['alpha_stoch_std']:.2f}"
                for r in results_by_N[N])
            print(row)
        print(f"{'─'*72}\n")


# ── cost / speedup enrichment ─────────────────────────────────────────────────

def enrich_with_speedup(
    results_by_key: dict,
    model,
    lam_target: float,
    device: "torch.device",
    gamma_max: int = 64,
) -> None:
    """Add c, gamma_star, speedup_star, speedup_gamma4 to every result dict in-place."""
    ffn_frac    = _ffn_param_fraction(model)
    frac_target = active_fraction(model, lam_target, device)["ffn"]
    for results in results_by_key.values():
        for r in results:
            c = draft_cost_ratio(r["ffn_active"], frac_target, ffn_frac)
            g_star, s_star = optimal_gamma_th311(r["alpha"], c, gamma_max)
            r["c"]              = c
            r["gamma_star"]     = g_star
            r["speedup_star"]   = s_star
            r["speedup_gamma4"] = speedup_th38(r["alpha"], 4, c)
            if "alpha_stoch" in r:
                g_stoch, s_stoch = optimal_gamma_th311(r["alpha_stoch"], c, gamma_max)
                r["gamma_star_stoch"]   = g_stoch
                r["speedup_star_stoch"] = s_stoch


def _heatmap_panel(ax, fig, lam_values, keys, matrix, key_label,
                   cbar_label, cmap, vmin, vmax, contour_at=None):
    """Cell-annotated pcolormesh heatmap shared by both plot types."""
    lv = np.asarray(lam_values)
    dx = np.diff(lv)
    x_edges = np.concatenate([[lv[0] - dx[0] * 0.5],
                               (lv[:-1] + lv[1:]) / 2,
                               [lv[-1] + dx[-1] * 0.5]])
    y_idx = np.arange(len(keys) + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_idx, matrix,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([str(k) for k in keys])
    ax.set_xlabel("λ_draft")
    ax.set_ylabel(key_label)
    span = vmax - vmin
    for i in range(len(keys)):
        for j in range(len(lam_values)):
            val = float(matrix[i, j])
            if not np.isnan(val):
                dark = (vmin + span * 0.2) < val < (vmin + span * 0.8)
                ax.text(lv[j], i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if dark else "white")
    if contour_at is not None:
        try:
            ax.contour(lv, np.arange(len(keys)), matrix,
                       levels=[contour_at], colors="black", linewidths=1.5)
        except Exception:
            pass


def make_combined_sweep_plot(
    results_by_key: dict[int, list[dict]],
    key_label: str,
    lam_target: float,
    out_path: "Path",
    cmap_name: str = "viridis",
) -> None:
    """Curves + heatmaps for α (and S* when available).

    Layout adapts to whether stochastic and speedup data are present:

    No stoch, no spd   — 2 rows × 1 col: α curves | α_hard heatmap
    No stoch, has spd  — 2 rows × 2 col: α curves, S* curves | α heatmap, S* heatmap
    Has stoch, no spd  — 2 rows × 2 col: α curves | α_hard heatmap, α_stoch heatmap
    Has stoch, has spd — 3 rows × 2 col: α curves, S* curves |
                                          α_hard heatmap, α_stoch heatmap |
                                          S* heatmap (full width)
    """
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec

    keys      = sorted(results_by_key.keys())
    lam_vals  = np.array([r["lam_draft"] for r in results_by_key[keys[0]]])
    alpha_mat = np.array([[r["alpha"] for r in results_by_key[k]] for k in keys])
    has_spd   = "speedup_star" in results_by_key[keys[0]][0]
    has_stoch = "alpha_stoch" in results_by_key[keys[0]][0]
    s_mat     = (np.array([[r["speedup_star"] for r in results_by_key[k]] for k in keys])
                 if has_spd else None)
    stoch_mat = (np.array([[r["alpha_stoch"] for r in results_by_key[k]] for k in keys])
                 if has_stoch else None)

    colors = cm.get_cmap(cmap_name)(np.linspace(0.1, 0.9, len(keys)))
    fig = plt.figure(figsize=(14 if (has_stoch or has_spd) else 7,
                              15 if (has_stoch and has_spd) else 10))
    fig.suptitle(
        f"α and S* vs λ_draft  by {key_label}  (λ_target={lam_target})\n"
        "Speedup: Theorems 3.8 & 3.11, Leviathan et al. 2023",
        fontsize=12)

    if has_stoch and has_spd:
        gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
        ax_alpha_c  = fig.add_subplot(gs[0, 0])
        ax_spd_c    = fig.add_subplot(gs[0, 1])
        ax_a_hard   = fig.add_subplot(gs[1, 0])
        ax_a_stoch  = fig.add_subplot(gs[1, 1])
        ax_spd_h    = fig.add_subplot(gs[2, :])
    elif has_stoch and not has_spd:
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        ax_alpha_c  = fig.add_subplot(gs[0, :])
        ax_a_hard   = fig.add_subplot(gs[1, 0])
        ax_a_stoch  = fig.add_subplot(gs[1, 1])
        ax_spd_c = ax_spd_h = None
    elif not has_stoch and has_spd:
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        ax_alpha_c  = fig.add_subplot(gs[0, 0])
        ax_spd_c    = fig.add_subplot(gs[0, 1])
        ax_a_hard   = fig.add_subplot(gs[1, 0])
        ax_spd_h    = fig.add_subplot(gs[1, 1])
        ax_a_stoch  = None
    else:
        gs = GridSpec(2, 1, figure=fig, hspace=0.4)
        ax_alpha_c  = fig.add_subplot(gs[0])
        ax_a_hard   = fig.add_subplot(gs[1])
        ax_spd_c = ax_spd_h = ax_a_stoch = None

    # ── α(λ) curves ──────────────────────────────────────────────────────────
    for i, k in enumerate(keys):
        alphas = [r["alpha"] for r in results_by_key[k]]
        ax_alpha_c.plot(lam_vals, alphas, color=colors[i], lw=2, marker="o",
                        ms=4, label=f"{key_label}={k} hard")
        if has_stoch:
            s_mu = np.array([r["alpha_stoch"]    for r in results_by_key[k]])
            s_sd = np.array([r["alpha_stoch_std"] for r in results_by_key[k]])
            ax_alpha_c.plot(lam_vals, s_mu, color=colors[i], lw=1.5, ls="--",
                            marker="x", ms=5, label=f"{key_label}={k} stoch")
            ax_alpha_c.fill_between(lam_vals, s_mu - s_sd, s_mu + s_sd,
                                    color=colors[i], alpha=0.08)
    ax_alpha_c.set_xlabel("λ_draft")
    ax_alpha_c.set_ylabel("α = 1 − TV(p, q)")
    ax_alpha_c.set_ylim(0, 1.05)
    ax_alpha_c.axhline(1.0, color="grey", lw=1, ls=":")
    ax_alpha_c.set_title(f"Acceptance rate α(λ)  by {key_label}"
                         + ("  [solid=hard, dashed=stoch]" if has_stoch else ""))
    ax_alpha_c.legend(fontsize=8)

    # ── α_hard heatmap ────────────────────────────────────────────────────────
    _heatmap_panel(ax_a_hard, fig, lam_vals, keys, alpha_mat,
                   key_label, "α = 1 − TV(p, q)",
                   cmap="RdYlGn", vmin=0, vmax=1)
    ax_a_hard.set_title(f"α_hard(λ, {key_label})  heatmap")

    # ── α_stoch heatmap ───────────────────────────────────────────────────────
    if has_stoch and ax_a_stoch is not None:
        _heatmap_panel(ax_a_stoch, fig, lam_vals, keys, stoch_mat,
                       key_label, "E_mask[α]",
                       cmap="RdYlGn", vmin=0, vmax=1)
        ax_a_stoch.set_title(f"α_stoch(λ, {key_label})  heatmap  [Bernoulli gates]")

    # ── S*(λ) curves ──────────────────────────────────────────────────────────
    if has_spd and ax_spd_c is not None:
        for i, k in enumerate(keys):
            s_vals = [r["speedup_star"] for r in results_by_key[k]]
            g_vals = [r["gamma_star"]   for r in results_by_key[k]]
            ln, = ax_spd_c.plot(lam_vals, s_vals, color=colors[i], lw=2,
                                marker="o", ms=4, label=f"{key_label}={k} hard")
            best_j = int(np.argmax(s_vals))
            if s_vals[best_j] > 1.0:
                ax_spd_c.annotate(f"γ*={g_vals[best_j]}",
                                  xy=(lam_vals[best_j], s_vals[best_j]),
                                  fontsize=7, color=ln.get_color(),
                                  xytext=(4, 4), textcoords="offset points")
            if has_stoch and "speedup_star_stoch" in results_by_key[k][0]:
                s_stoch = [r["speedup_star_stoch"] for r in results_by_key[k]]
                ax_spd_c.plot(lam_vals, s_stoch, color=colors[i], lw=1.5,
                              ls="--", marker="x", ms=5,
                              label=f"{key_label}={k} stoch")
        ax_spd_c.axhline(1.0, color="tomato", lw=1.5, ls="--", label="S=1 (no gain)")
        s_max_per_lam = np.nanmax(s_mat, axis=0)
        ax_spd_c.fill_between(lam_vals, 1, np.maximum(s_max_per_lam, 1),
                              where=s_max_per_lam > 1,
                              alpha=0.1, color="seagreen", label="any speedup region")
        ax_spd_c.set_xlabel("λ_draft")
        ax_spd_c.set_ylabel("Optimal speedup S*")
        ax_spd_c.set_title(f"Speedup S*(λ)  by {key_label}  [Th. 3.8/3.11]"
                           + ("  [solid=hard, dashed=stoch]" if has_stoch else ""))
        ax_spd_c.legend(fontsize=8)

    # ── S* heatmap ────────────────────────────────────────────────────────────
    if has_spd and ax_spd_h is not None:
        s_max = max(2.0, float(np.nanmax(s_mat)) + 0.1)
        _heatmap_panel(ax_spd_h, fig, lam_vals, keys, s_mat,
                       key_label, "Optimal speedup S*",
                       cmap="RdYlGn", vmin=0.5, vmax=s_max, contour_at=1.0)
        ax_spd_h.set_title(f"S*(λ, {key_label})  heatmap  [black contour: S*=1]")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nCombined plot saved → {out_path}")


def print_speedup_opportunities(
    results_by_key: dict,
    key_label: str,
    lam_target: float,
) -> None:
    """Ranked table of every (key, λ_draft) pair where S* > 1 (hard or stochastic)."""
    has_stoch = "speedup_star_stoch" in next(
        iter(next(iter(results_by_key.values()))), {})
    rows = [
        (r["speedup_star"], k, r)
        for k, results in results_by_key.items()
        for r in results
        if r.get("speedup_star", 0) > 1.0
           or r.get("speedup_star_stoch", 0) > 1.0
    ]
    if not rows:
        print(f"  No ({key_label}, λ) combinations with S* > 1 found.\n")
        return

    rows.sort(reverse=True)
    stoch_hdr = "  S*_stoch" if has_stoch else ""
    w = 94 if has_stoch else 84
    print(f"\n{'─'*w}")
    print(f"  Speedup opportunities  (S* > 1)  —  λ_target={lam_target}")
    print(f"{'─'*w}")
    print(f"  {'λ_draft':>8}  {key_label:>8}  {'FFN%':>6}  {'α_hard':>7}  "
          f"{'c':>6}  {'γ*':>5}  {'S*_hard':>8}{stoch_hdr}")
    print(f"{'─'*w}")
    for _, k, r in rows:
        stoch_col = (f"  {r['speedup_star_stoch']:>8.3f}" if has_stoch else "")
        print(f"  {r['lam_draft']:>8.3g}  {k:>8}  "
              f"{r['ffn_active']:>5.0%}  {r['alpha']:>7.3f}  "
              f"{r['c']:>6.3f}  {r['gamma_star']:>5d}  "
              f"{r['speedup_star']:>8.3f}{stoch_col}")
    print(f"{'─'*w}")
    _, best_k, best_r = rows[0]
    print(f"\n  Best (hard): λ_draft={best_r['lam_draft']:.3g}  {key_label}={best_k}  "
          f"→ S*={best_r['speedup_star']:.3f}  (γ*={best_r['gamma_star']})")
    if has_stoch:
        stoch_rows = sorted(
            ((r.get("speedup_star_stoch", 0), k, r)
             for k, results in results_by_key.items()
             for r in results
             if r.get("speedup_star_stoch", 0) > 1.0),
            reverse=True)
        if stoch_rows:
            _, bk, br = stoch_rows[0]
            print(f"  Best (stoch): λ_draft={br['lam_draft']:.3g}  {key_label}={bk}  "
                  f"→ S*_stoch={br['speedup_star_stoch']:.3f}  "
                  f"(γ*={br.get('gamma_star_stoch', '?')})")
    print()


# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(
    model,
    prompt: torch.Tensor,
    lam_target: float,
    lam_draft_values: list[float],
    gamma: int,
    temperature: float,
    max_new_tokens: int,
    n_trials: int,
    device: torch.device,
    n_stoch_trials: int = 0,
    top_k_vocab: int = 0,
    top_k_exact_fallback: float = 0.0,
) -> tuple[float, list[dict]]:
    """
    Returns (baseline_tok_per_s, list_of_result_dicts).

    baseline_tok_per_s: target-only (no speculation) tok/s averaged over n_trials.
    Each result dict has keys:
        lam_draft, tok_per_s, tok_per_s_std, accept_rate,
        expected_toks_per_call, ffn_active, speedup
    When n_stoch_trials > 0 (olmo3 model only), also adds:
        stoch_tok_per_s, stoch_tok_per_s_std, stoch_accept_rate,
        stoch_expected_toks_per_call, stoch_speedup
    """
    # Commit target masks once for the full model
    commit_masks(model, lam_target, device)

    print(f"\nBaseline: target-only  (λ={lam_target}, no speculation)")
    baseline_samples: list[float] = []
    for t in range(n_trials):
        _, stats = generate_greedy(
            model, prompt, max_new_tokens, lam_target,
            temperature, device, verbose=False)
        baseline_samples.append(stats["tok_per_s"])
        print(f"  trial {t+1}/{n_trials}: {stats['tok_per_s']:.1f} tok/s")
    baseline = float(np.mean(baseline_samples))
    print(f"  → mean {baseline:.1f} tok/s\n")

    do_stoch = n_stoch_trials > 0 and _MODEL_TYPE != "stoch"

    results: list[dict] = []
    for lam_draft in lam_draft_values:
        frac = active_fraction(model, lam_draft, device)
        ffn_pct = frac["ffn"]
        print(f"λ_draft={lam_draft:.3g}  FFN active={ffn_pct:.0%}")
        print("  Building compressed draft model …", flush=True)
        draft_model = _build_draft(model, lam_draft, device)
        draft_params = sum(p.numel() for p in set(draft_model.parameters()))
        target_params = sum(p.numel() for p in set(model.parameters()))
        print(f"  Compressed: {draft_params/1e6:.1f}M params  "
              f"(target: {target_params/1e6:.1f}M, "
              f"ratio: {draft_params/target_params:.2f}×)")
        # Re-commit target masks (build_compressed_olmo3 calls commit_masks internally)
        commit_masks(model, lam_target, device)

        # ── hard mask trials ──────────────────────────────────────────────
        print("  [hard mask trials]")
        tok_s_samples: list[float] = []
        accept_samples: list[float] = []
        etpc_samples: list[float] = []
        for t in range(n_trials):
            _, stats = generate_speculative(
                model, prompt, max_new_tokens,
                lam_draft, lam_target, gamma, temperature,
                device, verbose=False, draft_model=draft_model,
                top_k_vocab=top_k_vocab,
                top_k_exact_fallback=top_k_exact_fallback)
            tok_s_samples.append(stats["tok_per_s"])
            accept_samples.append(stats["accept_rate"])
            etpc_samples.append(stats["expected_toks_per_call"])
            print(f"  trial {t+1}/{n_trials}: {stats['tok_per_s']:.1f} tok/s  "
                  f"accept={stats['accept_rate']:.1%}  "
                  f"E[tok/call]={stats['expected_toks_per_call']:.2f}")

        mean_tok_s = float(np.mean(tok_s_samples))
        row: dict = {
            "lam_draft":              lam_draft,
            "tok_per_s":              mean_tok_s,
            "tok_per_s_std":          float(np.std(tok_s_samples)),
            "accept_rate":            float(np.mean(accept_samples)),
            "expected_toks_per_call": float(np.mean(etpc_samples)),
            "ffn_active":             ffn_pct,
            "param_ratio":            draft_params / target_params,
            "speedup":                mean_tok_s / baseline,
        }
        print(f"  → hard mean {mean_tok_s:.1f} tok/s  speedup {mean_tok_s/baseline:.2f}×")

        # ── stochastic mask trials ─────────────────────────────────────────
        if do_stoch:
            print("  [stochastic mask trials]")
            set_lam(model, lam_target)
            set_hard_mask(model, False)
            set_sample_mask(model, True)
            stoch_tok_s: list[float] = []
            stoch_accept: list[float] = []
            stoch_etpc: list[float] = []
            for t in range(n_stoch_trials):
                _, stats = generate_speculative(
                    model, prompt, max_new_tokens,
                    lam_draft, lam_target, gamma, temperature,
                    device, verbose=False, draft_model=draft_model,
                    top_k_vocab=top_k_vocab,
                top_k_exact_fallback=top_k_exact_fallback)
                stoch_tok_s.append(stats["tok_per_s"])
                stoch_accept.append(stats["accept_rate"])
                stoch_etpc.append(stats["expected_toks_per_call"])
                print(f"  stoch trial {t+1}/{n_stoch_trials}: "
                      f"{stats['tok_per_s']:.1f} tok/s  "
                      f"accept={stats['accept_rate']:.1%}  "
                      f"E[tok/call]={stats['expected_toks_per_call']:.2f}")
            commit_masks(model, lam_target, device)
            stoch_mean = float(np.mean(stoch_tok_s))
            row["stoch_tok_per_s"]              = stoch_mean
            row["stoch_tok_per_s_std"]          = float(np.std(stoch_tok_s))
            row["stoch_accept_rate"]            = float(np.mean(stoch_accept))
            row["stoch_expected_toks_per_call"] = float(np.mean(stoch_etpc))
            row["stoch_speedup"]                = stoch_mean / baseline
            print(f"  → stoch mean {stoch_mean:.1f} tok/s  speedup {stoch_mean/baseline:.2f}×")

        print()
        results.append(row)

    return baseline, results


# ── plot ──────────────────────────────────────────────────────────────────────

def make_plot(
    baseline: float,
    results: list[dict],
    lam_target: float,
    gamma: int,
    out_path: Path,
    alpha_results: list[dict] | None = None,
) -> None:
    lam_drafts  = [r["lam_draft"]              for r in results]
    tok_s       = [r["tok_per_s"]              for r in results]
    tok_s_std   = [r["tok_per_s_std"]          for r in results]
    accept      = [r["accept_rate"] * 100      for r in results]
    etpc        = [r["expected_toks_per_call"] for r in results]
    ffn_active  = [r["ffn_active"] * 100       for r in results]
    speedups    = [r["speedup"]                for r in results]
    has_stoch_tp = "stoch_tok_per_s" in results[0]

    fig, axes = plt.subplots(3, 1, figsize=(10 if has_stoch_tp else 8, 10), sharex=True)
    fig.suptitle(
        f"Speculative decoding sweep  (λ_target={lam_target}, γ={gamma})",
        fontsize=13, y=0.98)

    x = np.arange(len(lam_drafts))
    labels = [f"{lam:.3g}\n({f:.0f}% FFN)" for lam, f in zip(lam_drafts, ffn_active)]

    # ── panel 1: tok/s ────────────────────────────────────────────────────
    ax = axes[0]
    if has_stoch_tp:
        stoch_tok_s     = [r["stoch_tok_per_s"]     for r in results]
        stoch_tok_s_std = [r["stoch_tok_per_s_std"]  for r in results]
        stoch_speedups  = [r["stoch_speedup"]         for r in results]
        width = 0.38
        ax.bar(x - width / 2, tok_s, width,
               yerr=tok_s_std, capsize=3,
               color="steelblue", alpha=0.85, label="hard mask")
        ax.bar(x + width / 2, stoch_tok_s, width,
               yerr=stoch_tok_s_std, capsize=3,
               color="seagreen", alpha=0.85, label="stochastic mask")
        max_err = max(max(tok_s_std), max(stoch_tok_s_std)) if tok_s_std else 0.0
        for xi, (v, s) in enumerate(zip(tok_s, speedups)):
            ax.text(xi - width / 2, v + max_err * 0.15 + 0.5, f"{s:.2f}×",
                    ha="center", va="bottom", fontsize=7, color="steelblue")
        for xi, (v, s) in enumerate(zip(stoch_tok_s, stoch_speedups)):
            ax.text(xi + width / 2, v + max_err * 0.15 + 0.5, f"{s:.2f}×",
                    ha="center", va="bottom", fontsize=7, color="seagreen")
    else:
        ax.bar(x, tok_s, yerr=tok_s_std, capsize=4,
               color="steelblue", alpha=0.8, label="speculative")
        for xi, (v, s) in enumerate(zip(tok_s, speedups)):
            ax.text(xi, v + max(tok_s_std) * 0.1, f"{s:.2f}×",
                    ha="center", va="bottom", fontsize=8, color="steelblue")
    ax.axhline(baseline, color="tomato", linestyle="--", linewidth=1.5,
               label=f"target-only (λ={lam_target})")
    ax.set_ylabel("Tokens / second")
    ax.legend(fontsize=9)
    ax.set_title("Throughput  [hard vs stochastic mask]" if has_stoch_tp else "Throughput")

    # ── panel 2: accept rate (sampled + hard/stoch analytical overlay) ────
    ax = axes[1]
    ax.plot(x, accept, marker="o", color="steelblue", linewidth=2,
            label="sampled α (generative)")
    ax.fill_between(x, accept, alpha=0.15, color="steelblue")
    if alpha_results is not None:
        alpha_map = {r["lam_draft"]: r["alpha"] * 100 for r in alpha_results}
        analytical = [alpha_map.get(ld, float("nan")) for ld in lam_drafts]
        ax.plot(x, analytical, marker="x", color="darkorange", linewidth=1.5,
                linestyle="--", label="α_hard = 1 − TV(p,q)")
        if "alpha_stoch" in alpha_results[0]:
            stoch_map = {r["lam_draft"]: r["alpha_stoch"] * 100
                         for r in alpha_results}
            stoch_std_map = {r["lam_draft"]: r["alpha_stoch_std"] * 100
                             for r in alpha_results}
            s_mu  = [stoch_map.get(ld, float("nan"))     for ld in lam_drafts]
            s_std = [stoch_std_map.get(ld, float("nan")) for ld in lam_drafts]
            ax.plot(x, s_mu, marker="s", color="seagreen", linewidth=1.5,
                    linestyle="-.", label="α_stoch = E_mask[1−TV]")
            ax.fill_between(x,
                            [a - s for a, s in zip(s_mu, s_std)],
                            [a + s for a, s in zip(s_mu, s_std)],
                            alpha=0.12, color="seagreen")
    ax.set_ylabel("Accept rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="tomato", linestyle="--", linewidth=1, alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title("Draft token accept rate")

    # ── panel 3: E[tok/call] ──────────────────────────────────────────────
    ax = axes[2]
    ax.plot(x, etpc, marker="s", color="darkorange", linewidth=2,
            label="sampled")
    ax.fill_between(x, 1, etpc, alpha=0.15, color="darkorange")
    if alpha_results is not None:
        etpc_map = {r["lam_draft"]: r["expected_toks_per_call"]
                    for r in alpha_results}
        etpc_analytical = [etpc_map.get(ld, float("nan")) for ld in lam_drafts]
        ax.plot(x, etpc_analytical, marker="x", color="steelblue",
                linewidth=1.5, linestyle="--", label="theoretical γα + 1")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1,
               label="minimum (no benefit)")
    ax.axhline(gamma + 1, color="tomato", linestyle="--", linewidth=1,
               label=f"maximum (γ+1={gamma+1}, all accepted)")
    ax.set_ylabel("E[tokens / target call]\n(γ × α + 1)")
    ax.set_ylim(0, gamma + 1.5)
    ax.legend(fontsize=9)
    ax.set_title("Expected new tokens per target forward pass")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=8)
    axes[-1].set_xlabel("Draft λ  (FFN active %)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ── table ─────────────────────────────────────────────────────────────────────

def print_table(baseline: float, results: list[dict],
                lam_target: float, gamma: int,
                alpha_results: list[dict] | None = None) -> None:
    has_alpha_stoch = alpha_results is not None and "alpha_stoch" in alpha_results[0]
    has_stoch_tp    = "stoch_tok_per_s" in results[0]
    alpha_hard_map  = {r["lam_draft"]: r["alpha"]       for r in alpha_results} \
                      if alpha_results else {}
    alpha_stoch_map = {r["lam_draft"]: (r["alpha_stoch"], r["alpha_stoch_std"])
                       for r in alpha_results if "alpha_stoch" in r} \
                      if alpha_results else {}

    extra_cols = 0
    if has_alpha_stoch:
        extra_cols += 2   # α_stoch, ±
    if has_stoch_tp:
        extra_cols += 2   # stoch tok/s, stoch speedup
    w = 86 + extra_cols * 10
    print(f"\n{'─'*w}")
    print(f"  Baseline (target-only, λ={lam_target}): {baseline:.1f} tok/s")
    print(f"{'─'*w}")
    hdr = (f"  {'λ_draft':>8}  {'FFN%':>6}  {'params':>8}  {'tok/s':>8}  "
           f"{'speedup':>8}  {'accept%':>8}  {'α_hard':>7}  {'E[tok/call]':>12}")
    if has_alpha_stoch:
        hdr += f"  {'α_stoch':>8}  {'±':>6}"
    if has_stoch_tp:
        hdr += f"  {'stoch tok/s':>11}  {'stoch spd':>9}"
    print(hdr)
    print(f"{'─'*w}")
    for r in results:
        ld = r["lam_draft"]
        ah = alpha_hard_map.get(ld, float("nan"))
        row = (f"  {ld:>8.3g}  {r['ffn_active']:>5.0%}  "
               f"  {r['param_ratio']:>6.2f}×  {r['tok_per_s']:>7.1f}  "
               f"{r['speedup']:>8.2f}×  {r['accept_rate']:>7.1%}  {ah:>7.3f}  "
               f"{r['expected_toks_per_call']:>11.2f}")
        if has_alpha_stoch:
            mu, sd = alpha_stoch_map.get(ld, (float("nan"), float("nan")))
            row += f"  {mu:>8.3f}  {sd:>6.3f}"
        if has_stoch_tp:
            row += f"  {r['stoch_tok_per_s']:>10.1f}  {r['stoch_speedup']:>8.2f}×"
        print(row)
    print(f"{'─'*w}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep draft λ and benchmark speculative decoding performance")
    p.add_argument("--model_type",     default="olmo3", choices=["olmo3", "stoch"],
                   help="'olmo3': OLMo-3 checkpoint; 'stoch': PolyLM/ScaledPolyLM from stochastic_weight_test.py")
    p.add_argument("--model_tag",      default=None,
                   help="(stoch only) single variant to load, e.g. token_poly")
    p.add_argument("--model_tags",     nargs="+", default=None,
                   help="(stoch only) one or more variants to compare, e.g. token_poly hyper_joint")
    p.add_argument("--target_tag",     default=None,
                   help="(stoch only) use this model as the fixed target instead of model@lam_target (e.g. dense)")
    p.add_argument("--mode",           default="sweep",
                   choices=["sweep", "optimize", "context_sweep", "ram_budget",
                            "k_lambda_grid", "consensus_grid"],
                   help="'sweep': generative timing; 'optimize': analytical (Th.3.8/3.11); "
                        "'context_sweep': α(λ,T) across context lengths; "
                        "'ram_budget': analytical multi-draft speedup under a fixed GPU "
                        "memory budget (single vs distinct-copies vs shared-weights); "
                        "'k_lambda_grid': analytical speedup on a full (λ_draft × K) grid; "
                        "'consensus_grid': (λ × K) grid where K stochastic drafts decode a "
                        "token if they agree, else the full model is used")
    p.add_argument("--budget_gb",      type=float, default=48.0,
                   help="GPU memory budget in GB for ram_budget mode (default 48)")
    p.add_argument("--kv_seq_len",     type=int,   default=2048,
                   help="context length used to size the KV-cache reserve in ram_budget mode")
    p.add_argument("--k_max",          type=int,   default=256,
                   help="cap on the number of drafts considered in ram_budget mode")
    p.add_argument("--k_grid_min",     type=int,   default=2,
                   help="smallest K in k_lambda_grid mode (default 2)")
    p.add_argument("--k_grid_max",     type=int,   default=37,
                   help="largest K (inclusive) in k_lambda_grid mode (default 37)")
    p.add_argument("--no_consensus",   action="store_true",
                   help="ram_budget mode: skip the (empirical) consensus point and "
                        "keep the sweep fully analytical / fast")
    p.add_argument("--verify_points",  nargs="+", default=None, metavar="LAM,K",
                   help="run REAL timed consensus generation at these 'lam,K' points "
                        "to validate the grid (e.g. --verify_points 0.5,4 1.0,8). "
                        "Overrides --mode.")
    p.add_argument("--verify_gen",     type=int,   default=64,
                   help="tokens to generate per point in --verify_points (default 64)")
    p.add_argument("--verify_csv",     default=None,
                   help="CSV path for --verify_points results (default: derived from --csv)")
    p.add_argument("--ckpt_dir",       required=True)
    p.add_argument("--tokenizer_dir",  default=None)
    p.add_argument("--prompt",
                   default="The history of language models is")
    p.add_argument("--prompt_ids",     default=None,
                   help="space-separated token IDs (skip tokenizer)")
    p.add_argument("--lam_target",     type=float, default=0.1,
                   help="fixed target λ (dense model)")
    p.add_argument("--lam_draft_min",  type=float, default=None,
                   help="smallest draft λ to try (default: lam_target)")
    p.add_argument("--lam_draft_max",  type=float, default=3.0)
    p.add_argument("--n_lam",          type=int,   default=8,
                   help="number of draft λ values to sweep (log-spaced; ignored if --lam_step is set)")
    p.add_argument("--lam_step",       type=float, default=None,
                   help="linear step size between draft λ values (e.g. 0.02); "
                        "overrides --n_lam and uses np.arange(lam_min, lam_max, lam_step)")
    p.add_argument("--context_lengths", type=int, nargs="+",
                   default=[128, 512, 2048],
                   help="context lengths T to sweep in context_sweep mode (default: 128 512 2048)")
    p.add_argument("--decode_lengths", type=int, nargs="+", default=None,
                   help="tokens-decoded values N to sweep in context_sweep mode; "
                        "generates N tokens from the base prompt then measures α "
                        "on the extended context (e.g. --decode_lengths 0 50 200 500)")
    p.add_argument("--stochastic_samples", type=int, default=0,
                   help="if >0, also estimate E_mask[α] with Bernoulli-sampled gates "
                        "averaged over this many forward passes per (λ, T/N) point "
                        "(context_sweep mode only)")
    p.add_argument("--top_k_vocab",    type=int,   default=0,
                   help="if >0, restrict the target LM head to the top-K draft tokens "
                        "per step (sweep mode only). Reduces LM head FLOPs from "
                        "hidden×vocab to hidden×K_eff. "
                        "Output distribution is approximate (see speculative_step docs).")
    p.add_argument("--top_k_exact_fallback", type=float, default=0.0,
                   help="(used with --top_k_vocab) probability per round of falling back "
                        "to the full-vocab target pass, restoring the exact output-"
                        "distribution guarantee for that round. 0.0 = always approximate "
                        "(default); 1.0 = always exact (same as no top_k_vocab). "
                        "Expected LM head cost scales as (1-p)×K + p×vocab per round.")
    p.add_argument("--gamma",          type=int,   default=4,
                   help="draft tokens per round (sweep mode) or fixed γ overlay (optimize)")
    p.add_argument("--gamma_max",      type=int,   default=64,
                   help="max γ to consider when optimising (optimize mode)")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int,   default=100)
    p.add_argument("--n_trials",       type=int,   default=3,
                   help="trials per λ value (sweep mode only)")
    p.add_argument("--device",         default="auto")
    p.add_argument("--compile",        action="store_true",
                   help="apply torch.compile(mode='reduce-overhead') to each draft "
                        "model after structural pruning; requires PyTorch 2.0+ and "
                        "works best on CUDA/XPU")
    p.add_argument("--out",            default=None,
                   help="output plot filename (default: speculative_sweep.png or speculative_opt.png)")
    p.add_argument("--csv",            default=None,
                   help="if set, also write the per-λ results to this CSV path "
                        "(optimize and sweep modes). Columns are the union of all "
                        "result fields plus model tag / lam_target / baseline.")
    return p.parse_args()


def write_results_csv(
    rows: list[dict],
    out_path: Path,
    extra: dict | None = None,
) -> None:
    """Write a list of result dicts to a long-format CSV.

    The column set is the union of all keys across rows (first-seen order),
    optionally prefixed by constant columns in ``extra`` (e.g. model tag,
    lam_target, baseline). Missing keys are written as empty cells so that
    rows from different sweep modes can share one file.
    """
    if not rows:
        print(f"  (no rows to write to {out_path})")
        return

    extra = extra or {}
    cols: list[str] = list(extra.keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**extra, **r})
    print(f"  Wrote {len(rows)} rows → {out_path}")


def main() -> None:
    args = parse_args()

    global _compile_draft
    _compile_draft = args.compile

    if args.device == "auto":

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.compile and device.type not in ("cuda", "xpu"):
        print(f"  Note: --compile has no effect on {device.type}; skipping torch.compile")
        _compile_draft = False

    # Resolve the list of model tags to run
    if args.model_type == "stoch":
        tags = args.model_tags or ([args.model_tag] if args.model_tag else [None])
    else:
        tags = [None]

    # Load a fixed target model if --target_tag specified
    target_model = None
    target_label = None
    if args.target_tag and args.model_type == "stoch":
        print(f"Loading target model '{args.target_tag}' from {args.ckpt_dir} …")
        target_model, _ = load_checkpoint(args.ckpt_dir, device,
                                          model_tag=args.target_tag)
        target_model.eval()
        target_label = args.target_tag

    # Load first model just for prompt setup / single-tag path
    print(f"Loading checkpoint from {args.ckpt_dir} …")
    load_kwargs = {"model_tag": tags[0]} if args.model_type == "stoch" else {}
    model, ckpt_info = load_checkpoint(args.ckpt_dir, device, **load_kwargs)
    n_params = sum(p.numel() for p in set(model.parameters()))
    print(f"  {n_params/1e6:.1f}M parameters  device={device}")

    # tokenizer + prompt
    tokenizer = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer
        tok_path = Path(args.tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_path.resolve()) if tok_path.is_dir() else args.tokenizer_dir)

    if args.prompt_ids:
        prompt = torch.tensor(
            [int(x) for x in args.prompt_ids.split()], dtype=torch.long)
    elif tokenizer is not None:
        prompt = torch.tensor(
            tokenizer.encode(args.prompt, add_special_tokens=True),
            dtype=torch.long)
    elif args.model_type == "stoch":
        from stochastic_weight_test import build_vocab, build_tensor
        _vocab_n   = ckpt_info.get("vocab",   10_000)
        _seq_len   = ckpt_info.get("seq_len", 64)
        print(f"  Building WikiText-2 vocab (n={_vocab_n}) for prompt …")
        _vocab  = build_vocab(_vocab_n)
        _val_ds = build_tensor("validation", _vocab, _seq_len)
        _idx    = torch.randint(len(_val_ds), (1,)).item()
        _seq, _ = _val_ds[_idx]
        prompt  = _seq[:max(1, _seq_len // 2)]
        print(f"  Using WikiText-2 validation sequence as prompt.")
    else:
        prompt = torch.tensor([1], dtype=torch.long)
        print("  No tokenizer — using single BOS token as prompt.")

    prompt = prompt.to(device)
    print(f"  Prompt: {len(prompt)} tokens")

    lam_min = args.lam_draft_min if args.lam_draft_min is not None else args.lam_target
    if args.lam_step is not None:
        lam_draft_values = list(np.unique(np.round(
            np.arange(lam_min, args.lam_draft_max + args.lam_step * 0.5, args.lam_step),
            decimals=6)))
    else:
        lam_draft_values = list(np.unique(np.round(
            np.logspace(np.log10(max(lam_min, 1e-4)),
                        np.log10(args.lam_draft_max),
                        args.n_lam), decimals=4)))

    # ── verify_points: real timed consensus generation (overrides --mode) ──
    if args.verify_points:
        try:
            points = [(float(p.split(",")[0]), int(p.split(",")[1]))
                      for p in args.verify_points]
        except (ValueError, IndexError):
            raise SystemExit("--verify_points expects 'lam,K' pairs, "
                             "e.g. --verify_points 0.5,4 1.0,8")

        print(f"\nVerifying {len(points)} (λ, K) point(s) with real timed "
              f"consensus generation ({args.verify_gen} tokens each)")
        verify_results = verify_consensus_points(
            model, prompt,
            lam_target=args.lam_target,
            points=points,
            n_gen=args.verify_gen,
            temperature=args.temperature,
            device=device,
            target_model=target_model,
        )

        csv_path = args.verify_csv
        if csv_path is None and args.csv:
            csv_path = str(Path(args.csv).with_suffix("")) + "_verify.csv"
        if csv_path:
            write_results_csv(
                verify_results, Path(csv_path),
                extra={"mode": "verify_consensus",
                       "lam_target": args.lam_target,
                       "target_model": target_label or ""})
        return

    # ── ram_budget mode ───────────────────────────────────────────────────
    if args.mode == "ram_budget":
        out = Path(args.out) if args.out else Path("speculative_ram_budget.png")

        print(f"\nRAM-budget multi-draft sweep  "
              f"({len(lam_draft_values)} λ_draft values, γ_max={args.gamma_max})")
        if target_label:
            print(f"  Target model   : {target_label}  (fixed)")
        else:
            print(f"  lam_target     : {args.lam_target}")
        print(f"  λ range        : {lam_draft_values[0]:.3g} – {lam_draft_values[-1]:.3g}")

        budget_results = sweep_ram_budget(
            model, prompt,
            lam_target=args.lam_target,
            lam_draft_values=lam_draft_values,
            budget_gb=args.budget_gb,
            kv_seq_len=args.kv_seq_len,
            temperature=args.temperature,
            device=device,
            gamma_max=args.gamma_max,
            k_max=args.k_max,
            target_model=target_model,
            consensus=not args.no_consensus,
        )
        make_ram_budget_plot(budget_results, args.budget_gb, args.kv_seq_len, out)

        if args.csv:
            write_results_csv(
                budget_results, Path(args.csv),
                extra={"mode": "ram_budget",
                       "lam_target": args.lam_target,
                       "budget_gb": args.budget_gb,
                       "kv_seq_len": args.kv_seq_len,
                       "target_model": target_label or ""})
        return

    # ── k_lambda_grid mode ────────────────────────────────────────────────
    if args.mode == "k_lambda_grid":
        k_values = list(range(args.k_grid_min, args.k_grid_max + 1))

        print(f"\nK × λ grid sweep  "
              f"({len(lam_draft_values)} λ_draft × {len(k_values)} K, γ_max={args.gamma_max})")
        if target_label:
            print(f"  Target model   : {target_label}  (fixed)")
        else:
            print(f"  lam_target     : {args.lam_target}")
        print(f"  λ range        : {lam_draft_values[0]:.3g} – {lam_draft_values[-1]:.3g}")
        print(f"  K range        : {k_values[0]} – {k_values[-1]}")

        grid_results = sweep_k_lambda_grid(
            model, prompt,
            lam_target=args.lam_target,
            lam_draft_values=lam_draft_values,
            k_values=k_values,
            temperature=args.temperature,
            device=device,
            gamma_max=args.gamma_max,
            target_model=target_model,
        )

        if args.out:
            make_k_lambda_grid_plot(grid_results, Path(args.out))

        if args.csv:
            write_results_csv(
                grid_results, Path(args.csv),
                extra={"mode": "k_lambda_grid",
                       "lam_target": args.lam_target,
                       "target_model": target_label or ""})
        return

    # ── consensus_grid mode ───────────────────────────────────────────────
    if args.mode == "consensus_grid":
        k_values = list(range(args.k_grid_min, args.k_grid_max + 1))

        print(f"\nConsensus (K-agreement) grid sweep  "
              f"({len(lam_draft_values)} λ_draft × {len(k_values)} K)")
        if target_label:
            print(f"  Target model   : {target_label}  (fixed)")
        else:
            print(f"  lam_target     : {args.lam_target}")
        print(f"  λ range        : {lam_draft_values[0]:.3g} – {lam_draft_values[-1]:.3g}")
        print(f"  K range        : {k_values[0]} – {k_values[-1]}")

        consensus_results = sweep_consensus_grid(
            model, prompt,
            lam_target=args.lam_target,
            lam_draft_values=lam_draft_values,
            k_values=k_values,
            temperature=args.temperature,
            device=device,
            target_model=target_model,
        )

        if args.out:
            make_consensus_grid_plot(consensus_results, Path(args.out))

        if args.csv:
            write_results_csv(
                consensus_results, Path(args.csv),
                extra={"mode": "consensus_grid",
                       "lam_target": args.lam_target,
                       "target_model": target_label or ""})
        return

    # ── context_sweep mode ────────────────────────────────────────────────
    if args.mode == "context_sweep":
        base_out = Path(args.out) if args.out else None

        # ── context-length sub-sweep (always runs) ────────────────────────
        ctx_lengths = sorted(set(args.context_lengths))
        max_ctx = max(ctx_lengths)
        ctx_out = base_out or Path("speculative_context_sweep.png")

        print(f"\nContext sweep: {len(lam_draft_values)} λ values × {len(ctx_lengths)} context lengths")
        print(f"  λ range  : {lam_draft_values[0]:.3g} – {lam_draft_values[-1]:.3g}")
        print(f"  T values : {ctx_lengths}")
        print(f"  λ_target : {args.lam_target}")
        if len(prompt) < max_ctx:
            print(f"  Note: base prompt ({len(prompt)} tokens) < max T={max_ctx}; "
                  "will tile tokens to fill longer contexts.")
        print()

        results_by_ctx = sweep_alpha_contexts(
            model, prompt,
            lam_target=args.lam_target,
            lam_draft_values=lam_draft_values,
            context_lengths=ctx_lengths,
            temperature=args.temperature,
            device=device,
            n_stoch_samples=args.stochastic_samples,
        )
        enrich_with_speedup(results_by_ctx, model, args.lam_target, device,
                            gamma_max=args.gamma_max)
        print_context_sweep_table(results_by_ctx, args.lam_target)
        print_speedup_opportunities(results_by_ctx, "T", args.lam_target)
        make_combined_sweep_plot(results_by_ctx, "T", args.lam_target, ctx_out,
                                 cmap_name="viridis")

        # ── decode-length sub-sweep (runs if --decode_lengths given) ──────
        if args.decode_lengths:
            dec_lengths = sorted(set(args.decode_lengths))
            dec_out = (base_out.with_name(base_out.stem + "_decode" + base_out.suffix)
                       if base_out else Path("speculative_decode_sweep.png"))

            print(f"\nDecode sweep: {len(lam_draft_values)} λ values × {len(dec_lengths)} decode lengths")
            print(f"  N values : {dec_lengths}")
            print(f"  Base prompt: {len(prompt)} tokens\n")

            results_by_N = sweep_alpha_decode_lengths(
                model, prompt,
                lam_target=args.lam_target,
                lam_draft_values=lam_draft_values,
                decode_lengths=dec_lengths,
                temperature=args.temperature,
                device=device,
                n_stoch_samples=args.stochastic_samples,
            )
            enrich_with_speedup(results_by_N, model, args.lam_target, device,
                                gamma_max=args.gamma_max)
            print_decode_sweep_table(results_by_N, args.lam_target)
            print_speedup_opportunities(results_by_N, "N", args.lam_target)
            make_combined_sweep_plot(results_by_N, "N", args.lam_target, dec_out,
                                     cmap_name="plasma")

        return

    # ── optimize mode ──────────────────────────────────────────────────────
    if args.mode == "optimize":
        out = Path(args.out) if args.out else Path("speculative_opt.png")
        multi_tag = len(tags) > 1

        all_opt_results: dict[str, list[dict]] = {}
        for tag in tags:
            # Load model for this tag (reuse already-loaded model for first tag)
            if tag != tags[0] and args.model_type == "stoch":
                m, _ = load_checkpoint(args.ckpt_dir, device, model_tag=tag)
            else:
                m = model

            ffn_frac  = _ffn_param_fraction(m)
            norm_diag = compute_norm_entropy(m)
            tag_label = tag or "model"

            print(f"\nOptimise mode [{tag_label}]  "
                  f"({len(lam_draft_values)} λ_draft values, γ_max={args.gamma_max})")
            if target_label:
                print(f"  Target model   : {target_label}  (fixed)")
            else:
                print(f"  lam_target     : {args.lam_target}")
            print(f"  FFN param fraction : {ffn_frac:.3f}")
            print(f"  λ range            : {lam_draft_values[0]:.3g} – {lam_draft_values[-1]:.3g}")
            if norm_diag:
                print("\n  ── H(f_s) norm-entropy diagnostic ──────────────────────")
                print(f"  n neurons    : {norm_diag['n_neurons']:,}")
                print(f"  mean / std   : {norm_diag['mean']:.4f} / {norm_diag['std']:.4f}")
                print(f"  CV (σ/μ)     : {norm_diag['cv']:.4f}")
                print(f"  H(f_s)       : {norm_diag['H_fs']:.4f}  "
                      f"({'cliff' if norm_diag['H_fs'] < 0 else 'spread — may work'})")
                print(f"  {'─'*52}\n")

            print("λ-by-λ optimisation  (Th. 3.8 / 3.11):")
            opt_results = sweep_lambda_optimization(
                m, prompt,
                lam_target=args.lam_target,
                lam_draft_values=lam_draft_values,
                temperature=args.temperature,
                device=device,
                gamma_max=args.gamma_max,
                target_model=target_model,
            )
            print_optimization_table(opt_results, args.lam_target)
            all_opt_results[tag_label] = opt_results

        if multi_tag:
            make_comparison_plot(all_opt_results, args.lam_target, args.gamma_max,
                                 out, target_label=target_label)
        else:
            tag_label = tags[0] or "model"
            norm_diag = compute_norm_entropy(model)
            make_optimization_plot(all_opt_results[tag_label], args.lam_target,
                                   args.gamma_max, out, norm_diag=norm_diag)

        if args.csv:
            csv_rows = []
            for tag_label, opt_results in all_opt_results.items():
                for r in opt_results:
                    csv_rows.append({"model": tag_label, **r})
            write_results_csv(
                csv_rows, Path(args.csv),
                extra={"mode": "optimize",
                       "lam_target": args.lam_target,
                       "target_model": target_label or ""})
        return

    # ── sweep mode (default) ───────────────────────────────────────────────
    out = Path(args.out) if args.out else Path("speculative_sweep.png")
    print(f"\nSweep: {len(lam_draft_values)} λ_draft values: "
          f"{[f'{v:.3g}' for v in lam_draft_values]}")
    print(f"γ={args.gamma}  max_new_tokens={args.max_new_tokens}  "
          f"n_trials={args.n_trials}\n")

    # ── analytical α sweep (fast) ──────────────────────────────────────────
    print("Analytical α sweep  (1 − TV(p,q), no sampling):")
    alpha_results = sweep_alpha(
        model, prompt,
        lam_target=args.lam_target,
        lam_draft_values=lam_draft_values,
        temperature=args.temperature,
        device=device,
        gamma=args.gamma,
        n_stoch_samples=args.stochastic_samples,
    )

    # ── full generative sweep (timing) ────────────────────────────────────
    baseline, results = run_sweep(
        model, prompt,
        lam_target=args.lam_target,
        lam_draft_values=lam_draft_values,
        gamma=args.gamma,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        n_trials=args.n_trials,
        device=device,
        n_stoch_trials=args.stochastic_samples,
        top_k_vocab=args.top_k_vocab,
        top_k_exact_fallback=args.top_k_exact_fallback,
    )

    print_table(baseline, results, args.lam_target, args.gamma,
                alpha_results=alpha_results)
    make_plot(baseline, results, args.lam_target, args.gamma, out,
              alpha_results=alpha_results)

    if args.csv:
        # Merge analytical α (TV-distance, expected toks/call) into the timing
        # rows by lam_draft so the CSV carries both in one place.
        alpha_by_lam = {r["lam_draft"]: r for r in alpha_results}
        merged = []
        for r in results:
            a = alpha_by_lam.get(r["lam_draft"], {})
            merged.append({**a, **r})
        write_results_csv(
            merged, Path(args.csv),
            extra={"mode": "sweep",
                   "lam_target": args.lam_target,
                   "gamma": args.gamma,
                   "baseline_tok_per_s": baseline})


if __name__ == "__main__":
    main()
