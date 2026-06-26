# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "transformers>=5.0",
#   "numpy>=1.24",
# ]
# ///
"""
olmo3_speculative_decode.py

Speculative decoding with a single OLMo-3 stochastic-pruning checkpoint.

The same set of weights serves two roles by switching λ:

  draft  model  (high λ → aggressive pruning → sparse → fast)
  target model  (low  λ → minimal pruning   → dense → accurate)

Algorithm (Leviathan et al. 2022, "Fast Inference from Transformers via
Speculative Decoding"):

  Repeat until max_new_tokens:
    1. DRAFT: run draft model autoregressively for γ steps,
              collecting tokens x̃_i and their distributions q_i(·).
    2. VERIFY: run target model once on prefix + all γ draft tokens,
               collecting distributions p_i(·) at each position.
    3. ACCEPT/REJECT for i = 1..γ:
         r ~ U(0,1)
         if r ≤ p_i(x̃_i) / q_i(x̃_i):  accept x̃_i
         else: sample from norm(max(0, p_i − q_i)), stop round
       If all γ accepted: sample bonus token from p_{γ+1}.

  The output distribution is exactly the target distribution (proven).
  Expected new tokens per target call = γ × α + 1  (α = accept rate).

Usage:
  # generate from a text prompt
  uv run python olmo3_speculative_decode.py \\
      --ckpt_dir ./results/olmo3 \\
      --tokenizer_dir ./olmo3_data/tokenizer \\
      --prompt "The history of language models" \\
      --max_new_tokens 200 \\
      --lam_draft 2.0 --lam_target 0.1 --gamma 4

  # benchmark speculative vs target-only vs draft-only
  uv run python olmo3_speculative_decode.py \\
      --ckpt_dir ./results/olmo3 \\
      --tokenizer_dir ./olmo3_data/tokenizer \\
      --benchmark

  # work with raw token IDs (no tokenizer)
  uv run python olmo3_speculative_decode.py \\
      --ckpt_dir ./results/olmo3 \\
      --prompt_ids "1 2 3 4 5" \\
      --max_new_tokens 50

Note on wall-clock speedup:
  Real speedup requires either (a) a physically compressed draft model
  (remove pruned rows/columns) or (b) sparse kernel support.  With dense
  matmuls, PyTorch executes multiplications by zero and the masked model
  runs at similar speed to the dense one.  The benefit shown here is the
  correct accept-rate and output-distribution guarantee; combine with
  physical compression (see build_compressed in stochastic_weight_test.py)
  for actual throughput gains.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Olmo3Config, Olmo3ForCausalLM

# import gate classes from the training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from olmo3_mini_train import (
    BSplineScaledOlmo3Attn,
    BSplineScaledOlmo3MLP,
    PerHeadRMSNorm,
    _STOCH_TYPES,
    inject_embedding_amplitude,
    inject_per_head_qk_norm,
    inject_stochastic_attn,
    inject_stochastic_mlp,
    invalidate_spline_cache,
    set_hard_mask,
    set_lam,
    set_sample_mask,
    structural_prune,
)


# ── physical compression ───────────────────────────────────────────────────────

@torch.no_grad()
def build_compressed_olmo3(model: nn.Module, lam_val: float,
                            device: torch.device,
                            stochastic: bool = False) -> nn.Module:
    """Physically remove inactive units from all stochastic gates at lam_val.

    Returns a deep copy with smaller weight matrices — the draft model will
    run genuinely faster than the full-size target model.

    Delegates to structural_prune() from olmo3_mini_train, which slices
    gate_proj/up_proj rows and down_proj cols for FFN, and all four attention
    projections (q/k/v/o) for attention heads.

    stochastic: if True, the keep-mask is drawn from Bernoulli(ρ) instead of the
        hard ρ>0.5 threshold, so repeated calls yield *distinct* pruned subnets
        (a physically-compressed equivalent of stochastic gate sampling).
    """
    import copy

    m = copy.deepcopy(model).to(device)
    m.eval()
    structural_prune(m, lam_val, stochastic=stochastic)
    return m


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir: str | Path,
                    device: torch.device) -> tuple[nn.Module, dict]:
    """Reconstruct model from config.json + model.pt + train_args.json."""
    ckpt_dir   = Path(ckpt_dir).resolve()
    config     = Olmo3Config.from_pretrained(ckpt_dir)
    train_args = {}
    args_path  = ckpt_dir / "train_args.json"
    if args_path.exists():
        with open(args_path) as f:
            train_args = json.load(f)

    scale_output     = train_args.get("scale_output", True)
    embed_amp        = train_args.get("embed_amp", False)
    no_per_head_norm = train_args.get("no_per_head_norm", False)
    n_knots          = train_args.get("n_knots", 8)
    degree           = train_args.get("degree", 3)

    model = Olmo3ForCausalLM(config)
    inject_stochastic_mlp(model, scale_output=scale_output,
                          n_knots=n_knots, degree=degree)
    inject_stochastic_attn(model, scale_output=scale_output,
                           n_knots=n_knots, degree=degree)
    # Match the module structure used at training time *before* loading so the
    # checkpoint tensors line up.  Training (olmo3_mini_train.train) injects
    # per-head q/k norm by default — its checkpoint stores (head_dim,) weights,
    # not the flat (n_heads*head_dim,) of stock Olmo3RMSNorm.
    if not no_per_head_norm:
        inject_per_head_qk_norm(model)
    if embed_amp:
        inject_embedding_amplitude(model, n_knots=n_knots, degree=degree)

    state = torch.load(ckpt_dir / "model.pt", map_location="cpu",
                       weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")

    model = model.to(device)
    model.eval()
    return model, train_args


# ── mask control ──────────────────────────────────────────────────────────────

@torch.no_grad()
def commit_masks(model: nn.Module, lam_val: float,
                 device: torch.device) -> None:
    """Fix all gate modules to deterministic hard masks at lam_val.

    Sets hard=True and current_lam on every stochastic gate so subsequent
    forward passes use binary thresholding at ρ≥0.5 with no sampling.
    """
    lam = torch.tensor(lam_val, dtype=torch.float32, device=device)
    set_lam(model, lam)
    set_hard_mask(model, hard=True)


def active_fraction(model: nn.Module, lam_val: float,
                    device: torch.device) -> dict[str, float]:
    """Report fraction of active units per gate type at this λ."""
    lam = torch.tensor(lam_val, dtype=torch.float32, device=device)
    # Invalidate the spline cache for this λ. inclusion_probs() -> _tau() returns
    # a cached value if present (ignoring the lam arg), so a model previously
    # committed at a different λ would otherwise report a stale active fraction
    # (e.g. 100% at every λ after a λ=0 baseline pass). Use the cache-clearing
    # helper rather than set_lam() so we never cache a non-leaf tensor that would
    # break the copy.deepcopy() in build_compressed_olmo3().
    invalidate_spline_cache(model)
    result: dict[str, list[float]] = {"ffn": [], "attn": []}
    for m in model.modules():
        if isinstance(m, BSplineScaledOlmo3MLP):
            result["ffn"].append((m.inclusion_probs(lam) >= 0.5).float().mean().item())
        elif isinstance(m, BSplineScaledOlmo3Attn):
            result["attn"].append((m.inclusion_probs(lam) >= 0.5).float().mean().item())
    return {k: float(sum(v) / len(v)) if v else float("nan")
            for k, v in result.items()}


# ── forward utilities ─────────────────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def _restricted_lm_head(model: nn.Module, candidate_ids: torch.Tensor):
    """Temporarily replace model.lm_head.weight with only the candidate rows.

    The forward pass then produces logits of shape (..., K) instead of
    (..., vocab), where K = len(candidate_ids).  The KV cache entries are
    unaffected — they live in the transformer layers, not the LM head.

    Works with tie_word_embeddings=True because we only rebind the lm_head
    attribute; embed_tokens keeps the original weight tensor.
    """
    lm = model.lm_head

    orig_weight = lm.weight
    lm.weight = nn.Parameter(
        orig_weight.data[candidate_ids].contiguous(), requires_grad=False)

    try:
        yield
    finally:
        lm.weight = orig_weight


@torch.no_grad()
def _logprobs_at_all(model: nn.Module, tokens: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """Full forward pass; return log-softmax over vocab at every position.

    tokens: (T,) int64
    returns: (T, vocab) float32
    """
    out    = model(input_ids=tokens.unsqueeze(0))
    logits = out.logits[0].float()          # (T, vocab)
    if temperature != 1.0:
        logits = logits / temperature
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def _logprobs_last(model: nn.Module, tokens: torch.Tensor,
                   temperature: float) -> torch.Tensor:
    """Forward pass; return log-softmax at the last position only.

    tokens: (T,) int64
    returns: (vocab,) float32
    """
    return _logprobs_at_all(model, tokens, temperature)[-1]


def _sample(log_probs: torch.Tensor) -> int:
    return int(torch.multinomial(log_probs.exp().clamp(min=0), 1).item())


def _trim_kv_cache(past, max_seq_len: int):
    """Return a KV cache truncated to max_seq_len tokens.

    Handles the modern DynamicCache object (transformers >= 4.36) via its
    built-in crop() method, and the legacy tuple-of-tuples format.
    """
    import copy
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past, DynamicCache):
            trimmed = copy.deepcopy(past)
            trimmed.crop(max_seq_len)
            return trimmed
    except ImportError:
        pass
    # Legacy tuple-of-tuples: (layers, (k, v), batch, heads, seq, head_dim)
    return tuple(
        tuple(kv[..., :max_seq_len, :] for kv in layer)
        for layer in past
    )


@torch.no_grad()
def expected_acceptance_rate(
    target_model: nn.Module,
    draft_model: nn.Module,
    tokens: torch.Tensor,       # (T,) int64, on device
    lam_target: float,
    temperature: float,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Compute the theoretical per-token acceptance rate α from Section 3.2.

    The expected acceptance rate for a single token with target distribution p
    and draft distribution q is:

        α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)

    This requires only two forward passes and no sampling, giving the exact
    expectation rather than a noisy Monte Carlo estimate.

    Args:
        target_model: full model (already has masks committed at lam_target,
                      or will have commit_masks called internally)
        draft_model:  compressed/masked draft model
        tokens:       context sequence to evaluate on
        lam_target:   λ for target model (used to commit masks)
        temperature:  softmax temperature
        device:       torch device

    Returns:
        mean_alpha:  scalar mean of α across all token positions
        alpha_per_pos: (T-1,) tensor of per-position α values
    """
    commit_masks(target_model, lam_target, device)
    p = _logprobs_at_all(target_model, tokens, temperature).exp()  # (T, vocab)
    q = _logprobs_at_all(draft_model,  tokens, temperature).exp()  # (T, vocab)

    # α_i = Σ_x min(p_i(x), q_i(x))  — overlap between distributions
    alpha = torch.minimum(p, q).sum(dim=-1)   # (T,)

    # position 0 predicts token 1, so all T positions are valid
    mean_alpha = alpha.mean().item()
    return mean_alpha, alpha


@torch.no_grad()
def stochastic_expected_acceptance_rate(
    model: nn.Module,
    tokens: torch.Tensor,
    lam_target: float,
    lam_draft: float,
    temperature: float,
    device: torch.device,
    n_samples: int = 10,
) -> tuple[float, float]:
    """Estimate E_mask[α] where the draft draws a fresh Bernoulli subnet each sample.

    The target model keeps its hard mask at lam_target.  The draft model uses
    the same weights but samples a different random subnet on every forward pass;
    α is averaged across n_samples draws, giving an unbiased estimate of the
    expected TV overlap under the stochastic gating distribution.

    Returns (mean_alpha, std_alpha).
    """
    import copy

    # Target: hard mask — fixed for all samples
    commit_masks(model, lam_target, device)
    p = _logprobs_at_all(model, tokens, temperature).exp()   # (T, vocab), fixed

    # Draft: fresh Bernoulli sample on every forward pass
    draft_m = copy.deepcopy(model)
    lam_t = torch.tensor(lam_draft, dtype=torch.float32, device=device)
    set_lam(draft_m, lam_t)
    set_hard_mask(draft_m, hard=False)
    set_sample_mask(draft_m, sample=True)

    sample_alphas: list[float] = []
    for _ in range(n_samples):
        q     = _logprobs_at_all(draft_m, tokens, temperature).exp()  # (T, vocab)
        alpha = torch.minimum(p, q).sum(dim=-1).mean().item()
        sample_alphas.append(alpha)

    import numpy as _np
    return float(_np.mean(sample_alphas)), float(_np.std(sample_alphas))


# ── speculative decoding ──────────────────────────────────────────────────────

@torch.no_grad()
def speculative_step(
    model: nn.Module,
    prefix: torch.Tensor,       # (T,) int64, on device
    lam_draft: float,
    lam_target: float,
    gamma: int,
    temperature: float,
    device: torch.device,
    draft_model: nn.Module | None = None,
    target_past=None,           # KV cache for the prefix from prior rounds
    top_k_vocab: int = 0,       # 0 = full vocab; >0 = restrict target LM head
    top_k_exact_fallback: float = 0.0,  # probability of using full vocab (exact guarantee)
) -> tuple[list[int], int, object]:
    """
    One round of speculative decoding.

    If draft_model is provided it is used for the draft phase (e.g. a
    physically compressed copy built by build_compressed_olmo3); model is
    always used as the target.  If draft_model is None the original
    mask-switching approach is used (both roles handled by model).

    target_past: KV cache covering tokens up to (but not including) the
                 current prefix tail, carried across rounds by
                 generate_speculative.  On the first call pass None.

    top_k_vocab: if >0, restrict the target LM head to the union of the
                 top-K tokens from each draft position plus the actual draft
                 tokens.  This reduces the target LM head FLOPs from
                 (gamma+1)×hidden×vocab to (gamma+1)×hidden×K_eff where
                 K_eff ≤ K×gamma + gamma.  Acceptance/rejection is done over
                 the restricted vocabulary (an approximation: the true target
                 distribution assigns non-zero mass outside the candidate set).

    top_k_exact_fallback: only used when top_k_vocab > 0.  Each round,
                 independently with this probability, skip the restricted path
                 and run the full-vocab target pass instead.  This restores
                 the exact output-distribution guarantee for that round at the
                 cost of one full LM head call.  Expected LM head cost per
                 round scales as (1-p)×K + p×V.  Default 0.0 (always
                 approximate); set to e.g. 0.05 for a 5% exact-fallback rate.

    Returns:
        new_tokens    list of 1..gamma+1 accepted token IDs
        n_accepted    number of draft tokens that were accepted (not resampled)
        new_past      updated KV cache ending at the last accepted token
    """
    vocab = model.config.vocab_size

    # ── 1. draft phase (KV cached) ────────────────────────────────────────
    if draft_model is None:
        commit_masks(model, lam_draft, device)
        _draft = model
    else:
        _draft = draft_model

    draft_tokens = []
    draft_lp     = []        # list of (vocab,) tensors — full draft distributions

    ctx       = prefix.clone()
    draft_past = None
    # Prime the draft KV cache with the full prefix, then feed one token at a time.
    for _ in range(gamma):
        inp  = ctx.unsqueeze(0) if draft_past is None else ctx[-1:].unsqueeze(0)
        out  = _draft(input_ids=inp, past_key_values=draft_past, use_cache=True)
        lp   = torch.log_softmax(
            out.logits[0, -1].float() / temperature, dim=-1)
        draft_past = out.past_key_values
        tok  = _sample(lp)
        draft_tokens.append(tok)
        draft_lp.append(lp)
        ctx = torch.cat([ctx, torch.tensor([tok], device=device)])

    draft_lp_t = torch.stack(draft_lp)   # (gamma, vocab)

    # ── 2. target verification (KV cached, single pass over draft tokens) ─
    if draft_model is None:
        commit_masks(model, lam_target, device)

    # Feed the target model only the tokens it hasn't seen yet.
    # On the first round target_past=None → process the full prefix + draft.
    # On subsequent rounds target_past covers everything up to prefix[-1],
    # so we only need to process the γ draft tokens.
    draft_tensor = torch.tensor(draft_tokens, device=device)
    if target_past is None:
        target_inp = ctx.unsqueeze(0)                         # (1, T+gamma)
    else:
        target_inp = torch.cat(
            [prefix[-1:], draft_tensor]).unsqueeze(0)         # (1, 1+gamma)

    # ── optional: restricted LM head ──────────────────────────────────────
    use_restricted = (top_k_vocab > 0 and
                      (top_k_exact_fallback <= 0.0 or
                       torch.rand(1).item() > top_k_exact_fallback))
    if use_restricted:
        K = min(top_k_vocab, vocab)
        # candidate set: union of top-K at each draft position + draft tokens
        cands: set[int] = set(draft_tokens)
        for lp in draft_lp:
            cands.update(lp.topk(K).indices.tolist())
        candidate_ids = torch.tensor(sorted(cands), dtype=torch.long,
                                     device=device)   # (K_eff,)

        with _restricted_lm_head(model, candidate_ids):
            target_out = model(input_ids=target_inp,
                               past_key_values=target_past, use_cache=True)

        # logits shape: (1, *, K_eff)
        target_logits = target_out.logits[0, -(gamma + 1):].float()
        if temperature != 1.0:
            target_logits = target_logits / temperature
        target_lp_r = torch.log_softmax(target_logits, dim=-1)  # (gamma+1, K_eff)
        updated_past = target_out.past_key_values

        # fast lookup: vocab token id → position in candidate_ids
        cand_inv = {int(candidate_ids[j]): j for j in range(len(candidate_ids))}

        # ── 3a. accept / reject (restricted) ──────────────────────────────
        new_tokens = []
        n_accepted = 0

        for i, tok in enumerate(draft_tokens):
            j     = cand_inv[tok]
            q_tok = draft_lp_t[i, tok].exp().item()
            p_tok = target_lp_r[i, j].exp().item()

            r = torch.rand(1).item()
            if r <= min(1.0, p_tok / (q_tok + 1e-9)):
                new_tokens.append(tok)
                n_accepted += 1
            else:
                # corrected distribution over the restricted candidate set:
                # normalize q to the candidate set so both p and q sum to 1
                p_cand   = target_lp_r[i].exp()                # (K_eff,)
                q_cand   = draft_lp_t[i][candidate_ids].exp()  # (K_eff,)
                q_cand   = q_cand / q_cand.sum().clamp(min=1e-9)
                adjusted = (p_cand - q_cand).clamp(min=0.0)
                adj_sum  = adjusted.sum()
                if adj_sum < 1e-9:
                    adjusted = p_cand
                else:
                    adjusted = adjusted / adj_sum
                idx = int(torch.multinomial(adjusted, 1).item())
                new_tokens.append(int(candidate_ids[idx]))
                trimmed_past = _trim_kv_cache(updated_past, len(prefix) + i)
                return new_tokens, n_accepted, trimmed_past

        # all gamma tokens accepted → bonus from restricted target at position gamma
        bonus_idx = int(torch.multinomial(target_lp_r[gamma].exp(), 1).item())
        new_tokens.append(int(candidate_ids[bonus_idx]))
        return new_tokens, n_accepted, updated_past

    # ── full-vocab path (original) ────────────────────────────────────────
    target_out  = model(input_ids=target_inp,
                        past_key_values=target_past, use_cache=True)
    # logits at positions corresponding to p_1..p_{gamma+1}
    # last (gamma+1) logit rows cover prefix[-1]→draft[0], draft[0]→draft[1], …, draft[gamma-1]→bonus
    target_logits = target_out.logits[0, -(gamma + 1):].float()
    if temperature != 1.0:
        target_logits = target_logits / temperature
    target_lp_t = torch.log_softmax(target_logits, dim=-1)   # (gamma+1, vocab)
    updated_past = target_out.past_key_values

    # ── 3. accept / reject ────────────────────────────────────────────────
    new_tokens = []
    n_accepted = 0

    for i, tok in enumerate(draft_tokens):
        q_tok = draft_lp_t[i, tok].exp().item()
        p_tok = target_lp_t[i, tok].exp().item()

        r = torch.rand(1).item()
        if r <= min(1.0, p_tok / (q_tok + 1e-9)):
            new_tokens.append(tok)
            n_accepted += 1
        else:
            # resample from corrected distribution: norm(max(0, p - q))
            p_dist   = target_lp_t[i].exp()
            q_dist   = draft_lp_t[i].exp()
            adjusted = (p_dist - q_dist).clamp(min=0.0)
            adj_sum  = adjusted.sum()
            if adj_sum < 1e-9:
                adjusted = p_dist               # fallback: pure target
            else:
                adjusted = adjusted / adj_sum
            new_tokens.append(int(torch.multinomial(adjusted, 1).item()))
            # Trim the KV cache to cover only accepted tokens so the next
            # round's target pass starts from the right position.
            # updated_past covers prefix + all gamma draft tokens; we need
            # to keep only prefix + i tokens (i accepted so far).
            trimmed_past = _trim_kv_cache(updated_past, len(prefix) + i)
            return new_tokens, n_accepted, trimmed_past

    # all gamma tokens accepted → sample bonus token from target at position gamma
    new_tokens.append(_sample(target_lp_t[gamma]))
    return new_tokens, n_accepted, updated_past


# ── generation loops ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate_speculative(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    lam_draft: float,
    lam_target: float,
    gamma: int,
    temperature: float,
    device: torch.device,
    tokenizer=None,
    verbose: bool = True,
    draft_model: nn.Module | None = None,
    top_k_vocab: int = 0,
    top_k_exact_fallback: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Full speculative generation.  Returns (output_tokens, stats).

    Pass draft_model=build_compressed_olmo3(model, lam_draft, device) for
    genuine wall-clock speedup; omit to use the mask-switching fallback.

    top_k_vocab / top_k_exact_fallback: see speculative_step.
    """
    tokens         = prompt.to(device)
    total_accepted = 0
    total_proposed = 0
    n_rounds       = 0
    target_past    = None   # KV cache carried across rounds
    t0             = time.perf_counter()

    while (len(tokens) - len(prompt)) < max_new_tokens:
        remaining = max_new_tokens - (len(tokens) - len(prompt))
        g         = min(gamma, remaining)

        new_toks, n_acc, target_past = speculative_step(
            model, tokens, lam_draft, lam_target, g, temperature, device,
            draft_model=draft_model, target_past=target_past,
            top_k_vocab=top_k_vocab,
            top_k_exact_fallback=top_k_exact_fallback)

        tokens          = torch.cat([tokens,
                                     torch.tensor(new_toks, device=device)])
        total_accepted += n_acc
        total_proposed += g
        n_rounds       += 1

        if verbose and tokenizer is not None:
            print(tokenizer.decode(new_toks), end="", flush=True)

    elapsed     = time.perf_counter() - t0
    accept_rate = total_accepted / max(1, total_proposed)
    stats = {
        "method":       "speculative",
        "n_tokens":     max_new_tokens,
        "elapsed_s":    elapsed,
        "tok_per_s":    max_new_tokens / elapsed,
        "accept_rate":  accept_rate,
        "rounds":       n_rounds,
        "gamma":        gamma,
        "lam_draft":    lam_draft,
        "lam_target":   lam_target,
        # expected speedup vs target-only: gamma*alpha+1 tokens per target call
        "expected_toks_per_call": gamma * accept_rate + 1,
    }
    return tokens, stats


@torch.no_grad()
def generate_greedy(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    lam: float,
    temperature: float,
    device: torch.device,
    tokenizer=None,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Standard autoregressive generation at a fixed λ (baseline).

    Uses KV caching so the comparison with speculative decoding is fair —
    both methods benefit equally from prefix caching.
    """
    commit_masks(model, lam, device)
    tokens = prompt.to(device)
    t0     = time.perf_counter()

    past = None
    for _ in range(max_new_tokens):
        inp  = tokens.unsqueeze(0) if past is None else tokens[-1:].unsqueeze(0)
        out  = model(input_ids=inp, past_key_values=past, use_cache=True)
        lp   = torch.log_softmax(out.logits[0, -1].float() / temperature, dim=-1)
        past = out.past_key_values
        tok  = _sample(lp)
        tokens = torch.cat([tokens, torch.tensor([tok], device=device)])
        if verbose and tokenizer is not None:
            print(tokenizer.decode([tok]), end="", flush=True)

    elapsed = time.perf_counter() - t0
    stats   = {
        "method":    f"greedy_lam={lam}",
        "n_tokens":  max_new_tokens,
        "elapsed_s": elapsed,
        "tok_per_s": max_new_tokens / elapsed,
        "lam":       lam,
    }
    return tokens, stats


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(
    model: nn.Module,
    prompt: torch.Tensor,
    device: torch.device,
    lam_draft: float   = 2.0,
    lam_target: float  = 0.1,
    gamma: int         = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 100,
    tokenizer=None,
) -> None:
    """Compare speculative, target-only, and draft-only generation."""
    configs = [
        ("target only",  lambda: generate_greedy(
            model, prompt, max_new_tokens, lam_target,
            temperature, device, tokenizer, verbose=False)),
        ("draft only",   lambda: generate_greedy(
            model, prompt, max_new_tokens, lam_draft,
            temperature, device, tokenizer, verbose=False)),
        ("speculative",  lambda: generate_speculative(
            model, prompt, max_new_tokens, lam_draft, lam_target,
            gamma, temperature, device, tokenizer, verbose=False)),
    ]

    print(f"\n{'Method':<20} {'tok/s':>8}  {'accept%':>8}  "
          f"{'E[tok/call]':>12}  {'notes'}")
    print("-" * 68)

    # show active fractions
    frac_d = active_fraction(model, lam_draft, device)
    frac_t = active_fraction(model, lam_target, device)
    print(f"  draft  (λ={lam_draft}):  ffn={frac_d['ffn']:.0%} active")
    print(f"  target (λ={lam_target}): ffn={frac_t['ffn']:.0%} active\n")

    for name, fn in configs:
        _, stats = fn()
        accept = f"{stats.get('accept_rate', float('nan')):.1%}"
        etpc   = f"{stats.get('expected_toks_per_call', float('nan')):.2f}"
        print(f"  {name:<18} {stats['tok_per_s']:>8.1f}  {accept:>8}  {etpc:>12}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speculative decoding with an OLMo-3 stochastic checkpoint")
    p.add_argument("--ckpt_dir",       required=True,
                   help="checkpoint directory (contains model.pt, config.json)")
    p.add_argument("--tokenizer_dir",  default=None,
                   help="HuggingFace tokenizer directory (optional)")
    p.add_argument("--prompt",         default="The history of language models is",
                   help="text prompt (requires --tokenizer_dir)")
    p.add_argument("--prompt_ids",     default=None,
                   help="space-separated token IDs (alternative to --prompt)")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--lam_draft",      type=float, default=2.0,
                   help="λ for draft model (high = sparse = fast)")
    p.add_argument("--lam_target",     type=float, default=0.1,
                   help="λ for target model (low = dense = accurate)")
    p.add_argument("--gamma",          type=int,   default=4,
                   help="draft tokens per round (lookahead window)")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--benchmark",      action="store_true",
                   help="run timing comparison instead of generation")
    p.add_argument("--device",         default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading checkpoint from {args.ckpt_dir} …")
    model, train_args = load_checkpoint(args.ckpt_dir, device)
    n_params = sum(p.numel() for p in set(model.parameters()))
    print(f"  {n_params/1e6:.1f}M parameters  device={device}")

    tokenizer = None
    if args.tokenizer_dir:
        tok_path = Path(args.tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_path.resolve()) if tok_path.is_dir() else args.tokenizer_dir)

    # build prompt tensor
    if args.prompt_ids:
        prompt = torch.tensor([int(x) for x in args.prompt_ids.split()],
                               dtype=torch.long)
    elif tokenizer is not None:
        prompt = torch.tensor(
            tokenizer.encode(args.prompt, add_special_tokens=True),
            dtype=torch.long)
    else:
        # fallback: BOS token
        prompt = torch.tensor([1], dtype=torch.long)
        print("  No tokenizer/prompt_ids — using single BOS token as prompt.")

    print(f"  Prompt: {len(prompt)} tokens")
    print(f"  Draft  λ={args.lam_draft}  "
          f"({active_fraction(model, args.lam_draft, device)['ffn']:.0%} FFN active)")
    print(f"  Target λ={args.lam_target}  "
          f"({active_fraction(model, args.lam_target, device)['ffn']:.0%} FFN active)")
    print()

    if args.benchmark:
        benchmark(model, prompt, device,
                  lam_draft=args.lam_draft,
                  lam_target=args.lam_target,
                  gamma=args.gamma,
                  temperature=args.temperature,
                  max_new_tokens=args.max_new_tokens,
                  tokenizer=tokenizer)
        return

    print(f"{'─'*60}")
    print(f"Speculative generation  (γ={args.gamma}, "
          f"λ_draft={args.lam_draft}, λ_target={args.lam_target})\n")
    if tokenizer:
        print(tokenizer.decode(prompt.tolist()), end="")

    out, stats = generate_speculative(
        model, prompt, args.max_new_tokens,
        args.lam_draft, args.lam_target,
        args.gamma, args.temperature,
        device, tokenizer, verbose=True)

    print(f"\n{'─'*60}")
    print(f"  Tokens generated : {stats['n_tokens']}")
    print(f"  Time             : {stats['elapsed_s']:.1f}s  "
          f"({stats['tok_per_s']:.1f} tok/s)")
    print(f"  Accept rate      : {stats['accept_rate']:.1%}")
    print(f"  Rounds           : {stats['rounds']}  "
          f"(avg {stats['n_tokens']/stats['rounds']:.1f} tokens/round)")
    print(f"  E[tok/target call]: {stats['expected_toks_per_call']:.2f}  "
          f"(γ × α + 1 = {args.gamma} × {stats['accept_rate']:.2f} + 1)")


if __name__ == "__main__":
    main()
