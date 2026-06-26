#!/bin/bash
#SBATCH --job-name=ram_budget_pvc
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --account=157347369011
#SBATCH --output=ram_budget_pvc_%j.out
#SBATCH --error=ram_budget_pvc_%j.err
#SBATCH --mail-user=bobkubinec@gmail.com
#SBATCH --mail-type=all

# Analytical multi-draft speculative-decoding sweep under a fixed GPU memory
# budget (ram_budget mode). For each draft λ it computes how many drafts fit in
# the budget, K(λ), and compares three operating points:
#   single    — one draft (K=1)
#   distinct  — K(λ) physically pruned copies (K weight loads per step)
#   shared    — same K candidates from one masked dense copy (1 weight load)
#
# This mode is analytical (a few forward passes per λ), so it finishes in
# minutes and does not need --compile / generative timing.
#
# Override paths / parameters from the command line, e.g.:
#   BUDGET_GB=80 N_LAM=40 sbatch scripts/submit_ram_budget.sh

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Device : XPU (Intel PVC)"
echo ""

module purge
module load GCCcore/13.3.0
module load WebProxy
module load Python/3.12

source $SCRATCH/python_virtual_env/venv/bin/activate

# ── XPU preflight ───────────────────────────────────────────────────────────
# Fail fast (with a useful diagnostic) if the node reports no Intel GPU, rather
# than crashing deep inside model.to("xpu"). This catches unhealthy PVC nodes.
echo "SLURM GPU env : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK  GPU_DEVICE_ORDINAL=$GPU_DEVICE_ORDINAL"
command -v sycl-ls >/dev/null 2>&1 && { echo "sycl-ls:"; sycl-ls; }
python - <<'PY' || { echo "ERROR: torch sees no XPU on $(hostname). Likely a bad PVC node — resubmit."; exit 1; }
import torch
n = torch.xpu.device_count() if hasattr(torch, "xpu") else 0
print(f"torch={torch.__version__}  xpu.is_available={getattr(torch,'xpu',None) and torch.xpu.is_available()}  device_count={n}")
raise SystemExit(0 if n > 0 else 1)
PY
echo ""

CKPT_DIR=${CKPT_DIR:-$SCRATCH/demo_ckpt/olmo3_mini_1b_all}
DATA_DIR=${DATA_DIR:-/scratch/user/$USER/demo_data/olmo3_data}
TOKENIZER_DIR=${TOKENIZER_DIR:-$DATA_DIR/tokenizer}
OUT_DIR=${OUT_DIR:-$CKPT_DIR}

LAM_TARGET=${LAM_TARGET:-0.0}
LAM_DRAFT_MAX=${LAM_DRAFT_MAX:-3.0}
N_LAM=${N_LAM:-30}
GAMMA_MAX=${GAMMA_MAX:-64}
BUDGET_GB=${BUDGET_GB:-48}
KV_SEQ_LEN=${KV_SEQ_LEN:-2048}
K_MAX=${K_MAX:-256}

mkdir -p "$OUT_DIR"
OUT_PNG=$OUT_DIR/speculative_ram_budget.png
OUT_CSV=$OUT_DIR/speculative_ram_budget.csv

echo "Mode          : ram_budget"
echo "CkptDir       : $CKPT_DIR"
echo "TokenizerDir  : $TOKENIZER_DIR"
echo "OutPng        : $OUT_PNG"
echo "OutCsv        : $OUT_CSV"
echo "lam_target    : $LAM_TARGET"
echo "lam_draft_max : $LAM_DRAFT_MAX  ($N_LAM values, log-spaced)"
echo "budget_gb     : $BUDGET_GB  (KV reserve @ $KV_SEQ_LEN tokens, gamma_max=$GAMMA_MAX, k_max=$K_MAX)"
echo ""

# shellcheck disable=SC2086
python python/benchmark_speculative.py \
    --model_type     olmo3            \
    --mode           ram_budget       \
    --ckpt_dir       "$CKPT_DIR"      \
    --tokenizer_dir  "$TOKENIZER_DIR" \
    --device         xpu              \
    --lam_target     "$LAM_TARGET"    \
    --lam_draft_max  "$LAM_DRAFT_MAX" \
    --n_lam          "$N_LAM"         \
    --gamma_max      "$GAMMA_MAX"     \
    --budget_gb      "$BUDGET_GB"     \
    --kv_seq_len     "$KV_SEQ_LEN"    \
    --k_max          "$K_MAX"         \
    --out            "$OUT_PNG"       \
    --csv            "$OUT_CSV"

echo ""
echo "Done. Plot written to $OUT_PNG"
echo "      CSV  written to $OUT_CSV"
