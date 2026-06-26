#!/bin/bash
#SBATCH --job-name=spec_bench_pvc
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --account=157347369011
#SBATCH --output=spec_bench_pvc_%j.out
#SBATCH --error=spec_bench_pvc_%j.err
#SBATCH --mail-user=bobkubinec@gmail.com
#SBATCH --mail-type=all

# Benchmark speculative decoding for an OLMo3-mini checkpoint on Intel PVC XPU.
#
# Modes (set via $MODE env var):
#   optimize       — analytical (Th. 3.8 / 3.11), fast (minutes)              [default]
#   sweep          — full generative timing, ~30-60 min for n_lam draft values
#   context_sweep  — α(λ, T) across context lengths
#
# Override paths / parameters from the command line:
#   MODE=sweep CKPT_DIR=$SCRATCH/other_ckpt sbatch scripts/submit_benchmark_speculative.sh

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Device : XPU (Intel PVC)"
echo ""

module purge
module load GCCcore/13.3.0
module load WebProxy
module load Python/3.12

source $SCRATCH/python_virtual_env/venv/bin/activate

CKPT_DIR=${CKPT_DIR:-$SCRATCH/demo_ckpt/olmo3_mini_1b_all}
DATA_DIR=${DATA_DIR:-/scratch/user/$USER/demo_data/olmo3_data}
TOKENIZER_DIR=${TOKENIZER_DIR:-$DATA_DIR/tokenizer}
OUT_DIR=${OUT_DIR:-$CKPT_DIR}

MODE=${MODE:-sweep}
LAM_TARGET=${LAM_TARGET:-0.0}
LAM_DRAFT_MAX=${LAM_DRAFT_MAX:-3.0}
N_LAM=${N_LAM:-150}
GAMMA=${GAMMA:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
N_TRIALS=${N_TRIALS:-3}
GAMMA_MAX=${GAMMA_MAX:-64}

mkdir -p "$OUT_DIR"
OUT_PNG=$OUT_DIR/speculative_${MODE}.png
OUT_CSV=$OUT_DIR/speculative_${MODE}.csv

echo "Mode          : $MODE"
echo "CkptDir       : $CKPT_DIR"
echo "TokenizerDir  : $TOKENIZER_DIR"
echo "OutPng        : $OUT_PNG"
echo "OutCsv        : $OUT_CSV"
echo "lam_target    : $LAM_TARGET"
echo "lam_draft_max : $LAM_DRAFT_MAX  ($N_LAM values, log-spaced)"
echo "gamma         : $GAMMA  (gamma_max=$GAMMA_MAX)"
echo ""

# --compile gives a meaningful win on XPU for the generative sweep; in optimize
# mode each draft model only runs ~2 forwards so the compile cost rarely pays off.
COMPILE_FLAG=""
if [ "$MODE" = "sweep" ] || [ "$MODE" = "context_sweep" ]; then
    COMPILE_FLAG="--compile"
fi

# shellcheck disable=SC2086
python python/benchmark_speculative.py \
    --model_type     olmo3            \
    --mode           "$MODE"          \
    --ckpt_dir       "$CKPT_DIR"      \
    --tokenizer_dir  "$TOKENIZER_DIR" \
    --device         xpu              \
    --lam_target     "$LAM_TARGET"    \
    --lam_draft_max  "$LAM_DRAFT_MAX" \
    --n_lam          "$N_LAM"         \
    --gamma          "$GAMMA"         \
    --gamma_max      "$GAMMA_MAX"     \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --csv            "$OUT_CSV"        \
    --n_trials       "$N_TRIALS"      \
    --out            "$OUT_PNG"       \
    $COMPILE_FLAG

echo ""
echo "Done. Plot written to $OUT_PNG"
echo "      CSV  written to $OUT_CSV"
