#!/bin/bash
#SBATCH --job-name=olmo3_mini
#SBATCH --time=48:00:00
#SBATCH --mem=260G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:6
#SBATCH --account=157347369011
#SBATCH --output=olmo3_mini_%j.out
#SBATCH --error=olmo3_mini_%j.err
#SBATCH --mail-user=bobkubinec@gmail.com
#SBATCH --mail-type=all

# small 7GB = 5B tokens
DATA_DIR=/scratch/user/$USER/demo_data/olmo3_data
# big 187 GB = 50 GB from checkpoint
#DATA_DIR=/scratch/user/$USER/demo_data/olmo3_7b_data
# Use a size-specific checkpoint dir to avoid accidental cross-size resume.
# Change "1b" to "90m" if using the default 90M config.
CKPT_DIR=/scratch/user/$USER/demo_ckpt/olmo3_mini_1b_all

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Data   : $DATA_DIR"
echo "Ckpt   : $CKPT_DIR"
echo ""
echo "── NUMA topology ──────────────────────────────────────────────────"
numactl --hardware 2>/dev/null || echo "(numactl not available)"
echo "───────────────────────────────────────────────────────────────────"
echo ""

module purge
module load GCCcore/13.3.0
#module load intel/2024b
module load Python/3.12

source $SCRATCH/python_virtual_env/venv/bin/activate
#source /scratch/group/p.mth240050.000/cuda_venv/bin/activate

# ── oneCCL worker thread affinity (XPU only) ──────────────────────────────────
# CCL pins one worker thread per rank; without this it picks CPU IDs outside
# the SLURM cpuset → pthread_create EINVAL (error 22).
# One entry per rank (nproc_per_node=6), all within the 16 allocated CPUs.
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY=0,1,2,3,4,5
export CCL_ATL_TRANSPORT=ofi   # skip MPI probe, go straight to OFI/libfabric

# ── transformers>=4.48 must already be installed (compute nodes have no internet)
# Run once from a login node before submitting:
#   pip install --upgrade "transformers>=4.48"

# ── Step 1: tokenise WikiText-103 if not already done ─────────────────────────
# Requires a HuggingFace token stored in ~/.huggingface/token (run
# `huggingface-cli login` once from an interactive session beforehand).
# if [ ! -f "$DATA_DIR/train.bin" ]; then
#     echo "Tokenising WikiText-103 → $DATA_DIR"
#     python python/prepare_olmo3_mini.py --data_dir "$DATA_DIR"
#     echo "Tokenisation complete."
#     echo ""
# fi

# ── Step 2: train ─────────────────────────────────────────────────────────────
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 5000 ))

# PCIe topology: DDP (single all-reduce/step) beats FSDP FULL_SHARD (32 collectives/step).
# If you hit OOM, switch to: --fsdp --sharding grad_op --grad_ckpt (16 collectives/step)
#numactl --cpunodebind=0 --membind=0 \
# big model
  #--hidden 2048 --intermediate 8192 --n_layers 16 --n_heads 16  --batch 8 --seq_len 1024 \
# torchrun --nproc_per_node=1 python/olmo3_mini_train.py \
#   --accelerator cuda \
#   --restart \
#   --hidden 512 --intermediate 1376 --n_layers 12 --n_heads 8 --batch 16 --seq_len 512 \
#   --grad_ckpt \
#   --data_dir "$DATA_DIR" \
#   --ckpt_dir "$CKPT_DIR"

# no ddp, singl eGPU
#  --no_ddp \
#--fsdp --sharding full
#torchrun --nproc_per_node= python/olmo3_mini_train.py \
#python python/olmo3_mini_train.py \
#  --restart \

torchrun --nproc_per_node=6 python/olmo3_mini_train.py \
  --accelerator xpu \
  --hidden 2048 --intermediate 8192 --n_layers 16 --n_heads 16  --batch 4 --seq_len 1024 \
  --grad_ckpt \
  --n_knots 10 \
  --grad_accum 8 \
  --epochs 3 \
  --data_dir "$DATA_DIR" \
  --ckpt_dir "$CKPT_DIR"

# torchrun \
#     --nproc_per_node=4 \
#     --master_port=$MASTER_PORT \
#     python/olmo3_mini_train.py \
#         --data_dir  "$DATA_DIR"  \
#         --ckpt_dir  "$CKPT_DIR"  \
#         $MODEL_FLAGS             \
#         $PARALLEL_FLAGS          \
#         --lr             3e-4    \
#         --epochs            3 \
#         --weighted_nll

echo ""
echo "Training complete. Checkpoint written to $CKPT_DIR"
