#!/bin/bash
#SBATCH --job-name=stoch_compress
#SBATCH --time=48:00:00
#SBATCH --mem=480G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --account=157347369011
#SBATCH --output=stoch_xpu_%j.out
#SBATCH --error=stoch_xpu_%j.err
#SBATCH --mail-user=bobkubinec@gmail.com
#SBATCH --mail-type=all

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Device : XPU (Intel PVC)"
echo ""

module purge
module load GCCcore/13.3.0
module load WebProxy
module load Python/3.12

source $SCRATCH/python_virtual_env/venv/bin/activate

CKPT_DIR=$SCRATCH/demo_ckpt/olmo3_mini_1b_all
DATA_DIR=/scratch/user/$USER/demo_data/olmo3_data

python python/compression_comparison.py \
    --device  xpu \
    --ckpt "$CKPT_DIR/model.pt" \
    --out $CKPT_DIR/compression_comparison.json \
    --data_dir /scratch/user/$USER/demo_data/olmo3_data \
    --max_train_tokens 200_000_000 \
    --max_batches 50 \
    --n_resample 10

echo ""
echo "Done. Results written to $CKPT_DIR"
