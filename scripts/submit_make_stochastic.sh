#!/bin/bash
#SBATCH --job-name=stoch_compress
#SBATCH --time=24:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
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

python python/make_stochastic_results.py \
    --device  xpu \
    --ckpt $CKPT_DIR \
    --n_stability_draws 10 \
    --out $CKPT_DIR/stochastic_results.json \
    --data_dir /scratch/user/$USER/demo_data/olmo3_data \
    --prune_mode stochastic \
    --n_prune_draws 1 \
    --max_batches 50

echo ""
echo "Done. Results written to $CKPT_DIR"
