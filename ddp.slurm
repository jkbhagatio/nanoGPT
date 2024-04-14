#!/bin/bash
#SBATCH --job-name=ddp-training
#SBATCH --partition=a100
#SBATCH --nodes=2
#SBATCH --gres=gpu:2  # 2 gpus per node
#SBATCH --ntasks=4    # 4 processes per job
#SBATCH --array=0-26%3  # 27 jobs, max 3 in parallel (27 unique models, given hyperparemeter configurations)

# Set first node as the master
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Activate env
source /nfs/nhome/live/jbhagat/mambaforge/etc/profile.d/mamba.sh
mamba activate nanogpt

# Run ddp
srun python ddp.py \
    --config-idx="$SLURM_ARRAY_TASK_ID" \
    --world-size="$SLURM_NTASKS" \
    --rank="$SLURM_PROCID" \
    --master-addr="$MASTER_ADDR"
