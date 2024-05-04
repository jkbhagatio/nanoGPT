#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=215G
#SBATCH --gres=gpu:rtx5000:2   # gpus required for job
#SBATCH --output=/nfs/nhome/live/jbhagat/nanogpt_fsdp_runs/job_%j.out
#SBATCH --error=/nfs/nhome/live/jbhagat/nanogpt_fsdp_runs/job_%j.err

# Set first node as the master
HEAD_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(nslookup $HEAD_NODE_HOSTNAME | grep 'Address:' | awk 'NR==2 {print $2}')

# Echo vars to .out file
echo "HEAD_NODE_HOSTNAME: $HEAD_NODE_HOSTNAME, HEAD_NODE_IP: $HEAD_NODE_IP"

# Activate env
source /nfs/nhome/live/jbhagat/mambaforge/etc/profile.d/conda.sh
conda activate nanogpt

# Run fsdp
srun torchrun \
    --standalone \
    --nproc_per_node=2 \
    /nfs/nhome/live/jbhagat/nanoGPT/ddp_and_fsdp/fsdp.py
