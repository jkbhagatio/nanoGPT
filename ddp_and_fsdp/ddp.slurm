#!/bin/bash
#SBATCH --job-name=ddp-training
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --ntasks=2     # processes per job
#SBATCH --gres=gpu:2   # gpus total across nodes
#SBATCH --array=0-26%3  # jobs, % max in parallel (27 unique models, given hyperparemeter configurations)
#SBATCH --output=/nfs/nhome/live/jbhagat/nanogpt_ddp_runs/job_%j.out
#SBATCH --error=/nfs/nhome/live/jbhagat/nanogpt_ddp_runs/job_%j.err

# Set first node as the master
HEAD_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(nslookup $HEAD_NODE_HOSTNAME | grep 'Address:' | awk 'NR==2 {print $2}')

# Dynamically calculate number of processes per node, based on number of nodes assigned for this job
PROCS_PER_NODE=$(($SLURM_NTASKS / $SLURM_JOB_NUM_NODES))

# Echo vars to .out file
echo "HEAD_NODE_HOSTNAME: $HEAD_NODE_HOSTNAME, HEAD_NODE_IP: $HEAD_NODE_IP, PROCS_PER_NODE: $PROCS_PER_NODE"

# Activate env
source /nfs/nhome/live/jbhagat/mambaforge/etc/profile.d/conda.sh
conda activate nanogpt

# Run ddp
srun torchrun \
    --standalone \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc_per_node=${PROCS_PER_NODE} \
    /nfs/nhome/live/jbhagat/nanoGPT/ddp_and_fsdp/ddp.py \
        --train_config_idx="$SLURM_ARRAY_TASK_ID"

# rdzv args for multinode
#--rdzv_id=4444 \  
#--rdzv_backend="c10d" \
#--rdzv_endpoint="$HEAD_NODE_IP:44444" 
