"""Runs distributed training of NanoGPTs across multiple GPUs using PyTorch's DDP."""

import argparse
import itertools
import os
import subprocess

import torch
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, NAdam

from nanogpt import NanoGPT, train

# Constants
LR_SET = [5e-2, 1e-3, 1e-4]  # learning rate set
OPTIM_SET = [Adam, AdamW, NAdam]  # optimizer set
ARCH_SET = [  # model architecture set
    {"ctx_len": 1024, "emb_dim": 768, "n_heads": 12, "head_sz": 64},
    {"ctx_len": 2048, "emb_dim": 1024, "n_heads": 16, "head_sz": 64},
    {"ctx_len": 2048, "emb_dim": 1024, "n_heads": 20, "head_sz": 80},
]

def setup(rank, world_size):
    """Sets up the DDP environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("SLURMD_NODENAME")  # this node (hostname) as master
    os.environ["MASTER_PORT"] = "91827"
    # Create distributed process group.
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up and kills DDP environment."""
    destroy_process_group()

def train(rank, world_size, lr, optim, arch):
    setup(rank, world_size)

    # Assuming NanoGPT takes an argument for model configuration
    model = NanoGPT(model_config="Your model configuration here")
    model_ddp = DDP(model.cuda(rank), device_ids=[rank])

    # Placeholder for your training code
    # Train your model and return metrics
    metrics = {"accuracy": 0.99}  # Example metric

    model_size_gb = torch.tensor([0.0])
    if rank == 0:
        model_size_gb = torch.tensor([torch.cuda.memory_allocated(rank) / (1024**3)])
        wandb.init(project="your_wandb_project_name", entity="your_wandb_entity")
        wandb.log({"Model Size (GB)": model_size_gb.item(), "n_params": model.n_params, **metrics})

    cleanup()

if __name__ == "__main__":
    world_size = 4  # Number of processes to run
    for rank in range(world_size):
        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
