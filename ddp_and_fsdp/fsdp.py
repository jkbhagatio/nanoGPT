"""Runs distributed training of a sharded NanoGPT across multiple GPUs using PyTorch's FSDP."""

import argparse  # noqa: I001
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, ShardingStrategy)
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

# Import nanogpt from relative directory.
nanogpt_dir = Path.cwd().parent
sys.path.append(str(nanogpt_dir))
from nanogpt import NanoGPT, build_dataset  # noqa: E402, I001


CTX_LEN = 512
EMB_DIM = 2048
N_BLOCKS = 24
N_HEADS = 24
HEAD_SZ = 128
BATCH_SZ = 4
LR = 1e-4


def setup(backend: str):
    """Sets up the FSDP environment."""
    # Create distributed process group and set cuda device according to torchrun LOCAL_RANK env var.
    init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    """Cleans up and kills FSDP environment."""
    destroy_process_group()
    wandb.finish()


def train(
    model: nn.Module,  # model
    train_loader: DataLoader,  # batched dataset for training
    val_loader: DataLoader,  # batched dataset for validation
    optimizer: optim,  # optimizer
    loss_fn: nn.modules.loss,  # loss function
    global_rank: int,  # rank of current process across all nodes
    local_rank: int,  # rank of current process within node
    max_epochs: int = 5,  # max n training epochs
    max_batches: int = 500,  # max n batches to train
    val_chk_interval: int = 200,  # check val loss every `val_chk_interval` batches & print losses
    val_iter: int = 5,  # number of batches on val_loader to run and avg when computing val loss
    patience_thresh: int = 1e9,  # consecutive batches without val loss decrease for early stopping
    save_chkpt_dir: str = "",  # dir to save model checkpoint
    save_chkpt_thresh: float = 0.5,  # save model chkpnt every `save_chkpt_interval` loss decrease
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:  # -> loss, train_losses, val_losses
    """Trains a model, returns loss."""

    # <s Nested helper functions to make `train` more readable.
    @torch.no_grad()
    def estimate_losses(
        model, val_loader, val_losses, val_losses_avg, train_losses, train_losses_avg
    ):
        """Estimate losses on val_loader, and return val loss and train loss avg."""
        model.eval()
        for val_i, (x_val, y_val) in enumerate(val_loader):
            logits = model(x_val.to(local_rank))
            val_loss = loss_fn(logits.view(-1, n_tokens), y_val.to(local_rank).view(-1))
            val_losses.append(val_loss.item())
            if val_i >= (val_iter - 1):
                break
        val_losses_avg.append(np.mean(val_losses[-val_iter:]))
        train_losses_avg.append(np.mean(train_losses[-val_chk_interval:]))
        model.train()

    def apply_gradient_centralization(optimizer):
        """Applies gradient centralization to the optimizer.

        This function should be called before optimizer.step() in the training loop.
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # Compute the mean of the gradient
                    grad_mean = param.grad.data.mean(
                        dim=tuple(range(1, len(param.grad.shape))), keepdim=True
                    )
                    # Centralize the gradient
                    param.grad.data -= grad_mean

    # /s>
    # <s Trackers
    _ctx_len, n_tokens = model.module.ctx_len, model.module.n_tokens
    _batch_sz, n_batches = train_loader.batch_size, len(train_loader)
    batch_lim = min(max_batches, n_batches * max_epochs)
    patience_thresh *= val_chk_interval  # convert to batches within model validation block
    train_losses, val_losses, train_losses_avg, val_losses_avg = [], [], [], []
    init_loss, best_val_loss = float("inf"), float("inf")
    patience_ct = 0
    if global_rank == 0:
        wandb.log({"expected_total_batches": batch_lim})
    # /s>

    # <s Training loop
    start_t = time.time()
    for epoch in range(max_epochs):
        for batch_i, (x_train, y_train) in enumerate(train_loader):
            # <ss Model training.
            optimizer.zero_grad()
            logits = model(x_train.to(local_rank))  # -> [batch_sz, ctx_len, n_tokens], but...
            # must reshape to compare against batch_sz vector of targets for cross-entropy loss
            loss = loss_fn(logits.view(-1, n_tokens), y_train.to(local_rank).view(-1))
            loss.backward()
            apply_gradient_centralization(optimizer)
            optimizer.step()
            train_losses.append(loss.item())
            # /ss>
            # <ss Model validation.
            if val_chk_interval and batch_i % val_chk_interval == 0:
                # Estimate and print losses.
                estimate_losses(
                    model, val_loader, val_losses, val_losses_avg, train_losses, train_losses_avg
                )
                if global_rank == 0:
                    wandb.log({"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]})
                # Return if patience check reached (early stopping).
                patience_ct = (
                    0 if val_losses_avg[-1] < best_val_loss else patience_ct + val_chk_interval
                )
                best_val_loss = min(best_val_loss, val_losses_avg[-1])
                if patience_ct >= patience_thresh:
                    if global_rank == 0:
                        wandb.log(
                            {"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]}
                        )
                    return loss, train_losses_avg, val_losses_avg
            # Return if max_batches reached.
            if (batch_i + 1) * (epoch + 1) >= max_batches:
                if global_rank == 0:
                    wandb.log({"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]})
                return loss, train_losses_avg, val_losses_avg
            # Save checkpoint check.
            if (
                Path(save_chkpt_dir).exists()
                and (init_loss - loss.item()) > save_chkpt_thresh
                and global_rank == 0
            ):
                torch.save(
                    model.module.state_dict(),
                    Path(save_chkpt_dir) / f"model_chkpt_loss{loss.item():.3f}.pth",
                )
                init_loss = loss.item()
            # /ss>
            # <ss Progress metrics.
            if global_rank == 0:
                n_comp_batches = epoch * n_batches + batch_i + 1
                elapsed_t = time.time() - start_t
                avg_batch_t = elapsed_t / n_comp_batches
                est_total_t = avg_batch_t * batch_lim
                est_remaining_t = est_total_t - elapsed_t
                wandb.log(
                    {
                        "completed_batches": n_comp_batches,
                        "estimated_time_remaining": est_remaining_t,
                    }
                )
            # /ss> /s>
    # Return after max_epochs reached.
    if global_rank == 0:
        wandb.log(
            {
                "train_loss": train_losses_avg[-1],
                "val_loss": val_losses_avg[-1],
                "completed_batches": n_comp_batches,
                "estimated_time_remaining": est_remaining_t,
            }
        )
    if Path(save_chkpt_dir).exists() and local_rank == 0:
        torch.save(
            model.module.state_dict(),
            Path(save_chkpt_dir) / f"model_chkpt_loss{loss.item():.3f}.pth",
        )
    return loss, train_losses_avg, val_losses_avg


def main(
    backend: str,  # FSDP backend to use
    global_rank: int,  # rank of current process across all nodes
    local_rank: int,  # rank of current process within node
    text_file: str,  # path to text file to train on
):
    """Main function to run distributed training.

    Sets up DDP env, creates dataset from text file, creates and trains model, cleans up DDP env.
    """
    # Set up FSDP environment.
    setup(backend)
    # Set up dataset.
    with open(text_file) as f:
        text = f.read()
    tokens = sorted(set(text))
    X, Y = build_dataset(text_file, ctx_len=CTX_LEN)
    dataset = TensorDataset(X, Y)
    train_data, val_data = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SZ, shuffle=False, sampler=DistributedSampler(train_data)
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SZ, shuffle=False, sampler=DistributedSampler(val_data)
    )
    # Set up model.
    model = NanoGPT(
        n_tokens=len(tokens),
        ctx_len=CTX_LEN,
        emb_dim=EMB_DIM,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        head_sz=HEAD_SZ
    ).to(local_rank)
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
    # Initialize wandb config and run.
    param_bytes = 4  # 32-bit floats
    bytes_in_gb = 1024**3
    n_tot_params = sum(p.numel() for p in model.parameters())
    n_tot_params_b = round(n_tot_params / 1e9, 3)
    tot_sz_gb = n_tot_params * param_bytes / bytes_in_gb
    run_name = f"NADAMW-1e-4_{n_tot_params_b}B"
    if global_rank == 0:
        wandb_config = {
            "n_params_bil": n_tot_params_b,
            "sz_gb": tot_sz_gb,
            "lr": LR,
            "optim": "NADAMW",
            "completed_batches": 0,
            "expected_total_batches": None,  # set in `train` function
            "estimated_time_remaining": None,  # set in `train` function
        }
        model_arch_config = {
            "ctx_len": CTX_LEN,
            "emb_dim": EMB_DIM,
            "n_blocks": N_BLOCKS,
            "n_heads": N_HEADS,
            "head_sz": HEAD_SZ,
        }
        wandb_config.update(model_arch_config)
        # name: <optim>-<lr>_<n_tot_params_b>; e.g. Adam-0.005_0.122B
        wandb.init(project="NanoGPT-FSDP", entity="jkbhagatio", name=run_name, config=wandb_config)
    # Run training.
    optimizer = optim.NAdam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-8,
        momentum_decay=1e-4,
        decoupled_weight_decay=True
    )
    loss_fn = nn.CrossEntropyLoss()
    save_chkpt_dir = Path.home() / "nanogpt_fsdp_runs" / "chkpts" / run_name
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        global_rank,
        local_rank,
        save_chkpt_dir=save_chkpt_dir,
    )
    # Clean up FSDP environment.
    cleanup()


# Run training.
# 'config_idx', 'world_size', 'rank', 'MASTER_ADDR', and 'MASTER_PORT' set in slurm script.
if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description="Run FSDP distributed training of NanoGPT.")
    parser.add_argument(
        "--fsdp_backend",
        type=str,
        default="nccl",
        help="FSDP backend to use (typically 'nccl' on Unix-like system, 'gloo' on Windows).",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=(Path.cwd().parent / "data/tiny_austen.txt"),
        help="Path to text file to train on.",
    )
    args = parser.parse_args()
    # Get ranks from torchrun env vars.
    global_rank = int(os.environ["RANK"])  # rank of current process across all nodes
    local_rank = int(os.environ["LOCAL_RANK"])  # rank of current process within node.
    # Run FSDP training.
    main(args.fsdp_backend, global_rank, local_rank, args.text_file)
