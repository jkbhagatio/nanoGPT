"""Runs distributed training of NanoGPTs across multiple GPUs using PyTorch's DDP."""

import argparse
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import multiprocessing as mp
from torch import nn, optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, NAdam
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from nanogpt import NanoGPT, build_dataset

# Hyperparameters for model setup.
LR_SET = [5e-2, 1e-3, 1e-4]  # learning rate set
OPTIM_SET = [Adam, AdamW, NAdam]  # optimizer set
ARCH_SET = [  # model architecture set
    {"ctx_len": 1024, "emb_dim": 768, "n_heads": 12, "head_sz": 64},
    {"ctx_len": 2048, "emb_dim": 1024, "n_heads": 16, "head_sz": 64},
    {"ctx_len": 2048, "emb_dim": 1024, "n_heads": 20, "head_sz": 80},
]

def setup(rank, world_size, master_addr, master_port):
    """Sets up the DDP environment."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    # Create distributed process group.
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up and kills DDP environment."""
    destroy_process_group()
    wandb.finish()

def train(
    model: nn.Module,  # model
    train_loader: DataLoader,  # batched dataset for training
    val_loader: DataLoader,  # batched dataset for validation
    optimizer: optim,  # optimizer
    loss_fn: nn.modules.loss,  # loss function
    rank: int,  # rank of current process
    max_epochs: int = 5,  # max n training epochs
    max_batches: int = 1e9,  # max n batches to train
    val_chk_interval: int = 200,  # check val loss every `val_chk_interval` batches and print losses
    val_iter: int = 5,  # number of batches on val_loader to run and avg when computing val loss
    patience_thresh: int = 1e9,  # consecutive batches without val loss decrease for early stopping
    save_chkpt_dir: str = "",  # dir to save model checkpoint
    save_chkpt_thresh: float = 0.5,  # save model checkpoint every `save_chkpt_interval` loss decrease
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:  # -> loss, train_losses, val_losses
    """Trains a model, returns loss."""
    model = DDP(model, device_ids=[rank])
    # <s Nested helper functions to make `train` more readable.
    @torch.no_grad()
    def estimate_losses(model, val_loader, val_losses, val_losses_avg, train_losses, train_losses_avg):
        """Estimate losses on val_loader, and return val loss and train loss avg."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        for val_i, (x_val, y_val) in enumerate(val_loader):
            logits = model(x_val.to(device))
            val_loss = loss_fn(logits.view(-1, n_tokens), y_val.to(device).view(-1))
            val_losses.append(val_loss.item())
            if val_i >= (val_iter - 1):
                break
        val_losses_avg.append(np.mean(val_losses[-val_iter:]))
        train_losses_avg.append(np.mean(train_losses[-val_chk_interval:]))
        model.train()
    # /s>

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

    # <s Trackers
    _ctx_len, n_tokens  = model.ctx_len, model.n_tokens
    _batch_sz, n_batches = train_loader.batch_size, len(train_loader)
    batch_lim = min(max_batches, n_batches * max_epochs)
    patience_thresh *= val_chk_interval  # convert to batches within model validation block
    train_losses, val_losses, train_losses_avg, val_losses_avg = [], [], [], []
    init_loss, best_val_loss = float("inf"), float("inf")
    patience_ct = 0
    # /s>

    # <s Training loop
    start_t = time.time()
    for epoch in range(max_epochs):
        for batch_i, (x_train, y_train) in enumerate(train_loader):
            # <ss Model training.
            optimizer.zero_grad()
            logits = model(x_train.to(rank))  # -> [batch_sz, ctx_len, n_tokens], but...
            # must reshape to compare against batch_sz vector of targets for cross-entropy loss.
            loss = loss_fn(logits.view(-1, n_tokens), y_train.to(rank).view(-1))
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
                wandb.log({"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]})
                # Patience check for early stopping.
                patience_ct = (
                    0 if val_losses_avg[-1] < best_val_loss else patience_ct + val_chk_interval
                )
                best_val_loss = min(best_val_loss, val_losses_avg[-1])
                if patience_ct >= patience_thresh:
                    wandb.log({"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]})
                    return loss, train_losses_avg, val_losses_avg
            # Max batch check.
            if (batch_i + 1) * (epoch + 1) >= max_batches:
                wandb.log({"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]})
                return loss, train_losses_avg, val_losses_avg
            # Save checkpoint check.
            if (
                Path(save_chkpt_dir).exists()
                and (init_loss - loss.item()) > save_chkpt_thresh
                and rank == 0
            ):
                torch.save(
                    model.module.state_dict(),
                    Path(save_chkpt_dir) / f"model_chkpt_loss{loss.item():.3f}.pth"
                )
                init_loss = loss.item()
            # /ss>
            # <ss Progress metrics.
            n_comp_batches = epoch * n_batches + batch_i + 1
            elapsed_t = time.time() - start_t
            avg_batch_t = elapsed_t / n_comp_batches
            est_total_t = avg_batch_t * batch_lim
            est_remaining_t = est_total_t - elapsed_t
            wandb.log(
                {
                    "completed_batches": n_comp_batches,
                    "expected_total_batches": batch_lim,
                    "estimated_time_remaining": est_remaining_t,
                }
            )
            # /ss> /s>
    # Finished.
    wandb.log({"train_loss": train_losses_avg[-1], "val_loss": val_losses_avg[-1]})
    if Path(save_chkpt_dir).exists() and rank == 0:
        torch.save(
            model.module.state_dict(),
            Path(save_chkpt_dir) / f"model_chkpt_loss{loss.item():.3f}.pth"
        )
    return loss, train_losses_avg, val_losses_avg

def main(
    rank,
    world_size,
    master_addr,
    master_port,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    save_chkpt_dir
):
    setup(rank, world_size, master_addr, master_port)
    train(model, train_loader, val_loader, optimizer, loss_fn, rank, save_chkpt_dir=save_chkpt_dir)
    cleanup()

# Run training.
# 'config_idx', 'world_size', 'rank', 'MASTER_ADDR', and 'MASTER_PORT' set in slurm script.
if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description="Run DDP distributed training of NanoGPTs.")
    parser.add_argument("--config-idx", type=int, required=True, help="Index of config to run.")
    parser.add_argument(
        "--world-size", type=int, required=True, help="Number of processes to use for DDP."
    )
    parser.add_argument("--rank", type=int, required=True, help="Rank of current process.")
    parser.add_argument(
        "--master-addr", type=str, required=True, help="Master address (or hostname) for DDP."
    )
    parser.add_argument("--master-port", type=str, default="91827", help="Master port for DDP.")
    args = parser.parse_args()
    # Set config.
    configs = list(product(LR_SET, OPTIM_SET, ARCH_SET))
    config = configs[args.config_idx]
    # Set up dataset and model.
    txtfile = Path.cwd() / "data/tiny_austen.txt"
    with open(txtfile) as f:
        text = f.read()
    tokens = sorted(set(text))
    X, Y = build_dataset(txtfile, ctx_len=config[2]["ctx_len"])
    dataset = TensorDataset(X, Y)
    train_data, val_data = random_split(dataset, splits=[0.9, 0.1])
    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=False, sampler=DistributedSampler(train_data)
    )
    val_loader = DataLoader(
        val_data, batch_size=32, shuffle=False, sampler=DistributedSampler(val_data)
    )
    model = NanoGPT(n_tokens=len(tokens), **config[2])
    # Get model size.
    param_bytes = 4  # 32-bit floats
    bytes_in_gb = 1024 ** 3
    n_tot_params = sum(p.numel() for p in model.parameters())
    n_tot_params_b = round(n_tot_params / 1e9, 3)
    tot_sz_gb = n_tot_params * param_bytes / bytes_in_gb
    # Wandb config: model size, lr, optim, arch.
    wandb_config = {
        "n_params_bil": n_tot_params_b, "sz_gb": tot_sz_gb, "lr": config[0], "optim": config[1]
    }
    wandb_config.update(config[2])
    # name: <optim>-<lr>_<n_tot_params_b>; e.g. Adam-0.005_0.122B
    run_name = f"{config[1].__name__}-{config[0]}_{n_tot_params_b}B"
    wandb.init(
        project="NanoGPT-DDP",
        entity="jkbhagatio",
        name=run_name,
        config=wandb_config
    )
    # Setup DDP environment.
    setup(rank=args.rank, world_size=args.world_size, master_addr=args.master_addr, master_port=args.master_port)
    # Spawn and run training.
    optimizer = config[1](model.parameters(), lr=config[0])
    loss_fn = nn.CrossEntropyLoss()
    mp.spawn(
        main,
        args=(
            args.rank,
            args.world_size,
            args.master_addr,
            args.master_port,
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            run_name  # save_chkpt_dir
        ),
        nprocs=args.world_size,
        join=True
    )
    # Cleanup DDP environment.
    cleanup()
