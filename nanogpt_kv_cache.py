"""Code to build, train, and run NanoGPT."""

import argparse
from hmac import new
import json
from pathlib import Path
from warnings import warn

import numpy as np
import torch as t
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

# <s> Historical separate head and multi-head self-attention classes.


class _Head(nn.Module):  # type: ignore
    """Self-attention head."""

    def __init__(self, head_dim: int, emb_dim: int):
        """Initialize key, query, value matrices."""
        super().__init__()
        self.head_dim, self.emb_dim = head_dim, emb_dim
        self.key = nn.Linear(emb_dim, head_dim, bias=False)
        self.query = nn.Linear(emb_dim, head_dim, bias=False)
        self.value = nn.Linear(emb_dim, head_dim, bias=False)

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len emb_dim"],  # type: ignore
    ) -> Float[Tensor, "batch_sz seq_len head_dim"]:  # type: ignore
        """Compute self-attention output."""
        _batch_sz, seq_len, _emb_dim = x.shape
        q: Float[Tensor, "batch_sz seq_len head_dim"] = self.query(x)  # type: ignore
        k = self.key(x)
        v = self.value(x)
        k_q_sim: Float[Tensor, "batch_sz seq_len seq_len"] = (  # type: ignore
            q @ k.transpose(2, 1) / np.sqrt(self.head_dim)
        )  # scaled attn to preserve k, q var
        tril = t.tril(t.ones(seq_len, seq_len, device=x.device))  # mask: can't see future
        k_q_sim = k_q_sim.masked_fill(tril == 0, float("-inf"))
        # for each query token (dim=1), compute prob for all seq key tokens (dim=2)
        attn_weights = F.softmax(k_q_sim, dim=2)
        # get weighted sum of values
        attn_out: Float[Tensor, "batch_sz seq_len head_dim"] = attn_weights @ v  # type: ignore
        # Note, if *not* using this in MultiHead setting, we should proj back to emb_dim:
        # proj = nn.Linear(head_dim, emb_dim)
        # attn_out = proj(attn_out)
        return attn_out


class _MultiHead(nn.Module):  # type: ignore
    """Multi-head self-attention."""

    def __init__(self, n_heads: int, head_dim: int, emb_dim: int):
        """Initialize heads."""
        super().__init__()
        self.n_heads, self.head_dim, self.emb_dim = n_heads, head_dim, emb_dim
        self.heads = nn.ModuleList([_Head(head_dim, emb_dim) for _ in range(n_heads)])
        self.proj = nn.Linear(self.n_heads * self.head_dim, self.emb_dim)

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len emb_dim"],  # type: ignore
    ) -> Float[Tensor, "batch_sz seq_len emb_dim"]:  # type: ignore
        """Compute multi-head self-attention output."""
        attn_outs = [head(x) for head in self.heads]
        # concatenate head dimension
        attn_out: Float[Tensor, "batch_sz seq_len emb_dim"] = t.cat(attn_outs, dim=2)  # type: ignore
        # use linear projection to mix info across heads
        attn_out = self.proj(attn_out)
        return attn_out


# </s>


# <s> Transformer classes: MultiHead, Feedforward, Block, NanoGPT.


class MultiHead(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, n_heads: int, head_dim: int, emb_dim: int, max_cache_len: int):
        """Initialize multi-head attention matrices."""
        super().__init__()
        self.n_heads, self.head_dim, self.emb_dim = n_heads, head_dim, emb_dim
        self.n_heads_dim = n_heads * head_dim
        self.max_cache_len = max_cache_len

        self.key = nn.Linear(emb_dim, self.n_heads_dim, bias=False)
        self.query = nn.Linear(emb_dim, self.n_heads_dim, bias=False)
        self.value = nn.Linear(emb_dim, self.n_heads_dim, bias=False)
        self.proj = nn.Linear(self.n_heads_dim, emb_dim)

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len emb_dim"],  # type: ignore
        kv_cache: tuple[
            Float[Tensor, "batch_sz n_heads seq_len head_dim"],  # type: ignore
            Float[Tensor, "batch_sz n_heads seq_len head_dim"],  # type: ignore
        ]
        | None = None,
        use_cache: bool = False,
    ) -> Float[Tensor, "batch_sz seq_len emb_dim"]:  # type: ignore
        """Compute multi-head self-attention output."""
        batch_sz, seq_len, _emb_dim = x.shape

        # Compute Q, K, V for all heads at once
        q: Float[Tensor, "batch_sz seq_len n_heads_dim"] = self.query(x)  # type: ignore
        k = self.key(x)
        v = self.value(x)

        # Reshape to separate heads
        q: Float[Tensor, "batch_sz n_heads seq_len head_dim"] = q.view(  # type: ignore
            batch_sz, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(batch_sz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_sz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache with sliding window before computing attention
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # concatenate past K and V along seq_len dim
            k = t.cat([past_k, k], dim=2)
            v = t.cat([past_v, v], dim=2)
            # apply sliding window
            if k.shape[2] > self.max_cache_len:
                k = k[:, :, -self.max_cache_len :, :]
                v = v[:, :, -self.max_cache_len :, :]
        new_cache = (k, v) if use_cache else None

        # Compute attention scores
        k_q_sim: Float[Tensor, "batch_sz n_heads seq_len seq_len"] = (  # type: ignore
            q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)
        )

        # Apply causal mask: create a mask that allows attention to cached tokens and
        # masks out only future tokens in the current sequence.
        full_seq_len = k.shape[2]
        cur_pos = full_seq_len - seq_len
        if cur_pos > 0:
            mask = t.ones(seq_len, full_seq_len, device=x.device)
            tril = t.tril(t.ones(seq_len, full_seq_len, device=x.device))
            mask[:, :cur_pos] = tril  # mask out future tokens
        else:  # standard causal mask
            mask = t.tril(t.ones(seq_len, full_seq_len, device=x.device))

        k_q_sim = k_q_sim.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(k_q_sim, dim=-1)

        # Get attn out per head, combine heads, and mix info across heads with linear proj
        attn_out: Float[Tensor, "batch_sz n_heads seq_len head_dim"] = (  # type: ignore
            attn_weights @ v
        )
        attn_out: Float[Tensor, "batch_sz seq_len n_heads_dim"] = (  # type: ignore
            attn_out.transpose(1, 2).reshape(batch_sz, seq_len, self.n_heads_dim)
        )
        # Apply output projection
        attn_out: Float[Tensor, "batch_sz seq_len emb_dim"] = self.proj(attn_out)  # type: ignore
        return attn_out


class Feedforward(nn.Module):
    """Feedforward layer.

    "Linear layer sandwich" aka "inverted bottleneck":
    Dim fans out by factor `ff_dim` then back to `emb_dim`.
    Importantly, adds non-linearity (attn mechanism, besides softmax, is linear).
    ReLU or GeLU is preferred over sigmoid or tanh, to combat vanishing gradients.
    ("Position-wise Feed-Forward Networks" in "Attention is All You Need")
    """

    def __init__(self, emb_dim: int, ff_dim: int):
        """Initialize weights."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * ff_dim),
            nn.ReLU(),
            nn.Linear(emb_dim * ff_dim, emb_dim),
        )

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len emb_dim"],  # type: ignore
    ) -> Float[Tensor, "batch_sz seq_len emb_dim"]:  # type: ignore
        """Forward pass."""
        return self.layers(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation.

    Parts:
     - Multi-head self-attention
     - Position-wise feedforward network
     - Residual connections
     - Layer normalization (pre-norm formulation)
     - ~ Weight normalization ~ (not for now)
     - Dropout
    """

    def __init__(
        self, n_heads: int, head_dim: int, emb_dim: int, ff_dim: int, dropout: float
    ):
        """Self-attn -> pos-wise feedforward, each sandwiched by layer norm & dropout."""
        super().__init__()
        self.n_heads, self.head_dim, self.emb_dim, self.ff_dim = (
            n_heads,
            head_dim,
            emb_dim,
            ff_dim,
        )
        self.self_attn_ln = nn.LayerNorm(emb_dim)  # layer norm pre self-attention
        self.self_attn = MultiHead(
            n_heads, head_dim, emb_dim
        )  # multi-head self-attention
        self.self_attn_dropout = nn.Dropout(dropout)  # dropout after self-attention
        self.ff_ln = nn.LayerNorm(emb_dim)  # layer norm pre feedforward
        self.ff = Feedforward(emb_dim, ff_dim)  # position-wise feedforward
        self.ff_dropout = nn.Dropout(dropout)  # dropout after feedforward

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len emb_dim"],  # type: ignore
    ) -> Float[Tensor, "batch_sz seq_len emb_dim"]:  # type: ignore
        """Self-attn -> feedforward."""
        # layer-norm -> self-attn -> dropout + residual
        x = x + self.self_attn_dropout(self.self_attn(self.self_attn_ln(x)))
        # layer-norm -> feedforward -> dropout + residual
        x = x + self.ff_dropout(self.ff(self.ff_ln(x)))
        return x


"""
Create NanoGPT: Decoder-only Transformer.

In addition to our Transformer blocks, we need token embedding and positional embedding
layers, to compute the positional encodings that get passed to the attention units in the
transformer blocks.

We'll also apply weight init.

We want our output to be [batch_sz, seq_len, vocab_sz], because we want to predict the
next token for each token in the context.
"""


class NanoGPT(nn.Module):
    """NanoGPT: Decoder-only Transformer."""

    def __init__(
        self,
        vocab_sz,
        ctx_win=512,
        n_blocks=8,
        n_heads=8,
        head_dim=64,
        emb_dim=512,
        ff_dim=4,
        dropout=0.1,
    ):
        """Init token & pos embeddings, transformer blocks, & norm and out layers."""
        super().__init__()
        (
            self.vocab_sz,
            self.ctx_win,
            self.n_blocks,
            self.n_heads,
            self.head_dim,
            self.emb_dim,
            self.ff_dim,
        ) = (vocab_sz, ctx_win, n_blocks, n_heads, head_dim, emb_dim, ff_dim)
        if (n_heads * head_dim / emb_dim) != 1:
            warn(
                f"Ratio of n_heads X head_dim to emb_dim "
                f"{(n_heads * head_dim / emb_dim)}) is not 1",
                stacklevel=1,
            )
        self.tok_emb = nn.Embedding(vocab_sz, emb_dim)  # to learn token embeddings
        self.pos_emb = nn.Embedding(ctx_win, emb_dim)  # to learn positional embeddings
        self.blocks = nn.Sequential(  # Transformer blocks
            *[Block(n_heads, head_dim, emb_dim, ff_dim, dropout) for _ in range(n_blocks)]
        )
        self.f_ln = nn.LayerNorm(emb_dim)  # final layer norm
        self.f_dropout = nn.Dropout(dropout)  # final dropout
        self.out = nn.Linear(emb_dim, vocab_sz)  # final (raw) logits
        self.apply(self.xavier_init)

    @staticmethod
    def xavier_init(module, gain=1):
        """Applies Xavier initialization to all linear and embedding layer weights."""
        if isinstance(module, nn.Embedding | nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=gain)

    def forward(
        self,
        x: Float[Tensor, "batch_sz seq_len"],  # type: ignore
    ) -> Float[Tensor, "batch_sz seq_len vocab_sz"]:  # type: ignore
        """Feed pos encodings through transformer blocks and final norm and out layers."""
        _batch_sz, seq_len = x.shape

        # Compute positional encodings
        tok_emb = self.tok_emb(x)  # -> [batch_sz, seq_len, emb_dim]
        pos_emb = self.pos_emb.weight[0:seq_len]  # -> [seq_len, emb_dim]
        pos_enc = tok_emb + pos_emb  # -> [batch_sz, seq_len, emb_dim]

        # Go through transformer blocks and final linear layer
        logits = self.out(self.f_dropout(self.f_ln(self.blocks(pos_enc))))
        return logits


# </s>


# <s> Data loading, training, config, and utility functions


def build_dataset(
    txtfile: str | Path, ctx_win: int
) -> tuple[Float[Tensor, "n_examples ctx_win"], Int[Tensor, "n_examples ctx_win"]]:  # type: ignore
    """Build dataset from text file."""
    # Get tokens
    with open(txtfile) as f:
        text = f.read()
    tokens = sorted(set(text))

    # Encode tokens
    token_to_int = {t: i for i, t in enumerate(tokens)}
    encode = lambda tokens: [token_to_int[t] for t in tokens]
    data = t.tensor(encode(text), dtype=t.long)

    # Create X, Y pairs
    n_chars = len(text)
    n_examples = n_chars - ctx_win
    idxs: Int[Tensor, "n_examples ctx_win"] = (  # type: ignore
        t.arange(ctx_win + 1).unsqueeze(0) + t.arange(n_examples).unsqueeze(1)
    )
    # seq 'n' in Y is matched with seq 'n-1' in X
    X, Y = data[idxs[:, :-1]], data[idxs[:, 1:]]
    return X, Y


def build_dataloaders(X, Y, batch_sz=16, splits=None):
    """Build train, val, test dataloaders."""
    splits = [0.9, 0.05, 0.05] if splits is None else splits
    dataset = TensorDataset(X, Y)
    train_data, test_data, val_data = random_split(dataset, splits)
    train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_sz, shuffle=True)
    return train_loader, val_loader, test_loader


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


def train(
    model: nn.Module,  # model
    train_loader: DataLoader,  # batched dataset for training
    val_loader: DataLoader,  # batched dataset for validation
    optimizer: optim,  # optimizer
    loss_fn: nn.modules.loss,  # loss function
    max_epochs: int = 2,  # max n training epochs
    max_batches: int = 1_000_000_000,  # max n batches to train
    val_chk_interval: int = 200,  # n batches to check val loss and print losses
    val_iter: int = 5,  # n val data batches to run and avg when computing val loss
    patience_thresh: int = 1_000_000_000,  # n batches wo val loss decrease for early stop
    save_chkpt_dir: str = "",  # dir to save model checkpoints
    save_chkpt_thresh: float = 0.5,  # loss decrease before saving model checkpoint
) -> tuple[Tensor, list, list]:  # -> loss, train_losses, val_losses
    """Trains a model, returns loss."""
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # <ss> Nested helper functions to make `train` more readable.
    def print_losses(epoch, batch_i, train_losses_avg, val_losses_avg):
        """Print current average losses."""
        print(
            f"Epoch {epoch + 1}: Batch {batch_i + 1}:  "
            f"Loss = {train_losses_avg[-1]:.3f}, Val Loss = {val_losses_avg[-1]:.3f}"
        )

    @t.no_grad()
    def estimate_losses(
        model, val_loader, val_losses, val_losses_avg, train_losses, train_losses_avg
    ):
        """Estimate losses on val_loader, and return val loss and train loss avg."""
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        model.eval()
        for val_i, (x_val, y_val) in enumerate(val_loader):
            logits = model(x_val.to(device))
            val_loss = loss_fn(logits.view(-1, vocab_sz), y_val.to(device).view(-1))
            val_losses.append(val_loss.item())
            if val_i >= (val_iter - 1):
                break
        val_losses_avg.append(np.mean(val_losses[-val_iter:]))
        train_losses_avg.append(np.mean(train_losses[-val_chk_interval:]))
        model.train()

    # </ss>

    # <ss> Trackers
    _ctx_win, vocab_sz = model.ctx_win, model.vocab_sz
    _batch_sz, n_batches = train_loader.batch_size, len(train_loader)
    batch_lim = min(max_batches, n_batches * max_epochs)
    patience_thresh *= (
        val_chk_interval  # convert to batches within model validation block
    )
    train_losses, val_losses, train_losses_avg, val_losses_avg = [], [], [], []
    init_loss, best_val_loss = float("inf"), float("inf")
    patience_ct = 0
    # </ss>

    # <ss> Training loop
    for epoch in range(max_epochs):
        pbar = tqdm(enumerate(train_loader), total=batch_lim, desc="Batch progression")
        for batch_i, (x_train, y_train) in pbar:
            # Model training.
            optimizer.zero_grad()
            logits: Float[Tensor, "batch_sz seq_len vocab_sz"] = model(x_train.to(device))  # type: ignore
            # t.nn.CrossEntropyLoss requires 2d (batch_sz, n_class) logits and 1d labels,
            # so we must reshape logits and labels.
            loss = loss_fn(logits.view(-1, vocab_sz), y_train.to(device).view(-1))
            loss.backward()
            apply_gradient_centralization(optimizer)
            optimizer.step()
            train_losses.append(loss.item())

            # Model validation.
            if val_chk_interval and batch_i % val_chk_interval == 0:
                # Estimate and print losses.
                estimate_losses(
                    model,
                    val_loader,
                    val_losses,
                    val_losses_avg,
                    train_losses,
                    train_losses_avg,
                )
                print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)
                pbar.set_postfix_str(
                    f"Total Batch {(batch_i + 1) * (epoch + 1)} / {batch_lim}"
                )
                # Patience check for early stopping.
                patience_ct = (
                    0
                    if val_losses_avg[-1] < best_val_loss
                    else patience_ct + val_chk_interval
                )
                best_val_loss = min(best_val_loss, val_losses_avg[-1])
                if patience_ct >= patience_thresh:
                    print("Early stopping.")
                    print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)
                    return loss, train_losses_avg, val_losses_avg
            # Max batch check.
            if (batch_i + 1) * (epoch + 1) >= max_batches:
                print("Finished training:")
                print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)
                return loss, train_losses_avg, val_losses_avg
            # Save checkpoint check.
            if (Path(save_chkpt_dir).exists()) and (
                init_loss - loss.item() > save_chkpt_thresh
            ):
                t.save(
                    model.state_dict(),
                    Path(save_chkpt_dir) / f"model_chkpt_loss{loss.item():.3f}.pth",
                )
                init_loss = loss.item()
            # </ss>

    print("Finished training:")
    print_losses(epoch, batch_i, train_losses_avg, val_losses_avg)  # type: ignore
    return loss, train_losses_avg, val_losses_avg  # type: ignore


def print_model_summary(model):
    """Print model summary."""
    print(model)
    n_params_tot = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        n_params = parameter.numel()
        print(f"{name=}: {n_params=}")
        n_params_tot += n_params
    print(f"\n{n_params_tot / 1e6} M total parameters")


def generate(
    model: nn.Module,
    tokens: list[str],
    in_txt: str | None = None,
    n_tokens: int = 100,
    temp: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 42,
    print_gen: bool = True,
) -> str:
    """Generate text from a nanoGPT model."""
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # Set a random seed for generation
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # Create token_to_int, int_to_token dicts.
    token_to_int = {t: i for i, t in enumerate(tokens)}
    int_to_token = {i: t for t, i in token_to_int.items()}

    # Process input_text if provided, else start with "\n".
    if in_txt is not None:
        # Convert input text to tokens and encode.
        encode = lambda tokens: [token_to_int[t] for t in tokens]
        in_tkns = encode(in_txt)
        input_len = len(in_tkns)
        # Initialize output starting with input text.
        x = t.zeros((input_len + n_tokens,), dtype=t.long).to(device)
        x[:input_len] = t.tensor(in_tkns, dtype=t.long).to(device)
    else:
        # Initialize output starting with "\n".
        x = t.zeros((1 + n_tokens,), dtype=t.long).to(device)
        x[0] = token_to_int["\n"]
        input_len = 1

    # Run inference (generation) in eval mode
    model.eval()
    with t.no_grad():
        first_gen_idx, last_gen_idx = input_len - 1, input_len + n_tokens - 1
        for i in range(first_gen_idx, last_gen_idx):  # start gen after `input_len`
            model_first_ctx = 0 if i < model.ctx_win else i - model.ctx_win + 1
            # Feed in `x` w/ batch_sz 1
            logits: Float[Tensor, "batch_sz seq_len vocab_sz"] = (  # type: ignore
                model(x[model_first_ctx : (i + 1)].unsqueeze(0))
            )
            # Get logits for current last token (squeeze out seq_len), and scale by temp
            logits: Float[Tensor, "batch_sz vocab_sz"] = logits[:, -1, :] / temp  # type: ignore
            if top_k is not None:  # limit to top_k most likely tokens
                top_vals, top_idxs = logits.topk(top_k, dim=1)
                probs = F.softmax(top_vals, dim=1)  # compute top_k probs
                next_tkn_int = top_idxs.gather(1, t.multinomial(probs, 1))  # sample
            elif top_p is not None:  # nucleus sampling
                probs: Float[Tensor, "batch_sz vocab_sz"] = F.softmax(logits, dim=1)  # type: ignore
                sorted_probs: Float[Tensor, "batch_sz vocab_sz"]  # type: ignore
                sorted_idxs: Int[Tensor, "batch_sz vocab_sz"]  # type: ignore
                sorted_probs, sorted_idxs = t.sort(probs, descending=True, dim=1)  # type: ignore
                cumulative_probs: Float[Tensor, "batch_sz vocab_sz"] = (  # type: ignore
                    t.cumsum(sorted_probs, dim=1)
                )
                # Find indices where cumulative probability exceeds top_p
                nucleus_mask: Bool[Tensor, "batch_sz vocab_sz"] = cumulative_probs > top_p  # type: ignore
                # Keep at least one token (the most likely)
                nucleus_mask[:, 0] = False
                # Zero out probabilities outside nucleus
                sorted_probs[nucleus_mask] = 0.0
                # Renormalize probabilities
                sorted_probs = sorted_probs / sorted_probs.sum(dim=1, keepdim=True)
                # Sample from the nucleus
                sampled_sorted_idx = t.multinomial(sorted_probs, 1)
                next_tkn_int = sorted_idxs.gather(1, sampled_sorted_idx)
            else:
                probs = F.softmax(logits, dim=1)  # compute probs for all tokens
                next_tkn_int = t.multinomial(probs, 1)  # sample from probs
            x[i + 1] = next_tkn_int
            if print_gen:
                print(int_to_token[next_tkn_int.item()], end="")

    # Decode `x` and return it.
    decode = lambda ints: "".join([int_to_token[i] for i in ints])
    return decode(x.tolist())


# </s>


# Run gen on specified model if called as module
if __name__ == "__main__":
    # Parse command-line arguments.
    # Command-line arguments for:
    #   - dir to <model>.pth, <config>.json, and <tokens>.txt files
    #   - `in_txt`` for `generate`
    #   - `n_tokens` for `generate`
    #   - `temp` for `generate`
    #   - `top_k` for `generate`
    #   - `seed` for generate
    parser = argparse.ArgumentParser(description="Generate text with NanoGPT.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model, model config, & tokens files.",
    )
    parser.add_argument(
        "--in-txt", type=str, default=None, help="Input text for generation."
    )
    parser.add_argument(
        "--n-tokens", type=int, required=True, help="Number of tokens to generate."
    )
    parser.add_argument(
        "--temp", type=float, default=1.0, help="Temperature for generation."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top k tokens to sample from for generation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for generation."
    )
    args = parser.parse_args()

    # Configure model.
    print("Loading model...")
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # Ensure model_dir exists, and that there is exactly:
    # 1 .pth file, 1 .json file, and 1 .txt file.
    model_dir = Path(args.model_dir)
    exts = ["*.pth", "*.json", "*.txt"]
    files = []  # requisite files, in order: model (.pth), config (.json), tokens (.txt)
    for ext in exts:
        if len(list(model_dir.glob(ext))) != 1:
            raise ValueError(f"Expected exactly 1 {ext} file in {model_dir}.")
        else:
            files.append(list(model_dir.glob(ext))[0])
    # Initialize model
    with files[1].open() as f:
        model_config = json.load(f)
    with files[2].open() as f:
        tokens = f.read()
    model = NanoGPT(
        vocab_sz=model_config["vocab_sz"],
        ctx_win=model_config["ctx_win"],
        n_blocks=model_config["n_blocks"],
        n_heads=model_config["n_heads"],
        head_dim=model_config["head_dim"],
        emb_dim=model_config["emb_dim"],
        ff_dim=model_config["ff_dim"],
    ).to(device)
    model.load_state_dict(t.load(files[0], map_location=device))
    model.eval()

    # Generate text.
    print("Generating text...")
    gen = generate(
        model,
        tokens=list(tokens),
        in_txt=args.in_txt,
        n_tokens=args.n_tokens,
        temp=args.temp,
        top_k=args.top_k,
        seed=args.seed,
        print_gen=True,
    )
