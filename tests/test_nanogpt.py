"""Tests transformer components of NanoGPT."""


import sys
from pathlib import Path
filepath = Path(__file__)
sys.path.append(str(filepath.parent.parent.resolve()))

import torch

from nanogpt import Block, Feedforward, Head, MultiHead, NanoGPT, generate

# Setup.
tokens_file = filepath.parent.parent / "data/tiny_shakespeare_tokens.txt"
with open(tokens_file) as f:
    tokens = list(f.read())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nanogpt = NanoGPT(
    n_tokens=len(tokens),
    ctx_len=256,
    n_blocks=6,
    n_heads=8,
    head_sz=48,
    emb_dim=384,
    ff_dim=4,
    dropout=0.1,
).to(device)
batch_sz = 32


def test_Head(nanogpt=nanogpt, batch_sz=batch_sz):
    """Tests Head forward pass."""
    head = Head(nanogpt.head_sz, nanogpt.emb_dim).to(device)
    x = torch.rand(batch_sz, nanogpt.ctx_len, nanogpt.emb_dim).to(device)
    assert head(x).shape == (batch_sz, nanogpt.ctx_len, nanogpt.head_sz)


def test_MultiHead(nanogpt=nanogpt, batch_sz=batch_sz):
    """Tests MultiHead forward pass."""
    multi_head = MultiHead(nanogpt.n_heads, nanogpt.head_sz, nanogpt.emb_dim).to(device)
    x = torch.rand(batch_sz, nanogpt.ctx_len, nanogpt.emb_dim).to(device)
    assert multi_head(x).shape == (batch_sz, nanogpt.ctx_len, nanogpt.n_heads * nanogpt.head_sz)


def test_Feedforward(nanogpt=nanogpt, batch_sz=batch_sz):
    """Tests Feedforward forward pass."""
    feedforward = Feedforward(nanogpt.emb_dim, nanogpt.ff_dim).to(device)
    x = torch.rand(batch_sz, nanogpt.n_heads * nanogpt.head_sz, nanogpt.emb_dim).to(device)
    assert feedforward(x).shape == (batch_sz, nanogpt.emb_dim, nanogpt.emb_dim)


def test_Block(nanogpt=nanogpt, batch_sz=batch_sz):
    """Tests Block forward pass."""
    block = Block(
        nanogpt.n_heads, nanogpt.head_sz, nanogpt.emb_dim, nanogpt.ff_dim, dropout=0.1
    ).to(device)
    x = torch.rand(batch_sz, nanogpt.ctx_len, nanogpt.emb_dim).to(device)
    assert block(x).shape == (batch_sz, nanogpt.ctx_len, nanogpt.emb_dim)


def test_NanoGPT(nanogpt=nanogpt, batch_sz=batch_sz):
    """Tests NanoGPT forward pass."""
    x = torch.randint(0, nanogpt.n_tokens, (batch_sz, nanogpt.ctx_len)).to(device)
    assert nanogpt(x).shape == (batch_sz, nanogpt.ctx_len, nanogpt.n_tokens)


def test_generate(nanogpt=nanogpt, tokens=tokens):
    """Tests `generate`."""
    in_txt = "test"
    generated_text = generate(nanogpt, tokens, in_txt=in_txt, n_tokens=100, temp=0.5, top_k=50, seed=42)
    assert isinstance(generated_text, str)
    assert len(generated_text) > len(in_txt)
