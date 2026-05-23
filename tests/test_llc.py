"""Minimal end-to-end test for llc_estimation.py without transformer_lens.

Creates a tiny linear model + fake data and exercises the full estimate_llc
code path with tiny settings so it completes in <30s on CPU.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import numpy as np

# ── Tiny stand-in model (no transformer_lens needed) ──────────────────────────
class TinyModel(nn.Module):
    def __init__(self, vocab=131, out=113, ctx=3, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d * ctx, out)

    def forward(self, x):           # x: (batch, 3)
        e = self.embed(x)           # (batch, 3, d)
        return self.fc(e.flatten(1)) # (batch, 113)

# ── Fake data ─────────────────────────────────────────────────────────────────
torch.manual_seed(0)
N = 200
inputs = torch.randint(0, 113, (N, 3))
labels = torch.randint(0, 113, (N,))

model = TinyModel()

# ── Import and call estimate_llc ───────────────────────────────────────────────
from llc_estimation import estimate_llc

print("Running estimate_llc (2 chains, 10 draws, 5 burnin) ...")
result = estimate_llc(
    model, inputs, labels,
    epsilon=1e-4,
    nbeta=1.0,
    gamma=10.0,
    num_chains=2,
    num_draws=10,
    num_burnin_steps=5,
    device="cpu",
    batch_size=64,
)

print(f"llc_mean  = {result['llc_mean']:.4f}")
print(f"llc_std   = {result['llc_std']:.4f}")
print(f"traces shape = {result['loss_traces'].shape}")
print()
print("PASS — estimate_llc returned successfully.")
