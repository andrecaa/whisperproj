"""Sparse autoencoder on encoder activations (Phase E, PLAN §4E).

Standard top-k SAE: overcomplete dictionary, activations kept sparse by
keeping only the k largest feature pre-activations per sample (no L1 term
needed). Decoder rows are renormalized to unit norm after every step so
feature magnitudes live in the activations, not the dictionary.
"""

import numpy as np
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_dec)
        self.normalize_decoder()
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.t())

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        topk = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk.indices, torch.relu(topk.values))
        return acts

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        recon = acts @ self.W_dec + self.b_dec
        return recon, acts


def train_sae(
    X: torch.Tensor,       # [N, d] float32 (moved to device in batches)
    n_features: int,
    k: int,
    device: str,
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[TopKSAE, dict]:
    torch.manual_seed(seed)
    sae = TopKSAE(X.shape[1], n_features, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = len(X)
    var = X.var(dim=0).sum().item()

    # training loop: each epoch shuffles the frame order (perm) and walks
    # it in minibatches; the loss is plain reconstruction MSE, sparsity
    # comes for free from the top-k in encode()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            xb = X[perm[i:i + batch_size]].to(device)   # batch to GPU here,
            recon, _ = sae(xb)                          # X itself stays on CPU
            loss = ((recon - xb) ** 2).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()   # keep dictionary rows unit-norm
            total += loss.item() * len(xb)
        mse = total / n
        if epoch == 0 or epoch == epochs - 1:
            print(f"[sae] epoch {epoch + 1}/{epochs}: mse/dim "
                  f"{mse / X.shape[1]:.5f}")

    # final metrics: fraction of variance unexplained + dead features
    with torch.no_grad():
        errs, alive = 0.0, torch.zeros(n_features, dtype=torch.bool)
        for i in range(0, n, batch_size):
            xb = X[i:i + batch_size].to(device)
            recon, acts = sae(xb)
            errs += ((recon - xb) ** 2).sum().item()
            alive |= (acts > 0).any(dim=0).cpu()
    stats = {"fvu": errs / (var * n), "dead_features": int((~alive).sum())}
    print(f"[sae] FVU {stats['fvu']:.3f}, dead features "
          f"{stats['dead_features']}/{n_features}")
    return sae, stats
