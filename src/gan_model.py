"""
Conditional GAN for generating synthetic session-feature tensors.

Design notes (why this is structured the way it is):
  * We operate on the *normalized question-feature tensor* (shape
    [seq_len, n_features]), not raw clickstream. This is the exact
    representation the detector consumes, so it's what the GAN needs to
    match — and it avoids having to generate valid timestamps / item_ids
    which would be prone to hallucination.
  * Conditional on label y in {0, 1}: 0 = normal, 1 = cheating.
  * Seed data for y=1 comes from RealisticCheatingGenerator (rule-based).
    The GAN's job is to produce *variations* of those patterns, not to
    invent cheating from scratch — this is the anti-hallucination guardrail.
  * A learned padding-length head predicts effective sequence length per
    sample; positions beyond that are zeroed out (mirrors the real data).
  * Sanity filters (apply_sanity_filters) clip generated samples to the
    range observed in real data. Anything that falls outside is rejected.

This is a standard conditional DCGAN-style architecture adapted for
variable-length sequences with masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class ConditionalGenerator(nn.Module):
    """Noise + label -> (seq_len, n_features) tensor.

    Uses a small GRU decoder so the generator has a sequential inductive
    bias matching the detector's view of the data.
    """

    def __init__(self, noise_dim: int = 64, n_features: int = 10,
                 max_seq_len: int = 50, hidden_dim: int = 96, n_classes: int = 2):
        super().__init__()
        self.noise_dim = noise_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_features = n_features

        self.label_emb = nn.Embedding(n_classes, 16)
        self.init_fc = nn.Linear(noise_dim + 16, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                          batch_first=True, dropout=0.1)
        self.out_proj = nn.Linear(hidden_dim, n_features)
        # length head: outputs a scalar in (0, 1] * max_seq_len
        self.len_head = nn.Sequential(
            nn.Linear(noise_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = noise.size(0)
        lbl = self.label_emb(labels)
        z = torch.cat([noise, lbl], dim=1)
        h0 = torch.tanh(self.init_fc(z))  # (B, hidden)
        # Feed a learned constant token repeated across seq_len as input
        tokens = h0.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        out, _ = self.gru(tokens)
        x = self.out_proj(out)  # (B, max_seq_len, n_features)

        # Predict effective length in [5, max_seq_len]
        frac = self.len_head(z).squeeze(1)  # (B,)
        lengths = (frac * (self.max_seq_len - 5) + 5).long().clamp(5, self.max_seq_len)
        return x, lengths


class ConditionalDiscriminator(nn.Module):
    """(sequence, label) -> real/fake logit."""

    def __init__(self, n_features: int = 10, max_seq_len: int = 50,
                 hidden_dim: int = 96, n_classes: int = 2):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 16)
        self.gru = nn.GRU(n_features + 16, hidden_dim, num_layers=2,
                          batch_first=True, dropout=0.1, bidirectional=True)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        lbl = self.label_emb(labels).unsqueeze(1).expand(-1, seq_len, -1)
        h = torch.cat([x, lbl], dim=2)
        out, _ = self.gru(h)
        if lengths is not None:
            mask = (torch.arange(seq_len, device=x.device).unsqueeze(0)
                    < lengths.unsqueeze(1)).float().unsqueeze(2)
            pooled = (out * mask).sum(1) / lengths.unsqueeze(1).float().clamp(min=1)
        else:
            pooled = out.mean(1)
        return self.cls(pooled).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class GANConfig:
    noise_dim: int = 64
    n_features: int = 10
    max_seq_len: int = 50
    hidden_dim: int = 96
    n_classes: int = 2
    epochs: int = 40
    batch_size: int = 128
    lr: float = 2e-4
    beta1: float = 0.5
    d_updates_per_g: int = 1
    label_smoothing: float = 0.1


class ConditionalGANTrainer:
    def __init__(self, cfg: GANConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.use_amp = (device.type == 'cuda')
        self.scaler_D = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scaler_G = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.G = ConditionalGenerator(cfg.noise_dim, cfg.n_features,
                                       cfg.max_seq_len, cfg.hidden_dim,
                                       cfg.n_classes).to(device)
        self.D = ConditionalDiscriminator(cfg.n_features, cfg.max_seq_len,
                                           cfg.hidden_dim, cfg.n_classes).to(device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg.lr,
                                       betas=(cfg.beta1, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.lr,
                                       betas=(cfg.beta1, 0.999))

    def _apply_length_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = (torch.arange(seq_len, device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1)).float().unsqueeze(2)
        return x * mask

    def train(self, X: np.ndarray, y: np.ndarray, lengths: np.ndarray) -> dict:
        """Train on labeled feature tensors.

        X: (N, max_seq_len, n_features)  -- normalized features
        y: (N,) in {0, 1}                -- 0 normal, 1 cheating
        lengths: (N,)                    -- real sequence lengths
        """
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        L_t = torch.LongTensor(lengths)
        ds = TensorDataset(X_t, y_t, L_t)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True,
                            drop_last=True, pin_memory=True,
                            num_workers=2, persistent_workers=True)

        bce = nn.BCEWithLogitsLoss()
        history = {"d_loss": [], "g_loss": []}
        for epoch in range(self.cfg.epochs):
            d_losses, g_losses = [], []
            for x_real, lbl, lens in loader:
                x_real = x_real.to(self.device, non_blocking=True)
                lbl = lbl.to(self.device, non_blocking=True)
                lens = lens.to(self.device, non_blocking=True)
                bsz = x_real.size(0)

                # ----- Discriminator -----
                for _ in range(self.cfg.d_updates_per_g):
                    self.opt_D.zero_grad()
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        # Real
                        d_real = self.D(x_real, lbl, lens)
                        real_targets = torch.full((bsz,), 1 - self.cfg.label_smoothing,
                                                  device=self.device)
                        loss_real = bce(d_real, real_targets)
                        # Fake
                        noise = torch.randn(bsz, self.cfg.noise_dim, device=self.device)
                        fake_labels = torch.randint(0, self.cfg.n_classes, (bsz,),
                                                     device=self.device)
                        x_fake, fake_lens = self.G(noise, fake_labels)
                        x_fake = self._apply_length_mask(x_fake, fake_lens)
                        d_fake = self.D(x_fake.detach(), fake_labels, fake_lens)
                        loss_fake = bce(d_fake, torch.zeros(bsz, device=self.device))
                        d_loss = loss_real + loss_fake
                    self.scaler_D.scale(d_loss).backward()
                    self.scaler_D.step(self.opt_D)
                    self.scaler_D.update()

                # ----- Generator -----
                self.opt_G.zero_grad()
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    noise = torch.randn(bsz, self.cfg.noise_dim, device=self.device)
                    gen_labels = torch.randint(0, self.cfg.n_classes, (bsz,),
                                                device=self.device)
                    x_gen, gen_lens = self.G(noise, gen_labels)
                    x_gen = self._apply_length_mask(x_gen, gen_lens)
                    d_gen = self.D(x_gen, gen_labels, gen_lens)
                    g_loss = bce(d_gen, torch.ones(bsz, device=self.device))
                self.scaler_G.scale(g_loss).backward()
                self.scaler_G.step(self.opt_G)
                self.scaler_G.update()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            d_avg = float(np.mean(d_losses))
            g_avg = float(np.mean(g_losses))
            history["d_loss"].append(d_avg)
            history["g_loss"].append(g_avg)
            print(f"[GAN] epoch {epoch+1:03d}/{self.cfg.epochs}  "
                  f"D={d_avg:.4f}  G={g_avg:.4f}")
        return history

    @torch.no_grad()
    def generate(self, n: int, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n tensors with a given class label. Returns (X, lengths)."""
        self.G.eval()
        noise = torch.randn(n, self.cfg.noise_dim, device=self.device)
        lbl = torch.full((n,), int(label), dtype=torch.long, device=self.device)
        x, lens = self.G(noise, lbl)
        x = self._apply_length_mask(x, lens)
        self.G.train()
        return x.cpu().numpy(), lens.cpu().numpy()


# ---------------------------------------------------------------------------
# Sanity filters (anti-hallucination guardrails)
# ---------------------------------------------------------------------------


def compute_feature_bounds(X: np.ndarray, lengths: np.ndarray,
                            pad: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature min/max from real data (with padding around the range).

    Any generated sample with features outside [min - pad*range, max + pad*range]
    is considered a hallucination and rejected.
    """
    feats = []
    for i in range(len(X)):
        L = int(lengths[i])
        if L > 0:
            feats.append(X[i, :L])
    feats = np.concatenate(feats, axis=0)
    lo = feats.min(axis=0)
    hi = feats.max(axis=0)
    rng = hi - lo
    return lo - pad * rng, hi + pad * rng


def apply_sanity_filters(X_gen: np.ndarray, lengths_gen: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray,
                         max_rejection_rate: float = 0.5
                         ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reject samples with any out-of-range feature or NaN.

    Returns (X_kept, lengths_kept, rejection_rate). Prints a warning if the
    rejection rate exceeds max_rejection_rate — that's a sign the GAN is
    hallucinating and we should not trust its output.
    """
    n = len(X_gen)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        L = int(lengths_gen[i])
        if L <= 0 or not np.isfinite(X_gen[i, :L]).all():
            keep[i] = False
            continue
        sample = X_gen[i, :L]
        if (sample < lo).any() or (sample > hi).any():
            keep[i] = False
    rej = 1.0 - keep.mean()
    if rej > max_rejection_rate:
        print(f"[GAN sanity] WARNING: rejection rate {rej*100:.1f}% > "
              f"{max_rejection_rate*100:.0f}% — GAN may be hallucinating. "
              f"Consider retraining or reducing noise_dim.")
    else:
        print(f"[GAN sanity] rejection rate {rej*100:.1f}% (acceptable)")
    return X_gen[keep], lengths_gen[keep], rej
