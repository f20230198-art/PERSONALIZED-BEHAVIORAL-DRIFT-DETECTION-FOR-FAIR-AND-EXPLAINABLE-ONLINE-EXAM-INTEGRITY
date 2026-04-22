"""
Transformer Variational Autoencoder for sequential behavioral data.

Interface mirrors TransformerAutoencoder so it slots into the existing
training loop (forward(x, lengths) -> (reconstruction, latent)).

Why VAE over plain AE:
  * The VAE forces the latent distribution to be close to N(0, I), which
    acts as a regulariser and tends to produce a smoother latent manifold
    for anomaly detection (anomalies land further from the origin in KL
    terms, giving a second signal beyond reconstruction error).
  * We return the mean of q(z|x) as the "latent" the downstream code uses
    for Mahalanobis scoring, matching the TransformerAutoencoder API.
  * The Trainer already supports an optional custom loss via the model's
    `vae_loss` attribute; the training loop checks for it and, if present,
    uses it instead of plain MSE.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerVAE(nn.Module):
    def __init__(self, input_dim: int = 10, d_model: int = 96, nhead: int = 4,
                 num_layers: int = 2, latent_dim: int = 32,
                 max_seq_len: int = 50, dropout: float = 0.2,
                 kl_beta: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Variational bottleneck
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

        # Flag so Trainer knows to use the VAE loss (recon + KL)
        self.is_vae = True

    # --------------------------- encode / decode ---------------------------
    def _encode(self, x: torch.Tensor, lengths: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        h = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
        if lengths is not None:
            pad_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0)
                        >= lengths.unsqueeze(1))
        else:
            pad_mask = None
        enc = self.encoder(h, src_key_padding_mask=pad_mask)
        if lengths is not None:
            valid = (~pad_mask).unsqueeze(2).float()
            pooled = (enc * valid).sum(1) / lengths.unsqueeze(1).float().clamp(min=1)
        else:
            pooled = enc.mean(1)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-8.0, 8.0)  # numerical safety
        return mu, logvar

    def _reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu  # deterministic at eval time
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        h = self.from_latent(z).unsqueeze(1).repeat(1, seq_len, 1)
        h = h + self.pos_encoding[:, :seq_len, :]
        dec = self.decoder(h)
        return self.output_proj(dec)

    # --------------------------- public API ---------------------------
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        mu, logvar = self._encode(x, lengths)
        z = self._reparameterise(mu, logvar)
        recon = self._decode(z, seq_len)
        # Stash for loss computation
        self._last_mu = mu
        self._last_logvar = logvar
        return recon, mu

    def vae_loss(self, recon: torch.Tensor, target: torch.Tensor,
                 lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruction MSE (masked) + beta * KL divergence."""
        if lengths is not None:
            seq_len = target.size(1)
            mask = (torch.arange(seq_len, device=target.device).unsqueeze(0)
                    < lengths.unsqueeze(1)).float().unsqueeze(2)
            se = ((recon - target) ** 2) * mask
            denom = mask.sum().clamp(min=1.0) * target.size(2)
            recon_loss = se.sum() / denom
        else:
            recon_loss = F.mse_loss(recon, target)

        mu, logvar = self._last_mu, self._last_logvar
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_beta * kl
