"""
LSTM Variational Autoencoder.

Mirrors LSTMAutoencoder's interface (forward(x, lengths) -> (recon, latent))
so it drops into the existing Trainer. Implements the VAE loss via the
`vae_loss` method that Trainer picks up when model.is_vae is True.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMVAE(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 96,
                 latent_dim: int = 32, num_layers: int = 2,
                 dropout: float = 0.35, kl_beta: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.kl_beta = kl_beta
        self.is_vae = True

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    # ---------------- encode / decode ----------------
    def _encode(self, x: torch.Tensor, lengths: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lengths is not None and lengths.min() > 0:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            out, _ = self.encoder_lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)
            idx = idx.expand(-1, 1, self.hidden_dim).to(x.device)
            last = out.gather(1, idx).squeeze(1)
        else:
            out, _ = self.encoder_lstm(x)
            last = out[:, -1, :]
        mu = self.to_mu(last)
        logvar = self.to_logvar(last).clamp(-8.0, 8.0)
        return mu, logvar

    def _reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def _decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch = z.size(0)
        h_0 = self.from_latent(z).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_0 = torch.zeros_like(h_0)
        dec_in = torch.zeros(batch, seq_len, self.input_dim, device=z.device)
        out, _ = self.decoder_lstm(dec_in, (h_0, c_0))
        return self.output_fc(out)

    # ---------------- public ----------------
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(x, lengths)
        z = self._reparameterise(mu, logvar)
        recon = self._decode(z, x.size(1))
        self._last_mu = mu
        self._last_logvar = logvar
        return recon, mu

    def vae_loss(self, recon: torch.Tensor, target: torch.Tensor,
                 lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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
