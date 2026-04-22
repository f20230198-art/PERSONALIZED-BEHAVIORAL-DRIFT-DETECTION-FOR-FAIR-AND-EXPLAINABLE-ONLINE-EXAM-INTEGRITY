"""
Plain LSTM and Transformer classifiers (supervised baselines).

Ablation-study counterparts to the autoencoder models: same encoder
backbone, but trained directly on synthetic binary labels with a
classification head instead of an autoencoder reconstruction objective.

IMPORTANT CAVEAT: these models use the synthetic labels during training.
The autoencoders do not (they train on clean data only). Therefore these
classifiers represent a *supervised upper bound* for what detection
performance is achievable on this synthetic distribution — not an
unsupervised detector. Reporting them is informative precisely because
it shows how much signal the unsupervised AEs leave on the table.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PlainLSTMClassifier(nn.Module):
    """Plain LSTM encoder + linear classification head.

    No decoder, no reconstruction. Just: sequence -> last hidden -> logit.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 96,
                 num_layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if lengths is not None and lengths.min() > 0:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)
            idx = idx.expand(-1, 1, self.hidden_dim).to(x.device)
            last = out.gather(1, idx).squeeze(1)
        else:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
        return self.head(last).squeeze(1)  # logits (batch,)


class PlainTransformerClassifier(nn.Module):
    """Plain Transformer encoder + linear classification head."""

    def __init__(self, input_dim: int = 10, d_model: int = 96, nhead: int = 4,
                 num_layers: int = 2, max_seq_len: int = 100, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
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
        return self.head(pooled).squeeze(1)


# ---------------------------------------------------------------------------
# Trainer (separate from Trainer in src.train since that one is for AEs)
# ---------------------------------------------------------------------------


class ClassifierTrainer:
    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, weight_decay: float = 5e-4,
                 pos_weight: Optional[float] = None, gradient_clip: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def _run_batch(self, x, y, lens, train: bool):
        x = x.to(self.device); y = y.to(self.device); lens = lens.to(self.device)
        logits = self.model(x, lens)
        loss = self.loss_fn(logits, y.float())
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        return loss.item()

    def train(self, X_train, y_train, L_train, X_val, y_val, L_val,
              epochs: int = 30, batch_size: int = 256, patience: int = 7,
              save_path: Optional[str] = None) -> dict:
        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train),
            torch.LongTensor(L_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val),
            torch.LongTensor(L_val)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

        best_val = float('inf'); bad = 0
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            self.model.train()
            tl = [self._run_batch(x, y, L, True) for x, y, L in train_loader]
            self.model.eval()
            with torch.no_grad():
                vl = [self._run_batch(x, y, L, False) for x, y, L in val_loader]
            t_avg, v_avg = float(np.mean(tl)), float(np.mean(vl))
            history["train_loss"].append(t_avg)
            history["val_loss"].append(v_avg)
            self.scheduler.step(v_avg)
            print(f"  [Classifier] epoch {epoch+1:03d}/{epochs}  "
                  f"train={t_avg:.4f}  val={v_avg:.4f}")

            if v_avg < best_val:
                best_val = v_avg; bad = 0
                if save_path:
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
            else:
                bad += 1
                if bad >= patience:
                    print(f"  [Classifier] early stop at epoch {epoch+1}")
                    break
        if save_path:
            ckpt = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
        return history

    @torch.no_grad()
    def predict_scores(self, X, L, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(L))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        out = []
        for x, lens in loader:
            x = x.to(self.device); lens = lens.to(self.device)
            logits = self.model(x, lens)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out, axis=0)
