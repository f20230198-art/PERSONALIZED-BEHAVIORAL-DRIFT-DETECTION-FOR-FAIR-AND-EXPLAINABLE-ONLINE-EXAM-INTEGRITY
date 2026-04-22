"""
Model implementations for behavioral drift detection.

Models:
- LSTM Autoencoder (main model)
- Standard Autoencoder (baseline)
- Isolation Forest (baseline)
- One-Class SVM (baseline)
- Rule-Based detector (baseline)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import Dict, Tuple


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for sequential behavioral pattern learning.
    
    Architecture:
    - Encoder: 2-layer LSTM (input_dim -> hidden_dim -> latent_dim)
    - Decoder: 2-layer LSTM (latent_dim -> hidden_dim -> input_dim)
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, 
                 latent_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x, lengths=None):
        """Encode input sequence to latent representation.

        Args:
            x: (batch_size, seq_len, input_dim) input tensor
            lengths: (batch_size,) actual sequence lengths (for packed sequences)
        """
        if lengths is not None and lengths.min() > 0:
            # Pack sequences so the LSTM only processes valid timesteps
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.encoder_lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            # Gather the output at each sequence's actual last timestep
            batch_size = x.size(0)
            idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)
            idx = idx.expand(-1, 1, self.hidden_dim).to(x.device)
            last_hidden = lstm_out.gather(1, idx).squeeze(1)
        else:
            lstm_out, (hidden, cell) = self.encoder_lstm(x)
            last_hidden = lstm_out[:, -1, :]

        latent = self.encoder_fc(last_hidden)
        return latent
    
    def decode(self, latent, seq_len):
        """Decode latent representation back to sequence.

        The latent vector seeds the initial hidden state only.
        Zero vectors are fed as input at every decoder timestep, forcing the
        LSTM to rely solely on the hidden state to reconstruct the sequence.
        This creates a steeper reconstruction error gap between normal sessions
        (which the model learns well) and anomalous sessions (which it doesn't),
        improving both ROC-AUC and Precision@K.
        """
        batch_size = latent.size(0)

        # Project latent → initial hidden state (replicated for each layer)
        h_0 = self.decoder_fc(latent)                                        # (batch, hidden_dim)
        h_0 = h_0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous() # (layers, batch, hidden_dim)
        c_0 = torch.zeros_like(h_0)

        # Feed zeros at every timestep — decoder must rely on hidden state alone
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim, device=latent.device)

        lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
        output = self.output_fc(lstm_out)                                    # (batch, seq_len, input_dim)
        return output
    
    def forward(self, x, lengths=None):
        """Forward pass through autoencoder.

        Args:
            x: (batch, seq_len, input_dim) input sequences
            lengths: (batch,) optional actual sequence lengths
        """
        seq_len = x.size(1)
        latent = self.encode(x, lengths)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction, latent


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder for sequential behavioral pattern learning.

    Uses self-attention to capture dependencies between any two questions
    in the exam, regardless of distance.  Compared to the LSTM-AE, this
    model has no sequential inductive bias — temporal ordering is provided
    solely by learnable positional encodings.

    Architecture:
    - Encoder: Linear projection + positional encoding + Transformer encoder
               + masked mean pooling + linear bottleneck
    - Decoder: Linear expansion + positional encoding + Transformer decoder
               + linear output projection
    """

    def __init__(self, input_dim: int = 6, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, latent_dim: int = 16,
                 max_seq_len: int = 50, dropout: float = 0.2):
        super(TransformerAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bottleneck
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x, lengths=None):
        """Encode input sequence to latent representation via self-attention."""
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Padding mask: True = ignore position (PyTorch convention)
        if lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            pad_mask = None

        encoded = self.transformer_encoder(x, src_key_padding_mask=pad_mask)

        # Masked mean pooling over valid positions
        if lengths is not None:
            valid_mask = (~pad_mask).unsqueeze(2).float()  # (batch, seq, 1)
            pooled = (encoded * valid_mask).sum(1) / lengths.unsqueeze(1).float().clamp(min=1)
        else:
            pooled = encoded.mean(1)

        latent = self.to_latent(pooled)
        return latent

    def decode(self, latent, seq_len):
        """Decode latent representation back to sequence."""
        hidden = self.from_latent(latent)  # (batch, d_model)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        hidden = hidden + self.pos_encoding[:, :seq_len, :]
        decoded = self.transformer_decoder(hidden)
        output = self.output_proj(decoded)
        return output

    def forward(self, x, lengths=None):
        """Forward pass — same interface as LSTMAutoencoder."""
        seq_len = x.size(1)
        latent = self.encode(x, lengths)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction, latent


# ---------------------------------------------------------------------------
# Adversarial fairness components (in-processing fairness)
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    During the forward pass, acts as an identity.
    During the backward pass, negates gradients scaled by lambda_.
    This pushes the encoder to produce latent representations that
    a discriminator CANNOT use to predict demographic group.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DemographicDiscriminator(nn.Module):
    """Predicts demographic group from latent representation.

    Used with gradient reversal for in-processing fairness:
    the encoder learns representations that are demographic-invariant.
    """
    def __init__(self, latent_dim: int, num_groups: int):
        super(DemographicDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_groups)
        )

    def forward(self, latent, reversal_lambda=1.0):
        reversed_latent = GradientReversalFunction.apply(latent, reversal_lambda)
        return self.net(reversed_latent)


class StandardAutoencoder(nn.Module):
    """
    Standard feedforward autoencoder (baseline).
    
    Architecture: [input_dim] -> [64] -> [32] -> [64] -> [input_dim]
    """
    
    def __init__(self, input_dim: int = 6, hidden_dims: list = [64, 32, 64]):
        super(StandardAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:len(hidden_dims)//2 + 1]:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for hidden_dim in hidden_dims[len(hidden_dims)//2 + 1:]:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass."""
        # Flatten if needed
        if x.dim() == 3:
            batch_size, seq_len, input_dim = x.shape
            x = x.view(batch_size, -1)
            is_sequential = True
        else:
            is_sequential = False
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        # Reshape back if needed
        if is_sequential:
            reconstruction = reconstruction.view(batch_size, seq_len, input_dim)
        
        return reconstruction, latent


class IsolationForestDetector:
    """Isolation Forest anomaly detector (baseline)."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray):
        """Train the detector."""
        # Flatten sequential data if needed
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 = normal, -1 = anomaly)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return -self.model.score_samples(X)  # Negate to make higher = more anomalous


class OneClassSVMDetector:
    """One-Class SVM anomaly detector (baseline)."""
    
    def __init__(self, kernel: str = 'rbf', nu: float = 0.1):
        self.model = OneClassSVM(kernel=kernel, nu=nu)
    
    def fit(self, X: np.ndarray):
        """Train the detector."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 = normal, -1 = anomaly)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return -self.model.decision_function(X)  # Negate for consistent scoring


class RuleBasedDetector:
    """
    Rule-based anomaly detector (baseline).

    Flags sessions as anomalous if they violate heuristic rules on RAW
    (unnormalized) features.  Requires at least 2 of 3 rules to trigger
    (conjunction via voting) to avoid the massive false-positive rate
    that a single-rule OR produces on real EdNet data.

    Rules:
    - Response time too fast (mean < response_time_min seconds)
    - Too many answer changes (rate > answer_change_max per question)
    - High burst ratio (> burst_ratio_max of actions under 2s apart)
    """

    def __init__(self, response_time_min: float = 1.0,
                 answer_change_max: float = 2.0,
                 burst_ratio_max: float = 0.95):
        self.response_time_min = response_time_min
        self.answer_change_max = answer_change_max
        self.burst_ratio_max = burst_ratio_max

    def _rule_violations(self, features: np.ndarray,
                         feature_names: list) -> np.ndarray:
        """Count per-sample rule violations (0-3)."""
        scores = np.zeros(len(features))

        rt_mean_idx = feature_names.index('response_time_mean')
        answer_change_idx = feature_names.index('answer_change_rate')

        scores += (features[:, rt_mean_idx] < self.response_time_min).astype(float)
        scores += (features[:, answer_change_idx] > self.answer_change_max).astype(float)

        if 'burst_ratio' in feature_names:
            burst_idx = feature_names.index('burst_ratio')
            scores += (features[:, burst_idx] > self.burst_ratio_max).astype(float)

        return scores

    def predict(self, features: np.ndarray, feature_names: list) -> np.ndarray:
        """
        Predict anomalies based on rules (expects RAW unnormalized features).
        Flags as anomalous if ANY rule fires — thresholds should be set tight
        enough that a single violation is already strong evidence.

        Returns:
            predictions: 1 = normal, -1 = anomaly
        """
        scores = self._rule_violations(features, feature_names)
        predictions = np.ones(len(features))
        predictions[scores >= 1] = -1
        return predictions

    def score_samples(self, features: np.ndarray, feature_names: list) -> np.ndarray:
        """Get anomaly scores (count of rule violations, 0-3)."""
        return self._rule_violations(features, feature_names)


def compute_reconstruction_error(model: nn.Module, X: torch.Tensor,
                                device: torch.device,
                                lengths: torch.Tensor = None,
                                batch_size: int = 256) -> np.ndarray:
    """Compute per-sample reconstruction error, masked for padded positions (batched for memory safety)."""
    model.eval()
    out = []
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size].to(device)
            lb = lengths[i:i+batch_size].to(device) if lengths is not None else None

            if lb is not None:
                reconstruction, _ = model(xb, lb)
                bs, seq_len, feat_dim = xb.shape
                mask = torch.zeros(bs, seq_len, 1, device=device)
                for j in range(bs):
                    mask[j, :lb[j], :] = 1.0
                diff_sq = (xb - reconstruction) ** 2 * mask
                mse = diff_sq.sum(dim=(1, 2)) / (lb.float() * feat_dim)
            else:
                reconstruction, _ = model(xb)
                mse = torch.mean((xb - reconstruction) ** 2, dim=(1, 2))

            out.append(mse.cpu().numpy())
            del xb, reconstruction, mse
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)
