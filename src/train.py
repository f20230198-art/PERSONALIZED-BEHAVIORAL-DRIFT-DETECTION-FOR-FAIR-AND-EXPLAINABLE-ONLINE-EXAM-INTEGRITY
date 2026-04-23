"""
Training script for LSTM / Transformer Autoencoders with optional
adversarial fairness regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import os


class Trainer:
    """Trainer with masked loss, gradient clipping, LR scheduling,
    and optional adversarial demographic-invariance regularization."""

    def __init__(self, model: nn.Module, config: Dict, device: torch.device,
                 discriminator: Optional[nn.Module] = None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.use_amp = (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Optimizer (model + discriminator params if present)
        params = list(self.model.parameters())
        self.discriminator = None
        if discriminator is not None:
            self.discriminator = discriminator.to(device)
            params += list(self.discriminator.parameters())
        self.optimizer = optim.Adam(
            params, lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )

        # LR scheduler: halve LR when val loss plateaus for 5 epochs
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Fairness training settings
        fairness_cfg = config.get('fairness_training', {})
        self.fairness_lambda = fairness_cfg.get('lambda', 0.0)
        self.fairness_warmup = fairness_cfg.get('warmup_epochs', 10)
        self.adv_loss_fn = nn.CrossEntropyLoss() if self.discriminator else None

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = config['training']['early_stopping_patience']
        self.patience_counter = 0

        # History
        self.train_losses = []
        self.val_losses = []

    def _masked_mse(self, reconstruction, target, lengths=None):
        """Compute MSE loss, masked for padded positions."""
        if lengths is not None:
            batch_size, seq_len, feat_dim = target.shape
            mask = torch.zeros(batch_size, seq_len, 1, device=target.device)
            for i in range(batch_size):
                mask[i, :lengths[i], :] = 1.0
            diff_sq = (target - reconstruction) ** 2 * mask
            loss = diff_sq.sum() / (mask.sum() * feat_dim)
        else:
            loss = torch.mean((target - reconstruction) ** 2)
        return loss

    def _get_reversal_lambda(self, epoch: int) -> float:
        """Ramp up the gradient reversal strength after warm-up."""
        if epoch < self.fairness_warmup:
            return 0.0
        progress = (epoch - self.fairness_warmup) / max(1, 50)
        return float(self.fairness_lambda * min(1.0, progress))

    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> float:
        """Train for one epoch."""
        self.model.train()
        if self.discriminator:
            self.discriminator.train()
        total_loss = 0.0
        rev_lambda = self._get_reversal_lambda(epoch)

        for batch in train_loader:
            X = batch[0].to(self.device, non_blocking=True)
            lengths = batch[1].to(self.device, non_blocking=True) if len(batch) > 1 else None
            demo_labels = batch[2].to(self.device, non_blocking=True) if len(batch) > 2 else None

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                reconstruction, latent = self.model(X, lengths)
                if getattr(self.model, 'is_vae', False) and hasattr(self.model, 'vae_loss'):
                    recon_loss = self.model.vae_loss(reconstruction, X, lengths)
                else:
                    recon_loss = self._masked_mse(reconstruction, X, lengths)

                # Adversarial fairness loss
                adv_loss = torch.tensor(0.0, device=self.device)
                if self.discriminator is not None and demo_labels is not None and rev_lambda > 0:
                    demo_pred = self.discriminator(latent, reversal_lambda=rev_lambda)
                    adv_loss = self.adv_loss_fn(demo_pred, demo_labels)

                loss = recon_loss + rev_lambda * adv_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += recon_loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                X = batch[0].to(self.device, non_blocking=True)
                lengths = batch[1].to(self.device, non_blocking=True) if len(batch) > 1 else None

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    reconstruction, _ = self.model(X, lengths)
                    if getattr(self.model, 'is_vae', False) and hasattr(self.model, 'vae_loss'):
                        loss = self.model.vae_loss(reconstruction, X, lengths)
                    else:
                        loss = self._masked_mse(reconstruction, X, lengths)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, save_dir: str) -> Dict[str, list]:
        """Train the model with early stopping and LR scheduling."""
        print(f"Training for up to {epochs} epochs...")
        if self.discriminator:
            print(f"  Adversarial fairness: lambda={self.fairness_lambda}, "
                  f"warmup={self.fairness_warmup} epochs")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch=epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Step the LR scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - "
                  f"Val: {val_loss:.6f} - LR: {current_lr:.2e}")

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'))
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Periodic checkpoints
            if (epoch + 1) % self.config['output']['checkpoint_frequency'] == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        print(f"Training complete. Best validation loss: {self.best_val_loss:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        if self.discriminator:
            state['discriminator_state_dict'] = self.discriminator.state_dict()
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        if self.discriminator and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f"Loaded checkpoint from {filepath}")


def compute_mahalanobis_scores(latents: np.ndarray,
                               train_latents: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance of each latent vector from the training distribution.

    Normal sessions cluster tightly in latent space; anomalous sessions produce
    latent vectors that land far from the training cluster.  Combined with
    reconstruction error, this gives a second independent signal.

    Uses a regularized covariance estimate to handle near-singular matrices.
    """
    mu = np.mean(train_latents, axis=0)
    cov = np.cov(train_latents, rowvar=False)

    # Regularize: add small diagonal to avoid singular matrix
    cov += np.eye(cov.shape[0]) * 1e-4

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal only
        cov_inv = np.diag(1.0 / (np.diag(cov) + 1e-4))

    diff = latents - mu  # (N, latent_dim)
    # Mahalanobis: sqrt(diff @ cov_inv @ diff.T) per sample
    scores = np.sqrt(np.einsum('ni,ij,nj->n', diff, cov_inv, diff))
    return scores


def _extract_latents(model: nn.Module, X: torch.Tensor, device: torch.device,
                     lengths: torch.Tensor = None, batch_size: int = 256) -> np.ndarray:
    """Extract latent representations for all samples (batched to avoid OOM)."""
    model.eval()
    out = []
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size].to(device)
            if lengths is not None:
                lb = lengths[i:i+batch_size].to(device)
                _, latents = model(xb, lb)
            else:
                _, latents = model(xb)
            out.append(latents.cpu().numpy())
            del xb, latents
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def compute_drift_scores(model: nn.Module, X: torch.Tensor, device: torch.device,
                        train_errors: np.ndarray = None,
                        lengths: torch.Tensor = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute drift scores using Z-score normalization of reconstruction errors.

    DriftScore(s, e*) = (L_recon(s, e*) - mu) / sigma

    Args:
        model: Trained autoencoder
        X: Input data (batch, seq_len, features)
        device: PyTorch device
        train_errors: Training reconstruction errors for computing mean/std
        lengths: (batch,) actual sequence lengths for masked MSE
    """
    from src.models import compute_reconstruction_error

    X = X.to(device)
    if lengths is not None:
        lengths = lengths.to(device)
    errors = compute_reconstruction_error(model, X, device, lengths)

    if train_errors is not None:
        # Robust z-score: median/MAD is more stable than mean/std for
        # skewed MSE distributions, preserving the normal-vs-anomalous gap.
        median = np.median(train_errors)
        mad = np.median(np.abs(train_errors - median)) * 1.4826  # scale to match std for normal dist
        drift_scores = (errors - median) / (mad if mad > 0 else 1.0)
    else:
        drift_scores = errors

    return errors, drift_scores


def compute_blended_drift_scores(model: nn.Module, X: torch.Tensor, device: torch.device,
                                  train_errors: np.ndarray,
                                  train_student_ids: np.ndarray,
                                  eval_student_ids: np.ndarray,
                                  lengths: torch.Tensor = None,
                                  n_min: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute personalized blended drift scores.

    DriftScore(s, e*) = λ_s · DriftScore_personal(s, e*) + (1 - λ_s) · DriftScore_pop(s, e*)

    where λ_s = min(1, n_s / N_min) and n_s is the number of historical sessions
    for student s in the training set.

    Students with ≥ N_min training sessions get fully personalized scoring.
    Students with fewer sessions smoothly blend personal and population baselines.

    Args:
        model: Trained autoencoder
        X: Input data (batch, seq_len, features)
        device: PyTorch device
        train_errors: Training reconstruction errors (one per clean training session)
        train_student_ids: Student ID for each training session (parallel to train_errors)
        eval_student_ids: Student ID for each evaluation session (parallel to X)
        lengths: Actual sequence lengths for masked MSE
        n_min: Minimum sessions for full personalization (default: 5)

    Returns:
        raw_errors: Raw reconstruction errors
        blended_scores: Personalized blended drift scores
    """
    from src.models import compute_reconstruction_error
    from collections import defaultdict

    X_dev = X.to(device)
    if lengths is not None:
        lengths = lengths.to(device)
    errors = compute_reconstruction_error(model, X_dev, device, lengths)

    # --- Population-level statistics (robust: median/MAD) ---
    pop_median = np.median(train_errors)
    pop_mad = np.median(np.abs(train_errors - pop_median)) * 1.4826
    if pop_mad == 0:
        pop_mad = 1.0

    # --- Per-student statistics from training errors ---
    student_errors = defaultdict(list)
    for err, sid in zip(train_errors, train_student_ids):
        student_errors[sid].append(err)

    student_stats = {}  # sid -> (median, mad, n_sessions)
    for sid, errs in student_errors.items():
        errs = np.array(errs)
        s_median = np.median(errs)
        s_mad = np.median(np.abs(errs - s_median)) * 1.4826
        if s_mad == 0:
            s_mad = pop_mad * 0.5  # Regularize: use fraction of pop MAD
        student_stats[sid] = (s_median, s_mad, len(errs))

    # --- Compute blended scores for each eval session ---
    blended_scores = np.zeros(len(errors))
    for i, (err, sid) in enumerate(zip(errors, eval_student_ids)):
        # Population z-score
        pop_score = (err - pop_median) / pop_mad

        if sid in student_stats:
            s_median, s_mad, n_s = student_stats[sid]
            # Personal z-score
            personal_score = (err - s_median) / s_mad
            # Lambda blending: smooth transition from population to personal
            lambda_s = min(1.0, n_s / n_min)
            blended_scores[i] = lambda_s * personal_score + (1 - lambda_s) * pop_score
        else:
            # No history → pure population score
            blended_scores[i] = pop_score

    return errors, blended_scores


def personalize_scores(train_scores: np.ndarray,
                       eval_scores: np.ndarray,
                       train_student_ids: np.ndarray,
                       eval_student_ids: np.ndarray,
                       n_min: int = 5) -> np.ndarray:
    """
    Apply personalized (per-student) blended scoring to ANY model's raw scores.

    Same λ-blend logic as compute_blended_drift_scores, but generalized to work
    with any scoring method (supervised classifier outputs, IF scores, OC-SVM
    decision function, StandardAE reconstruction errors, etc).

    DriftScore(s) = λ_s · personal_zscore(s) + (1 - λ_s) · pop_zscore(s)
    λ_s = min(1, n_s / n_min)

    Args:
        train_scores: Raw scores on CLEAN training sessions (for computing personal baselines)
        eval_scores: Raw scores on evaluation sessions to personalize
        train_student_ids: Student IDs parallel to train_scores
        eval_student_ids: Student IDs parallel to eval_scores
        n_min: Minimum sessions for full personalization (default: 5)

    Returns:
        Personalized blended scores (same shape as eval_scores)
    """
    from collections import defaultdict

    train_scores = np.asarray(train_scores)
    eval_scores = np.asarray(eval_scores)

    # Population stats
    pop_median = np.median(train_scores)
    pop_mad = np.median(np.abs(train_scores - pop_median)) * 1.4826
    if pop_mad == 0:
        pop_mad = 1.0

    # Per-student stats
    student_scores = defaultdict(list)
    for s, sid in zip(train_scores, train_student_ids):
        student_scores[sid].append(s)

    student_stats = {}
    for sid, scores in student_scores.items():
        scores = np.array(scores)
        s_median = np.median(scores)
        s_mad = np.median(np.abs(scores - s_median)) * 1.4826
        if s_mad == 0:
            s_mad = pop_mad * 0.5
        student_stats[sid] = (s_median, s_mad, len(scores))

    # Blend
    out = np.zeros(len(eval_scores))
    for i, (score, sid) in enumerate(zip(eval_scores, eval_student_ids)):
        pop_z = (score - pop_median) / pop_mad
        if sid in student_stats:
            s_median, s_mad, n_s = student_stats[sid]
            personal_z = (score - s_median) / s_mad
            lam = min(1.0, n_s / n_min)
            out[i] = lam * personal_z + (1 - lam) * pop_z
        else:
            out[i] = pop_z
    return out


def compute_combined_scores(model: nn.Module, X: torch.Tensor, device: torch.device,
                             train_errors: np.ndarray,
                             train_student_ids: np.ndarray,
                             eval_student_ids: np.ndarray,
                             train_X: torch.Tensor,
                             lengths: torch.Tensor = None,
                             train_lengths: torch.Tensor = None,
                             recon_weight: float = 0.4,
                             mahal_weight: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """Combine blended reconstruction drift scores with Mahalanobis latent-space scores.

    Final score = recon_weight * z(recon_error) + mahal_weight * z(mahalanobis)

    Both components are z-scored to the same scale before combining.
    This gives two independent signals:
      - Reconstruction error: how poorly the model rebuilds the sequence
      - Mahalanobis distance: how far the latent vector is from normal clusters

    Args:
        model: Trained autoencoder
        X: Evaluation data (batch, seq_len, features)
        device: PyTorch device
        train_errors: Training reconstruction errors
        train_student_ids: Student IDs for training sessions
        eval_student_ids: Student IDs for evaluation sessions
        train_X: Training data tensor (for extracting train latents)
        lengths: Eval sequence lengths
        train_lengths: Train sequence lengths
        recon_weight: Weight for reconstruction error component (default 0.4)
        mahal_weight: Weight for Mahalanobis component (default 0.6)

    Returns:
        raw_errors: Raw reconstruction errors
        combined_scores: Weighted combination of both signals
    """
    # Component 1: blended reconstruction drift scores
    raw_errors, blended_scores = compute_blended_drift_scores(
        model, X, device, train_errors, train_student_ids, eval_student_ids, lengths
    )

    # Component 2: Mahalanobis distance in latent space
    train_latents = _extract_latents(model, train_X, device, train_lengths)
    eval_latents = _extract_latents(model, X, device, lengths)
    mahal_scores = compute_mahalanobis_scores(eval_latents, train_latents)

    # Z-score both to the same scale
    def _zscore(arr):
        median = np.median(arr)
        mad = np.median(np.abs(arr - median)) * 1.4826
        return (arr - median) / (mad if mad > 0 else 1.0)

    recon_z = _zscore(blended_scores)
    mahal_z = _zscore(mahal_scores)

    combined = recon_weight * recon_z + mahal_weight * mahal_z
    return raw_errors, combined


def train_baseline_models(X_train: np.ndarray, config: Dict) -> Dict:
    """Train all baseline models."""
    from src.models import IsolationForestDetector, OneClassSVMDetector, RuleBasedDetector
    
    print("Training baseline models...")
    
    baselines = {}
    
    # Isolation Forest
    if_config = [b for b in config['baselines'] if b['name'] == 'IsolationForest'][0]
    iso_forest = IsolationForestDetector(
        contamination=if_config['contamination'],
        n_estimators=if_config['n_estimators']
    )
    iso_forest.fit(X_train)
    baselines['IsolationForest'] = iso_forest
    print("✓ Isolation Forest trained")
    
    # One-Class SVM
    svm_config = [b for b in config['baselines'] if b['name'] == 'OneClassSVM'][0]
    ocsvm = OneClassSVMDetector(
        kernel=svm_config['kernel'],
        nu=svm_config['nu']
    )
    ocsvm.fit(X_train)
    baselines['OneClassSVM'] = ocsvm
    print("✓ One-Class SVM trained")
    
    # Rule-Based (no training needed)
    rule_config = [b for b in config['baselines'] if b['name'] == 'RuleBased'][0]
    rule_based = RuleBasedDetector(
        response_time_min=rule_config['thresholds']['response_time_min'],
        answer_change_max=rule_config['thresholds']['answer_change_max'],
        burst_ratio_max=rule_config['thresholds'].get('burst_ratio_max', 0.95)
    )
    baselines['RuleBased'] = rule_based
    print("✓ Rule-Based detector initialized")
    
    # Standard Autoencoder (with early stopping on a held-out validation split)
    print("Training Standard Autoencoder...")
    from src.models import StandardAutoencoder

    ae_config = [b for b in config['baselines'] if b['name'] == 'StandardAutoencoder'][0]
    std_ae = StandardAutoencoder(
        input_dim=X_train.shape[-1],
        hidden_dims=ae_config['hidden_dims']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    std_ae = std_ae.to(device)

    optimizer = optim.Adam(std_ae.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Hold out 15% of training data for validation-based early stopping
    n_total = len(X_train)
    n_val_ae = max(1, int(n_total * 0.15))
    perm = np.random.permutation(n_total)
    val_idx_ae = perm[:n_val_ae]
    train_idx_ae = perm[n_val_ae:]

    X_train_ae = torch.FloatTensor(X_train[train_idx_ae]).to(device)
    X_val_ae = torch.FloatTensor(X_train[val_idx_ae]).to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    ae_patience = 10
    best_state = None

    for epoch in range(100):
        std_ae.train()
        reconstruction, _ = std_ae(X_train_ae)
        loss = criterion(reconstruction, X_train_ae)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation loss
        std_ae.eval()
        with torch.no_grad():
            val_recon, _ = std_ae(X_val_ae)
            val_loss = criterion(val_recon, X_val_ae).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in std_ae.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/100 - Train: {loss.item():.6f} - Val: {val_loss:.6f}")

        if patience_counter >= ae_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_state is not None:
        std_ae.load_state_dict(best_state)

    baselines['StandardAutoencoder'] = std_ae
    print("✓ Standard Autoencoder trained")
    
    return baselines
