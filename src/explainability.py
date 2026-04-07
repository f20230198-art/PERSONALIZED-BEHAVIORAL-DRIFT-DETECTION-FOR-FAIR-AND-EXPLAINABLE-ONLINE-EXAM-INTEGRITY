"""
Explainability module using SHAP for behavioral drift detection.

Two explanation strategies are provided:

1. **SequentialSHAPExplainer** (primary): Aggregates per-question features
   into session-level summaries (mean/std across timesteps per feature),
   then uses KernelSHAP on those summaries. This explains the model's
   actual behaviour on sequential data without the invalid length-1 hack.

2. **SHAPExplainer** (legacy): Kept for backward compatibility with session-
   level baselines.
"""

import numpy as np
import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


class SHAPExplainer:
    """SHAP-based explainability for drift detection models.

    Works on session-level (flat) features — appropriate for baselines
    (StandardAutoencoder, IsolationForest, etc.) but NOT for the LSTM/
    Transformer models which operate on sequential data.
    """

    def __init__(self, model, feature_names: List[str], config: Dict, device: torch.device):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.device = device
        self.explainer = None

    def create_prediction_function(self):
        """Create a prediction function compatible with SHAP."""
        def predict_fn(X):
            self.model.eval()
            if X.ndim == 2:
                X = X[:, np.newaxis, :]
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                reconstruction, _ = self.model(X_tensor)
                errors = torch.mean((X_tensor - reconstruction) ** 2, dim=(1, 2))
            return errors.cpu().numpy()
        return predict_fn

    def init_explainer(self, background_data: np.ndarray):
        """Initialize SHAP KernelExplainer."""
        if background_data.ndim == 3:
            background_data = background_data.squeeze(1)
        n_samples = self.config['explainability']['kernel_explainer_samples']
        if len(background_data) > n_samples:
            indices = np.random.choice(len(background_data), n_samples, replace=False)
            background_data = background_data[indices]
        prediction_fn = self.create_prediction_function()
        print(f"Initializing SHAP explainer with {len(background_data)} background samples...")
        self.explainer = shap.KernelExplainer(prediction_fn, background_data)
        print("SHAP explainer initialized")
    
    def explain_instance(self, instance: np.ndarray) -> shap.Explanation:
        """
        Explain a single instance.
        
        Args:
            instance: (n_features,) or (1, n_features) array
        
        Returns:
            shap_values: SHAP explanation object
        """
        if instance.ndim == 3:
            instance = instance.squeeze(1)
        elif instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(instance)
        
        return shap_values
    
    def explain_batch(self, instances: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        Explain a batch of instances.
        
        Args:
            instances: (n_samples, n_features) array
            max_samples: Maximum samples to explain (SHAP can be slow)
        
        Returns:
            shap_values: (n_samples, n_features) array of SHAP values
        """
        if instances.ndim == 3:
            instances = instances.squeeze(1)
        
        # Limit number of samples
        if len(instances) > max_samples:
            indices = np.random.choice(len(instances), max_samples, replace=False)
            instances = instances[indices]
        
        print(f"Computing SHAP values for {len(instances)} instances...")
        shap_values = self.explainer.shap_values(instances)
        print("SHAP computation complete")
        
        return np.array(shap_values)
    
    def get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        Compute global feature importance from SHAP values.
        
        Args:
            shap_values: (n_samples, n_features) array
        
        Returns:
            importance: Dictionary mapping feature names to importance scores
        """
        # Mean absolute SHAP value per feature
        abs_shap = np.abs(shap_values)
        mean_importance = np.mean(abs_shap, axis=0)
        
        importance = {
            feature: float(score)
            for feature, score in zip(self.feature_names, mean_importance)
        }
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def plot_feature_importance(self, importance: Dict[str, float], save_path: str = None):
        """Plot global feature importance."""
        plt.figure(figsize=(10, 6))
        
        features = list(importance.keys())
        scores = list(importance.values())
        
        sns.barplot(x=scores, y=features, palette='viridis')
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Global Feature Importance for Drift Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        
        plt.close()
    
    def plot_summary(self, shap_values: np.ndarray, instances: np.ndarray, 
                    save_path: str = None):
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: (n_samples, n_features) SHAP values
            instances: (n_samples, n_features) feature values
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        if instances.ndim == 3:
            instances = instances.squeeze(1)
        
        shap.summary_plot(
            shap_values, 
            instances, 
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP summary plot to {save_path}")
        
        plt.close()
    
    def plot_waterfall(self, instance: np.ndarray, shap_value: np.ndarray, 
                      save_path: str = None):
        """
        Create SHAP waterfall plot for a single instance.
        
        Args:
            instance: (n_features,) array
            shap_value: (n_features,) array of SHAP values
            save_path: Path to save plot
        """
        if instance.ndim > 1:
            instance = instance.flatten()
        if shap_value.ndim > 1:
            shap_value = shap_value.flatten()
        
        # Create explanation object
        base_value = self.explainer.expected_value
        
        plt.figure(figsize=(12, 8))
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_value))[::-1]
        
        # Plot waterfall
        cumulative = base_value
        y_pos = 0
        
        for idx in sorted_indices:
            feature = self.feature_names[idx]
            value = instance[idx]
            shap_val = shap_value[idx]
            
            color = 'red' if shap_val > 0 else 'blue'
            plt.barh(y_pos, shap_val, left=cumulative, color=color, alpha=0.7)
            plt.text(cumulative + shap_val/2, y_pos, f'{feature}\\n{value:.3f}', 
                    ha='center', va='center', fontsize=9)
            
            cumulative += shap_val
            y_pos += 1
        
        plt.axvline(base_value, color='black', linestyle='--', label='Base value')
        plt.xlabel('Drift Score Contribution', fontsize=12)
        plt.title('SHAP Waterfall Plot: Feature Contributions to Anomaly Score', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved waterfall plot to {save_path}")
        
        plt.close()
    
    def explain_top_anomalies(self, X: np.ndarray, scores: np.ndarray, 
                             top_k: int = 10, save_dir: str = None):
        """
        Explain the top-k most anomalous instances.
        
        Args:
            X: Feature matrix
            scores: Anomaly scores
            top_k: Number of top anomalies to explain
            save_dir: Directory to save explanation plots
        """
        # Get top-k anomalies
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        print(f"Explaining top {top_k} anomalies...")
        
        explanations = []
        
        for rank, idx in enumerate(top_indices):
            instance = X[idx]
            score = scores[idx]
            
            # Get SHAP explanation
            shap_value = self.explain_instance(instance)
            
            if shap_value.ndim > 1:
                shap_value = shap_value.flatten()
            if instance.ndim > 1:
                instance = instance.flatten()
            
            explanations.append({
                'rank': rank + 1,
                'index': int(idx),
                'anomaly_score': float(score),
                'shap_values': shap_value,
                'features': instance
            })
            
            # Save waterfall plot
            if save_dir:
                plot_path = os.path.join(save_dir, f'anomaly_rank_{rank+1}_explanation.png')
                self.plot_waterfall(instance, shap_value, save_path=plot_path)
        
        print(f"Explained top {top_k} anomalies")
        
        return explanations


class SequentialSHAPExplainer:
    """SHAP explainer for LSTM / Transformer autoencoders on sequential data.

    Strategy: aggregate each (seq_len, n_features) sequence into a fixed-size
    summary vector by computing the *mean* and *std* of each feature across
    valid (non-padded) timesteps.  KernelSHAP then operates on these 2*F
    summary features, and the resulting SHAP values tell us which *feature
    statistics* most drove the reconstruction error up.

    This avoids the invalid length-1 squeeze that the legacy SHAPExplainer
    performed when called on sequential models.
    """

    def __init__(self, model, question_feature_names: List[str],
                 config: Dict, device: torch.device):
        self.model = model
        self.q_feature_names = question_feature_names
        # Summary feature names: mean and std per question-level feature
        self.feature_names = (
            [f"{n}_mean" for n in question_feature_names] +
            [f"{n}_std" for n in question_feature_names]
        )
        self.config = config
        self.device = device
        self.explainer = None
        # Store the raw sequential background data + lengths for the
        # prediction wrapper (needed to map summary → sequence → error)
        self._bg_seq = None
        self._bg_lengths = None

    @staticmethod
    def _summarize(X_seq: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Convert (N, T, F) sequences to (N, 2*F) mean/std summaries."""
        N, T, F = X_seq.shape
        summaries = np.zeros((N, 2 * F))
        for i in range(N):
            L = int(lengths[i]) if lengths is not None else T
            valid = X_seq[i, :L, :]
            summaries[i, :F] = valid.mean(axis=0)
            summaries[i, F:] = valid.std(axis=0)
        return summaries

    def _make_predict_fn(self, base_seq: np.ndarray, base_lengths: np.ndarray):
        """Return a function that maps summary features → reconstruction error.

        For each sample, we rebuild a synthetic sequence from the summary
        statistics (mean broadcast to each timestep) and pass it through
        the model.  This is an approximation, but it ensures the model
        receives inputs of the correct shape and the SHAP values reflect
        how shifting each feature's *distribution* affects error.

        NOTE: KernelSHAP can pass very large batches (thousands of rows)
        through this function.  We process in mini-batches on CPU to avoid
        GPU OOM errors.
        """
        F = base_seq.shape[2]
        T = base_seq.shape[1]
        med_len = int(np.median(base_lengths))
        BATCH_SIZE = 64  # small enough for any GPU/CPU

        # Run inference on CPU to avoid OOM — SHAP batches can be huge
        cpu_device = torch.device('cpu')
        cpu_model = self.model.cpu().eval()

        def predict_fn(summaries: np.ndarray) -> np.ndarray:
            N = summaries.shape[0]
            all_errors = []

            for start in range(0, N, BATCH_SIZE):
                end = min(start + BATCH_SIZE, N)
                batch_means = summaries[start:end, :F]
                batch_size = end - start

                seqs = np.tile(batch_means[:, np.newaxis, :], (1, T, 1))
                X_tensor = torch.FloatTensor(seqs)
                L_tensor = torch.LongTensor([med_len] * batch_size)

                with torch.no_grad():
                    reconstruction, _ = cpu_model(X_tensor, L_tensor)
                    mask = torch.zeros(batch_size, T, 1)
                    mask[:, :med_len, :] = 1.0
                    diff_sq = (X_tensor - reconstruction) ** 2 * mask
                    errors = diff_sq.sum(dim=(1, 2)) / (med_len * F)

                all_errors.append(errors.numpy())

            return np.concatenate(all_errors)

        return predict_fn

    def init_explainer(self, X_seq: np.ndarray, lengths: np.ndarray):
        """Initialize KernelSHAP on summary features.

        Uses shap.kmeans to compress background data, which dramatically
        speeds up KernelSHAP and avoids the large-background-data warning.
        """
        n_bg = self.config['explainability']['kernel_explainer_samples']
        if len(X_seq) > n_bg:
            idx = np.random.choice(len(X_seq), n_bg, replace=False)
            X_seq = X_seq[idx]
            lengths = lengths[idx]

        self._bg_seq = X_seq
        self._bg_lengths = lengths
        bg_summary = self._summarize(X_seq, lengths)

        predict_fn = self._make_predict_fn(X_seq, lengths)

        # Compress background to 50 cluster centroids for speed
        bg_compressed = shap.kmeans(bg_summary, min(50, len(bg_summary)))
        print(f"Initializing sequential SHAP explainer with {len(bg_summary)} "
              f"background samples compressed to 50 centroids "
              f"({len(self.feature_names)} summary features)...")
        self.explainer = shap.KernelExplainer(predict_fn, bg_compressed)
        print("Sequential SHAP explainer initialized")

    def explain_batch(self, X_seq: np.ndarray, lengths: np.ndarray,
                      max_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Explain a batch; returns (shap_values, summary_features)."""
        if len(X_seq) > max_samples:
            idx = np.random.choice(len(X_seq), max_samples, replace=False)
            X_seq = X_seq[idx]
            lengths = lengths[idx]
        summaries = self._summarize(X_seq, lengths)
        print(f"Computing sequential SHAP values for {len(summaries)} instances...")
        shap_values = self.explainer.shap_values(summaries)
        return np.array(shap_values), summaries

    def get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Mean |SHAP| per summary feature."""
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        importance = {f: float(s) for f, s in zip(self.feature_names, mean_abs)}
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def plot_feature_importance(self, importance: Dict[str, float],
                                save_path: str = None):
        """Bar chart of feature importance."""
        plt.figure(figsize=(10, 8))
        features = list(importance.keys())
        scores = list(importance.values())
        sns.barplot(x=scores, y=features, palette='viridis')
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature (mean/std summary)', fontsize=12)
        plt.title('Sequential Model: Feature Importance via SHAP',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        plt.close()

    def plot_summary(self, shap_values: np.ndarray, summaries: np.ndarray,
                     save_path: str = None):
        """SHAP beeswarm summary plot."""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, summaries,
                          feature_names=self.feature_names, show=False)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP summary plot to {save_path}")
        plt.close()


def generate_explanation_report(explainer, X_test: np.ndarray,
                                scores: np.ndarray, predictions: np.ndarray,
                                output_dir: str,
                                lengths: np.ndarray = None):
    """
    Generate comprehensive SHAP explanation report.

    Supports both SequentialSHAPExplainer and legacy SHAPExplainer.

    Args:
        explainer: Initialized SHAP explainer (either type)
        X_test: Test features (sequential or flat)
        scores: Anomaly scores
        predictions: Binary predictions
        output_dir: Directory to save outputs
        lengths: Sequence lengths (required for SequentialSHAPExplainer)
    """
    os.makedirs(output_dir, exist_ok=True)
    is_sequential = isinstance(explainer, SequentialSHAPExplainer)

    # 1. Explain batch of detected anomalies
    anomaly_indices = np.where(predictions == 1)[0]
    if len(anomaly_indices) > 50:
        anomaly_indices = np.random.choice(anomaly_indices, 50, replace=False)

    anomalies = X_test[anomaly_indices]

    if is_sequential:
        anom_lengths = lengths[anomaly_indices] if lengths is not None else None
        shap_values, summaries = explainer.explain_batch(
            anomalies, anom_lengths, max_samples=50
        )
    else:
        shap_values = explainer.explain_batch(anomalies, max_samples=50)
        summaries = anomalies

    # 2. Global feature importance
    importance = explainer.get_feature_importance(shap_values)
    explainer.plot_feature_importance(
        importance, save_path=os.path.join(output_dir, 'feature_importance.png')
    )

    # 3. SHAP summary plot
    explainer.plot_summary(
        shap_values, summaries,
        save_path=os.path.join(output_dir, 'shap_summary.png')
    )

    # 4. Explain top anomalies (only for legacy explainer which has the method)
    if hasattr(explainer, 'explain_top_anomalies'):
        explainer.explain_top_anomalies(
            X_test, scores, top_k=5,
            save_dir=os.path.join(output_dir, 'top_anomalies')
        )

    print(f"SHAP explanation report generated in {output_dir}")
