"""
Publication-quality visualization module for behavioral drift detection.

Generates all figures needed for the research paper:
1. ROC curves (all models overlay)
2. Precision-Recall curves (all models overlay)
3. Model comparison bar chart (F1, Precision, Recall, ROC-AUC)
4. Confusion matrices (grid of all models)
5. Training loss curves (LSTM-AE + Transformer)
6. Fairness comparison (before/after calibration)
7. Score distribution (normal vs anomalous per model)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from typing import Dict, List, Optional
import os

# ──────────────────────────────────────────────────────────────────────────
# Global style — clean, publication-ready
# ──────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colorblind-friendly palette
MODEL_COLORS = {
    'Plain-Transformer (Ours)': '#E63946',   # red — proposed model
    'Plain-LSTM': '#457B9D',                  # steel blue
    'LSTM-AE': '#2A9D8F',                     # teal
    'StandardAutoencoder': '#E9C46A',         # gold
    'IsolationForest': '#264653',             # dark teal
    'OneClassSVM': '#F4A261',                 # orange
    'RuleBased': '#8D99AE',                   # grey
}

MODEL_ORDER = [
    'Plain-Transformer (Ours)',
    'Plain-LSTM',
    'LSTM-AE',
    'StandardAutoencoder',
    'IsolationForest',
    'OneClassSVM',
    'RuleBased',
]


def _get_color(name):
    return MODEL_COLORS.get(name, '#333333')


def _ordered(results: dict) -> list:
    """Return model names sorted by MODEL_ORDER, then alphabetically."""
    ordered = [m for m in MODEL_ORDER if m in results]
    ordered += sorted(set(results) - set(ordered))
    return ordered


# ──────────────────────────────────────────────────────────────────────────
# 1. ROC Curves
# ──────────────────────────────────────────────────────────────────────────
def plot_roc_curves(y_test: np.ndarray, all_scores: Dict[str, np.ndarray],
                    save_path: str):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name in _ordered(all_scores):
        scores = all_scores[name]
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if '(Ours)' in name else 1.5
        ls = '-' if '(Ours)' in name else '--'
        ax.plot(fpr, tpr, color=_get_color(name), lw=lw, linestyle=ls,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved ROC curves → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 2. Precision-Recall Curves
# ──────────────────────────────────────────────────────────────────────────
def plot_pr_curves(y_test: np.ndarray, all_scores: Dict[str, np.ndarray],
                   save_path: str):
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    baseline_pr = np.mean(y_test)

    for name in _ordered(all_scores):
        scores = all_scores[name]
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        lw = 2.5 if '(Ours)' in name else 1.5
        ls = '-' if '(Ours)' in name else '--'
        ax.plot(recall, precision, color=_get_color(name), lw=lw, linestyle=ls,
                label=f'{name} (AP = {ap:.3f})')

    ax.axhline(baseline_pr, color='k', ls=':', lw=1, alpha=0.5,
               label=f'Random ({baseline_pr:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — All Models')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved PR curves → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 3. Model Comparison Bar Chart
# ──────────────────────────────────────────────────────────────────────────
def plot_model_comparison(all_results: Dict[str, Dict], save_path: str):
    """Grouped bar chart comparing F1, Precision, Recall, ROC-AUC."""
    metrics_to_plot = ['f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
    metric_labels = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']
    models = _ordered(all_results)
    n_models = len(models)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_models)
    width = 0.15
    offsets = np.arange(n_metrics) - (n_metrics - 1) / 2

    bar_colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [all_results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + offsets[i] * width, values, width, label=label,
                      color=bar_colors[i], edgecolor='white', linewidth=0.5)
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7,
                        rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend(loc='upper right', ncol=n_metrics, fontsize=8)
    ax.set_ylim([0, 1.15])

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved model comparison → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 4. Confusion Matrices (grid)
# ──────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(y_test: np.ndarray,
                            all_predictions: Dict[str, np.ndarray],
                            save_path: str):
    """Grid of confusion matrices for all models."""
    models = _ordered(all_predictions)
    n = len(models)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(models):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        preds = all_predictions[name]
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    cbar=False)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved confusion matrices → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 5. Training Loss Curves
# ──────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: Dict[str, list], model_name: str,
                         save_path: str):
    """Training and validation loss curves."""
    if not history:
        print(f"  Skipping training curves for {model_name} — no history available")
        return

    # Accept both key formats: Trainer uses 'train_losses'/'val_losses' (plural),
    # ClassifierTrainer uses 'train_loss'/'val_loss' (singular).
    train_key = 'train_loss' if 'train_loss' in history else 'train_losses'
    val_key = 'val_loss' if 'val_loss' in history else 'val_losses'

    if train_key not in history:
        print(f"  Skipping training curves for {model_name} — no history available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history[train_key]) + 1)

    ax.plot(epochs, history[train_key], color='#E63946', lw=2, label='Train Loss')
    if val_key in history:
        ax.plot(epochs, history[val_key], color='#457B9D', lw=2, label='Val Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Curves — {model_name}')
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved training curves → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 6. Score Distributions (normal vs anomalous)
# ──────────────────────────────────────────────────────────────────────────
def plot_score_distributions(y_test: np.ndarray,
                             all_scores: Dict[str, np.ndarray],
                             save_path: str):
    """Score distributions split by true label for each model."""
    models = _ordered(all_scores)
    n = len(models)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(models):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        scores = all_scores[name]
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]

        ax.hist(normal_scores, bins=50, alpha=0.6, color='#2A9D8F',
                label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color='#E63946',
                label='Anomaly', density=True)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle('Score Distributions — Normal vs Anomaly', fontsize=14,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved score distributions → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 7. Fairness Comparison (before vs after calibration)
# ──────────────────────────────────────────────────────────────────────────
def plot_fairness_comparison(fairness_results: Dict, save_path: str):
    """Bar chart showing fairness metrics before and after calibration."""
    before = fairness_results.get('before_calibration', {})
    after = fairness_results.get('after_calibration', {})

    if not before:
        print("  Skipping fairness plot — no results available")
        return

    attributes = list(before.keys())
    n_attrs = len(attributes)

    fig, axes = plt.subplots(1, n_attrs, figsize=(5 * n_attrs, 5))
    if n_attrs == 1:
        axes = [axes]

    for i, attr in enumerate(attributes):
        ax = axes[i]
        before_dp = float(before[attr]['demographic_parity']['ratio'])
        before_eo = float(before[attr]['equalized_odds']['max_difference'])
        after_dp = float(after[attr]['demographic_parity']['ratio']) if attr in after else before_dp
        after_eo = float(after[attr]['equalized_odds']['max_difference']) if attr in after else before_eo

        x = np.arange(2)
        width = 0.3
        bars1 = ax.bar(x - width/2, [before_dp, before_eo], width,
                       label='Before', color='#E63946', alpha=0.8)
        bars2 = ax.bar(x + width/2, [after_dp, after_eo], width,
                       label='After', color='#2A9D8F', alpha=0.8)

        # Value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(['DP Ratio\n(↑ better)', 'EO Diff\n(↓ better)'])
        ax.set_title(f'{attr.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylim([0, 1.2])
        ax.legend()

    fig.suptitle('Fairness Metrics — Before vs After Calibration',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved fairness comparison → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 8. Precision@K Comparison
# ──────────────────────────────────────────────────────────────────────────
def plot_precision_at_k(all_results: Dict[str, Dict], save_path: str):
    """Bar chart of Precision@10 and Precision@50 per model."""
    models = _ordered(all_results)
    p10 = [all_results[m].get('precision_at_10', 0) for m in models]
    p50 = [all_results[m].get('precision_at_50', 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, p10, width, label='P@10', color='#E63946', edgecolor='white')
    ax.bar(x + width/2, p50, width, label='P@50', color='#457B9D', edgecolor='white')

    for i in range(len(models)):
        ax.text(x[i] - width/2, p10[i] + 0.02, f'{p10[i]:.2f}',
                ha='center', va='bottom', fontsize=9)
        ax.text(x[i] + width/2, p50[i] + 0.02, f'{p50[i]:.2f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Precision')
    ax.set_title('Precision@K — Operational Metric')
    ax.legend()
    ax.set_ylim([0, 1.15])

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved Precision@K → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Master function — generate ALL figures
# ──────────────────────────────────────────────────────────────────────────
def generate_all_plots(processed_data: dict, config: dict):
    """Generate all paper figures from evaluation results."""
    plots_dir = config['output']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    y_test = processed_data['y_test']
    all_scores = processed_data.get('test_scores', {})
    all_results = processed_data.get('evaluation_results', {})
    all_predictions = processed_data.get('test_predictions', {})
    fairness_results = processed_data.get('fairness_results', {})

    if not all_scores:
        print("  ERROR: No test scores found. Run evaluate first.")
        return

    # 1. ROC Curves
    plot_roc_curves(y_test, all_scores,
                    os.path.join(plots_dir, 'roc_curves.png'))

    # 2. Precision-Recall Curves
    plot_pr_curves(y_test, all_scores,
                   os.path.join(plots_dir, 'pr_curves.png'))

    # 3. Model Comparison Bar Chart
    plot_model_comparison(all_results,
                          os.path.join(plots_dir, 'model_comparison.png'))

    # 4. Confusion Matrices
    if all_predictions:
        plot_confusion_matrices(y_test, all_predictions,
                                os.path.join(plots_dir, 'confusion_matrices.png'))

    # 5. Training Loss Curves
    lstm_history = processed_data.get('lstm_history', {})
    plot_training_curves(lstm_history, 'LSTM Autoencoder',
                         os.path.join(plots_dir, 'training_curves_lstm_ae.png'))

    # Transformer classifier training curves
    tf_history = processed_data.get('tf_classifier_history', {})
    plot_training_curves(tf_history, 'Plain Transformer Classifier',
                         os.path.join(plots_dir, 'training_curves_transformer.png'))

    # LSTM classifier training curves
    lstm_cls_history = processed_data.get('lstm_classifier_history', {})
    plot_training_curves(lstm_cls_history, 'Plain LSTM Classifier',
                         os.path.join(plots_dir, 'training_curves_lstm_cls.png'))

    # 6. Score Distributions
    plot_score_distributions(y_test, all_scores,
                             os.path.join(plots_dir, 'score_distributions.png'))

    # 7. Fairness Comparison
    if fairness_results:
        plot_fairness_comparison(fairness_results,
                                 os.path.join(plots_dir, 'fairness_comparison.png'))

    # 8. Precision@K
    plot_precision_at_k(all_results,
                        os.path.join(plots_dir, 'precision_at_k.png'))

    print("\n  All figures saved to: " + plots_dir)
    print("=" * 60)
