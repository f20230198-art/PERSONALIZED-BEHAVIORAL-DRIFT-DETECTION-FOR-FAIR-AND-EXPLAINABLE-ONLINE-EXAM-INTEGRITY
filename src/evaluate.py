"""Evaluation metrics and functions for anomaly detection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, Tuple


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray,
                           k: int = 10) -> float:
    """Compute Precision@K: fraction of true anomalies in the top-K scored sessions.

    This metric is operationally meaningful — instructors will only review
    the handful of highest-scoring sessions, so we want those sessions
    to actually be anomalous.

    Args:
        y_true: True labels (0 = normal, 1 = anomaly)
        y_scores: Anomaly scores (higher = more anomalous)
        k: Number of top-scoring sessions to consider

    Returns:
        Precision@K value in [0, 1]
    """
    if len(y_scores) == 0 or k <= 0:
        return 0.0
    k = min(k, len(y_scores))
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    return float(np.mean(y_true[top_k_indices]))



def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_scores: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics for anomaly detection.
    
    Args:
        y_true: True labels (0 = normal, 1 = anomaly)
        y_pred: Predicted labels (0 = normal, 1 = anomaly)
        y_scores: Anomaly scores (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to binary if needed
    y_pred_binary = (y_pred > 0).astype(int) if np.all(np.isin(y_pred, [-1, 1])) else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0)
    }
    
    # Add ROC-AUC and PR-AUC if scores are available
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except ValueError as e:
            # Typically: only one class present in y_true (e.g. edge-case split)
            print(f"  Warning: Could not compute AUC metrics: {e}")
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

        # Precision@K — operationally meaningful metric
        metrics['precision_at_10'] = compute_precision_at_k(y_true, y_scores, k=10)
        metrics['precision_at_50'] = compute_precision_at_k(y_true, y_scores, k=50)
    
    # Confusion matrix — handle edge cases where only one class is present
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # False positive rate and true positive rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def evaluate_model(model, X_test, y_test, threshold, device, model_type='lstm',
                   train_errors=None, lengths=None):
    """
    Evaluate a model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Anomaly detection threshold for THIS model
        device: PyTorch device
        model_type: 'lstm', 'standard_ae', or 'sklearn'
        train_errors: Training errors for z-score normalization (LSTM/AE only)
        lengths: Sequence lengths for LSTM models
    """
    import torch
    from src.train import compute_drift_scores

    if model_type == 'lstm':
        X_tensor = torch.FloatTensor(X_test).to(device)
        L_tensor = torch.LongTensor(lengths).to(device) if lengths is not None else None
        errors, scores = compute_drift_scores(model, X_tensor, device, train_errors, L_tensor)
        predictions = (scores > threshold).astype(int)

    elif model_type == 'standard_ae':
        X_tensor = torch.FloatTensor(X_test).to(device)
        errors, scores = compute_drift_scores(model, X_tensor, device, train_errors)
        predictions = (scores > threshold).astype(int)

    elif model_type == 'sklearn':
        scores = model.score_samples(X_test)
        predictions = (scores > threshold).astype(int)

    elif model_type == 'rule_based':
        from src.feature_extraction import BehavioralFeatureExtractor
        feature_names = BehavioralFeatureExtractor().feature_names
        pred_labels = model.predict(X_test, feature_names)
        scores = model.score_samples(X_test, feature_names)
        predictions = (pred_labels == -1).astype(int)

    metrics = compute_classification_metrics(y_test, predictions, scores)
    return metrics, scores, predictions


def select_optimal_threshold(y_val, val_scores, method='f1'):
    """
    Select optimal threshold for anomaly detection.

    Args:
        y_val: Validation labels
        val_scores: Validation anomaly scores
        method: 'f1_weighted' (default, class-imbalance aware), 'f1',
                'precision', 'recall', or 'percentile'

    Returns:
        optimal_threshold: Best threshold value
    """
    if method == 'percentile':
        threshold = np.percentile(val_scores, 95)

    else:
        # Grid search across the full score distribution
        thresholds = np.unique(np.percentile(val_scores, np.linspace(1, 99, 200)))
        best_score = 0
        best_threshold = thresholds[0]

        # Compute class weight for imbalance-aware optimization
        # Weight the anomaly class proportionally to its under-representation
        pos_rate = np.mean(y_val)
        anomaly_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)  # inverse frequency

        for thresh in thresholds:
            preds = (val_scores > thresh).astype(int)

            if method == 'f1_weighted':
                # Weighted F-beta — beta > 1 values recall more, but capped at
                # 1.5 to avoid the extreme recall-bias that F2 produces (which
                # tanks precision to ~0.30 and flags too many normals).
                prec = precision_score(y_val, preds, zero_division=0)
                rec = recall_score(y_val, preds, zero_division=0)
                beta = min(anomaly_weight, 1.5)  # cap at F1.5
                if prec + rec > 0:
                    score = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
                else:
                    score = 0.0
            elif method == 'f1':
                score = f1_score(y_val, preds, zero_division=0)
            elif method == 'precision':
                score = precision_score(y_val, preds, zero_division=0)
            elif method == 'recall':
                score = recall_score(y_val, preds, zero_division=0)
            else:
                score = f1_score(y_val, preds, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        threshold = best_threshold

    return threshold


def compare_models(results: Dict[str, Dict]) -> None:
    """
    Print comparison table of all models.
    
    Args:
        results: Dictionary mapping model names to their metrics
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Table header
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>9} {'PR-AUC':>9} {'P@10':>6} {'P@50':>6} {'Accuracy':>10}")
    print("-"*97)

    # Table rows
    for model_name, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        auc = metrics.get('roc_auc', 0)
        pr_auc = metrics.get('pr_auc', 0)
        p10 = metrics.get('precision_at_10', 0)
        p50 = metrics.get('precision_at_50', 0)

        print(f"{model_name:<25} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {auc:>9.4f} {pr_auc:>9.4f} {p10:>6.2f} {p50:>6.2f} {acc:>10.4f}")

    print("="*97 + "\n")


def compute_ablation_study(model_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Perform ablation study by training models with different feature subsets.
    
    This is a placeholder - in practice, you'd retrain models with:
    - All 6 features
    - Without response time features
    - Without answer change features
    - etc.
    
    Returns:
        ablation_results: Performance with different feature sets
    """
    ablation_results = {
        'all_features': model_results,
        'without_response_time': {},  # Would train without these features
        'without_answer_changes': {},
        'without_keystroke': {},
        'behavioral_only': {},  # Only behavioral, no timing
        'timing_only': {}  # Only timing, no behavioral
    }
    
    print("Note: Full ablation study requires retraining with feature subsets")
    
    return ablation_results
