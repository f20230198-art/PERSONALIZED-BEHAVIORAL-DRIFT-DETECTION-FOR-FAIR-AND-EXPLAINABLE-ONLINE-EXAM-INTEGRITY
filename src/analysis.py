"""Reviewer-driven analyses: bootstrap CIs, calibration, per-family,
cold-start buckets, and FP/FN error inspection.

All functions operate on already-computed test scores and labels — no
retraining required. Designed to be called once after evaluate_models().
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_metric_ci(y_true: np.ndarray, y_scores: np.ndarray,
                        metric: str = "roc_auc",
                        n_iter: int = 1000, alpha: float = 0.05,
                        seed: int = 42,
                        threshold: float = None) -> Dict[str, float]:
    """Compute point estimate and 95% bootstrap CI for a metric.

    metric: one of {"roc_auc", "pr_auc", "f1", "precision", "recall"}.
    For threshold-based metrics, `threshold` must be provided.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        ys = y_scores[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            if metric == "roc_auc":
                stats.append(roc_auc_score(yt, ys))
            elif metric == "pr_auc":
                stats.append(average_precision_score(yt, ys))
            else:
                yp = (ys > threshold).astype(int)
                if metric == "f1":
                    stats.append(f1_score(yt, yp, zero_division=0))
                elif metric == "precision":
                    stats.append(precision_score(yt, yp, zero_division=0))
                elif metric == "recall":
                    stats.append(recall_score(yt, yp, zero_division=0))
        except Exception:
            continue

    stats = np.array(stats)
    if metric == "roc_auc":
        point = roc_auc_score(y_true, y_scores)
    elif metric == "pr_auc":
        point = average_precision_score(y_true, y_scores)
    else:
        yp = (y_scores > threshold).astype(int)
        if metric == "f1":
            point = f1_score(y_true, yp, zero_division=0)
        elif metric == "precision":
            point = precision_score(y_true, yp, zero_division=0)
        elif metric == "recall":
            point = recall_score(y_true, yp, zero_division=0)

    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return {"point": float(point), "ci_low": lo, "ci_high": hi,
            "n_iter": int(len(stats))}


def bootstrap_all_models(scores_by_model: Dict[str, np.ndarray],
                         predictions_by_model: Dict[str, np.ndarray],
                         y_test: np.ndarray,
                         thresholds: Dict[str, float] = None,
                         n_iter: int = 1000) -> Dict[str, Dict]:
    """Run bootstrap CIs for ROC-AUC, PR-AUC, F1 across all models."""
    out = {}
    for name, scores in scores_by_model.items():
        if scores is None:
            continue
        # Recover threshold from predictions if not given
        if thresholds and name in thresholds:
            t = thresholds[name]
        else:
            preds = predictions_by_model.get(name)
            if preds is not None and preds.sum() > 0:
                t = float(scores[preds == 1].min())
            else:
                t = float(np.median(scores))
        out[name] = {
            "roc_auc": bootstrap_metric_ci(y_test, scores, "roc_auc", n_iter=n_iter),
            "pr_auc":  bootstrap_metric_ci(y_test, scores, "pr_auc",  n_iter=n_iter),
            "f1":      bootstrap_metric_ci(y_test, scores, "f1",      n_iter=n_iter, threshold=t),
        }
    return out


# ---------------------------------------------------------------------------
# Calibration: reliability curve + Expected Calibration Error
# ---------------------------------------------------------------------------
def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, observed_freq, predicted_mean, counts)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    obs = np.zeros(n_bins)
    pred = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        cnt[b] = mask.sum()
        if cnt[b] > 0:
            obs[b] = y_true[mask].mean()
            pred[b] = y_prob[mask].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, obs, pred, cnt


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    centers, obs, pred, cnt = calibration_curve(y_true, y_prob, n_bins)
    n = max(int(cnt.sum()), 1)
    ece = 0.0
    for b in range(n_bins):
        if cnt[b] > 0:
            ece += (cnt[b] / n) * abs(obs[b] - pred[b])
    return float(ece)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     out_path: str, title: str = "Calibration (Reliability Diagram)",
                     n_bins: int = 10) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers, obs, pred, cnt = calibration_curve(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    mask = cnt > 0
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(pred[mask], obs[mask], "o-", label=f"Model (ECE={ece:.4f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ece


# ---------------------------------------------------------------------------
# Per-anomaly-family breakdown
# ---------------------------------------------------------------------------
def per_family_metrics(y_test: np.ndarray, scores: np.ndarray,
                       preds: np.ndarray, families: np.ndarray) -> Dict[str, Dict]:
    """For each family, compute detection rate (recall) on its members,
    plus the family's prevalence in the test set.

    Note: ROC-AUC requires both classes; we compute it by combining
    each family's anomalies vs. all 'normal' sessions.
    """
    families = np.asarray(families)
    out = {}
    normal_mask = (families == "normal")
    n_normal = int(normal_mask.sum())
    for fam in np.unique(families):
        if fam == "normal":
            continue
        fam_mask = (families == fam)
        n_fam = int(fam_mask.sum())
        if n_fam == 0:
            continue
        fam_recall = float(preds[fam_mask].mean())
        # Per-family AUC: family anomalies + all normals
        sub_idx = fam_mask | normal_mask
        sub_y = y_test[sub_idx]
        sub_s = scores[sub_idx]
        try:
            fam_auc = float(roc_auc_score(sub_y, sub_s)) if sub_y.sum() > 0 else 0.0
        except Exception:
            fam_auc = 0.0
        out[str(fam)] = {
            "n_anomalies": n_fam,
            "n_normals_compared": n_normal,
            "recall": fam_recall,
            "roc_auc_vs_normals": fam_auc,
        }
    return out


# ---------------------------------------------------------------------------
# Cold-start: bucket test sessions by # prior train sessions per student
# ---------------------------------------------------------------------------
def cold_start_buckets(y_test: np.ndarray, scores: np.ndarray, preds: np.ndarray,
                       sid_test: np.ndarray, sid_train: np.ndarray,
                       buckets=((0, 0), (1, 2), (3, 4), (5, 999))) -> Dict[str, Dict]:
    """Bucket test sessions by how many prior (train) sessions the student had.

    buckets: list of (low, high) inclusive ranges.
    """
    sid_test = np.asarray(sid_test)
    # Count prior sessions per student in train
    prior_counts: Dict = {}
    for sid in sid_train:
        prior_counts[sid] = prior_counts.get(sid, 0) + 1

    test_priors = np.array([prior_counts.get(s, 0) for s in sid_test])
    out = {}
    for lo, hi in buckets:
        mask = (test_priors >= lo) & (test_priors <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        yt = y_test[mask]; ys = scores[mask]; yp = preds[mask]
        try:
            auc = float(roc_auc_score(yt, ys)) if (yt.sum() > 0 and yt.sum() < len(yt)) else 0.0
        except Exception:
            auc = 0.0
        f1 = float(f1_score(yt, yp, zero_division=0))
        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        key = f"prior_{lo}-{hi}"
        out[key] = {"n_test": n, "n_anomalies": int(yt.sum()),
                    "roc_auc": auc, "f1": f1,
                    "precision": prec, "recall": rec}
    return out


# ---------------------------------------------------------------------------
# Error analysis: top FPs and FNs
# ---------------------------------------------------------------------------
def top_errors(y_test: np.ndarray, scores: np.ndarray, preds: np.ndarray,
               families: np.ndarray = None, sid_test: np.ndarray = None,
               k: int = 5) -> Dict[str, List[Dict]]:
    """Return top-k highest-scored false positives and lowest-scored false negatives."""
    families = np.asarray(families) if families is not None else None
    sid_test = np.asarray(sid_test) if sid_test is not None else None

    fp_mask = (preds == 1) & (y_test == 0)
    fn_mask = (preds == 0) & (y_test == 1)

    fp_idx = np.where(fp_mask)[0]
    fp_idx = fp_idx[np.argsort(scores[fp_idx])[::-1][:k]]
    fn_idx = np.where(fn_mask)[0]
    fn_idx = fn_idx[np.argsort(scores[fn_idx])[:k]]

    def _entry(i):
        e = {"index": int(i), "score": float(scores[i]), "label": int(y_test[i])}
        if families is not None: e["family"] = str(families[i])
        if sid_test is not None: e["student_id"] = str(sid_test[i])
        return e

    return {
        "top_false_positives": [_entry(i) for i in fp_idx],
        "top_false_negatives": [_entry(i) for i in fn_idx],
    }


# ---------------------------------------------------------------------------
# Convenience: run all analyses for the main (Transformer) model
# ---------------------------------------------------------------------------
def run_full_analysis(processed_data: Dict, config: Dict,
                      main_model_key: str = "Plain-Transformer (Ours)") -> Dict:
    """Run bootstrap, calibration, per-family, cold-start, and error analysis
    on the main model's saved test scores. Returns a single dict and writes
    JSON + plot artifacts to results/metrics/ and paper/figures/.
    """
    metrics_dir = config["output"]["metrics_dir"]
    plots_dir = config["output"]["plots_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    y_test = np.asarray(processed_data["y_test"])
    test_scores = processed_data.get("test_scores", {})
    test_preds = processed_data.get("test_predictions", {})
    families = np.asarray(processed_data.get("fam_test", np.array([])))
    sid_test = np.asarray(processed_data.get("sid_test", np.array([])))
    sid_train = np.asarray(processed_data.get("sid_train", np.array([])))

    report: Dict = {}

    # 1. Bootstrap CIs for all models
    print("\n[Analysis] Bootstrap 95% CIs (1000 iters)...")
    report["bootstrap_ci"] = bootstrap_all_models(test_scores, test_preds, y_test,
                                                  n_iter=1000)
    for m, cis in report["bootstrap_ci"].items():
        roc = cis["roc_auc"]
        print(f"  {m:<28} ROC-AUC = {roc['point']:.4f} "
              f"[{roc['ci_low']:.4f}, {roc['ci_high']:.4f}]")

    # 2. Calibration on the main supervised model (probabilities in [0,1])
    if main_model_key in test_scores:
        main_scores = np.asarray(test_scores[main_model_key])
        # Only score arrays that look like probabilities can be calibrated directly
        if main_scores.min() >= 0.0 and main_scores.max() <= 1.0:
            print(f"\n[Analysis] Calibration curve for {main_model_key}...")
            cal_path = os.path.join(plots_dir, "calibration_curve.png")
            ece = plot_calibration(y_test, main_scores, cal_path,
                                   title=f"Calibration: {main_model_key}")
            report["calibration"] = {"model": main_model_key, "ece": ece,
                                     "plot": cal_path}
            print(f"  ECE = {ece:.4f}  ->  {cal_path}")

    # 3. Per-family breakdown
    if main_model_key in test_scores and families.size:
        print(f"\n[Analysis] Per-family breakdown for {main_model_key}...")
        report["per_family"] = per_family_metrics(
            y_test, np.asarray(test_scores[main_model_key]),
            np.asarray(test_preds[main_model_key]), families
        )
        for fam, m in report["per_family"].items():
            print(f"  {fam:<25} n={m['n_anomalies']:<5} "
                  f"recall={m['recall']:.3f}  AUC={m['roc_auc_vs_normals']:.3f}")

    # 4. Cold-start buckets
    if main_model_key in test_scores and sid_test.size and sid_train.size:
        print(f"\n[Analysis] Cold-start buckets for {main_model_key}...")
        report["cold_start"] = cold_start_buckets(
            y_test, np.asarray(test_scores[main_model_key]),
            np.asarray(test_preds[main_model_key]),
            sid_test, sid_train
        )
        for bucket, m in report["cold_start"].items():
            print(f"  {bucket:<15} n={m['n_test']:<5} "
                  f"AUC={m['roc_auc']:.3f} F1={m['f1']:.3f}")

    # 5. Top FP/FN error analysis
    if main_model_key in test_scores:
        print(f"\n[Analysis] Top FP/FN inspection for {main_model_key}...")
        report["error_analysis"] = top_errors(
            y_test, np.asarray(test_scores[main_model_key]),
            np.asarray(test_preds[main_model_key]),
            families if families.size else None,
            sid_test if sid_test.size else None,
            k=5
        )

    # Save the full report
    out_path = os.path.join(metrics_dir, "analysis_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Analysis] Full report saved to {out_path}")
    return report
