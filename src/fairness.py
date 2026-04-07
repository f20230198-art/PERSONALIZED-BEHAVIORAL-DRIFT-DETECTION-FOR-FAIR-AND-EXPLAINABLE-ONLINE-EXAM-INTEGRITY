"""
Fairness analysis and bias mitigation for drift detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class FairnessAnalyzer:
    """Analyze and mitigate fairness issues in anomaly detection."""
    
    def __init__(self, sensitive_attributes: List[str], config: Dict):
        self.sensitive_attributes = sensitive_attributes
        self.config = config
        self.group_thresholds = {}
    
    def compute_demographic_parity(self, predictions: np.ndarray, 
                                   demographics: List[Dict], 
                                   attribute: str) -> Dict[str, float]:
        """
        Compute Demographic Parity: P(Ŷ=1 | A=a) for each group a.
        
        Demographic parity is satisfied when positive prediction rate is
        similar across all demographic groups.
        
        Returns:
            positive_rates: Positive prediction rate per group
        """
        groups = defaultdict(list)
        
        # Group predictions by demographic attribute
        for i, demo in enumerate(demographics):
            if attribute in demo:
                group_value = demo[attribute]
                groups[group_value].append(predictions[i])
        
        # Compute positive rate per group
        positive_rates = {}
        for group, preds in groups.items():
            positive_rate = np.mean(preds)
            positive_rates[str(group)] = positive_rate
        
        return positive_rates
    
    def compute_demographic_parity_ratio(self, positive_rates: Dict[str, float]) -> float:
        """
        Compute Demographic Parity Ratio: min_rate / max_rate.
        
        Ratio closer to 1.0 indicates better fairness.
        Threshold: >= 0.80 is acceptable.
        """
        rates = list(positive_rates.values())
        if len(rates) < 2:
            return 1.0
        
        return min(rates) / max(rates) if max(rates) > 0 else 1.0
    
    def compute_equalized_odds(self, predictions: np.ndarray, 
                              ground_truth: np.ndarray,
                              demographics: List[Dict], 
                              attribute: str) -> Dict[str, Dict[str, float]]:
        """
        Compute Equalized Odds: TPR and FPR per demographic group.
        
        Equalized odds is satisfied when TPR and FPR are similar across groups.
        
        Returns:
            rates: {'group': {'tpr': x, 'fpr': y}}
        """
        groups = defaultdict(lambda: {'preds': [], 'labels': []})
        
        # Group by demographic attribute
        for i, demo in enumerate(demographics):
            if attribute in demo:
                group_value = demo[attribute]
                groups[group_value]['preds'].append(predictions[i])
                groups[group_value]['labels'].append(ground_truth[i])
        
        # Compute TPR and FPR per group
        rates = {}
        for group, data in groups.items():
            preds = np.array(data['preds'])
            labels = np.array(data['labels'])
            
            # True positives and false positives
            tp = np.sum((preds == 1) & (labels == 1))
            fn = np.sum((preds == 0) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            tn = np.sum((preds == 0) & (labels == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            rates[str(group)] = {'tpr': tpr, 'fpr': fpr}
        
        return rates
    
    def compute_equalized_odds_difference(self, eo_rates: Dict[str, Dict[str, float]]) -> float:
        """
        Compute max difference in TPR and FPR across groups.
        
        Lower difference indicates better fairness.
        Threshold: <= 0.10 is acceptable.
        """
        tprs = [r['tpr'] for r in eo_rates.values()]
        fprs = [r['fpr'] for r in eo_rates.values()]
        
        if len(tprs) < 2:
            return 0.0
        
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        
        # Return max difference
        return max(tpr_diff, fpr_diff)
    
    def analyze_fairness(self, predictions: np.ndarray, 
                        ground_truth: np.ndarray,
                        demographics: List[Dict]) -> Dict[str, Dict]:
        """
        Comprehensive fairness analysis across all sensitive attributes.
        
        Returns:
            fairness_metrics: Nested dictionary of fairness metrics per attribute
        """
        fairness_metrics = {}
        
        for attribute in self.sensitive_attributes:
            # Demographic Parity
            dp_rates = self.compute_demographic_parity(predictions, demographics, attribute)
            dp_ratio = self.compute_demographic_parity_ratio(dp_rates)
            
            # Equalized Odds
            eo_rates = self.compute_equalized_odds(predictions, ground_truth, demographics, attribute)
            eo_diff = self.compute_equalized_odds_difference(eo_rates)
            
            fairness_metrics[attribute] = {
                'demographic_parity': {
                    'positive_rates': dp_rates,
                    'ratio': dp_ratio,
                    'fair': dp_ratio >= self.config['fairness']['dp_threshold']
                },
                'equalized_odds': {
                    'rates': eo_rates,
                    'max_difference': eo_diff,
                    'fair': eo_diff <= self.config['fairness']['eo_threshold']
                }
            }
        
        return fairness_metrics
    
    def calibrate_thresholds(self, scores: np.ndarray, 
                            demographics: List[Dict],
                            global_threshold: float,
                            attribute: str) -> Dict[str, float]:
        """
        Calibrate detection thresholds per demographic group for fairness.
        
        Threshold adjustment:
            τ_g = τ + α * (D̄_g - D̄_global)
        
        where:
            τ_g: threshold for group g
            τ: global threshold
            α: adjustment strength (from config)
            D̄_g: mean drift score for group g
            D̄_global: global mean drift score
        
        Returns:
            group_thresholds: Adjusted threshold per group
        """
        alpha = self.config['fairness']['threshold_adjustment_alpha']
        
        # Compute global mean
        global_mean = np.mean(scores)
        
        # Group scores by demographic
        groups = defaultdict(list)
        for i, demo in enumerate(demographics):
            if attribute in demo:
                group_value = demo[attribute]
                groups[group_value].append(scores[i])
        
        # Compute adjusted thresholds
        group_thresholds = {}
        for group, group_scores in groups.items():
            group_mean = np.mean(group_scores)
            adjusted_threshold = global_threshold + alpha * (group_mean - global_mean)
            group_thresholds[str(group)] = adjusted_threshold
        
        return group_thresholds

    def alpha_grid_search(self, scores: np.ndarray, ground_truth: np.ndarray,
                          demographics: List[Dict], global_threshold: float,
                          attribute: str,
                          alpha_values: np.ndarray = None) -> Tuple[float, Dict[str, float]]:
        """Search for the optimal α that balances fairness and detection quality.

        Tries each α, applies group-specific thresholds, and selects the α
        that maximizes min(DPR, 1-EOD) subject to:
          - DPR ≥ dp_threshold (default 0.80)
          - EOD ≤ eo_threshold (default 0.10)

        If no α satisfies both constraints, returns the one that minimizes
        the combined violation.

        Args:
            scores: Anomaly scores for evaluation set
            ground_truth: True labels
            demographics: Demographic info per session
            global_threshold: Global detection threshold
            attribute: Sensitive attribute to calibrate
            alpha_values: Array of α values to try (default: 0.0 to 1.0 step 0.05)

        Returns:
            best_alpha: Optimal α value
            best_thresholds: Group thresholds at optimal α
        """
        if alpha_values is None:
            alpha_values = np.arange(0.0, 1.05, 0.05)

        dp_target = self.config['fairness']['dp_threshold']
        eo_target = self.config['fairness']['eo_threshold']

        best_alpha = 0.0
        best_score = -np.inf
        best_thresholds = {}

        original_alpha = self.config['fairness']['threshold_adjustment_alpha']

        for alpha in alpha_values:
            # Temporarily set alpha for calibrate_thresholds
            self.config['fairness']['threshold_adjustment_alpha'] = alpha
            group_thresholds = self.calibrate_thresholds(
                scores, demographics, global_threshold, attribute
            )
            fair_preds = self.apply_fair_predictions(
                scores, demographics, attribute, group_thresholds
            )

            # Compute fairness metrics
            dp_rates = self.compute_demographic_parity(fair_preds, demographics, attribute)
            dp_ratio = self.compute_demographic_parity_ratio(dp_rates)
            eo_rates = self.compute_equalized_odds(fair_preds, ground_truth, demographics, attribute)
            eo_diff = self.compute_equalized_odds_difference(eo_rates)

            # Composite score: higher is better
            # Reward meeting both constraints; penalize violations
            meets_dp = dp_ratio >= dp_target
            meets_eo = eo_diff <= eo_target
            fairness_score = min(dp_ratio, 1.0 - eo_diff)

            # Also consider detection quality (F1)
            from sklearn.metrics import f1_score as sk_f1
            f1 = sk_f1(ground_truth, fair_preds, zero_division=0)
            combined = 0.6 * fairness_score + 0.4 * f1  # balance fairness vs detection

            if combined > best_score:
                best_score = combined
                best_alpha = float(alpha)
                best_thresholds = group_thresholds

        # Restore original alpha
        self.config['fairness']['threshold_adjustment_alpha'] = original_alpha

        return best_alpha, best_thresholds
    
    def apply_fair_predictions(self, scores: np.ndarray,
                              demographics: List[Dict],
                              attribute: str,
                              group_thresholds: Dict[str, float]) -> np.ndarray:
        """
        Apply group-specific thresholds to make fair predictions.
        
        Args:
            scores: Anomaly scores
            demographics: List of demographic dictionaries
            attribute: Sensitive attribute to use
            group_thresholds: Threshold per group
        
        Returns:
            fair_predictions: Binary predictions with fairness calibration
        """
        predictions = np.zeros(len(scores), dtype=int)
        
        for i, demo in enumerate(demographics):
            if attribute in demo:
                group_value = str(demo[attribute])
                if group_value in group_thresholds:
                    threshold = group_thresholds[group_value]
                    predictions[i] = 1 if scores[i] > threshold else 0
        
        return predictions
    
    def print_fairness_report(self, fairness_metrics: Dict[str, Dict]):
        """Print comprehensive fairness report."""
        print("\n" + "="*80)
        print("FAIRNESS ANALYSIS REPORT")
        print("="*80)
        
        for attribute, metrics in fairness_metrics.items():
            print(f"\nSensitive Attribute: {attribute}")
            print("-" * 80)
            
            # Demographic Parity
            dp = metrics['demographic_parity']
            print(f"  Demographic Parity Ratio: {dp['ratio']:.4f}")
            print(f"  Fair: {dp['fair']} (threshold: {self.config['fairness']['dp_threshold']})")
            print(f"  Positive Rates by Group:")
            for group, rate in dp['positive_rates'].items():
                print(f"    {group}: {rate:.4f}")
            
            # Equalized Odds
            eo = metrics['equalized_odds']
            print(f"\n  Equalized Odds Max Difference: {eo['max_difference']:.4f}")
            print(f"  Fair: {eo['fair']} (threshold: {self.config['fairness']['eo_threshold']})")
            print(f"  TPR/FPR by Group:")
            for group, rates in eo['rates'].items():
                print(f"    {group}: TPR={rates['tpr']:.4f}, FPR={rates['fpr']:.4f}")
        
        print("="*80 + "\n")


def compare_fairness_before_after(fairness_before: Dict, fairness_after: Dict):
    """
    Compare fairness metrics before and after calibration.
    
    Args:
        fairness_before: Fairness metrics with global threshold
        fairness_after: Fairness metrics with calibrated thresholds
    """
    print("\n" + "="*80)
    print("FAIRNESS COMPARISON: Before vs After Calibration")
    print("="*80)
    
    for attribute in fairness_before.keys():
        print(f"\nAttribute: {attribute}")
        
        dp_before = fairness_before[attribute]['demographic_parity']['ratio']
        dp_after = fairness_after[attribute]['demographic_parity']['ratio']
        
        eo_before = fairness_before[attribute]['equalized_odds']['max_difference']
        eo_after = fairness_after[attribute]['equalized_odds']['max_difference']
        
        print(f"  Demographic Parity Ratio:")
        print(f"    Before: {dp_before:.4f}")
        print(f"    After:  {dp_after:.4f}")
        print(f"    Change: {dp_after - dp_before:+.4f}")
        
        print(f"  Equalized Odds Max Difference:")
        print(f"    Before: {eo_before:.4f}")
        print(f"    After:  {eo_after:.4f}")
        print(f"    Change: {eo_after - eo_before:+.4f}")
    
    print("="*80 + "\n")
