"""
Advanced metrics for LLM fine-tuning evaluation.
Restored from previous ensemble system for comprehensive evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well the predicted probabilities match the actual accuracy.
    Lower ECE indicates better calibration.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (lower is better, 0 is perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between accuracy and confidence across all bins.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        MCE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, error)
            
    return max_error


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score for probability predictions.
    
    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier score (lower is better, 0 is perfect)
    """
    return np.mean((y_prob - y_true) ** 2)


def compute_reliability_diagram_data(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute data for reliability (calibration) diagram.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin data for plotting
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        
        if i == n_bins - 1:  # Include upper boundary in last bin
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            bin_accuracies.append(y_true[in_bin].mean())
            bin_confidences.append(y_prob[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    return {
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts),
        'bin_centers': bin_centers,
        'ece': compute_ece(y_true, y_prob, n_bins),
        'mce': compute_mce(y_true, y_prob, n_bins)
    }


def compute_abstention_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abstentions: np.ndarray,
    costs: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute metrics for models with abstention capability.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        abstentions: Binary array indicating abstentions
        costs: Optional cost dictionary with keys:
            - 'misclassification': Cost of wrong prediction
            - 'abstention': Cost of abstaining
            - 'correct': Reward for correct prediction
            
    Returns:
        Dictionary of abstention-aware metrics
    """
    # Default costs
    if costs is None:
        costs = {
            'misclassification': 1.0,
            'abstention': 0.3,
            'correct': 0.0
        }
    
    n_samples = len(y_true)
    n_abstentions = abstentions.sum()
    n_predictions = n_samples - n_abstentions
    
    metrics = {
        'abstention_rate': n_abstentions / n_samples if n_samples > 0 else 0,
        'n_abstentions': int(n_abstentions),
        'n_predictions': int(n_predictions)
    }
    
    if n_predictions > 0:
        # Accuracy on non-abstained samples
        mask = ~abstentions.astype(bool)
        accuracy = (y_true[mask] == y_pred[mask]).mean()
        metrics['accuracy_on_predictions'] = accuracy
        
        # Compute costs
        total_cost = 0
        n_correct = 0
        n_incorrect = 0
        
        for i in range(n_samples):
            if abstentions[i]:
                total_cost += costs['abstention']
            elif y_true[i] == y_pred[i]:
                total_cost += costs['correct']
                n_correct += 1
            else:
                total_cost += costs['misclassification']
                n_incorrect += 1
        
        metrics['total_cost'] = total_cost
        metrics['avg_cost_per_sample'] = total_cost / n_samples
        metrics['n_correct'] = n_correct
        metrics['n_incorrect'] = n_incorrect
        
        # Effective accuracy (treating abstentions as incorrect)
        metrics['effective_accuracy'] = n_correct / n_samples if n_samples > 0 else 0
    else:
        # All samples abstained
        metrics['accuracy_on_predictions'] = 0
        metrics['total_cost'] = n_abstentions * costs['abstention']
        metrics['avg_cost_per_sample'] = costs['abstention']
        metrics['n_correct'] = 0
        metrics['n_incorrect'] = 0
        metrics['effective_accuracy'] = 0
    
    return metrics


def compute_risk_aware_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_matrix: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute risk-aware metrics for high-stakes classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        risk_matrix: Risk/cost matrix where element [i,j] is the cost
                    of predicting class j when true class is i
        class_names: Optional class names
        
    Returns:
        Dictionary of risk-aware metrics
    """
    if class_names is None:
        class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
    
    n_classes = len(class_names)
    
    # Default risk matrix (higher penalty for underestimating risk)
    if risk_matrix is None:
        risk_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                if i < j:  # Underestimating risk
                    risk_matrix[i, j] = (j - i) * 2.0
                elif i > j:  # Overestimating risk
                    risk_matrix[i, j] = (i - j) * 0.5
                # Correct prediction has 0 cost
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    # Compute total risk
    total_risk = np.sum(cm * risk_matrix)
    avg_risk = total_risk / len(y_true)
    
    # Per-class risks
    class_risks = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_pred = y_pred[class_mask]
            class_risk = sum(risk_matrix[i, j] for j in class_pred)
            class_risks[f'risk_{class_name}'] = class_risk / class_mask.sum()
    
    # Risk reduction compared to worst case
    worst_case_risk = np.max(risk_matrix) * len(y_true)
    risk_reduction = 1.0 - (total_risk / worst_case_risk)
    
    return {
        'total_risk': total_risk,
        'average_risk': avg_risk,
        'risk_reduction': risk_reduction,
        'confusion_matrix': cm.tolist(),
        **class_risks
    }


def compute_confidence_metrics(
    confidences: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute metrics related to model confidence.
    
    Args:
        confidences: Predicted confidence scores
        y_true: Optional true labels
        y_pred: Optional predicted labels
        
    Returns:
        Dictionary of confidence metrics
    """
    metrics = {
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences)),
        'median_confidence': float(np.median(confidences))
    }
    
    # Confidence quartiles
    quartiles = np.percentile(confidences, [25, 50, 75])
    metrics['confidence_q1'] = float(quartiles[0])
    metrics['confidence_q2'] = float(quartiles[1])
    metrics['confidence_q3'] = float(quartiles[2])
    
    # Confidence distribution
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(confidences, bins=bins)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        metrics[f'conf_{low:.1f}_{high:.1f}'] = int(hist[i])
    
    # If we have true labels and predictions
    if y_true is not None and y_pred is not None:
        correct = y_true == y_pred
        
        # Average confidence for correct vs incorrect
        if correct.sum() > 0:
            metrics['mean_confidence_correct'] = float(np.mean(confidences[correct]))
        if (~correct).sum() > 0:
            metrics['mean_confidence_incorrect'] = float(np.mean(confidences[~correct]))
        
        # Confidence-accuracy correlation
        if len(np.unique(correct)) > 1:
            try:
                from scipy.stats import pearsonr
                corr, p_value = pearsonr(confidences, correct.astype(float))
                metrics['confidence_accuracy_correlation'] = float(corr)
                metrics['confidence_accuracy_p_value'] = float(p_value)
            except (ImportError, Exception):
                # Fallback to simple correlation if scipy fails
                corr = np.corrcoef(confidences, correct.astype(float))[0, 1]
                metrics['confidence_accuracy_correlation'] = float(corr) if not np.isnan(corr) else 0.0
    
    return metrics


class MetricsAggregator:
    """Aggregate and track metrics over time."""
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize metrics aggregator.
        
        Args:
            save_path: Optional path to save metrics
        """
        self.metrics_history = []
        self.save_path = save_path
        
    def add_metrics(self, metrics: Dict[str, Any], epoch: Optional[int] = None):
        """Add metrics to history."""
        if epoch is not None:
            metrics['epoch'] = epoch
        
        import time
        metrics['timestamp'] = time.time()
        
        self.metrics_history.append(metrics)
        
        if self.save_path:
            self.save()
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Dict[str, Any]:
        """
        Get the best metrics based on a specific metric.
        
        Args:
            metric_name: Name of the metric to optimize
            mode: 'max' or 'min'
            
        Returns:
            Best metrics dictionary
        """
        if not self.metrics_history:
            return {}
        
        valid_metrics = [m for m in self.metrics_history if metric_name in m]
        if not valid_metrics:
            return {}
        
        if mode == 'max':
            return max(valid_metrics, key=lambda x: x[metric_name])
        else:
            return min(valid_metrics, key=lambda x: x[metric_name])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        
        # Collect all metric names
        all_keys = set()
        for metrics in self.metrics_history:
            all_keys.update(metrics.keys())
        
        # Compute statistics for numeric metrics
        for key in all_keys:
            if key in ['timestamp', 'epoch']:
                continue
                
            values = []
            for metrics in self.metrics_history:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(metrics[key])
            
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
        
        return summary
    
    def save(self, path: Optional[Path] = None):
        """Save metrics history to file."""
        save_path = path or self.save_path
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
    
    def load(self, path: Path):
        """Load metrics history from file."""
        with open(path, 'r') as f:
            self.metrics_history = json.load(f)