"""Evaluation metrics for the ensemble system."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..voters.classical.calibrate import compute_ece

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Comprehensive evaluation for the ensemble system."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        self.metrics_history = []
        
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate predictions with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            sample_weights: Sample weights (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, 
            average=None,
            sample_weight=sample_weights,
            labels=range(len(self.class_names))
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = int(support[i])
        
        # Aggregate metrics
        metrics['f1_macro'] = f1.mean()
        metrics['f1_weighted'] = np.average(f1, weights=support)
        metrics['precision_macro'] = precision.mean()
        metrics['recall_macro'] = recall.mean()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Probability-based metrics if available
        if y_prob is not None:
            # ECE (Expected Calibration Error)
            if len(y_prob.shape) == 1:
                metrics['ece'] = compute_ece(y_true, y_prob)
            else:
                # Multiclass ECE
                max_probs = y_prob.max(axis=1)
                metrics['ece'] = compute_ece(y_true, max_probs)
            
            # AUC-ROC for each class
            if len(self.class_names) == 2:
                # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multiclass - one-vs-rest AUC
                for i, class_name in enumerate(self.class_names):
                    y_binary = (y_true == i).astype(int)
                    if len(np.unique(y_binary)) > 1:  # Need both classes
                        metrics[f'auc_{class_name}'] = roc_auc_score(
                            y_binary, y_prob[:, i]
                        )
            
            # Mean confidence
            metrics['mean_confidence'] = y_prob.max(axis=1).mean()
            metrics['std_confidence'] = y_prob.max(axis=1).std()
        
        return metrics
    
    def evaluate_with_slices(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        slice_membership: Dict[str, np.ndarray],
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate with slice-level metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            slice_membership: Dict mapping slice names to boolean masks
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with overall and per-slice metrics
        """
        results = {}
        
        # Overall metrics
        results['overall'] = self.evaluate_predictions(y_true, y_pred, y_prob)
        results['overall']['n_samples'] = len(y_true)
        
        # Per-slice metrics
        for slice_name, mask in slice_membership.items():
            if mask.sum() > 0:
                slice_y_true = y_true[mask]
                slice_y_pred = y_pred[mask]
                slice_y_prob = y_prob[mask] if y_prob is not None else None
                
                slice_metrics = self.evaluate_predictions(
                    slice_y_true, slice_y_pred, slice_y_prob
                )
                slice_metrics['n_samples'] = mask.sum()
                slice_metrics['fraction_of_data'] = mask.mean()
                
                results[f'slice_{slice_name}'] = slice_metrics
        
        # Find worst slice
        worst_slice = None
        worst_f1 = 1.0
        
        for key, metrics in results.items():
            if key.startswith('slice_'):
                if metrics['f1_weighted'] < worst_f1:
                    worst_f1 = metrics['f1_weighted']
                    worst_slice = key.replace('slice_', '')
        
        results['worst_slice'] = {
            'name': worst_slice,
            'f1': worst_f1,
            'gap_from_overall': results['overall']['f1_weighted'] - worst_f1
        }
        
        return results
    
    def evaluate_cascade(
        self,
        predictions: List[Dict[str, Any]],
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate cascade-specific metrics.
        
        Args:
            predictions: List of prediction dictionaries with tier info
            y_true: True labels
            
        Returns:
            Cascade evaluation metrics
        """
        metrics = {}
        
        # Extract information
        tiers = []
        abstentions = []
        costs = []
        latencies = []
        llm_calls = []
        
        for pred in predictions:
            tiers.append(pred.get('tier', -1))
            abstentions.append(pred.get('abstain', False))
            costs.append(pred.get('cost_cents', 0))
            latencies.append(pred.get('latency_ms', 0))
            llm_calls.append('llm' in pred.get('model_id', '').lower())
        
        # Tier distribution
        tier_counts = pd.Series(tiers).value_counts()
        for tier in [0, 1, 2]:
            metrics[f'tier_{tier}_fraction'] = tier_counts.get(tier, 0) / len(tiers)
        
        # Abstention metrics
        metrics['abstention_rate'] = np.mean(abstentions)
        metrics['coverage'] = 1 - metrics['abstention_rate']
        
        # Cost metrics
        metrics['mean_cost_cents'] = np.mean(costs)
        metrics['total_cost_cents'] = np.sum(costs)
        metrics['cost_per_1k'] = metrics['mean_cost_cents'] * 10  # Convert to per 1K
        
        # Latency metrics
        metrics['mean_latency_ms'] = np.mean(latencies)
        metrics['p50_latency_ms'] = np.percentile(latencies, 50)
        metrics['p95_latency_ms'] = np.percentile(latencies, 95)
        metrics['p99_latency_ms'] = np.percentile(latencies, 99)
        
        # LLM usage
        metrics['llm_call_rate'] = np.mean(llm_calls)
        
        # Accuracy on non-abstained samples
        non_abstained_idx = [i for i, a in enumerate(abstentions) if not a]
        if non_abstained_idx:
            y_pred_covered = []
            y_true_covered = []
            
            for idx in non_abstained_idx:
                pred = predictions[idx]
                if 'decision' in pred:
                    # Convert decision to numeric
                    decision = pred['decision']
                    if decision in self.class_names:
                        y_pred_covered.append(self.class_names.index(decision))
                        y_true_covered.append(y_true[idx])
            
            if y_pred_covered:
                metrics['accuracy_on_covered'] = accuracy_score(y_true_covered, y_pred_covered)
                metrics['f1_on_covered'] = f1_score(
                    y_true_covered, y_pred_covered, average='weighted'
                )
        
        return metrics
    
    def evaluate_voters(
        self,
        voter_outputs_list: List[Dict[str, Dict]],
        y_true: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual voter performance.
        
        Args:
            voter_outputs_list: List of voter outputs for each sample
            y_true: True labels
            
        Returns:
            Per-voter metrics
        """
        # Collect predictions per voter
        voter_predictions = {}
        voter_abstentions = {}
        voter_costs = []
        voter_latencies = []
        
        for voter_outputs in voter_outputs_list:
            for voter_id, output in voter_outputs.items():
                if voter_id not in voter_predictions:
                    voter_predictions[voter_id] = []
                    voter_abstentions[voter_id] = []
                    voter_costs.append((voter_id, []))
                    voter_latencies.append((voter_id, []))
                
                if output.get('abstain'):
                    voter_predictions[voter_id].append(-1)  # Abstention marker
                    voter_abstentions[voter_id].append(True)
                else:
                    probs = output.get('probs', {})
                    if probs:
                        pred = max(probs, key=probs.get)
                        pred_idx = self.class_names.index(pred) if pred in self.class_names else -1
                        voter_predictions[voter_id].append(pred_idx)
                    else:
                        voter_predictions[voter_id].append(-1)
                    voter_abstentions[voter_id].append(False)
                
                # Track costs and latencies
                for i, (vid, costs) in enumerate(voter_costs):
                    if vid == voter_id:
                        voter_costs[i][1].append(output.get('cost_cents', 0))
                        break
                
                for i, (vid, lats) in enumerate(voter_latencies):
                    if vid == voter_id:
                        voter_latencies[i][1].append(output.get('latency_ms', 0))
                        break
        
        # Compute metrics per voter
        voter_metrics = {}
        
        for voter_id, predictions in voter_predictions.items():
            # Filter non-abstained predictions
            valid_idx = [i for i, p in enumerate(predictions) if p != -1]
            
            if valid_idx:
                y_pred_valid = [predictions[i] for i in valid_idx]
                y_true_valid = [y_true[i] for i in valid_idx]
                
                metrics = {
                    'accuracy': accuracy_score(y_true_valid, y_pred_valid),
                    'f1_weighted': f1_score(y_true_valid, y_pred_valid, average='weighted'),
                    'coverage': len(valid_idx) / len(predictions),
                    'abstention_rate': np.mean(voter_abstentions[voter_id])
                }
            else:
                metrics = {
                    'accuracy': 0.0,
                    'f1_weighted': 0.0,
                    'coverage': 0.0,
                    'abstention_rate': 1.0
                }
            
            # Add cost and latency
            for vid, costs in voter_costs:
                if vid == voter_id and costs:
                    metrics['mean_cost_cents'] = np.mean(costs)
                    break
            
            for vid, lats in voter_latencies:
                if vid == voter_id and lats:
                    metrics['mean_latency_ms'] = np.mean(lats)
                    metrics['p95_latency_ms'] = np.percentile(lats, 95)
                    break
            
            voter_metrics[voter_id] = metrics
        
        return voter_metrics
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_format: str = 'markdown'
    ) -> str:
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results dictionary
            output_format: 'markdown' or 'html'
            
        Returns:
            Formatted report string
        """
        if output_format == 'markdown':
            return self._generate_markdown_report(results)
        elif output_format == 'html':
            return self._generate_html_report(results)
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report."""
        lines = ["# Ensemble Evaluation Report\n"]
        
        # Overall metrics
        if 'overall' in results:
            lines.append("## Overall Performance")
            overall = results['overall']
            lines.append(f"- **Accuracy**: {overall.get('accuracy', 0):.3f}")
            lines.append(f"- **F1 (weighted)**: {overall.get('f1_weighted', 0):.3f}")
            lines.append(f"- **F1 (macro)**: {overall.get('f1_macro', 0):.3f}")
            
            if 'ece' in overall:
                lines.append(f"- **ECE**: {overall['ece']:.4f}")
            
            lines.append("")
        
        # Cascade metrics
        if 'cascade' in results:
            lines.append("## Cascade Performance")
            cascade = results['cascade']
            lines.append(f"- **Coverage**: {cascade.get('coverage', 0):.1%}")
            lines.append(f"- **Abstention Rate**: {cascade.get('abstention_rate', 0):.1%}")
            lines.append(f"- **LLM Call Rate**: {cascade.get('llm_call_rate', 0):.1%}")
            lines.append(f"- **Mean Cost (cents)**: {cascade.get('mean_cost_cents', 0):.3f}")
            lines.append(f"- **P95 Latency (ms)**: {cascade.get('p95_latency_ms', 0):.1f}")
            lines.append("")
        
        # Per-class performance
        if 'overall' in results:
            lines.append("## Per-Class Metrics")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|----|---------| ")
            
            for class_name in self.class_names:
                prec = results['overall'].get(f'precision_{class_name}', 0)
                rec = results['overall'].get(f'recall_{class_name}', 0)
                f1 = results['overall'].get(f'f1_{class_name}', 0)
                supp = results['overall'].get(f'support_{class_name}', 0)
                lines.append(f"| {class_name} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {supp} |")
            
            lines.append("")
        
        # Slice performance
        slice_results = {k: v for k, v in results.items() if k.startswith('slice_')}
        if slice_results:
            lines.append("## Slice Performance")
            lines.append("| Slice | F1 | Accuracy | Samples |")
            lines.append("|-------|----|-----------|---------| ")
            
            for slice_key, metrics in slice_results.items():
                slice_name = slice_key.replace('slice_', '')
                f1 = metrics.get('f1_weighted', 0)
                acc = metrics.get('accuracy', 0)
                n = metrics.get('n_samples', 0)
                lines.append(f"| {slice_name} | {f1:.3f} | {acc:.3f} | {n} |")
            
            lines.append("")
            
            if 'worst_slice' in results:
                worst = results['worst_slice']
                lines.append(f"**Worst Slice**: {worst['name']} (F1: {worst['f1']:.3f})")
                lines.append("")
        
        # Voter performance
        if 'voters' in results:
            lines.append("## Individual Voter Performance")
            lines.append("| Voter | Accuracy | F1 | Coverage | Cost |")
            lines.append("|-------|----------|-------|----------|------|")
            
            for voter_id, metrics in results['voters'].items():
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_weighted', 0)
                cov = metrics.get('coverage', 0)
                cost = metrics.get('mean_cost_cents', 0)
                lines.append(f"| {voter_id} | {acc:.3f} | {f1:.3f} | {cov:.1%} | {cost:.4f} |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report."""
        # Convert markdown to HTML (simplified)
        markdown_report = self._generate_markdown_report(results)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ensemble Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
            </style>
        </head>
        <body>
            <pre>{markdown_report}</pre>
        </body>
        </html>
        """
        
        return html