"""
Metrics computation module for evaluation.

This module provides comprehensive metrics computation including
accuracy, precision, recall, F1, and specialized metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class MetricsComputer(BaseComponent):
    """Compute various evaluation metrics for model predictions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics computer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics_config = self.config.get('metrics', {})
        
        # Metric computation settings
        self.average = self.metrics_config.get('average', 'weighted')
        self.compute_confidence_metrics = self.metrics_config.get('confidence_metrics', True)
        self.compute_abstention_metrics = self.metrics_config.get('abstention_metrics', True)
        
        # Thresholds
        self.confidence_threshold = self.metrics_config.get('confidence_threshold', 0.5)
        self.abstention_threshold = self.metrics_config.get('abstention_threshold', 0.3)
        
        # Label information
        self.label_names = self.config.get('labels', [])
        self.n_classes = len(self.label_names) if self.label_names else 2
        
        # Results cache
        self.computed_metrics = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.metrics_config = self.config.get('metrics', {})
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.computed_metrics.clear()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "MetricsComputer"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def compute_all_metrics(self,
                           predictions: Union[List, np.ndarray],
                           labels: Union[List, np.ndarray],
                           probabilities: Optional[Union[List, np.ndarray]] = None,
                           sample_weights: Optional[Union[List, np.ndarray]] = None) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        if probabilities is not None:
            probabilities = np.array(probabilities)
        
        if sample_weights is not None:
            sample_weights = np.array(sample_weights)
        
        # Basic classification metrics
        basic_metrics = self.compute_classification_metrics(
            predictions, labels, sample_weights
        )
        metrics.update(basic_metrics)
        
        # Confusion matrix
        cm = self.compute_confusion_matrix(predictions, labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        per_class = self.compute_per_class_metrics(predictions, labels)
        metrics['per_class_metrics'] = per_class
        
        # Confidence-based metrics
        if probabilities is not None and self.compute_confidence_metrics:
            conf_metrics = self.compute_confidence_metrics_impl(
                predictions, labels, probabilities
            )
            metrics.update(conf_metrics)
        
        # Abstention metrics
        if self.compute_abstention_metrics:
            abst_metrics = self.compute_abstention_metrics_impl(
                predictions, labels, probabilities
            )
            metrics.update(abst_metrics)
        
        # Cache results
        self.computed_metrics = metrics
        
        return metrics
    
    def compute_classification_metrics(self,
                                      predictions: np.ndarray,
                                      labels: np.ndarray,
                                      sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            predictions: Predictions
            labels: Ground truth
            sample_weights: Optional weights
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            # Accuracy
            metrics['accuracy'] = accuracy_score(
                labels, predictions, sample_weight=sample_weights
            )
            
            # Precision, Recall, F1
            metrics['precision'] = precision_score(
                labels, predictions, 
                average=self.average,
                sample_weight=sample_weights,
                zero_division=0
            )
            
            metrics['recall'] = recall_score(
                labels, predictions,
                average=self.average,
                sample_weight=sample_weights,
                zero_division=0
            )
            
            metrics['f1_score'] = f1_score(
                labels, predictions,
                average=self.average,
                sample_weight=sample_weights,
                zero_division=0
            )
            
            # Macro and micro averages
            metrics['precision_macro'] = precision_score(
                labels, predictions, average='macro', zero_division=0
            )
            metrics['recall_macro'] = recall_score(
                labels, predictions, average='macro', zero_division=0
            )
            metrics['f1_macro'] = f1_score(
                labels, predictions, average='macro', zero_division=0
            )
            
            metrics['precision_micro'] = precision_score(
                labels, predictions, average='micro', zero_division=0
            )
            metrics['recall_micro'] = recall_score(
                labels, predictions, average='micro', zero_division=0
            )
            metrics['f1_micro'] = f1_score(
                labels, predictions, average='micro', zero_division=0
            )
            
        except Exception as e:
            logger.error(f"Error computing classification metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def compute_confusion_matrix(self,
                                predictions: np.ndarray,
                                labels: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predictions
            labels: Ground truth
            
        Returns:
            Confusion matrix
        """
        try:
            cm = confusion_matrix(labels, predictions)
            return cm
        except Exception as e:
            logger.error(f"Error computing confusion matrix: {e}")
            return np.zeros((self.n_classes, self.n_classes))
    
    def compute_per_class_metrics(self,
                                 predictions: np.ndarray,
                                 labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Args:
            predictions: Predictions
            labels: Ground truth
            
        Returns:
            Per-class metrics dictionary
        """
        per_class = {}
        
        try:
            # Get unique classes
            unique_classes = np.unique(np.concatenate([predictions, labels]))
            
            for class_idx in unique_classes:
                # Binary classification for this class
                binary_preds = (predictions == class_idx).astype(int)
                binary_labels = (labels == class_idx).astype(int)
                
                class_name = self.label_names[class_idx] if class_idx < len(self.label_names) else f"class_{class_idx}"
                
                per_class[class_name] = {
                    'precision': precision_score(binary_labels, binary_preds, zero_division=0),
                    'recall': recall_score(binary_labels, binary_preds, zero_division=0),
                    'f1_score': f1_score(binary_labels, binary_preds, zero_division=0),
                    'support': int(np.sum(labels == class_idx))
                }
                
        except Exception as e:
            logger.error(f"Error computing per-class metrics: {e}")
        
        return per_class
    
    def compute_confidence_metrics_impl(self,
                                       predictions: np.ndarray,
                                       labels: np.ndarray,
                                       probabilities: np.ndarray) -> Dict[str, float]:
        """
        Compute confidence-based metrics.
        
        Args:
            predictions: Predictions
            labels: Ground truth
            probabilities: Confidence scores
            
        Returns:
            Confidence metrics
        """
        metrics = {}
        
        try:
            # Average confidence
            metrics['avg_confidence'] = float(np.mean(probabilities))
            metrics['std_confidence'] = float(np.std(probabilities))
            
            # Confidence for correct/incorrect predictions
            correct_mask = predictions == labels
            incorrect_mask = ~correct_mask
            
            if np.any(correct_mask):
                metrics['avg_confidence_correct'] = float(np.mean(probabilities[correct_mask]))
            
            if np.any(incorrect_mask):
                metrics['avg_confidence_incorrect'] = float(np.mean(probabilities[incorrect_mask]))
            
            # High confidence accuracy
            high_conf_mask = probabilities >= self.confidence_threshold
            if np.any(high_conf_mask):
                metrics['high_confidence_accuracy'] = accuracy_score(
                    labels[high_conf_mask], predictions[high_conf_mask]
                )
                metrics['high_confidence_ratio'] = float(np.mean(high_conf_mask))
            
            # Low confidence accuracy
            low_conf_mask = probabilities < self.confidence_threshold
            if np.any(low_conf_mask):
                metrics['low_confidence_accuracy'] = accuracy_score(
                    labels[low_conf_mask], predictions[low_conf_mask]
                )
                metrics['low_confidence_ratio'] = float(np.mean(low_conf_mask))
            
        except Exception as e:
            logger.error(f"Error computing confidence metrics: {e}")
        
        return metrics
    
    def compute_abstention_metrics_impl(self,
                                       predictions: np.ndarray,
                                       labels: np.ndarray,
                                       probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute abstention-related metrics.
        
        Args:
            predictions: Predictions (may include abstention class)
            labels: Ground truth
            probabilities: Confidence scores
            
        Returns:
            Abstention metrics
        """
        metrics = {}
        
        try:
            # Check for abstention class (-1 or specific value)
            abstention_value = -1  # Convention for abstention
            abstained_mask = predictions == abstention_value
            
            metrics['abstention_rate'] = float(np.mean(abstained_mask))
            
            # Accuracy on non-abstained samples
            if np.any(~abstained_mask):
                metrics['accuracy_non_abstained'] = accuracy_score(
                    labels[~abstained_mask], predictions[~abstained_mask]
                )
            
            # Would-be accuracy if forced to predict
            if probabilities is not None and np.any(abstained_mask):
                # For abstained samples, what would accuracy be?
                # This is hypothetical based on confidence
                high_conf_abstained = probabilities[abstained_mask] >= self.confidence_threshold
                metrics['potential_accuracy_abstained'] = float(np.mean(high_conf_abstained))
            
            # Coverage (non-abstention rate)
            metrics['coverage'] = 1.0 - metrics['abstention_rate']
            
        except Exception as e:
            logger.error(f"Error computing abstention metrics: {e}")
        
        return metrics
    
    def compute_calibration_metrics(self,
                                   probabilities: np.ndarray,
                                   labels: np.ndarray,
                                   n_bins: int = 10) -> Dict[str, float]:
        """
        Compute calibration metrics (ECE, MCE).
        
        Args:
            probabilities: Predicted probabilities
            labels: Ground truth labels
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration metrics
        """
        from ..metrics import compute_ece, compute_mce
        
        metrics = {}
        
        try:
            # Expected Calibration Error
            metrics['ece'] = compute_ece(probabilities, labels, n_bins)
            
            # Maximum Calibration Error
            metrics['mce'] = compute_mce(probabilities, labels, n_bins)
            
        except Exception as e:
            logger.error(f"Error computing calibration metrics: {e}")
            metrics['ece'] = 0.0
            metrics['mce'] = 0.0
        
        return metrics
    
    def get_classification_report(self,
                                 predictions: np.ndarray,
                                 labels: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            predictions: Predictions
            labels: Ground truth
            
        Returns:
            Classification report string
        """
        try:
            report = classification_report(
                labels, predictions,
                target_names=self.label_names if self.label_names else None,
                zero_division=0
            )
            return report
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return f"Error generating report: {e}"