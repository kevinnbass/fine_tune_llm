"""
Unified Metrics Computation Interface.

This module provides a centralized system for all metric computations
across the platform with consistent interfaces and extensible framework.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from datetime import datetime
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    LOSS = "loss"
    CALIBRATION = "calibration"
    CONFIDENCE = "confidence"
    ENTROPY = "entropy"
    CUSTOM = "custom"


class MetricLevel(Enum):
    """Metric computation level."""
    SAMPLE = "sample"
    BATCH = "batch"
    EPOCH = "epoch"
    DATASET = "dataset"
    GLOBAL = "global"


class MetricAggregation(Enum):
    """Metric aggregation methods."""
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    STD = "std"
    LAST = "last"
    WEIGHTED_MEAN = "weighted_mean"


@dataclass
class MetricResult:
    """Result from metric computation."""
    name: str
    value: float
    type: MetricType
    level: MetricLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_interval: Optional[Tuple[float, float]] = None
    standard_error: Optional[float] = None


@dataclass
class MetricConfig:
    """Configuration for metric computation."""
    name: str
    type: MetricType
    enabled: bool = True
    compute_confidence: bool = False
    confidence_level: float = 0.95
    aggregation: MetricAggregation = MetricAggregation.MEAN
    window_size: Optional[int] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """
    Base class for all metrics.
    
    Provides common interface for metric computation.
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """Initialize base metric."""
        self.config = config or MetricConfig(
            name=self.__class__.__name__,
            type=MetricType.CUSTOM
        )
        self.reset()
    
    @abstractmethod
    def update(self, predictions: Any, targets: Any, **kwargs):
        """Update metric with new predictions and targets."""
        pass
    
    @abstractmethod
    def compute(self) -> MetricResult:
        """Compute final metric value."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.config.name}({self.config.type.value})"


class AccuracyMetric(BaseMetric):
    """Accuracy metric implementation."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """Initialize accuracy metric."""
        if config is None:
            config = MetricConfig(name="accuracy", type=MetricType.ACCURACY)
        super().__init__(config)
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, **kwargs):
        """Update accuracy counts."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
        if targets.ndim > 1:
            targets = np.argmax(targets, axis=-1)
        
        self.correct += np.sum(predictions == targets)
        self.total += len(predictions)
    
    def compute(self) -> MetricResult:
        """Compute accuracy."""
        if self.total == 0:
            value = 0.0
        else:
            value = self.correct / self.total
        
        result = MetricResult(
            name=self.config.name,
            value=value,
            type=self.config.type,
            level=MetricLevel.DATASET,
            metadata={
                'correct': self.correct,
                'total': self.total
            }
        )
        
        if self.config.compute_confidence and self.total > 0:
            # Wilson score interval for binomial proportion
            from scipy import stats
            z = stats.norm.ppf((1 + self.config.confidence_level) / 2)
            p = value
            n = self.total
            
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4*n**2)) / denominator
            
            result.confidence_interval = (center - margin, center + margin)
            result.standard_error = np.sqrt(p * (1 - p) / n)
        
        return result


class PrecisionRecallF1Metric(BaseMetric):
    """Combined precision, recall, and F1 metric."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """Initialize metric."""
        if config is None:
            config = MetricConfig(name="precision_recall_f1", type=MetricType.F1)
        super().__init__(config)
        self.num_classes = None
    
    def reset(self):
        """Reset metric state."""
        self.true_positives = None
        self.false_positives = None
        self.false_negatives = None
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, **kwargs):
        """Update confusion matrix counts."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
        if targets.ndim > 1:
            targets = np.argmax(targets, axis=-1)
        
        # Determine number of classes
        if self.num_classes is None:
            self.num_classes = max(predictions.max(), targets.max()) + 1
            self.true_positives = np.zeros(self.num_classes)
            self.false_positives = np.zeros(self.num_classes)
            self.false_negatives = np.zeros(self.num_classes)
        
        # Update counts for each class
        for c in range(self.num_classes):
            pred_c = predictions == c
            target_c = targets == c
            
            self.true_positives[c] += np.sum(pred_c & target_c)
            self.false_positives[c] += np.sum(pred_c & ~target_c)
            self.false_negatives[c] += np.sum(~pred_c & target_c)
    
    def compute(self) -> MetricResult:
        """Compute precision, recall, and F1."""
        if self.true_positives is None:
            return MetricResult(
                name=self.config.name,
                value=0.0,
                type=self.config.type,
                level=MetricLevel.DATASET
            )
        
        # Compute per-class metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
            recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Aggregate based on configuration
        if self.config.aggregation == MetricAggregation.MEAN:
            avg_precision = precision.mean()
            avg_recall = recall.mean()
            avg_f1 = f1.mean()
        elif self.config.aggregation == MetricAggregation.WEIGHTED_MEAN:
            support = self.true_positives + self.false_negatives
            total_support = support.sum()
            if total_support > 0:
                avg_precision = np.sum(precision * support) / total_support
                avg_recall = np.sum(recall * support) / total_support
                avg_f1 = np.sum(f1 * support) / total_support
            else:
                avg_precision = avg_recall = avg_f1 = 0.0
        else:
            avg_precision = precision.mean()
            avg_recall = recall.mean()
            avg_f1 = f1.mean()
        
        return MetricResult(
            name=self.config.name,
            value=avg_f1,
            type=self.config.type,
            level=MetricLevel.DATASET,
            metadata={
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'per_class_precision': precision.tolist(),
                'per_class_recall': recall.tolist(),
                'per_class_f1': f1.tolist()
            }
        )


class CalibrationMetric(BaseMetric):
    """Calibration metrics (ECE, MCE)."""
    
    def __init__(self, n_bins: int = 10, config: Optional[MetricConfig] = None):
        """Initialize calibration metric."""
        if config is None:
            config = MetricConfig(name="calibration", type=MetricType.CALIBRATION)
        super().__init__(config)
        self.n_bins = n_bins
    
    def reset(self):
        """Reset metric state."""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, **kwargs):
        """Store predictions and targets."""
        self.predictions.extend(predictions.tolist() if hasattr(predictions, 'tolist') else predictions)
        self.targets.extend(targets.tolist() if hasattr(targets, 'tolist') else targets)
    
    def compute(self) -> MetricResult:
        """Compute ECE and MCE."""
        if not self.predictions:
            return MetricResult(
                name=self.config.name,
                value=0.0,
                type=self.config.type,
                level=MetricLevel.DATASET
            )
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Get probability of predicted class
        if predictions.ndim > 1:
            confidences = np.max(predictions, axis=-1)
            pred_classes = np.argmax(predictions, axis=-1)
        else:
            confidences = predictions
            pred_classes = (predictions > 0.5).astype(int)
        
        if targets.ndim > 1:
            targets = np.argmax(targets, axis=-1)
        
        accuracies = (pred_classes == targets).astype(float)
        
        # Compute ECE and MCE
        ece = 0.0
        mce = 0.0
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_metrics = []
        
        for i in range(self.n_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if bin_mask.sum() > 0:
                bin_accuracy = accuracies[bin_mask].mean()
                bin_confidence = confidences[bin_mask].mean()
                bin_count = bin_mask.sum()
                
                bin_ece = np.abs(bin_accuracy - bin_confidence) * (bin_count / len(predictions))
                bin_mce = np.abs(bin_accuracy - bin_confidence)
                
                ece += bin_ece
                mce = max(mce, bin_mce)
                
                bin_metrics.append({
                    'bin': i,
                    'accuracy': bin_accuracy,
                    'confidence': bin_confidence,
                    'count': bin_count,
                    'ece_contribution': bin_ece
                })
        
        return MetricResult(
            name=self.config.name,
            value=ece,  # Primary value is ECE
            type=self.config.type,
            level=MetricLevel.DATASET,
            metadata={
                'ece': ece,
                'mce': mce,
                'n_bins': self.n_bins,
                'bin_metrics': bin_metrics
            }
        )


class EntropyMetric(BaseMetric):
    """Entropy-based uncertainty metric."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """Initialize entropy metric."""
        if config is None:
            config = MetricConfig(name="entropy", type=MetricType.ENTROPY)
        super().__init__(config)
    
    def reset(self):
        """Reset metric state."""
        self.entropies = []
    
    def update(self, predictions: np.ndarray, targets: Any = None, **kwargs):
        """Compute and store entropy values."""
        predictions = np.array(predictions)
        
        if predictions.ndim == 1:
            # Binary case
            p = predictions
            entropy = -p * np.log(p + 1e-10) - (1 - p) * np.log(1 - p + 1e-10)
        else:
            # Multi-class case
            entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=-1)
        
        self.entropies.extend(entropy.tolist() if hasattr(entropy, 'tolist') else [entropy])
    
    def compute(self) -> MetricResult:
        """Compute average entropy."""
        if not self.entropies:
            return MetricResult(
                name=self.config.name,
                value=0.0,
                type=self.config.type,
                level=MetricLevel.DATASET
            )
        
        entropies = np.array(self.entropies)
        
        return MetricResult(
            name=self.config.name,
            value=entropies.mean(),
            type=self.config.type,
            level=MetricLevel.DATASET,
            metadata={
                'mean': entropies.mean(),
                'std': entropies.std(),
                'min': entropies.min(),
                'max': entropies.max(),
                'median': np.median(entropies)
            }
        )


class UnifiedMetricsComputer:
    """
    Unified metrics computation system.
    
    Provides centralized metric computation with pluggable metrics,
    automatic aggregation, and comprehensive reporting.
    """
    
    def __init__(self):
        """Initialize unified metrics computer."""
        # Metric registry
        self.registered_metrics: Dict[str, Type[BaseMetric]] = {}
        self.active_metrics: Dict[str, BaseMetric] = {}
        
        # Metric history
        self.history: Dict[str, List[MetricResult]] = defaultdict(list)
        
        # Aggregation settings
        self.aggregation_windows: Dict[str, int] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register built-in metrics
        self._register_builtin_metrics()
        
        logger.info("Initialized UnifiedMetricsComputer")
    
    def _register_builtin_metrics(self):
        """Register built-in metric types."""
        self.register_metric("accuracy", AccuracyMetric)
        self.register_metric("precision_recall_f1", PrecisionRecallF1Metric)
        self.register_metric("calibration", CalibrationMetric)
        self.register_metric("entropy", EntropyMetric)
        
        logger.info(f"Registered {len(self.registered_metrics)} built-in metrics")
    
    def register_metric(self, name: str, metric_class: Type[BaseMetric]) -> bool:
        """
        Register new metric type.
        
        Args:
            name: Metric name
            metric_class: Metric class
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.registered_metrics[name] = metric_class
            
            logger.info(f"Registered metric: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register metric {name}: {e}")
            return False
    
    def activate_metric(self, 
                       name: str,
                       config: Optional[MetricConfig] = None,
                       **kwargs) -> bool:
        """
        Activate metric for computation.
        
        Args:
            name: Metric name
            config: Metric configuration
            **kwargs: Additional arguments for metric initialization
            
        Returns:
            True if activated successfully
        """
        try:
            with self._lock:
                if name not in self.registered_metrics:
                    logger.error(f"Metric not registered: {name}")
                    return False
                
                # Create metric instance
                metric_class = self.registered_metrics[name]
                
                if config is None:
                    config = MetricConfig(name=name, type=MetricType.CUSTOM)
                
                metric = metric_class(config=config, **kwargs)
                self.active_metrics[name] = metric
            
            logger.info(f"Activated metric: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate metric {name}: {e}")
            return False
    
    def deactivate_metric(self, name: str) -> bool:
        """
        Deactivate metric.
        
        Args:
            name: Metric name
            
        Returns:
            True if deactivated successfully
        """
        try:
            with self._lock:
                if name in self.active_metrics:
                    del self.active_metrics[name]
                    logger.info(f"Deactivated metric: {name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to deactivate metric {name}: {e}")
            return False
    
    def update(self, 
              predictions: Any,
              targets: Any,
              metrics: Optional[List[str]] = None,
              **kwargs):
        """
        Update metrics with new data.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metrics: Specific metrics to update (all if None)
            **kwargs: Additional metric-specific arguments
        """
        with self._lock:
            metrics_to_update = metrics or list(self.active_metrics.keys())
            
            for metric_name in metrics_to_update:
                if metric_name in self.active_metrics:
                    try:
                        metric = self.active_metrics[metric_name]
                        metric.update(predictions, targets, **kwargs)
                        
                    except Exception as e:
                        logger.error(f"Error updating metric {metric_name}: {e}")
    
    def compute(self, 
               metrics: Optional[List[str]] = None,
               reset_after: bool = True) -> Dict[str, MetricResult]:
        """
        Compute metric values.
        
        Args:
            metrics: Specific metrics to compute (all if None)
            reset_after: Whether to reset metrics after computation
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        with self._lock:
            metrics_to_compute = metrics or list(self.active_metrics.keys())
            
            for metric_name in metrics_to_compute:
                if metric_name in self.active_metrics:
                    try:
                        metric = self.active_metrics[metric_name]
                        result = metric.compute()
                        results[metric_name] = result
                        
                        # Store in history
                        self.history[metric_name].append(result)
                        
                        # Apply windowing if configured
                        if metric_name in self.aggregation_windows:
                            window_size = self.aggregation_windows[metric_name]
                            if len(self.history[metric_name]) > window_size:
                                self.history[metric_name] = self.history[metric_name][-window_size:]
                        
                        # Reset if requested
                        if reset_after:
                            metric.reset()
                        
                    except Exception as e:
                        logger.error(f"Error computing metric {metric_name}: {e}")
        
        return results
    
    def reset(self, metrics: Optional[List[str]] = None):
        """
        Reset metric states.
        
        Args:
            metrics: Specific metrics to reset (all if None)
        """
        with self._lock:
            metrics_to_reset = metrics or list(self.active_metrics.keys())
            
            for metric_name in metrics_to_reset:
                if metric_name in self.active_metrics:
                    self.active_metrics[metric_name].reset()
    
    def get_history(self, 
                   metric_name: str,
                   last_n: Optional[int] = None) -> List[MetricResult]:
        """
        Get metric history.
        
        Args:
            metric_name: Metric name
            last_n: Number of recent results to return
            
        Returns:
            List of metric results
        """
        with self._lock:
            history = self.history.get(metric_name, [])
            
            if last_n is not None:
                return history[-last_n:]
            return history.copy()
    
    def aggregate_history(self, 
                         metric_name: str,
                         aggregation: MetricAggregation = MetricAggregation.MEAN,
                         last_n: Optional[int] = None) -> Optional[float]:
        """
        Aggregate metric history.
        
        Args:
            metric_name: Metric name
            aggregation: Aggregation method
            last_n: Number of recent results to aggregate
            
        Returns:
            Aggregated value or None
        """
        history = self.get_history(metric_name, last_n)
        
        if not history:
            return None
        
        values = [result.value for result in history]
        
        if aggregation == MetricAggregation.MEAN:
            return np.mean(values)
        elif aggregation == MetricAggregation.SUM:
            return np.sum(values)
        elif aggregation == MetricAggregation.MAX:
            return np.max(values)
        elif aggregation == MetricAggregation.MIN:
            return np.min(values)
        elif aggregation == MetricAggregation.MEDIAN:
            return np.median(values)
        elif aggregation == MetricAggregation.STD:
            return np.std(values)
        elif aggregation == MetricAggregation.LAST:
            return values[-1] if values else None
        else:
            return np.mean(values)
    
    def clear_history(self, metrics: Optional[List[str]] = None):
        """
        Clear metric history.
        
        Args:
            metrics: Specific metrics to clear (all if None)
        """
        with self._lock:
            if metrics:
                for metric_name in metrics:
                    if metric_name in self.history:
                        self.history[metric_name].clear()
            else:
                self.history.clear()
    
    def set_aggregation_window(self, metric_name: str, window_size: int):
        """
        Set aggregation window for metric.
        
        Args:
            metric_name: Metric name
            window_size: Window size for history
        """
        with self._lock:
            self.aggregation_windows[metric_name] = window_size
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Returns:
            Summary dictionary
        """
        summary = {}
        
        with self._lock:
            for metric_name, metric in self.active_metrics.items():
                history = self.history.get(metric_name, [])
                
                if history:
                    latest = history[-1]
                    summary[metric_name] = {
                        'latest_value': latest.value,
                        'latest_timestamp': latest.timestamp,
                        'history_length': len(history),
                        'mean': self.aggregate_history(metric_name, MetricAggregation.MEAN),
                        'std': self.aggregate_history(metric_name, MetricAggregation.STD),
                        'min': self.aggregate_history(metric_name, MetricAggregation.MIN),
                        'max': self.aggregate_history(metric_name, MetricAggregation.MAX)
                    }
                    
                    if latest.metadata:
                        summary[metric_name]['metadata'] = latest.metadata
        
        return summary
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics data.
        
        Returns:
            Complete metrics export
        """
        export = {
            'timestamp': datetime.now().isoformat(),
            'active_metrics': list(self.active_metrics.keys()),
            'registered_metrics': list(self.registered_metrics.keys()),
            'summary': self.get_summary(),
            'history': {}
        }
        
        with self._lock:
            for metric_name, results in self.history.items():
                export['history'][metric_name] = [
                    {
                        'value': r.value,
                        'timestamp': r.timestamp.isoformat(),
                        'type': r.type.value,
                        'level': r.level.value,
                        'metadata': r.metadata
                    }
                    for r in results
                ]
        
        return export


# Global instance
_metrics_computer = None

def get_metrics_computer() -> UnifiedMetricsComputer:
    """Get global metrics computer instance."""
    global _metrics_computer
    if _metrics_computer is None:
        _metrics_computer = UnifiedMetricsComputer()
    return _metrics_computer


# Convenience functions

def compute_metrics(predictions: Any, 
                   targets: Any,
                   metrics: List[str],
                   **kwargs) -> Dict[str, MetricResult]:
    """
    Compute specified metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metrics: List of metric names
        **kwargs: Additional metric arguments
        
    Returns:
        Dictionary of metric results
    """
    computer = get_metrics_computer()
    
    # Activate metrics if not already active
    for metric_name in metrics:
        if metric_name not in computer.active_metrics:
            computer.activate_metric(metric_name)
    
    # Update and compute
    computer.update(predictions, targets, metrics=metrics, **kwargs)
    return computer.compute(metrics=metrics)


def register_custom_metric(name: str, metric_class: Type[BaseMetric]) -> bool:
    """
    Register custom metric globally.
    
    Args:
        name: Metric name
        metric_class: Metric class
        
    Returns:
        True if registered successfully
    """
    computer = get_metrics_computer()
    return computer.register_metric(name, metric_class)