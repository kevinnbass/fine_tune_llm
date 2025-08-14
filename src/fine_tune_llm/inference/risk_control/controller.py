"""
Risk-controlled prediction implementation.

This module provides risk-aware prediction capabilities with statistical
guarantees on error rates and cost-sensitive decision making.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import logging
from abc import ABC, abstractmethod

from ...core.interfaces import BaseComponent
from ...core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class BaseRiskController(BaseComponent, ABC):
    """Abstract base class for risk controllers."""
    
    def __init__(self, 
                 risk_level: float = 0.1,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base risk controller.
        
        Args:
            risk_level: Target risk level (probability of error)
            config: Configuration dictionary
        """
        self.risk_level = risk_level
        self.config = config or {}
        
        # Calibration data
        self.calibration_scores = None
        self.calibration_labels = None
        self.is_calibrated = False
        
        # Risk control threshold
        self.threshold = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.risk_level = self.config.get('risk_level', self.risk_level)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.calibration_scores = None
        self.calibration_labels = None
    
    @property
    def name(self) -> str:
        """Component name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    @abstractmethod
    def compute_risk_scores(self, 
                           probabilities: np.ndarray,
                           labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute risk scores for samples.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels (for calibration)
            
        Returns:
            Risk scores
        """
        pass
    
    def calibrate(self, 
                 probabilities: np.ndarray,
                 labels: np.ndarray):
        """
        Calibrate the risk controller.
        
        Args:
            probabilities: Calibration probabilities
            labels: Calibration labels
        """
        self.calibration_scores = probabilities
        self.calibration_labels = labels
        
        # Compute risk scores
        risk_scores = self.compute_risk_scores(probabilities, labels)
        
        # Find threshold that controls risk at desired level
        n = len(risk_scores)
        k = int(np.ceil((n + 1) * (1 - self.risk_level))) - 1
        self.threshold = np.sort(risk_scores)[k] if k < n else np.max(risk_scores)
        
        self.is_calibrated = True
        
        logger.info(f"Calibrated {self.name} with {n} samples, "
                   f"threshold: {self.threshold:.4f}")
    
    def predict(self, 
               probabilities: np.ndarray,
               cost_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make risk-controlled predictions.
        
        Args:
            probabilities: Test probabilities
            cost_matrix: Optional cost matrix for decision making
            
        Returns:
            Predictions (-1 for abstention, class index otherwise)
        """
        if not self.is_calibrated:
            raise ValueError("Controller must be calibrated before making predictions")
        
        # Compute risk scores for test data
        risk_scores = self.compute_risk_scores(probabilities)
        
        # Make predictions based on risk threshold
        predictions = []
        
        for i, (prob, risk) in enumerate(zip(probabilities, risk_scores)):
            if risk <= self.threshold:
                # Low risk - make prediction
                if cost_matrix is not None:
                    pred = self._cost_sensitive_prediction(prob, cost_matrix)
                else:
                    pred = np.argmax(prob)
            else:
                # High risk - abstain
                pred = -1
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _cost_sensitive_prediction(self, 
                                  probabilities: np.ndarray,
                                  cost_matrix: np.ndarray) -> int:
        """
        Make cost-sensitive prediction.
        
        Args:
            probabilities: Probability vector
            cost_matrix: Cost matrix (rows=true, cols=predicted)
            
        Returns:
            Predicted class that minimizes expected cost
        """
        # Expected cost for each prediction
        expected_costs = np.dot(probabilities, cost_matrix)
        return np.argmin(expected_costs)
    
    def compute_coverage(self, 
                        predictions: np.ndarray,
                        true_labels: np.ndarray) -> float:
        """
        Compute coverage (1 - abstention rate).
        
        Args:
            predictions: Predictions (-1 for abstention)
            true_labels: True labels
            
        Returns:
            Coverage rate
        """
        non_abstained = predictions != -1
        return np.mean(non_abstained)
    
    def compute_conditional_accuracy(self, 
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray) -> float:
        """
        Compute accuracy on non-abstained predictions.
        
        Args:
            predictions: Predictions (-1 for abstention)
            true_labels: True labels
            
        Returns:
            Conditional accuracy
        """
        non_abstained = predictions != -1
        if not np.any(non_abstained):
            return 0.0
        
        return np.mean(predictions[non_abstained] == true_labels[non_abstained])
    
    def compute_risk_metrics(self, 
                           predictions: np.ndarray,
                           true_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive risk metrics.
        
        Args:
            predictions: Predictions (-1 for abstention)
            true_labels: True labels
            
        Returns:
            Dictionary of risk metrics
        """
        coverage = self.compute_coverage(predictions, true_labels)
        conditional_accuracy = self.compute_conditional_accuracy(predictions, true_labels)
        
        # Overall accuracy (treating abstentions as errors)
        overall_accuracy = np.mean(predictions == true_labels)
        
        # Error rate on predictions made
        non_abstained = predictions != -1
        if np.any(non_abstained):
            error_rate = 1 - conditional_accuracy
        else:
            error_rate = 0.0
        
        return {
            'coverage': coverage,
            'conditional_accuracy': conditional_accuracy,
            'overall_accuracy': overall_accuracy,
            'error_rate': error_rate,
            'abstention_rate': 1 - coverage,
            'risk_level_target': self.risk_level,
            'risk_controlled': error_rate <= self.risk_level + 0.01  # Small tolerance
        }


class ConfidenceRiskController(BaseRiskController):
    """
    Risk controller based on prediction confidence.
    
    Uses 1 - max_probability as the risk score.
    """
    
    def compute_risk_scores(self, 
                           probabilities: np.ndarray,
                           labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute confidence-based risk scores."""
        # Risk is inversely related to confidence
        confidences = np.max(probabilities, axis=1)
        return 1 - confidences


class LossRiskController(BaseRiskController):
    """
    Risk controller based on expected loss.
    
    Uses the expected loss of the prediction as the risk score.
    """
    
    def __init__(self, 
                 risk_level: float = 0.1,
                 loss_function: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize loss-based risk controller.
        
        Args:
            risk_level: Target risk level
            loss_function: Loss function (default: 0-1 loss)
            config: Configuration dictionary
        """
        super().__init__(risk_level, config)
        self.loss_function = loss_function or self._zero_one_loss
    
    def _zero_one_loss(self, prediction: int, true_label: int) -> float:
        """0-1 loss function."""
        return 0.0 if prediction == true_label else 1.0
    
    def compute_risk_scores(self, 
                           probabilities: np.ndarray,
                           labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute loss-based risk scores."""
        if labels is None:
            # For test data, use expected loss of argmax prediction
            predictions = np.argmax(probabilities, axis=1)
            # Expected loss approximation
            return 1 - np.max(probabilities, axis=1)
        else:
            # For calibration, compute actual loss
            predictions = np.argmax(probabilities, axis=1)
            losses = []
            for pred, true_label in zip(predictions, labels):
                loss = self.loss_function(pred, true_label)
                losses.append(loss)
            return np.array(losses)


class EntropyRiskController(BaseRiskController):
    """
    Risk controller based on prediction entropy.
    
    Uses the entropy of the probability distribution as risk score.
    """
    
    def compute_risk_scores(self, 
                           probabilities: np.ndarray,
                           labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute entropy-based risk scores."""
        # Compute entropy
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        # Normalize to [0, 1] range
        max_entropy = np.log(probabilities.shape[1])
        normalized_entropies = entropies / max_entropy
        
        return normalized_entropies


class RiskControlledPredictor:
    """
    Main risk-controlled prediction interface.
    
    Provides unified access to different risk control methods with
    statistical guarantees and cost-sensitive decision making.
    """
    
    METHODS = {
        'confidence': ConfidenceRiskController,
        'loss': LossRiskController,
        'entropy': EntropyRiskController
    }
    
    def __init__(self, 
                 method: str = 'confidence',
                 risk_level: float = 0.1,
                 **kwargs):
        """
        Initialize risk-controlled predictor.
        
        Args:
            method: Risk control method ('confidence', 'loss', 'entropy')
            risk_level: Target risk level
            **kwargs: Method-specific parameters
        """
        if method not in self.METHODS:
            raise ConfigurationError(f"Unknown risk control method: {method}. "
                                   f"Available: {list(self.METHODS.keys())}")
        
        self.method = method
        self.controller = self.METHODS[method](risk_level=risk_level, **kwargs)
        
        # Cost matrix for cost-sensitive decisions
        self.default_cost_matrix = None
        
        logger.info(f"Initialized RiskControlledPredictor with method: {method}")
    
    def calibrate(self, 
                 probabilities: np.ndarray,
                 labels: np.ndarray):
        """Calibrate the risk controller."""
        self.controller.calibrate(probabilities, labels)
    
    def predict(self, 
               probabilities: np.ndarray,
               cost_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Make risk-controlled predictions."""
        cost_matrix = cost_matrix or self.default_cost_matrix
        return self.controller.predict(probabilities, cost_matrix)
    
    def predict_with_confidence(self, 
                              probabilities: np.ndarray,
                              return_risk_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with risk score information.
        
        Args:
            probabilities: Input probabilities
            return_risk_scores: Whether to return risk scores
            
        Returns:
            Predictions and optionally risk scores
        """
        risk_scores = self.controller.compute_risk_scores(probabilities)
        predictions = self.controller.predict(probabilities)
        
        if return_risk_scores:
            return predictions, risk_scores
        else:
            return predictions
    
    def set_cost_matrix(self, cost_matrix: np.ndarray):
        """Set default cost matrix for cost-sensitive decisions."""
        self.default_cost_matrix = cost_matrix
    
    def compute_metrics(self, 
                       predictions: np.ndarray,
                       true_labels: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive risk control metrics."""
        return self.controller.compute_risk_metrics(predictions, true_labels)
    
    def get_info(self) -> Dict[str, Any]:
        """Get risk controller information."""
        return {
            'method': self.method,
            'risk_level': self.controller.risk_level,
            'calibrated': self.controller.is_calibrated,
            'threshold': float(self.controller.threshold) if self.controller.threshold is not None else None,
            'has_cost_matrix': self.default_cost_matrix is not None
        }
    
    def set_risk_level(self, risk_level: float):
        """Update risk level and re-calibrate if possible."""
        self.controller.risk_level = risk_level
        
        # Re-calibrate if possible
        if (self.controller.is_calibrated and 
            self.controller.calibration_scores is not None and
            self.controller.calibration_labels is not None):
            self.controller.calibrate(
                self.controller.calibration_scores,
                self.controller.calibration_labels
            )
    
    def analyze_risk_coverage_tradeoff(self, 
                                     probabilities: np.ndarray,
                                     true_labels: np.ndarray,
                                     risk_levels: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """
        Analyze the risk-coverage tradeoff for different risk levels.
        
        Args:
            probabilities: Test probabilities
            true_labels: True labels
            risk_levels: List of risk levels to test
            
        Returns:
            Dictionary with risk levels, coverages, and accuracies
        """
        if risk_levels is None:
            risk_levels = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        original_risk_level = self.controller.risk_level
        
        results = {
            'risk_levels': risk_levels,
            'coverages': [],
            'conditional_accuracies': [],
            'overall_accuracies': []
        }
        
        for risk_level in risk_levels:
            # Update risk level
            self.set_risk_level(risk_level)
            
            # Make predictions
            predictions = self.predict(probabilities)
            
            # Compute metrics
            metrics = self.compute_metrics(predictions, true_labels)
            
            results['coverages'].append(metrics['coverage'])
            results['conditional_accuracies'].append(metrics['conditional_accuracy'])
            results['overall_accuracies'].append(metrics['overall_accuracy'])
        
        # Restore original risk level
        self.set_risk_level(original_risk_level)
        
        return results
    
    # Delegate other methods to the controller
    def __getattr__(self, name):
        """Forward undefined attributes to the controller."""
        return getattr(self.controller, name)