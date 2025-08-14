"""
Conformal prediction implementation for uncertainty quantification.

This module provides conformal prediction methods that offer statistical
guarantees on prediction sets and coverage rates.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from abc import ABC, abstractmethod

from ...core.interfaces import BaseComponent
from ...core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class BaseConformalPredictor(BaseComponent, ABC):
    """Abstract base class for conformal predictors."""
    
    def __init__(self, 
                 alpha: float = 0.1,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha is the target coverage)
            config: Configuration dictionary
        """
        self.alpha = alpha
        self.config = config or {}
        
        # Calibration data
        self.calibration_scores = None
        self.calibration_labels = None
        self.is_calibrated = False
        
        # Quantiles for prediction sets
        self.quantile = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.alpha = self.config.get('alpha', self.alpha)
    
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
    def compute_conformity_scores(self, 
                                 probabilities: np.ndarray,
                                 labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute conformity scores.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels (for calibration)
            
        Returns:
            Conformity scores
        """
        pass
    
    def calibrate(self, 
                 calibration_scores: np.ndarray,
                 calibration_labels: Optional[np.ndarray] = None):
        """
        Calibrate the conformal predictor.
        
        Args:
            calibration_scores: Scores from calibration data
            calibration_labels: Labels from calibration data
        """
        self.calibration_scores = np.array(calibration_scores)
        if calibration_labels is not None:
            self.calibration_labels = np.array(calibration_labels)
        
        # Compute conformity scores
        conformity_scores = self.compute_conformity_scores(
            self.calibration_scores, self.calibration_labels
        )
        
        # Compute quantile
        n = len(conformity_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(conformity_scores, level)
        
        self.is_calibrated = True
        
        logger.info(f"Calibrated {self.name} with {n} samples, "
                   f"quantile: {self.quantile:.4f}")
    
    def predict(self, 
               test_probabilities: np.ndarray) -> List[List[int]]:
        """
        Generate prediction sets.
        
        Args:
            test_probabilities: Test set probabilities
            
        Returns:
            List of prediction sets (lists of class indices)
        """
        if not self.is_calibrated:
            raise ValueError("Predictor must be calibrated before making predictions")
        
        # Compute conformity scores for test data
        test_scores = self.compute_conformity_scores(test_probabilities)
        
        # Generate prediction sets
        prediction_sets = []
        n_classes = test_probabilities.shape[1] if test_probabilities.ndim > 1 else 1
        
        for i, score in enumerate(test_scores):
            # Find classes that satisfy the conformity condition
            prediction_set = []
            
            for class_idx in range(n_classes):
                # Check if class should be included in prediction set
                if self._include_in_set(test_probabilities[i], class_idx, score):
                    prediction_set.append(class_idx)
            
            # Ensure non-empty prediction set
            if not prediction_set:
                prediction_set = [np.argmax(test_probabilities[i])]
            
            prediction_sets.append(prediction_set)
        
        return prediction_sets
    
    @abstractmethod
    def _include_in_set(self, 
                       probabilities: np.ndarray,
                       class_idx: int,
                       conformity_score: float) -> bool:
        """
        Determine if a class should be included in the prediction set.
        
        Args:
            probabilities: Probability vector for this sample
            class_idx: Class index to check
            conformity_score: Conformity score for this sample
            
        Returns:
            Whether to include the class
        """
        pass
    
    def compute_coverage(self, 
                        prediction_sets: List[List[int]],
                        true_labels: np.ndarray) -> float:
        """
        Compute empirical coverage of prediction sets.
        
        Args:
            prediction_sets: List of prediction sets
            true_labels: True labels
            
        Returns:
            Coverage rate
        """
        covered = 0
        for pred_set, true_label in zip(prediction_sets, true_labels):
            if true_label in pred_set:
                covered += 1
        
        return covered / len(prediction_sets)
    
    def compute_average_size(self, prediction_sets: List[List[int]]) -> float:
        """
        Compute average prediction set size.
        
        Args:
            prediction_sets: List of prediction sets
            
        Returns:
            Average set size
        """
        return np.mean([len(pred_set) for pred_set in prediction_sets])
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information."""
        if not self.is_calibrated:
            return {'calibrated': False}
        
        return {
            'calibrated': True,
            'alpha': self.alpha,
            'target_coverage': 1 - self.alpha,
            'quantile': float(self.quantile),
            'calibration_size': len(self.calibration_scores) if self.calibration_scores is not None else 0
        }


class LACConformalPredictor(BaseConformalPredictor):
    """
    Least Ambiguous Set-valued Classifier (LAC) conformal predictor.
    
    Uses 1 - p_y as the conformity score, where p_y is the predicted
    probability of the true class.
    """
    
    def compute_conformity_scores(self, 
                                 probabilities: np.ndarray,
                                 labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute LAC conformity scores."""
        if labels is None:
            # For test data, use maximum probability
            return 1 - np.max(probabilities, axis=1)
        else:
            # For calibration data, use true label probability
            scores = []
            for i, label in enumerate(labels):
                if label < probabilities.shape[1]:
                    scores.append(1 - probabilities[i, label])
                else:
                    scores.append(1.0)  # Maximum non-conformity for invalid labels
            return np.array(scores)
    
    def _include_in_set(self, 
                       probabilities: np.ndarray,
                       class_idx: int,
                       conformity_score: float) -> bool:
        """Include class if its non-conformity score is below quantile."""
        class_nonconformity = 1 - probabilities[class_idx]
        return class_nonconformity <= self.quantile


class APSConformalPredictor(BaseConformalPredictor):
    """
    Adaptive Prediction Sets (APS) conformal predictor.
    
    Uses cumulative probability ordering to create more efficient
    prediction sets than LAC.
    """
    
    def compute_conformity_scores(self, 
                                 probabilities: np.ndarray,
                                 labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute APS conformity scores."""
        if labels is None:
            # For test data, use the score for the predicted class
            return 1 - np.max(probabilities, axis=1)
        else:
            # For calibration, compute cumulative probability until true label
            scores = []
            for i, label in enumerate(labels):
                if label < probabilities.shape[1]:
                    # Sort probabilities in descending order
                    sorted_probs = np.sort(probabilities[i])[::-1]
                    sorted_indices = np.argsort(probabilities[i])[::-1]
                    
                    # Find position of true label
                    true_label_pos = np.where(sorted_indices == label)[0][0]
                    
                    # Cumulative probability up to and including true label
                    cumulative_prob = np.sum(sorted_probs[:true_label_pos + 1])
                    scores.append(1 - cumulative_prob)
                else:
                    scores.append(1.0)
            
            return np.array(scores)
    
    def _include_in_set(self, 
                       probabilities: np.ndarray,
                       class_idx: int,
                       conformity_score: float) -> bool:
        """Include classes based on cumulative probability ordering."""
        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities)[::-1]
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Find position of this class
        class_pos = np.where(sorted_indices == class_idx)[0][0]
        
        # Cumulative probability up to this class
        cumulative_prob = np.sum(sorted_probs[:class_pos + 1])
        
        # Include if cumulative non-conformity is below quantile
        return (1 - cumulative_prob) <= self.quantile


class RAPSConformalPredictor(BaseConformalPredictor):
    """
    Regularized Adaptive Prediction Sets (RAPS) conformal predictor.
    
    Adds regularization to APS to encourage smaller prediction sets.
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 k_reg: int = 5,
                 lambda_reg: float = 0.01,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAPS predictor.
        
        Args:
            alpha: Miscoverage level
            k_reg: Regularization parameter k
            lambda_reg: Regularization strength
            config: Configuration dictionary
        """
        super().__init__(alpha, config)
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
    
    def compute_conformity_scores(self, 
                                 probabilities: np.ndarray,
                                 labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute RAPS conformity scores with regularization."""
        if labels is None:
            # For test data
            return 1 - np.max(probabilities, axis=1)
        else:
            # For calibration with regularization
            scores = []
            for i, label in enumerate(labels):
                if label < probabilities.shape[1]:
                    # Sort probabilities
                    sorted_probs = np.sort(probabilities[i])[::-1]
                    sorted_indices = np.argsort(probabilities[i])[::-1]
                    
                    # Find true label position
                    true_label_pos = np.where(sorted_indices == label)[0][0]
                    
                    # Cumulative probability
                    cumulative_prob = np.sum(sorted_probs[:true_label_pos + 1])
                    
                    # Add regularization
                    reg_term = self.lambda_reg * max(true_label_pos + 1 - self.k_reg, 0)
                    
                    scores.append(1 - cumulative_prob + reg_term)
                else:
                    scores.append(1.0)
            
            return np.array(scores)
    
    def _include_in_set(self, 
                       probabilities: np.ndarray,
                       class_idx: int,
                       conformity_score: float) -> bool:
        """Include classes with regularization consideration."""
        # Sort probabilities
        sorted_probs = np.sort(probabilities)[::-1]
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Find position of this class
        class_pos = np.where(sorted_indices == class_idx)[0][0]
        
        # Cumulative probability
        cumulative_prob = np.sum(sorted_probs[:class_pos + 1])
        
        # Add regularization
        reg_term = self.lambda_reg * max(class_pos + 1 - self.k_reg, 0)
        
        # Include if regularized score is below quantile
        return (1 - cumulative_prob + reg_term) <= self.quantile


class ConformalPredictor:
    """
    Factory class for creating conformal predictors.
    
    Supports multiple conformal prediction methods with unified interface.
    """
    
    METHODS = {
        'lac': LACConformalPredictor,
        'aps': APSConformalPredictor,
        'raps': RAPSConformalPredictor
    }
    
    def __init__(self, 
                 method: str = 'lac',
                 alpha: float = 0.1,
                 **kwargs):
        """
        Initialize conformal predictor.
        
        Args:
            method: Conformal prediction method ('lac', 'aps', 'raps')
            alpha: Miscoverage level
            **kwargs: Method-specific parameters
        """
        if method not in self.METHODS:
            raise ConfigurationError(f"Unknown conformal method: {method}. "
                                   f"Available: {list(self.METHODS.keys())}")
        
        self.method = method
        self.predictor = self.METHODS[method](alpha=alpha, **kwargs)
        
        logger.info(f"Initialized ConformalPredictor with method: {method}")
    
    def calibrate(self, 
                 probabilities: np.ndarray,
                 labels: np.ndarray):
        """Calibrate the conformal predictor."""
        self.predictor.calibrate(probabilities, labels)
    
    def predict(self, probabilities: np.ndarray) -> List[List[int]]:
        """Generate prediction sets."""
        return self.predictor.predict(probabilities)
    
    def compute_coverage(self, 
                        prediction_sets: List[List[int]],
                        true_labels: np.ndarray) -> float:
        """Compute empirical coverage."""
        return self.predictor.compute_coverage(prediction_sets, true_labels)
    
    def compute_average_size(self, prediction_sets: List[List[int]]) -> float:
        """Compute average set size."""
        return self.predictor.compute_average_size(prediction_sets)
    
    def get_info(self) -> Dict[str, Any]:
        """Get predictor information."""
        info = self.predictor.get_calibration_info()
        info['method'] = self.method
        return info
    
    def set_alpha(self, alpha: float):
        """Update miscoverage level."""
        self.predictor.alpha = alpha
        
        # Re-calibrate if possible
        if self.predictor.is_calibrated and self.predictor.calibration_scores is not None:
            self.predictor.calibrate(
                self.predictor.calibration_scores,
                self.predictor.calibration_labels
            )
    
    # Delegate other methods to the underlying predictor
    def __getattr__(self, name):
        """Forward undefined attributes to the predictor."""
        return getattr(self.predictor, name)