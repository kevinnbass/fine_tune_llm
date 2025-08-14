"""
Conformal prediction for risk-controlled inference.
Restored and adapted from ensemble system for LLM fine-tuning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Split conformal prediction for controlled abstention in high-stakes scenarios."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        calibration_ratio: float = 0.2,
        method: str = "lac"  # "lac" (least ambiguous set) or "aps" (adaptive prediction sets)
    ):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha confidence)
            calibration_ratio: Fraction of data to use for calibration
            method: Conformal method to use
        """
        self.alpha = alpha
        self.calibration_ratio = calibration_ratio
        self.method = method
        
        # Calibration data
        self.calibration_scores = None
        self.quantile = None
        self.is_calibrated = False
        
    def compute_nonconformity_scores(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute nonconformity scores for calibration or inference.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels for calibration (optional)
            
        Returns:
            Nonconformity scores
        """
        if self.method == "lac":  # Least Ambiguous Set
            # Score = 1 - probability of true class
            if labels is not None:
                scores = 1.0 - probs[np.arange(len(labels)), labels]
            else:
                # For inference, use max probability
                scores = 1.0 - np.max(probs, axis=1)
        
        elif self.method == "aps":  # Adaptive Prediction Sets
            if labels is not None:
                # Score = sum of probabilities of classes ranked higher than true class
                sorted_indices = np.argsort(-probs, axis=1)
                scores = np.zeros(len(labels))
                
                for i in range(len(labels)):
                    true_class = labels[i]
                    true_rank = np.where(sorted_indices[i] == true_class)[0][0]
                    # Sum probabilities of higher-ranked classes
                    scores[i] = np.sum(probs[i, sorted_indices[i][:true_rank]])
            else:
                # For inference, return probabilities of second-best class
                sorted_probs = np.sort(probs, axis=1)
                scores = sorted_probs[:, -2]  # Second highest probability
        
        else:
            raise ValueError(f"Unknown conformal method: {self.method}")
        
        return scores
    
    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        validation_probs: Optional[np.ndarray] = None,
        validation_labels: Optional[np.ndarray] = None
    ):
        """
        Calibrate the conformal predictor.
        
        Args:
            probs: Predicted probabilities on calibration set
            labels: True labels for calibration set
            validation_probs: Optional separate validation set probabilities
            validation_labels: Optional separate validation set labels
        """
        # Use provided validation set or split the data
        if validation_probs is not None and validation_labels is not None:
            cal_probs, cal_labels = validation_probs, validation_labels
        else:
            # Split data for calibration
            n_cal = int(len(probs) * self.calibration_ratio)
            indices = np.random.permutation(len(probs))
            cal_indices = indices[:n_cal]
            
            cal_probs = probs[cal_indices]
            cal_labels = labels[cal_indices]
        
        # Compute nonconformity scores on calibration set
        self.calibration_scores = self.compute_nonconformity_scores(cal_probs, cal_labels)
        
        # Compute quantile threshold
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        self.is_calibrated = True
        
        logger.info(f"Calibrated conformal predictor with alpha={self.alpha}, quantile={self.quantile:.4f}")
    
    def predict_sets(
        self,
        probs: np.ndarray,
        return_sizes: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate prediction sets with coverage guarantees.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            return_sizes: Whether to also return set sizes
            
        Returns:
            Prediction sets as boolean array (n_samples, n_classes)
            Optionally also set sizes (n_samples,)
        """
        if not self.is_calibrated:
            raise ValueError("Conformal predictor must be calibrated first")
        
        n_samples, n_classes = probs.shape
        prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)
        
        if self.method == "lac":
            # Include all classes with probability >= (max_prob - quantile)
            max_probs = np.max(probs, axis=1, keepdims=True)
            prediction_sets = probs >= (max_probs - self.quantile)
        
        elif self.method == "aps":
            # Include classes in descending order until cumulative probability > (1 - quantile)
            sorted_indices = np.argsort(-probs, axis=1)
            
            for i in range(n_samples):
                cumulative_prob = 0.0
                for j in range(n_classes):
                    class_idx = sorted_indices[i, j]
                    cumulative_prob += probs[i, class_idx]
                    prediction_sets[i, class_idx] = True
                    
                    if cumulative_prob >= (1.0 - self.quantile):
                        break
        
        if return_sizes:
            set_sizes = np.sum(prediction_sets, axis=1)
            return prediction_sets, set_sizes
        
        return prediction_sets
    
    def should_abstain(
        self,
        probs: np.ndarray,
        max_set_size: int = 1,
        min_confidence: float = 0.5
    ) -> np.ndarray:
        """
        Determine which predictions should be abstained based on prediction set size.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            max_set_size: Maximum allowed prediction set size
            min_confidence: Minimum confidence for non-abstention
            
        Returns:
            Boolean array indicating which samples to abstain on
        """
        if not self.is_calibrated:
            raise ValueError("Conformal predictor must be calibrated first")
        
        prediction_sets, set_sizes = self.predict_sets(probs, return_sizes=True)
        max_confidences = np.max(probs, axis=1)
        
        # Abstain if set size is too large or confidence is too low
        abstain = (set_sizes > max_set_size) | (max_confidences < min_confidence)
        
        return abstain
    
    def evaluate_coverage(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate coverage properties of the conformal predictor.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
            
        Returns:
            Dictionary of coverage metrics
        """
        if not self.is_calibrated:
            raise ValueError("Conformal predictor must be calibrated first")
        
        prediction_sets, set_sizes = self.predict_sets(probs, return_sizes=True)
        
        # Check if true label is in prediction set for each sample
        covered = prediction_sets[np.arange(len(labels)), labels]
        
        coverage = np.mean(covered)
        avg_set_size = np.mean(set_sizes)
        
        # Size distribution
        size_counts = np.bincount(set_sizes.astype(int))
        size_distribution = {}
        for size, count in enumerate(size_counts):
            if count > 0:
                size_distribution[f'size_{size}'] = int(count)
        
        return {
            'coverage': float(coverage),
            'target_coverage': 1.0 - self.alpha,
            'avg_set_size': float(avg_set_size),
            'median_set_size': float(np.median(set_sizes)),
            'max_set_size': int(np.max(set_sizes)),
            'quantile_threshold': float(self.quantile),
            **size_distribution
        }


class RiskControlledPredictor:
    """Risk-controlled conformal prediction for high-stakes applications."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        risk_matrix: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize risk-controlled predictor.
        
        Args:
            alpha: Risk level (expected proportion of high-risk errors)
            risk_matrix: Risk/cost matrix (n_classes, n_classes)
            class_names: Class names for interpretation
        """
        self.alpha = alpha
        self.risk_matrix = risk_matrix
        self.class_names = class_names or ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        
        # Default risk matrix (higher penalty for underestimating risk)
        if self.risk_matrix is None:
            n_classes = len(self.class_names)
            self.risk_matrix = np.zeros((n_classes, n_classes))
            for i in range(n_classes):
                for j in range(n_classes):
                    if i < j:  # Underestimating risk (more severe)
                        self.risk_matrix[i, j] = (j - i) * 2.0
                    elif i > j:  # Overestimating risk (less severe)
                        self.risk_matrix[i, j] = (i - j) * 0.5
        
        self.conformal = ConformalPredictor(alpha=alpha)
        self.risk_threshold = None
        
    def compute_risk_scores(
        self,
        probs: np.ndarray,
        prediction_sets: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute risk scores for each prediction.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            prediction_sets: Prediction sets (n_samples, n_classes)
            labels: True labels for calibration (optional)
            
        Returns:
            Risk scores for each sample
        """
        n_samples, n_classes = probs.shape
        risk_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            if labels is not None:
                true_class = labels[i]
                # Risk if we predict each class in the set
                set_risks = []
                for j in range(n_classes):
                    if prediction_sets[i, j]:
                        risk = self.risk_matrix[true_class, j]
                        set_risks.append(risk * probs[i, j])
                
                # Expected risk over the prediction set
                if set_risks:
                    risk_scores[i] = np.sum(set_risks) / np.sum(prediction_sets[i])
            else:
                # For inference, compute expected risk
                expected_risk = 0.0
                for true_j in range(n_classes):
                    for pred_j in range(n_classes):
                        if prediction_sets[i, pred_j]:
                            risk_contribution = (
                                probs[i, true_j] *  # P(true class = j)
                                probs[i, pred_j] *  # P(predict class = j | in set)
                                self.risk_matrix[true_j, pred_j]
                            )
                            expected_risk += risk_contribution
                
                risk_scores[i] = expected_risk
        
        return risk_scores
    
    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ):
        """
        Calibrate both conformal prediction and risk control.
        
        Args:
            probs: Predicted probabilities on calibration set
            labels: True labels for calibration set
        """
        # First calibrate conformal predictor
        self.conformal.calibrate(probs, labels)
        
        # Get prediction sets
        prediction_sets = self.conformal.predict_sets(probs)
        
        # Compute risk scores
        risk_scores = self.compute_risk_scores(probs, prediction_sets, labels)
        
        # Set risk threshold to control risk level
        self.risk_threshold = np.quantile(risk_scores, 1.0 - self.alpha)
        
        logger.info(f"Calibrated risk-controlled predictor with risk_threshold={self.risk_threshold:.4f}")
    
    def predict_with_risk_control(
        self,
        probs: np.ndarray,
        abstain_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with risk control.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            abstain_threshold: Optional risk threshold for abstention
            
        Returns:
            Dictionary with predictions, sets, risks, and abstentions
        """
        if not self.conformal.is_calibrated or self.risk_threshold is None:
            raise ValueError("Predictor must be calibrated first")
        
        # Get conformal prediction sets
        prediction_sets = self.conformal.predict_sets(probs)
        
        # Compute risk scores
        risk_scores = self.compute_risk_scores(probs, prediction_sets)
        
        # Make predictions (highest probability class in set)
        predictions = np.zeros(len(probs), dtype=int)
        for i in range(len(probs)):
            valid_classes = np.where(prediction_sets[i])[0]
            if len(valid_classes) > 0:
                best_class = valid_classes[np.argmax(probs[i, valid_classes])]
                predictions[i] = best_class
        
        # Determine abstentions based on risk
        if abstain_threshold is None:
            abstain_threshold = self.risk_threshold
        
        abstentions = risk_scores > abstain_threshold
        
        return {
            'predictions': predictions,
            'prediction_sets': prediction_sets,
            'risk_scores': risk_scores,
            'abstentions': abstentions,
            'confidences': np.max(probs * prediction_sets, axis=1)
        }


def compute_prediction_intervals(
    predictions: np.ndarray,
    alpha: float = 0.1,
    method: str = "quantile"
) -> Dict[str, np.ndarray]:
    """
    Compute prediction intervals for regression-like outputs.
    
    Args:
        predictions: Point predictions (n_samples,)
        alpha: Miscoverage level
        method: Method for computing intervals
        
    Returns:
        Dictionary with lower and upper bounds
    """
    if method == "quantile":
        # Simple quantile-based intervals
        residuals = predictions - np.mean(predictions)
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        lower_bound = predictions + np.quantile(residuals, lower_q)
        upper_bound = predictions + np.quantile(residuals, upper_q)
    
    else:
        raise ValueError(f"Unknown interval method: {method}")
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'width': upper_bound - lower_bound
    }