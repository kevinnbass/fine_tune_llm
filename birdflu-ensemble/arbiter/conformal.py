"""Conformal prediction for risk-controlled abstention."""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Split conformal prediction for controlled abstention."""
    
    def __init__(self, config_path: str = "configs/conformal.yaml"):
        """
        Initialize conformal predictor.
        
        Args:
            config_path: Path to conformal configuration
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)['conformal']
        
        self.method = self.config['method']
        self.calibration_split = self.config['calibration_split']
        self.global_config = self.config['global']
        self.per_slice_config = self.config.get('per_slice', {})
        
        # Thresholds will be computed during calibration
        self.global_threshold = None
        self.slice_thresholds = {}
        self.calibration_scores = None
        
    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        slices: Optional[Dict[str, np.ndarray]] = None,
        critical_pairs: Optional[List[Tuple[int, int, float]]] = None
    ):
        """
        Calibrate conformal thresholds.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
            slices: Dict mapping slice names to boolean masks
            critical_pairs: List of (from_class, to_class, weight) for critical errors
        """
        logger.info("Calibrating conformal thresholds")
        
        # Split calibration set
        n_samples = len(labels)
        n_cal = int(n_samples * self.calibration_split)
        
        cal_idx, _ = train_test_split(
            np.arange(n_samples),
            test_size=1-self.calibration_split,
            stratify=labels,
            random_state=42
        )
        
        cal_probs = probs[cal_idx]
        cal_labels = labels[cal_idx]
        
        # Compute conformity scores
        if self.method == 'split_conformal':
            # Use 1 - max_prob as conformity score (lower = more confident)
            self.calibration_scores = 1 - cal_probs.max(axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute global threshold
        self.global_threshold = self._compute_threshold(
            cal_probs,
            cal_labels,
            self.global_config['max_error_rate'],
            critical_pairs
        )
        
        logger.info(f"Global threshold: {self.global_threshold:.4f}")
        
        # Compute per-slice thresholds
        if slices:
            for slice_name, mask in slices.items():
                if slice_name in self.per_slice_config:
                    slice_cal_mask = mask[cal_idx]
                    if slice_cal_mask.sum() > 10:  # Need enough samples
                        slice_probs = cal_probs[slice_cal_mask]
                        slice_labels = cal_labels[slice_cal_mask]
                        
                        slice_config = self.per_slice_config[slice_name]
                        slice_threshold = self._compute_threshold(
                            slice_probs,
                            slice_labels,
                            slice_config['max_error_rate'],
                            critical_pairs
                        )
                        
                        self.slice_thresholds[slice_name] = slice_threshold
                        logger.info(f"Slice '{slice_name}' threshold: {slice_threshold:.4f}")
    
    def _compute_threshold(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        max_error_rate: float,
        critical_pairs: Optional[List[Tuple[int, int, float]]] = None
    ) -> float:
        """
        Compute threshold to achieve target error rate.
        
        Args:
            probs: Probabilities
            labels: True labels
            max_error_rate: Maximum allowed error rate
            critical_pairs: Critical error pairs with weights
            
        Returns:
            Threshold value
        """
        # Get predictions and compute scores
        preds = probs.argmax(axis=1)
        scores = 1 - probs.max(axis=1)  # Higher score = less confident
        
        # Compute weighted errors
        errors = []
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                # Check if this is a critical error
                weight = 1.0
                if critical_pairs:
                    for from_class, to_class, crit_weight in critical_pairs:
                        if labels[i] == from_class and preds[i] == to_class:
                            weight = crit_weight
                            break
                errors.append((scores[i], weight))
            else:
                errors.append((scores[i], 0.0))
        
        # Sort by score
        errors.sort(key=lambda x: x[0])
        
        # Find threshold that satisfies error rate
        cumulative_error = 0.0
        cumulative_weight = 0.0
        threshold = 0.0
        
        for score, error_weight in errors:
            cumulative_error += error_weight
            cumulative_weight += 1.0
            
            error_rate = cumulative_error / max(cumulative_weight, 1.0)
            
            if error_rate > max_error_rate:
                # Use previous threshold
                break
            
            threshold = score
        
        # Ensure minimum coverage
        min_coverage = self.global_config.get('min_coverage', 0.85)
        coverage = (scores <= threshold).mean()
        
        if coverage < min_coverage:
            # Adjust threshold to meet coverage requirement
            threshold_idx = int(len(scores) * (1 - min_coverage))
            sorted_scores = np.sort(scores)
            threshold = sorted_scores[min(threshold_idx, len(sorted_scores)-1)]
        
        return threshold
    
    def predict(
        self,
        probs: np.ndarray,
        slice_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with conformal abstention.
        
        Args:
            probs: Predicted probabilities
            slice_name: Optional slice name for specific threshold
            
        Returns:
            Prediction dictionary
        """
        if self.global_threshold is None:
            raise ValueError("Must calibrate before prediction")
        
        # Get appropriate threshold
        if slice_name and slice_name in self.slice_thresholds:
            threshold = self.slice_thresholds[slice_name]
            logger.debug(f"Using slice threshold for '{slice_name}': {threshold:.4f}")
        else:
            threshold = self.global_threshold
        
        # Compute conformity score
        max_prob = probs.max()
        score = 1 - max_prob
        
        # Check threshold
        if score > threshold:
            # Abstain
            return {
                'abstain': True,
                'reason': f'Conformity score {score:.4f} > threshold {threshold:.4f}',
                'max_prob': float(max_prob),
                'threshold': float(threshold),
                'tier': 2
            }
        
        # Make prediction
        class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        pred_idx = probs.argmax()
        
        return {
            'decision': class_names[pred_idx],
            'confidence': float(max_prob),
            'probs': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
            'conformity_score': float(score),
            'threshold': float(threshold),
            'tier': 2,
            'abstain': False
        }
    
    def evaluate_coverage(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        slices: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate coverage and error rates.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            slices: Optional slice masks
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Global metrics
        global_metrics = self._compute_metrics(probs, labels, self.global_threshold)
        metrics['global'] = global_metrics
        
        # Per-slice metrics
        if slices:
            for slice_name, mask in slices.items():
                if mask.sum() > 0:
                    slice_probs = probs[mask]
                    slice_labels = labels[mask]
                    
                    threshold = self.slice_thresholds.get(slice_name, self.global_threshold)
                    slice_metrics = self._compute_metrics(slice_probs, slice_labels, threshold)
                    metrics[f'slice_{slice_name}'] = slice_metrics
        
        return metrics
    
    def _compute_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Compute coverage and error metrics."""
        scores = 1 - probs.max(axis=1)
        abstain_mask = scores > threshold
        predict_mask = ~abstain_mask
        
        coverage = predict_mask.mean()
        
        if predict_mask.sum() > 0:
            preds = probs[predict_mask].argmax(axis=1)
            true_labels = labels[predict_mask]
            error_rate = (preds != true_labels).mean()
        else:
            error_rate = 0.0
        
        return {
            'coverage': float(coverage),
            'abstention_rate': float(abstain_mask.mean()),
            'error_rate': float(error_rate),
            'n_samples': len(labels),
            'n_predictions': int(predict_mask.sum()),
            'n_abstentions': int(abstain_mask.sum())
        }
    
    def adaptive_update(
        self,
        new_probs: np.ndarray,
        new_labels: np.ndarray,
        learning_rate: float = 0.01
    ):
        """
        Adaptively update thresholds (optional advanced feature).
        
        Args:
            new_probs: New predictions
            new_labels: New labels
            learning_rate: Update rate
        """
        if not self.config.get('adaptive', {}).get('enabled', False):
            return
        
        # Compute new threshold on recent data
        new_threshold = self._compute_threshold(
            new_probs,
            new_labels,
            self.global_config['max_error_rate'],
            None
        )
        
        # Exponential moving average update
        self.global_threshold = (
            (1 - learning_rate) * self.global_threshold +
            learning_rate * new_threshold
        )
        
        logger.debug(f"Updated global threshold to {self.global_threshold:.4f}")
    
    def save(self, path: str):
        """Save calibrated thresholds."""
        save_dict = {
            'global_threshold': self.global_threshold,
            'slice_thresholds': self.slice_thresholds,
            'calibration_scores': self.calibration_scores,
            'config': self.config
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved conformal thresholds to {path}")
    
    def load(self, path: str):
        """Load calibrated thresholds."""
        import pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.global_threshold = save_dict['global_threshold']
        self.slice_thresholds = save_dict['slice_thresholds']
        self.calibration_scores = save_dict['calibration_scores']
        self.config = save_dict['config']
        
        logger.info(f"Loaded conformal thresholds from {path}")


class AdaptiveConformal(ConformalPredictor):
    """
    Adaptive conformal prediction with learnable thresholds.
    (Advanced feature for future enhancement)
    """
    
    def __init__(self, config_path: str = "configs/conformal.yaml"):
        """Initialize adaptive conformal predictor."""
        super().__init__(config_path)
        
        self.window_size = self.config.get('adaptive', {}).get('window_size', 5000)
        self.update_frequency = self.config.get('adaptive', {}).get('update_frequency', 1000)
        
        self.recent_predictions = []
        self.recent_labels = []
        self.update_counter = 0
    
    def update(self, probs: np.ndarray, label: int):
        """
        Update with new observation.
        
        Args:
            probs: Prediction probabilities
            label: True label
        """
        self.recent_predictions.append(probs)
        self.recent_labels.append(label)
        
        # Maintain window
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)
            self.recent_labels.pop(0)
        
        self.update_counter += 1
        
        # Update thresholds periodically
        if self.update_counter % self.update_frequency == 0:
            self._update_thresholds()
    
    def _update_thresholds(self):
        """Update thresholds based on recent data."""
        if len(self.recent_predictions) < 100:
            return  # Not enough data
        
        recent_probs = np.array(self.recent_predictions)
        recent_labels = np.array(self.recent_labels)
        
        # Compute new threshold
        new_threshold = self._compute_threshold(
            recent_probs,
            recent_labels,
            self.global_config['max_error_rate'],
            None
        )
        
        # Smooth update
        lr = self.config.get('adaptive', {}).get('learning_rate', 0.01)
        self.global_threshold = (1 - lr) * self.global_threshold + lr * new_threshold
        
        logger.debug(f"Adaptively updated threshold to {self.global_threshold:.4f}")