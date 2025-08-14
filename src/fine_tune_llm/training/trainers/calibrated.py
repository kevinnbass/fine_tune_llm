"""
Calibrated trainer for advanced model training with calibration awareness.

This module provides calibration-aware training capabilities including
ECE/MCE monitoring, abstention-aware loss, and conformal prediction integration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from transformers import Trainer
import logging

from .base import BaseTrainer
from ...core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class CalibratedTrainer(Trainer):
    """
    Enhanced trainer with calibration awareness and advanced metrics.
    
    This trainer extends the standard Transformers Trainer with:
    - ECE/MCE monitoring and learning rate adjustment
    - Abstention-aware loss functions with confidence weighting
    - Advanced metrics aggregation and tracking
    - Conformal prediction integration during training
    """
    
    def __init__(self, 
                 metrics_aggregator=None, 
                 conformal_predictor=None,
                 calibration_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize calibrated trainer.
        
        Args:
            metrics_aggregator: Advanced metrics aggregation component
            conformal_predictor: Conformal prediction component
            calibration_config: Calibration-specific configuration
            **kwargs: Standard Trainer arguments
        """
        super().__init__(**kwargs)
        
        # Advanced components
        self.metrics_aggregator = metrics_aggregator
        self.conformal_predictor = conformal_predictor
        
        # Calibration configuration
        self.calibration_config = calibration_config or {}
        
        # Calibration monitoring
        self.ece_threshold = self.calibration_config.get('ece_threshold', 0.05)
        self.mce_threshold = self.calibration_config.get('mce_threshold', 0.10)
        self.adjustment_threshold = self.calibration_config.get('adjustment_threshold', 0.05)
        self.lr_reduction_factor = self.calibration_config.get('lr_reduction_factor', 0.8)
        
        # Abstention configuration
        self.abstention_config = self.calibration_config.get('abstention_loss', {})
        self.enable_abstention = self.abstention_config.get('enabled', False)
        self.confidence_threshold = self.abstention_config.get('confidence_threshold', 0.7)
        self.abstention_penalty = self.abstention_config.get('abstention_penalty', 0.3)
        self.uncertainty_weight = self.abstention_config.get('uncertainty_weight', 0.1)
        
        # Advanced metrics tracking
        self.enable_advanced_metrics = self.calibration_config.get('enable_advanced_metrics', True)
        self.conformal_prediction = self.calibration_config.get('conformal_prediction', False)
        self.risk_controlled_training = self.calibration_config.get('risk_controlled_training', False)
        
        # Metrics history for trend analysis
        self.ece_history = []
        self.mce_history = []
        self.last_adjustment_step = 0
        
        logger.info(f"Initialized CalibratedTrainer with advanced features:")
        logger.info(f"  - Abstention loss: {self.enable_abstention}")
        logger.info(f"  - Advanced metrics: {self.enable_advanced_metrics}")
        logger.info(f"  - Conformal prediction: {self.conformal_prediction}")
        logger.info(f"  - Risk-controlled training: {self.risk_controlled_training}")
    
    def evaluate(self, 
                 eval_dataset=None, 
                 ignore_keys=None, 
                 metric_key_prefix="eval"):
        """
        Enhanced evaluation with calibration metrics.
        
        Args:
            eval_dataset: Evaluation dataset
            ignore_keys: Keys to ignore in metrics
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Evaluation metrics including calibration scores
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # Get predictions and labels for calibration analysis
        predictions, labels = self._get_predictions_and_labels(eval_dataloader)
        
        if predictions is not None and labels is not None:
            # Compute calibration metrics
            calibration_metrics = self._compute_calibration_metrics(predictions, labels)
            output.metrics.update(calibration_metrics)
            
            # Update history
            self.ece_history.append(calibration_metrics.get('eval_ece', 0.0))
            self.mce_history.append(calibration_metrics.get('eval_mce', 0.0))
            
            # Check for learning rate adjustment
            if self.should_adjust_learning_rate():
                self.adjust_learning_rate()
            
            # Conformal prediction calibration
            if self.conformal_prediction and self.conformal_predictor:
                try:
                    conformal_metrics = self._calibrate_conformal_predictor(predictions, labels)
                    output.metrics.update(conformal_metrics)
                except Exception as e:
                    logger.warning(f"Conformal calibration failed: {e}")
            
            # Advanced metrics aggregation
            if self.enable_advanced_metrics and self.metrics_aggregator:
                try:
                    advanced_metrics = self.metrics_aggregator.compute_advanced_metrics(
                        predictions=predictions,
                        labels=labels,
                        probabilities=torch.softmax(torch.tensor(predictions), dim=-1).numpy()
                    )
                    output.metrics.update({f"eval_{k}": v for k, v in advanced_metrics.items()})
                except Exception as e:
                    logger.warning(f"Advanced metrics computation failed: {e}")
        
        self.log(output.metrics)
        
        return output
    
    def _get_predictions_and_labels(self, dataloader):
        """
        Extract predictions and labels from dataloader.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Tuple of (predictions, labels)
        """
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    predictions = torch.softmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                
                if 'labels' in batch:
                    all_labels.extend(batch['labels'].cpu().numpy())
        
        if not all_predictions or not all_labels:
            return None, None
        
        return np.array(all_predictions), np.array(all_labels)
    
    def _compute_calibration_metrics(self, predictions, labels, n_bins=10):
        """
        Compute calibration metrics (ECE, MCE, Brier score).
        
        Args:
            predictions: Predicted probabilities
            labels: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics
        """
        try:
            # Import calibration metrics
            from ...evaluation.metrics import compute_ece, compute_mce, compute_brier_score
            
            # Get confidence scores (max probability)
            confidences = np.max(predictions, axis=1)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Compute metrics
            metrics = {}
            
            # Expected Calibration Error
            ece = compute_ece(confidences, predicted_classes == labels, n_bins)
            metrics['ece'] = float(ece)
            
            # Maximum Calibration Error  
            mce = compute_mce(confidences, predicted_classes == labels, n_bins)
            metrics['mce'] = float(mce)
            
            # Brier Score
            brier = compute_brier_score(predictions, labels)
            metrics['brier_score'] = float(brier)
            
            # Average confidence
            metrics['avg_confidence'] = float(np.mean(confidences))
            metrics['confidence_std'] = float(np.std(confidences))
            
            return metrics
            
        except ImportError:
            logger.warning("Advanced calibration metrics not available")
            return {}
        except Exception as e:
            logger.error(f"Error computing calibration metrics: {e}")
            return {}
    
    def should_adjust_learning_rate(self):
        """
        Check if learning rate should be adjusted based on calibration trends.
        
        Returns:
            Bool indicating if adjustment is needed
        """
        if len(self.ece_history) < 3:  # Need at least 3 evaluations
            return False
        
        # Check if we recently adjusted
        if self.state.global_step - self.last_adjustment_step < 200:
            return False
        
        # Check for ECE trend
        recent_ece = self.ece_history[-3:]
        if all(ece > self.adjustment_threshold for ece in recent_ece):
            # ECE consistently high
            return True
        
        # Check for increasing trend
        if recent_ece[-1] > recent_ece[0] * 1.2:  # 20% increase
            return True
        
        return False
    
    def adjust_learning_rate(self, factor=None):
        """
        Adjust learning rate for better calibration.
        
        Args:
            factor: Reduction factor (default from config)
        """
        if factor is None:
            factor = self.lr_reduction_factor
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * factor
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.last_adjustment_step = self.state.global_step
        
        logger.info(f"Adjusted learning rate: {current_lr:.2e} -> {new_lr:.2e} "
                   f"(factor: {factor}) at step {self.state.global_step}")
    
    def compute_abstention_aware_loss(self, model, inputs, return_outputs=False):
        """
        Compute abstention-aware loss with confidence weighting.
        
        Args:
            model: Model instance
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor and optionally model outputs
        """
        if not self.enable_abstention:
            return self.compute_loss(model, inputs, return_outputs)
        
        # Forward pass
        outputs = model(**inputs)
        
        if 'labels' in inputs:
            labels = inputs['labels']
            logits = outputs.logits
            
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                    labels.view(-1), 
                                    reduction='none')
            
            # Compute confidence scores
            probs = F.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            
            # Confidence weighting
            confidence_weights = torch.where(
                max_probs > self.confidence_threshold,
                torch.ones_like(max_probs),
                torch.ones_like(max_probs) * (1.0 + self.abstention_penalty)
            )
            
            # Weighted loss
            weighted_loss = ce_loss.view(-1) * confidence_weights
            
            # Uncertainty penalty (entropy-based)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            uncertainty_penalty = entropy * self.uncertainty_weight
            
            # Combined loss
            total_loss = weighted_loss + uncertainty_penalty
            loss = total_loss.mean()
            
            # Update outputs
            outputs.loss = loss
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with abstention awareness if enabled.
        
        Args:
            model: Model instance
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss tensor and optionally outputs
        """
        if self.enable_abstention:
            return self.compute_abstention_aware_loss(model, inputs, return_outputs)
        else:
            return super().compute_loss(model, inputs, return_outputs)
    
    def _calibrate_conformal_predictor(self, predictions, labels):
        """
        Calibrate conformal predictor during training.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Conformal prediction metrics
        """
        if self.conformal_predictor is None:
            return {}
        
        try:
            # Convert to appropriate format for conformal predictor
            scores = np.max(predictions, axis=1)  # Confidence scores
            
            # Calibrate predictor
            self.conformal_predictor.calibrate(scores, labels)
            
            # Get prediction sets for evaluation
            prediction_sets = self.conformal_predictor.predict(scores)
            
            # Compute coverage
            coverage = np.mean([labels[i] in pred_set for i, pred_set in enumerate(prediction_sets)])
            
            # Compute average set size
            avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
            
            return {
                'conformal_coverage': float(coverage),
                'conformal_avg_set_size': float(avg_set_size),
                'conformal_efficiency': float(1.0 / avg_set_size) if avg_set_size > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Conformal calibration error: {e}")
            return {}
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with calibration information."""
        # Add calibration trend information
        if self.ece_history:
            logs['ece_trend'] = len(self.ece_history)
            logs['ece_latest'] = self.ece_history[-1]
            
            if len(self.ece_history) >= 2:
                logs['ece_change'] = self.ece_history[-1] - self.ece_history[-2]
        
        # Add learning rate adjustment info
        logs['last_lr_adjustment'] = self.last_adjustment_step
        logs['steps_since_adjustment'] = self.state.global_step - self.last_adjustment_step
        
        # Call parent logging
        super().log(logs)