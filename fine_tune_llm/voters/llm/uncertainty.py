"""Uncertainty-aware fine-tuning for high-stakes precision optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MCDropoutWrapper(nn.Module):
    """Monte Carlo Dropout wrapper for uncertainty estimation."""
    
    def __init__(self, model, num_samples: int = 5, dropout_rate: float = 0.1):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Enable dropout during inference
        self._enable_mc_dropout()
        
    def _enable_mc_dropout(self):
        """Enable dropout layers during inference for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active
                
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with uncertainty estimation."""
        if self.training:
            # Regular forward during training
            return self.model(input_ids, attention_mask, labels=labels, **kwargs)
        else:
            # MC Dropout sampling during inference
            outputs_list = []
            
            for _ in range(self.num_samples):
                with torch.no_grad():
                    # Try different calling patterns for compatibility
                    try:
                        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
                    except TypeError:
                        # Fallback for models that don't accept attention_mask as kwarg
                        try:
                            outputs = self.model(input_ids, **kwargs)
                        except TypeError:
                            # Last resort - positional only
                            outputs = self.model(input_ids)
                    
                    outputs_list.append(outputs.logits)
            
            # Stack predictions
            logits_stack = torch.stack(outputs_list)
            
            # Calculate mean and uncertainty
            mean_logits = logits_stack.mean(dim=0)
            
            # Uncertainty as entropy of predictions
            probs_stack = F.softmax(logits_stack, dim=-1)
            mean_probs = probs_stack.mean(dim=0)
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
            
            # Normalized uncertainty (0-1)
            max_entropy = torch.log(torch.tensor(mean_logits.shape[-1], dtype=torch.float))
            uncertainty = entropy / max_entropy
            
            # Create output object with uncertainty
            class UncertainOutput:
                def __init__(self, logits, uncertainty):
                    self.logits = logits
                    self.uncertainty = uncertainty
                    self.loss = None
            
            return UncertainOutput(mean_logits, uncertainty)
    
    def generate_with_uncertainty(self, input_ids, attention_mask=None, max_new_tokens=256, **kwargs):
        """Generate text with uncertainty estimation."""
        try:
            all_outputs = []
            all_scores = []
            
            for _ in range(self.num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **kwargs
                    )
                    all_outputs.append(outputs.sequences)
                    if hasattr(outputs, 'scores'):
                        all_scores.append(torch.stack(outputs.scores))
            
            # Calculate uncertainty from output diversity
            # Simple approach: check agreement among samples
            unique_outputs = len(set([o.tolist()[0] for o in all_outputs]))
            uncertainty = 1.0 - (1.0 / unique_outputs) if unique_outputs > 0 else 0.0
            
            # Return most common output
            from collections import Counter
            output_counts = Counter([tuple(o.tolist()[0]) for o in all_outputs])
            most_common = output_counts.most_common(1)[0][0]
            final_output = torch.tensor([list(most_common)])
            
            return final_output, torch.tensor(uncertainty)
            
        except Exception as e:
            logger.error(f"Error in generate_with_uncertainty: {e}")
            raise


class DeepEnsembleWrapper:
    """Deep ensemble wrapper for uncertainty estimation with multiple models."""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.num_models = len(models)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through ensemble."""
        outputs_list = []
        
        for model in self.models:
            outputs = model(input_ids, attention_mask, labels=labels, **kwargs)
            outputs_list.append(outputs.logits)
        
        # Stack predictions
        logits_stack = torch.stack(outputs_list)
        
        # Calculate mean and uncertainty
        mean_logits = logits_stack.mean(dim=0)
        std_logits = logits_stack.std(dim=0)
        
        # Uncertainty as prediction variance
        uncertainty = std_logits.mean(dim=-1)
        
        class EnsembleOutput:
            def __init__(self, logits, uncertainty):
                self.logits = logits
                self.uncertainty = uncertainty
                self.loss = None if labels is None else F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return EnsembleOutput(mean_logits, uncertainty)


def compute_uncertainty_aware_loss(outputs, labels, config):
    """
    Compute loss with uncertainty-based penalties for high-stakes precision.
    
    Args:
        outputs: Model outputs with uncertainty
        labels: Ground truth labels
        config: Configuration with high_stakes settings
        
    Returns:
        Total loss with uncertainty penalties
    """
    try:
        base_loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        if not hasattr(outputs, 'uncertainty'):
            return base_loss
        
        uncertainty_config = config.get('high_stakes', {}).get('uncertainty', {})
        
        if uncertainty_config.get('enabled', False):
            # Get predictions
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # Identify false positives (assuming binary classification; adjust for multi-class)
            # Assuming positive class = 1 (relevant for bird flu)
            positive_class = 1
            fp_mask = ((preds == positive_class) & (labels != positive_class)).float()
            
            # Penalize overconfident false positives
            # Low uncertainty on FPs is bad for high-stakes
            overconfident_fp = fp_mask * (1 - outputs.uncertainty.mean())
            fp_penalty = overconfident_fp.mean() * uncertainty_config.get('fp_penalty_weight', 2.0)
            
            # Add calibration loss to ensure uncertainty matches accuracy
            confidence = 1 - outputs.uncertainty
            accuracy = (preds == labels).float()
            calibration_loss = F.mse_loss(confidence, accuracy) * 0.5
            
            total_loss = base_loss + fp_penalty + calibration_loss
            
            # Log components for debugging
            if overconfident_fp.mean() > 0:
                logger.info(f"Uncertainty loss components - Base: {base_loss:.4f}, FP penalty: {fp_penalty:.4f}, Calibration: {calibration_loss:.4f}")
            
            return total_loss
        
        return base_loss
        
    except Exception as e:
        logger.error(f"Error computing uncertainty-aware loss: {e}")
        return base_loss


def should_abstain(uncertainty: float, config: Dict) -> Tuple[bool, str]:
    """
    Determine if model should abstain based on uncertainty.
    
    Args:
        uncertainty: Uncertainty score (0-1)
        config: Configuration with abstention threshold
        
    Returns:
        Tuple of (should_abstain, reason)
    """
    uncertainty_config = config.get('high_stakes', {}).get('uncertainty', {})
    
    if not uncertainty_config.get('enabled', False):
        return False, ""
    
    threshold = uncertainty_config.get('abstention_threshold', 0.7)
    
    if uncertainty > threshold:
        return True, f"High uncertainty ({uncertainty:.2f} > {threshold}) - abstaining for precision safety"
    
    return False, ""


class UncertaintyCalibrator:
    """Calibrate uncertainty estimates to match actual accuracy."""
    
    def __init__(self):
        self.calibration_data = []
        
    def add_sample(self, uncertainty: float, is_correct: bool):
        """Add a sample for calibration."""
        self.calibration_data.append({
            'uncertainty': uncertainty,
            'correct': is_correct
        })
        
    def compute_expected_calibration_error(self, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score (lower is better)
        """
        if not self.calibration_data:
            return 0.0
        
        # Convert to arrays
        uncertainties = np.array([d['uncertainty'] for d in self.calibration_data])
        correctness = np.array([d['correct'] for d in self.calibration_data])
        confidences = 1 - uncertainties
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if bin_mask.sum() > 0:
                bin_confidence = confidences[bin_mask].mean()
                bin_accuracy = correctness[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(confidences)
                
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def plot_calibration(self, save_path: Optional[str] = None):
        """Plot calibration diagram."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.calibration_data:
                return
            
            uncertainties = np.array([d['uncertainty'] for d in self.calibration_data])
            correctness = np.array([d['correct'] for d in self.calibration_data])
            confidences = 1 - uncertainties
            
            # Bin data
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            
            bin_accuracies = []
            bin_confidences = []
            
            for i in range(n_bins):
                bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                if bin_mask.sum() > 0:
                    bin_accuracies.append(correctness[bin_mask].mean())
                    bin_confidences.append(confidences[bin_mask].mean())
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(bin_centers[i])
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            plt.plot(bin_confidences, bin_accuracies, 'ro-', label='Model calibration')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title('Calibration Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Calibration plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for calibration plotting")
        except Exception as e:
            logger.error(f"Error plotting calibration: {e}")


def compute_uncertainty_aware_loss(outputs, labels, config):
    """
    Compute loss with uncertainty awareness and false positive penalties.
    
    Args:
        outputs: Model outputs with uncertainty estimates
        labels: Target labels  
        config: Configuration dictionary
        
    Returns:
        Uncertainty-aware loss tensor
    """
    uncertainty_config = config.get('high_stakes', {}).get('uncertainty', {})
    
    if not uncertainty_config.get('enabled', False):
        # Standard cross-entropy loss
        return F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                              labels.view(-1))
    
    # Standard loss
    base_loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                               labels.view(-1), reduction='none')
    
    # Get uncertainty if available
    if hasattr(outputs, 'uncertainty'):
        uncertainty = outputs.uncertainty.view(-1)
        
        # Apply false positive penalty
        fp_penalty_weight = uncertainty_config.get('fp_penalty_weight', 2.0)
        
        # Identify potential false positives (high confidence on wrong predictions)
        probs = F.softmax(outputs.logits.view(-1, outputs.logits.size(-1)), dim=-1)
        predicted_classes = probs.argmax(dim=-1)
        
        # Penalty for confident wrong predictions
        wrong_predictions = (predicted_classes != labels.view(-1)).float()
        high_confidence = (1.0 - uncertainty) > 0.7
        fp_penalty = wrong_predictions * high_confidence.float() * fp_penalty_weight
        
        # Combine losses
        total_loss = base_loss + fp_penalty
    else:
        total_loss = base_loss
    
    return total_loss.mean()


def should_abstain(uncertainty_score, config):
    """
    Determine if the model should abstain based on uncertainty.
    
    Args:
        uncertainty_score: Uncertainty score (0-1)
        config: Configuration dictionary
        
    Returns:
        Tuple of (should_abstain: bool, reason: str)
    """
    uncertainty_config = config.get('high_stakes', {}).get('uncertainty', {})
    
    if not uncertainty_config.get('enabled', False):
        return False, "Uncertainty-aware training disabled"
    
    threshold = uncertainty_config.get('abstention_threshold', 0.7)
    
    if uncertainty_score >= threshold:
        return True, f"High uncertainty ({uncertainty_score:.3f} >= {threshold})"
    
    return False, f"Low uncertainty ({uncertainty_score:.3f} < {threshold})"