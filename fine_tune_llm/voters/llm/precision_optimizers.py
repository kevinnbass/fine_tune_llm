"""Precision optimization modules including ORPO, pruning, and memory-efficient methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from transformers import Trainer, TrainingArguments
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


class ORPOTrainer:
    """Odds-Ratio Preference Optimization for high-stakes precision alignment."""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.orpo_config = config['alignment']['orpo']
        self.beta = self.orpo_config['beta']
        self.factuality_threshold = self.orpo_config['factuality_threshold']
        
    def compute_orpo_loss(self, chosen_logits, rejected_logits, labels):
        """
        Compute ORPO loss for preference optimization.
        
        Args:
            chosen_logits: Logits for preferred responses
            rejected_logits: Logits for rejected responses
            labels: Ground truth labels
            
        Returns:
            ORPO loss value
        """
        # Compute log probabilities
        chosen_logprobs = F.log_softmax(chosen_logits, dim=-1)
        rejected_logprobs = F.log_softmax(rejected_logits, dim=-1)
        
        # Get probabilities for actual tokens
        chosen_probs = torch.gather(chosen_logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)
        rejected_probs = torch.gather(rejected_logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)
        
        # Compute odds ratio
        odds_chosen = torch.exp(chosen_probs) / (1 - torch.exp(chosen_probs) + 1e-8)
        odds_rejected = torch.exp(rejected_probs) / (1 - torch.exp(rejected_probs) + 1e-8)
        odds_ratio = odds_chosen / (odds_rejected + 1e-8)
        
        # ORPO loss: maximize log odds ratio for chosen over rejected
        orpo_loss = -torch.log(odds_ratio + 1e-8).mean()
        
        # Add factuality regularization for high-stakes
        factuality_penalty = self.compute_factuality_penalty(chosen_logits)
        
        return orpo_loss + self.beta * factuality_penalty
    
    def compute_factuality_penalty(self, logits):
        """
        Compute penalty for non-factual outputs in high-stakes scenarios.
        
        Args:
            logits: Model output logits
            
        Returns:
            Factuality penalty
        """
        # Simple entropy-based penalty (lower entropy = more confident = potentially less factual)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Penalize overconfidence (low entropy)
        min_entropy = 0.1  # Minimum acceptable entropy
        penalty = F.relu(min_entropy - entropy).mean()
        
        return penalty
    
    def train_step(self, batch):
        """
        Single training step with ORPO.
        
        Args:
            batch: Training batch with chosen and rejected examples
            
        Returns:
            Loss value
        """
        # Process chosen and rejected examples
        chosen_inputs = self.tokenizer(
            batch['chosen_text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        rejected_inputs = self.tokenizer(
            batch['rejected_text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Forward passes
        chosen_outputs = self.model(**chosen_inputs, labels=chosen_inputs['input_ids'])
        rejected_outputs = self.model(**rejected_inputs, labels=rejected_inputs['input_ids'])
        
        # Compute ORPO loss
        loss = self.compute_orpo_loss(
            chosen_outputs.logits,
            rejected_outputs.logits,
            chosen_inputs['input_ids']
        )
        
        return loss


class PrecisionPruner:
    """Model pruning optimized for maintaining precision in high-stakes tasks."""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.pruning_config = config['pruning']
        self.prune_ratio = self.pruning_config['ratio']
        self.method = self.pruning_config['method']
        self.importance_type = self.pruning_config['importance_type']
        
    def compute_importance_scores(self, eval_dataset) -> Dict[str, float]:
        """
        Compute importance scores for parameters based on precision impact.
        
        Args:
            eval_dataset: Evaluation dataset for measuring precision
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        importance_scores = {}
        
        # Get baseline precision
        baseline_precision = self.evaluate_precision(eval_dataset)
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name:
                continue
                
            # Temporarily zero out parameter
            original_data = param.data.clone()
            param.data.zero_()
            
            # Measure precision drop
            perturbed_precision = self.evaluate_precision(eval_dataset)
            importance = baseline_precision - perturbed_precision
            
            # Restore parameter
            param.data = original_data
            
            importance_scores[name] = importance
            
        return importance_scores
    
    def evaluate_precision(self, eval_dataset) -> float:
        """
        Evaluate model precision on dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Precision score
        """
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_dataset:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        return correct / total if total > 0 else 0
    
    def prune_model(self, eval_dataset=None):
        """
        Prune model parameters while maintaining precision.
        
        Args:
            eval_dataset: Optional evaluation dataset for importance scoring
        """
        if self.method == 'magnitude':
            self.magnitude_pruning()
        elif self.method == 'precision' and eval_dataset:
            self.precision_based_pruning(eval_dataset)
        else:
            self.random_pruning()
            
        logger.info(f"Pruned {self.prune_ratio * 100}% of parameters")
        
    def magnitude_pruning(self):
        """Prune parameters with smallest magnitudes."""
        all_weights = []
        
        # Collect all weights
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.abs().flatten())
        
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, self.prune_ratio)
        
        # Apply pruning mask
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = param.data.abs() > threshold
                param.data *= mask.float()
                
    def precision_based_pruning(self, eval_dataset):
        """Prune parameters with lowest impact on precision."""
        importance_scores = self.compute_importance_scores(eval_dataset)
        
        # Sort by importance
        sorted_params = sorted(importance_scores.items(), key=lambda x: x[1])
        
        # Prune least important parameters
        num_to_prune = int(len(sorted_params) * self.prune_ratio)
        params_to_prune = sorted_params[:num_to_prune]
        
        for param_name, _ in params_to_prune:
            # Zero out parameter
            for name, param in self.model.named_parameters():
                if name == param_name:
                    param.data.zero_()
                    

class EfficientAttentionSkipper:
    """Skip attention heads based on confidence for efficiency."""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.eas_config = config['eas']
        self.skip_ratio = self.eas_config['skip_ratio']
        self.confidence_based = self.eas_config['confidence_based']
        self.head_importance = {}
        
    def compute_head_importance(self, eval_dataset):
        """
        Compute importance of each attention head.
        
        Args:
            eval_dataset: Evaluation dataset
        """
        # Track attention weights for each head
        attention_scores = {}
        
        def hook_fn(module, input, output, layer_idx, head_idx):
            """Hook to capture attention scores."""
            if (layer_idx, head_idx) not in attention_scores:
                attention_scores[(layer_idx, head_idx)] = []
            attention_scores[(layer_idx, head_idx)].append(output[0].mean().item())
        
        # Register hooks (simplified - actual implementation would be model-specific)
        hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(
                    lambda m, i, o, l=layer_idx: hook_fn(m, i, o, l, 0)
                )
                hooks.append(hook)
        
        # Run evaluation
        self.model.eval()
        with torch.no_grad():
            for batch in eval_dataset:
                _ = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute importance as variance of attention scores
        for key, scores in attention_scores.items():
            self.head_importance[key] = np.var(scores)
            
    def apply_skipping(self, eval_dataset=None):
        """
        Apply attention head skipping.
        
        Args:
            eval_dataset: Optional evaluation dataset for importance computation
        """
        if self.confidence_based and eval_dataset:
            self.compute_head_importance(eval_dataset)
            
            # Skip heads with lowest importance
            sorted_heads = sorted(self.head_importance.items(), key=lambda x: x[1])
            num_to_skip = int(len(sorted_heads) * self.skip_ratio)
            heads_to_skip = sorted_heads[:num_to_skip]
            
            # Apply masking (simplified - actual implementation would be model-specific)
            for (layer_idx, head_idx), _ in heads_to_skip:
                logger.info(f"Skipping attention head {head_idx} in layer {layer_idx}")
                # In practice, would modify attention mask or architecture
        else:
            # Random skipping
            logger.info(f"Randomly skipping {self.skip_ratio * 100}% of attention heads")


class MemoryEfficientOptimizers:
    """LOMO and MeZO optimizers for memory-efficient training."""
    
    @staticmethod
    def get_lomo_optimizer(model, config):
        """
        Get LOMO (Low-Memory Optimization) optimizer.
        
        Args:
            model: Model to optimize
            config: Configuration dictionary
            
        Returns:
            LOMO optimizer
        """
        lomo_config = config['lomo']
        
        class LOMOOptimizer(torch.optim.Optimizer):
            """Low-memory SGD variant with adaptive learning rates."""
            
            def __init__(self, params, lr=1e-3, adaptive_lr=True):
                defaults = dict(lr=lr, adaptive_lr=adaptive_lr)
                super().__init__(params, defaults)
                
            def step(self, closure=None):
                """Single optimization step with memory efficiency."""
                loss = None
                if closure is not None:
                    loss = closure()
                    
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                            
                        # Adaptive learning rate based on gradient magnitude
                        if group['adaptive_lr']:
                            grad_norm = p.grad.norm()
                            effective_lr = group['lr'] / (1 + grad_norm)
                        else:
                            effective_lr = group['lr']
                            
                        # Low-memory update: immediate application without momentum
                        p.data.add_(p.grad, alpha=-effective_lr)
                        
                        # Clear gradient immediately to save memory
                        p.grad = None
                        
                return loss
        
        return LOMOOptimizer(
            model.parameters(),
            lr=config['training']['learning_rate'],
            adaptive_lr=lomo_config['adaptive_lr']
        )
    
    @staticmethod
    def get_mezo_optimizer(model, config):
        """
        Get MeZO (Memory-Efficient Zeroth-Order) optimizer.
        
        Args:
            model: Model to optimize
            config: Configuration dictionary
            
        Returns:
            MeZO optimizer
        """
        mezo_config = config['mezo']
        
        class MeZOOptimizer(torch.optim.Optimizer):
            """Zeroth-order optimizer using only forward passes."""
            
            def __init__(self, params, lr=1e-3, perturbation_scale=1e-3):
                defaults = dict(lr=lr, perturbation_scale=perturbation_scale)
                super().__init__(params, defaults)
                
            def step(self, closure):
                """Optimization step using finite differences."""
                if closure is None:
                    raise ValueError("MeZO requires closure for forward passes")
                    
                for group in self.param_groups:
                    for p in group['params']:
                        if p.requires_grad:
                            # Store original parameters
                            original = p.data.clone()
                            
                            # Positive perturbation
                            perturbation = torch.randn_like(p.data) * group['perturbation_scale']
                            p.data.add_(perturbation)
                            loss_pos = closure()
                            
                            # Negative perturbation
                            p.data = original - perturbation
                            loss_neg = closure()
                            
                            # Estimate gradient
                            grad_estimate = (loss_pos - loss_neg) / (2 * group['perturbation_scale'])
                            
                            # Update parameters
                            p.data = original - group['lr'] * grad_estimate * perturbation
                            
                return (loss_pos + loss_neg) / 2
        
        return MeZOOptimizer(
            model.parameters(),
            lr=config['training']['learning_rate'],
            perturbation_scale=mezo_config['perturbation_scale']
        )