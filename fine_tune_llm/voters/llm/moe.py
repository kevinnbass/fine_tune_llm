"""Mixture of Experts (MoE) implementation for precision-optimized fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class PrecisionGatedMoE(nn.Module):
    """Mixture of Experts with precision-based gating for high-stakes tasks."""
    
    def __init__(self, config: Dict[str, Any], hidden_size: int):
        super().__init__()
        self.config = config
        self.num_experts = config['moe']['num_experts']
        self.top_k = config['moe']['top_k']
        self.precision_gate = config['moe']['precision_gate']
        
        # Initialize gating network
        self.gate = nn.Linear(hidden_size, self.num_experts)
        
        # Track expert precision scores (initialized to 1.0)
        self.register_buffer('expert_precision', torch.ones(self.num_experts))
        self.register_buffer('expert_counts', torch.zeros(self.num_experts))
        
    def forward(self, hidden_states: torch.Tensor, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Route inputs through experts based on gating scores.
        
        Args:
            hidden_states: Input hidden states for gating
            expert_outputs: List of outputs from each expert
            
        Returns:
            Weighted combination of expert outputs
        """
        batch_size = hidden_states.size(0)
        
        # Compute gating scores
        gate_scores = self.gate(hidden_states)  # [batch_size, num_experts]
        
        # Apply precision-based adjustment if enabled
        if self.precision_gate and self.training:
            # Adjust scores based on historical precision
            gate_scores = gate_scores * self.expert_precision.unsqueeze(0)
        
        # Apply softmax for probability distribution
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Combine expert outputs
        output = torch.zeros_like(expert_outputs[0])
        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                weight = top_k_probs[i, j]
                output[i] += weight * expert_outputs[expert_idx][i]
                
                # Track expert usage
                if self.training:
                    self.expert_counts[expert_idx] += 1
        
        return output
    
    def update_expert_precision(self, expert_idx: int, precision: float):
        """Update precision score for a specific expert."""
        # Exponential moving average
        alpha = 0.1
        self.expert_precision[expert_idx] = (
            (1 - alpha) * self.expert_precision[expert_idx] + alpha * precision
        )
        

class MoEWrapper:
    """Wrapper for Mixture of Experts models with precision optimization."""
    
    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if not self.config['moe']['enabled']:
            raise ValueError("MoE is not enabled in configuration")
        
        self.num_experts = self.config['moe']['num_experts']
        self.experts = []
        self.tokenizers = []
        self.gate = None
        
    def initialize_experts(self):
        """Initialize expert models."""
        logger.info(f"Initializing {self.num_experts} expert models...")
        
        # Load expert models (can be different architectures)
        for i in range(self.num_experts):
            # For demo, using same model; in production, use different specialized models
            model_id = self.config['model_options'][self.config['selected_model']]['model_id']
            
            expert = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            
            self.experts.append(expert)
            self.tokenizers.append(tokenizer)
            
        # Initialize gating network
        hidden_size = self.experts[0].config.hidden_size
        self.gate = PrecisionGatedMoE(self.config, hidden_size)
        
        logger.info("MoE initialization complete")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Combined output from experts
        """
        # Get hidden states from first layer for gating
        # In practice, might use a separate encoder
        with torch.no_grad():
            hidden_states = self.experts[0].model.embed_tokens(input_ids)
            hidden_states = hidden_states.mean(dim=1)  # Pool for gating
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(input_ids=input_ids, attention_mask=attention_mask)
            expert_outputs.append(output.logits)
        
        # Route through gating network
        combined_output = self.gate(hidden_states, expert_outputs)
        
        return combined_output
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training step with precision-optimized routing.
        
        Args:
            batch: Training batch with input_ids, attention_mask, labels
            
        Returns:
            Loss value
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self.forward(input_ids, attention_mask)
        
        # Compute loss with precision optimization
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Add load balancing loss to ensure all experts are used
        if self.config['moe'].get('expert_capacity_factor', 1.0) > 1.0:
            # Compute load balancing loss
            expert_usage = self.gate.expert_counts / self.gate.expert_counts.sum()
            target_usage = torch.ones_like(expert_usage) / self.num_experts
            balance_loss = F.mse_loss(expert_usage, target_usage)
            loss = loss + 0.01 * balance_loss  # Small weight for balance loss
        
        return loss
    
    def evaluate_expert_precision(self, eval_dataset) -> Dict[int, float]:
        """
        Evaluate precision of each expert individually.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary mapping expert index to precision score
        """
        precisions = {}
        
        for idx, expert in enumerate(self.experts):
            # Evaluate each expert
            correct = 0
            total = 0
            
            for batch in eval_dataset:
                with torch.no_grad():
                    output = expert(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    preds = torch.argmax(output.logits, dim=-1)
                    correct += (preds == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
            
            precision = correct / total if total > 0 else 0
            precisions[idx] = precision
            
            # Update gate's precision tracking
            self.gate.update_expert_precision(idx, precision)
            
        return precisions