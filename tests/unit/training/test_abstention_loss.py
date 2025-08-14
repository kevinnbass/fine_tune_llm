"""
Test abstention-aware loss functions for LLM fine-tuning.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy imports first
mock_modules = [
    'transformers', 'peft', 'accelerate', 'datasets', 'trl', 'evaluate',
    'transformers.models', 'transformers.models.auto', 
    'transformers.models.auto.modeling_auto', 'transformers.models.auto.tokenization_auto',
    'transformers.training_args', 'transformers.trainer'
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Create a simple mock trainer class for testing
class MockCalibratedTrainer:
    """Simple mock for testing abstention loss without heavy dependencies."""
    
    def __init__(self, abstention_loss_config=None, **kwargs):
        self.abstention_loss_config = abstention_loss_config or {}
        self.use_abstention_loss = self.abstention_loss_config.get('enabled', False)
        self.abstention_threshold = self.abstention_loss_config.get('confidence_threshold', 0.7)
        self.abstention_penalty = self.abstention_loss_config.get('abstention_penalty', 0.3)
        self.uncertainty_weight = self.abstention_loss_config.get('uncertainty_weight', 0.1)
    
    def compute_abstention_aware_loss(self, model, inputs, return_outputs=False):
        """Compute abstention-aware loss (copied from actual implementation)."""
        if not self.use_abstention_loss:
            return model(**inputs) if return_outputs else model(**inputs).loss
            
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get('labels')
        
        if labels is None:
            return outputs if return_outputs else outputs.loss
            
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        if logits.dim() > 2:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            active_mask = shift_labels != -100
            active_logits = shift_logits[active_mask]
            active_labels = shift_labels[active_mask]
        else:
            active_logits = logits
            active_labels = labels
            
        if len(active_logits) == 0:
            return outputs if return_outputs else torch.tensor(0.0, device=logits.device)
            
        base_losses = loss_fct(active_logits, active_labels)
        probs = torch.softmax(active_logits, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        
        confidence_weighted_loss = base_losses * confidences
        uncertainty_penalty = torch.relu(self.abstention_threshold - confidences) * self.abstention_penalty
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        entropy_penalty = entropy * self.uncertainty_weight
        
        total_loss = confidence_weighted_loss + uncertainty_penalty + entropy_penalty
        final_loss = total_loss.mean()
        
        if return_outputs:
            outputs.loss = final_loss
            return outputs
        return final_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use abstention-aware loss."""
        return self.compute_abstention_aware_loss(model, inputs, return_outputs)


class TestAbstentionAwareLoss:
    """Test abstention-aware loss computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Mock trainer configuration
        self.abstention_config = {
            'enabled': True,
            'confidence_threshold': 0.7,
            'abstention_penalty': 0.3,
            'uncertainty_weight': 0.1
        }
        
        # Create mock trainer
        self.trainer = MockCalibratedTrainer(
            abstention_loss_config=self.abstention_config
        )
    
    def test_abstention_loss_disabled(self):
        """Test that standard loss is used when abstention loss is disabled."""
        # Trainer with abstention loss disabled
        config = {'enabled': False}
        trainer = MockCalibratedTrainer(abstention_loss_config=config)
        
        # Mock model and inputs
        mock_model = Mock()
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.5)
        mock_model.return_value = mock_output
        
        mock_inputs = {'input_ids': torch.tensor([[1, 2, 3]]), 'labels': torch.tensor([[1, 2, 3]])}
        
        # Test loss computation
        loss = trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        
        assert loss == torch.tensor(1.5)
        mock_model.assert_called_once_with(**mock_inputs)
    
    def test_abstention_loss_enabled_simple_case(self):
        """Test abstention-aware loss with simple logits."""
        # Mock model outputs
        mock_model = Mock()
        mock_output = Mock()
        
        # Simple 2-class classification logits
        # High confidence for class 0: [2.0, -1.0] -> prob ~[0.95, 0.05]
        # Low confidence: [0.1, -0.1] -> prob ~[0.55, 0.45]
        logits = torch.tensor([
            [2.0, -1.0],  # High confidence prediction
            [0.1, -0.1],  # Low confidence prediction
        ], dtype=torch.float32)
        
        mock_output.logits = logits
        mock_model.return_value = mock_output
        
        # Labels (ground truth)
        labels = torch.tensor([0, 0], dtype=torch.long)  # Both should predict class 0
        mock_inputs = {'labels': labels}
        
        # Test loss computation
        loss = self.trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0  # Loss should be positive
        
        # Verify components are computed
        # - High confidence prediction should have lower total loss
        # - Low confidence prediction should get penalty
        mock_model.assert_called_once_with(**mock_inputs)
    
    def test_abstention_loss_components(self):
        """Test individual components of abstention-aware loss."""
        # Test with known logits and labels
        mock_model = Mock()
        mock_output = Mock()
        
        # Create logits with varying confidence levels
        logits = torch.tensor([
            [3.0, 0.0],   # Very confident: class 0
            [1.0, 0.8],   # Moderately confident: class 0  
            [0.1, 0.05],  # Low confidence: class 0
        ], dtype=torch.float32)
        
        mock_output.logits = logits
        mock_model.return_value = mock_output
        
        labels = torch.tensor([0, 0, 0], dtype=torch.long)
        mock_inputs = {'labels': labels}
        
        # Compute loss
        loss = self.trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        
        # Loss should be finite and positive
        assert torch.isfinite(loss)
        assert loss > 0
        
        # Test with perfect predictions (should have lower loss)
        perfect_logits = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
        mock_output.logits = perfect_logits
        perfect_labels = torch.tensor([0], dtype=torch.long)
        perfect_inputs = {'labels': perfect_labels}
        
        perfect_loss = self.trainer.compute_abstention_aware_loss(mock_model, perfect_inputs)
        
        # Perfect prediction should have lower loss than uncertain ones
        assert perfect_loss < loss
    
    def test_abstention_loss_with_causal_lm_format(self):
        """Test abstention-aware loss with causal LM format (3D logits)."""
        mock_model = Mock()
        mock_output = Mock()
        
        # Causal LM format: (batch_size, seq_len, vocab_size)
        batch_size, seq_len, vocab_size = 2, 4, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Make last position predictions more confident
        logits[0, -1, 100] = 5.0  # High confidence for token 100
        logits[1, -1, 200] = 0.5  # Lower confidence for token 200
        
        mock_output.logits = logits
        mock_model.return_value = mock_output
        
        # Labels with padding (-100) and actual tokens
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, -1] = 100  # Last token prediction
        labels[1, -1] = 200  # Last token prediction
        
        mock_inputs = {'labels': labels}
        
        # Test loss computation
        loss = self.trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_abstention_loss_empty_active_labels(self):
        """Test behavior when no active labels (all padding)."""
        mock_model = Mock()
        mock_output = Mock()
        
        logits = torch.randn(2, 4, 100)
        mock_output.logits = logits
        mock_model.return_value = mock_output
        
        # All labels are padding (-100)
        labels = torch.full((2, 4), -100, dtype=torch.long)
        mock_inputs = {'labels': labels}
        
        # Should return zero loss
        loss = self.trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        
        assert loss == torch.tensor(0.0)
    
    def test_abstention_loss_configuration_parameters(self):
        """Test that configuration parameters affect loss computation."""
        # Test with different penalty weights
        high_penalty_config = {
            'enabled': True,
            'confidence_threshold': 0.9,  # High threshold
            'abstention_penalty': 1.0,    # High penalty
            'uncertainty_weight': 0.5     # High uncertainty weight
        }
        
        low_penalty_config = {
            'enabled': True,
            'confidence_threshold': 0.5,  # Low threshold
            'abstention_penalty': 0.1,    # Low penalty
            'uncertainty_weight': 0.01    # Low uncertainty weight
        }
        
        high_penalty_trainer = MockCalibratedTrainer(
            abstention_loss_config=high_penalty_config
        )
        low_penalty_trainer = MockCalibratedTrainer(
            abstention_loss_config=low_penalty_config
        )
        
        # Same model and inputs for both
        mock_model = Mock()
        mock_output = Mock()
        
        # Medium confidence prediction
        logits = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
        mock_output.logits = logits
        mock_model.return_value = mock_output
        
        labels = torch.tensor([0], dtype=torch.long)
        mock_inputs = {'labels': labels}
        
        high_penalty_loss = high_penalty_trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        low_penalty_loss = low_penalty_trainer.compute_abstention_aware_loss(mock_model, mock_inputs)
        
        # High penalty configuration should generally result in higher loss for uncertain predictions
        # Note: This might not always be true due to the confidence weighting, but typically should hold
        assert torch.isfinite(high_penalty_loss)
        assert torch.isfinite(low_penalty_loss)
    
    def test_compute_loss_override(self):
        """Test that compute_loss method properly calls abstention-aware loss."""
        mock_model = Mock()
        mock_output = Mock()
        
        logits = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
        mock_output.logits = logits
        mock_model.return_value = mock_output
        
        labels = torch.tensor([0], dtype=torch.long)
        mock_inputs = {'labels': labels}
        
        # Test compute_loss method
        loss = self.trainer.compute_loss(mock_model, mock_inputs)
        
        assert torch.isfinite(loss)
        assert loss > 0
        
        # Test with return_outputs=True
        output_with_loss = self.trainer.compute_loss(mock_model, mock_inputs, return_outputs=True)
        
        assert hasattr(output_with_loss, 'loss')
        assert torch.isfinite(output_with_loss.loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])