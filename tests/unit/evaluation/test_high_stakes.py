"""Tests for high-stakes precision and auditability features."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
import yaml

# Test imports
try:
    from voters.llm.uncertainty import MCDropoutWrapper, compute_uncertainty_aware_loss, should_abstain
    from voters.llm.fact_check import RELIANCEFactChecker, create_factual_test_data
    from voters.llm.high_stakes_audit import BiasAuditor, ExplainableReasoning, ProceduralAlignment, VerifiableTraining
    HIGH_STAKES_AVAILABLE = True
except ImportError:
    HIGH_STAKES_AVAILABLE = False
    pytest.skip("High-stakes modules not available", allow_module_level=True)

from transformers import AutoModelForCausalLM, AutoTokenizer


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    def __init__(self, vocab_size=1000, hidden_size=768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, self.hidden_size)
        logits = self.linear(hidden)
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1)
            )
        
        class MockOutput:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss
        
        return MockOutput(logits, loss)
    
    def generate(self, input_ids, **kwargs):
        # Simple generation for testing
        batch_size = input_ids.size(0)
        max_new_tokens = kwargs.get('max_new_tokens', 10)
        
        generated = torch.randint(0, self.vocab_size, (batch_size, max_new_tokens))
        full_sequence = torch.cat([input_ids, generated], dim=1)
        
        class GenerateOutput:
            def __init__(self, sequences):
                self.sequences = sequences
        
        return GenerateOutput(full_sequence)


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.vocab_size = 1000
        
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            # Simple tokenization
            tokens = torch.randint(0, self.vocab_size, (1, 10))
        else:
            tokens = torch.randint(0, self.vocab_size, (len(text), 10))
            
        return {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens)}
    
    def decode(self, tokens, **kwargs):
        return "Generated text response"


@pytest.fixture
def mock_model():
    """Fixture providing mock model."""
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        'high_stakes': {
            'uncertainty': {
                'enabled': True,
                'method': 'mc_dropout',
                'num_samples': 3,
                'abstention_threshold': 0.7,
                'fp_penalty_weight': 2.0
            },
            'factual': {
                'enabled': True,
                'reliance_steps': 2,
                'fact_penalty_weight': 2.0,
                'self_consistency_threshold': 0.8
            },
            'bias_audit': {
                'enabled': True,
                'audit_categories': ['gender', 'race'],
                'bias_threshold': 0.1,
                'mitigation_weight': 1.5
            },
            'explainable': {
                'enabled': True,
                'chain_of_thought': True,
                'reasoning_steps': 2,
                'faithfulness_check': True
            },
            'procedural': {
                'enabled': True,
                'domain': 'medical',
                'compliance_weight': 2.0
            },
            'verifiable': {
                'enabled': True,
                'hash_artifacts': True,
                'cryptographic_proof': True,
                'audit_log': 'test_audit.jsonl'
            }
        }
    }


class TestUncertainty:
    """Test uncertainty-aware features."""
    
    def test_mc_dropout_wrapper(self, mock_model):
        """Test MC Dropout wrapper."""
        wrapper = MCDropoutWrapper(mock_model, num_samples=3)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        
        # Set wrapper to eval mode to enable MC sampling
        wrapper.eval()
        
        # Test forward pass
        outputs = wrapper(input_ids, attention_mask)
        
        assert hasattr(outputs, 'logits')
        assert hasattr(outputs, 'uncertainty')
        assert outputs.logits.shape == (2, 10, 1000)
    
    def test_uncertainty_loss(self, test_config):
        """Test uncertainty-aware loss computation."""
        # Mock outputs with uncertainty
        class MockUncertainOutput:
            def __init__(self):
                self.logits = torch.randn(2, 10, 1000)
                self.uncertainty = torch.rand(2, 10)  # Random uncertainty
        
        outputs = MockUncertainOutput()
        labels = torch.randint(0, 2, (2, 10))  # Binary labels
        
        loss = compute_uncertainty_aware_loss(outputs, labels, test_config)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_abstention_logic(self, test_config):
        """Test abstention logic."""
        # High uncertainty should trigger abstention
        should_abstain_result, reason = should_abstain(0.8, test_config)
        assert should_abstain_result is True
        assert "high uncertainty" in reason.lower()
        
        # Low uncertainty should not trigger abstention
        should_abstain_result, reason = should_abstain(0.3, test_config)
        assert should_abstain_result is False


class TestFactualAccuracy:
    """Test factual accuracy features."""
    
    def test_fact_checker_initialization(self, mock_model, mock_tokenizer, test_config):
        """Test RELIANCE fact checker initialization."""
        fact_checker = RELIANCEFactChecker(mock_model, mock_tokenizer, test_config)
        
        assert fact_checker.reliance_steps == 2
        assert fact_checker.consistency_threshold == 0.8
    
    def test_reasoning_step_splitting(self, mock_model, mock_tokenizer, test_config):
        """Test reasoning step extraction."""
        fact_checker = RELIANCEFactChecker(mock_model, mock_tokenizer, test_config)
        
        text = "First, we analyze the symptoms. Second, we check the test results. Therefore, the diagnosis is clear."
        steps = fact_checker.split_into_reasoning_steps(text, 3)
        
        assert len(steps) > 0
        assert any("symptoms" in step for step in steps)
    
    def test_factual_test_data(self):
        """Test factual test data creation."""
        test_data = create_factual_test_data()
        
        assert len(test_data) > 0
        assert all('text' in item for item in test_data)
        assert all('label' in item for item in test_data)
        assert all('expected_score' in item for item in test_data)


class TestBiasAuditing:
    """Test bias auditing features."""
    
    def test_bias_auditor_initialization(self, test_config):
        """Test bias auditor initialization."""
        auditor = BiasAuditor(test_config)
        
        assert 'gender' in auditor.audit_categories
        assert 'race' in auditor.audit_categories
        assert auditor.bias_threshold == 0.1
    
    def test_gender_bias_detection(self, test_config):
        """Test gender bias detection."""
        auditor = BiasAuditor(test_config)
        
        # Text with gender imbalance
        biased_text = "He is a great doctor. His expertise is excellent."
        predictions = torch.randn(1, 10, 2)
        
        bias_scores = auditor.detect_bias(biased_text, predictions)
        
        assert 'gender' in bias_scores
        assert bias_scores['gender'] >= 0
    
    def test_bias_mitigation_loss(self, test_config):
        """Test bias mitigation loss."""
        auditor = BiasAuditor(test_config)
        
        text = "The man is very qualified for this position."
        predictions = torch.randn(1, 10, 2)
        
        loss = auditor.compute_bias_mitigation_loss(text, predictions)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestExplainableReasoning:
    """Test explainable reasoning features."""
    
    def test_explainable_reasoning_init(self, mock_model, mock_tokenizer, test_config):
        """Test explainable reasoning initialization."""
        explainer = ExplainableReasoning(mock_model, mock_tokenizer, test_config)
        
        assert explainer.chain_of_thought is True
        assert explainer.min_steps == 2
        assert explainer.faithfulness_check is True
    
    def test_step_extraction(self, mock_model, mock_tokenizer, test_config):
        """Test reasoning step extraction."""
        explainer = ExplainableReasoning(mock_model, mock_tokenizer, test_config)
        
        text = "Step 1: Analyze the data. Step 2: Draw conclusions."
        steps = explainer._extract_reasoning_steps(text)
        
        assert len(steps) > 0


class TestProceduralAlignment:
    """Test procedural alignment features."""
    
    def test_procedural_alignment_init(self, test_config):
        """Test procedural alignment initialization."""
        alignment = ProceduralAlignment(test_config)
        
        assert alignment.domain == 'medical'
        assert alignment.compliance_weight == 2.0
    
    def test_compliance_checking(self, test_config):
        """Test compliance checking."""
        alignment = ProceduralAlignment(test_config)
        
        # Text with medical disclaimers
        compliant_text = "Please consult with your healthcare provider before taking any medication."
        is_compliant, score, missing = alignment.check_compliance(compliant_text)
        
        assert isinstance(is_compliant, bool)
        assert 0 <= score <= 1


class TestVerifiableTraining:
    """Test verifiable training features."""
    
    def test_verifiable_training_init(self, test_config):
        """Test verifiable training initialization."""
        verifier = VerifiableTraining(test_config)
        
        assert verifier.verifiable_config['enabled'] is True
        assert verifier.verifiable_config['hash_artifacts'] is True
    
    def test_artifact_hashing(self, test_config):
        """Test artifact hashing."""
        verifier = VerifiableTraining(test_config)
        
        # Test data hashing
        test_data = {"key": "value", "number": 123}
        hash_value = verifier.hash_artifact(test_data, 'data')
        
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0
    
    def test_training_proof(self, test_config, mock_model):
        """Test training proof creation."""
        verifier = VerifiableTraining(test_config)
        
        train_data = [{"text": "test", "label": "test"}]
        config = {"param": "value"}
        
        proof = verifier.create_training_proof(mock_model, train_data, config)
        
        assert 'timestamp' in proof
        assert 'model_hash' in proof
        assert 'data_hash' in proof
        assert 'config_hash' in proof


class TestIntegration:
    """Integration tests."""
    
    def test_high_stakes_config_loading(self, test_config):
        """Test that high-stakes config loads properly."""
        # Test that all required sections exist
        assert 'high_stakes' in test_config
        
        high_stakes = test_config['high_stakes']
        required_sections = ['uncertainty', 'factual', 'bias_audit', 'explainable', 'procedural', 'verifiable']
        
        for section in required_sections:
            assert section in high_stakes
            assert 'enabled' in high_stakes[section]
    
    def test_feature_interaction(self, mock_model, mock_tokenizer, test_config):
        """Test interaction between multiple features."""
        # Initialize multiple components
        fact_checker = RELIANCEFactChecker(mock_model, mock_tokenizer, test_config)
        bias_auditor = BiasAuditor(test_config)
        
        # Test text
        text = "The patient should consult a doctor for proper medical advice."
        predictions = torch.randn(1, 10, 2)
        
        # Test factual accuracy
        is_factual, fact_score, _ = fact_checker.reliance_fact_check(text)
        
        # Test bias detection
        bias_scores = bias_auditor.detect_bias(text, predictions)
        
        # Both should work without interference
        assert isinstance(is_factual, bool)
        assert isinstance(fact_score, float)
        assert isinstance(bias_scores, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])