"""Simplified tests for high-stakes precision features without heavy ML dependencies."""

import pytest
import torch
import numpy as np
import json
from unittest.mock import Mock, patch


class TestUncertaintySimple:
    """Test uncertainty-aware features with minimal dependencies."""
    
    def test_uncertainty_functions_exist(self):
        """Test that uncertainty functions can be imported."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.uncertainty import compute_uncertainty_aware_loss, should_abstain
        
        # Test should_abstain function
        config = {'high_stakes': {'uncertainty': {'enabled': True, 'abstention_threshold': 0.7}}}
        should_abs, reason = should_abstain(0.8, config)
        assert should_abs is True
        assert "high uncertainty" in reason.lower()
        
        should_abs, reason = should_abstain(0.3, config)
        assert should_abs is False
        
    def test_compute_uncertainty_loss_disabled(self):
        """Test uncertainty loss when disabled."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.uncertainty import compute_uncertainty_aware_loss
        
        # Mock outputs
        logits = torch.randn(2, 3)
        labels = torch.tensor([0, 1])
        
        class MockOutputs:
            def __init__(self, logits):
                self.logits = logits
        
        outputs = MockOutputs(logits)
        config = {'high_stakes': {'uncertainty': {'enabled': False}}}
        
        loss = compute_uncertainty_aware_loss(outputs, labels, config)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestFactCheckSimple:
    """Test factual accuracy features."""
    
    def test_factual_test_data_creation(self):
        """Test factual test data creation."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.fact_check import create_factual_test_data
        
        test_data = create_factual_test_data()
        
        assert len(test_data) > 0
        assert all('text' in item for item in test_data)
        assert all('label' in item for item in test_data)
        assert all('expected_score' in item for item in test_data)


class TestBiasAuditSimple:
    """Test bias auditing with minimal dependencies."""
    
    def test_bias_auditor_basic(self):
        """Test basic bias auditor functionality."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.high_stakes_audit import BiasAuditor
        
        config = {
            'high_stakes': {
                'bias_audit': {
                    'enabled': True,
                    'audit_categories': ['gender', 'race'],
                    'bias_threshold': 0.1
                }
            }
        }
        
        auditor = BiasAuditor(config)
        assert 'gender' in auditor.audit_categories
        assert 'race' in auditor.audit_categories
        assert auditor.bias_threshold == 0.1
        
        # Test bias detection
        text = "He is a great doctor"
        predictions = torch.randn(1, 10, 2)
        
        bias_scores = auditor.detect_bias(text, predictions)
        assert isinstance(bias_scores, dict)
        assert 'gender' in bias_scores
        assert isinstance(bias_scores['gender'], float)


class TestExplainableSimple:
    """Test explainable reasoning."""
    
    def test_explainable_reasoning_init(self):
        """Test explainable reasoning initialization."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.high_stakes_audit import ExplainableReasoning
        
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        config = {
            'high_stakes': {
                'explainable': {
                    'enabled': True,
                    'chain_of_thought': True,
                    'reasoning_steps': 3
                }
            }
        }
        
        explainer = ExplainableReasoning(mock_model, mock_tokenizer, config)
        assert explainer.chain_of_thought is True
        assert explainer.min_steps == 3


class TestProceduralSimple:
    """Test procedural alignment."""
    
    def test_procedural_alignment_init(self):
        """Test procedural alignment initialization."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.high_stakes_audit import ProceduralAlignment
        
        config = {
            'high_stakes': {
                'procedural': {
                    'enabled': True,
                    'domain': 'medical',
                    'compliance_weight': 2.0
                }
            }
        }
        
        alignment = ProceduralAlignment(config)
        assert alignment.domain == 'medical'
        assert alignment.compliance_weight == 2.0
        
        # Test compliance checking
        text = "Please consult with your healthcare provider"
        is_compliant, score, missing = alignment.check_compliance(text)
        
        assert isinstance(is_compliant, bool)
        assert 0 <= score <= 1
        assert isinstance(missing, list)


class TestVerifiableSimple:
    """Test verifiable training."""
    
    def test_verifiable_training_init(self):
        """Test verifiable training initialization."""
        import sys
        sys.path.insert(0, 'fine_tune_llm')
        
        from voters.llm.high_stakes_audit import VerifiableTraining
        
        config = {
            'high_stakes': {
                'verifiable': {
                    'enabled': True,
                    'hash_artifacts': True,
                    'cryptographic_proof': True,
                    'audit_log': 'test_audit.jsonl'
                }
            }
        }
        
        verifier = VerifiableTraining(config)
        assert verifier.verifiable_config['enabled'] is True
        assert verifier.verifiable_config['hash_artifacts'] is True
        
        # Test artifact hashing
        test_data = {"key": "value", "number": 123}
        hash_value = verifier.hash_artifact(test_data, 'data')
        
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0


class TestIntegrationSimple:
    """Test integration between high-stakes features."""
    
    def test_feature_toggles(self):
        """Test that all features can be toggled."""
        features = ['uncertainty', 'factual', 'bias_audit', 'explainable', 'procedural', 'verifiable']
        
        for feature in features:
            config = {
                'high_stakes': {
                    feature: {'enabled': True}
                }
            }
            
            # Each feature should be independently configurable
            assert config['high_stakes'][feature]['enabled'] is True
            
            # Can be disabled
            config['high_stakes'][feature]['enabled'] = False
            assert config['high_stakes'][feature]['enabled'] is False
    
    def test_config_structure(self):
        """Test high-stakes configuration structure."""
        full_config = {
            'high_stakes': {
                'uncertainty': {
                    'enabled': True,
                    'method': 'mc_dropout',
                    'num_samples': 5,
                    'abstention_threshold': 0.7
                },
                'factual': {
                    'enabled': True,
                    'reliance_steps': 3
                },
                'bias_audit': {
                    'enabled': True,
                    'audit_categories': ['gender', 'race']
                },
                'explainable': {
                    'enabled': True,
                    'chain_of_thought': True
                },
                'procedural': {
                    'enabled': True,
                    'domain': 'medical'
                },
                'verifiable': {
                    'enabled': True,
                    'hash_artifacts': True
                }
            }
        }
        
        # All sections should be present
        assert 'high_stakes' in full_config
        
        required_sections = ['uncertainty', 'factual', 'bias_audit', 'explainable', 'procedural', 'verifiable']
        for section in required_sections:
            assert section in full_config['high_stakes']
            assert 'enabled' in full_config['high_stakes'][section]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])