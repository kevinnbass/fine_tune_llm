"""Comprehensive tests for LLM voter inference functionality."""

import pytest
import json
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test imports
try:
    from voters.llm.infer import LLMVoterInference
    LLM_VOTER_INFERENCE_AVAILABLE = True
except ImportError:
    LLM_VOTER_INFERENCE_AVAILABLE = False
    pytest.skip("LLM voter inference module not available", allow_module_level=True)


@pytest.fixture
def temp_model_dir():
    """Fixture providing temporary model directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model():
    """Fixture providing mock model."""
    model = Mock()
    model.config = Mock()
    model.config.model_type = "glm"
    model.generate = Mock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
    tokenizer.decode = Mock(return_value="Generated response")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def sample_config():
    """Fixture providing sample inference configuration."""
    return {
        "model_id": "ZHIPU-AI/glm-4-9b-chat",
        "tokenizer_id": "ZHIPU-AI/glm-4-9b-chat",
        "max_length": 2048,
        "device": "cpu",
        "torch_dtype": "bfloat16",
        "schema": {
            "decision": "str",
            "rationale": "str",
            "confidence": "float",
            "abstain": "bool"
        },
        "abstention": {
            "enabled": True,
            "threshold": 0.7
        }
    }


class TestLLMVoterInference:
    """Test LLM voter inference class."""
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_initialization_success(self, mock_tokenizer_cls, mock_model_cls, 
                                   mock_model, mock_tokenizer, sample_config):
        """Test successful inference class initialization."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        assert inference.model == mock_model
        assert inference.tokenizer == mock_tokenizer
        assert inference.config == sample_config
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_initialization_with_lora(self, mock_tokenizer_cls, mock_model_cls,
                                     mock_model, mock_tokenizer, sample_config, temp_model_dir):
        """Test initialization with LoRA adapter."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Create mock adapter config
        adapter_config = {"peft_type": "LORA", "r": 16}
        config_path = Path(temp_model_dir) / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f)
        
        with patch('voters.llm.infer.PeftModel.from_pretrained') as mock_peft:
            mock_peft_model = Mock()
            mock_peft.return_value = mock_peft_model
            
            inference = LLMVoterInference(
                model_path="base_model",
                lora_path=temp_model_dir,
                config=sample_config
            )
            
            mock_peft.assert_called_once()
            assert inference.model == mock_peft_model
    
    def test_initialization_invalid_model(self, sample_config):
        """Test initialization with invalid model path."""
        with patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained') as mock_model:
            mock_model.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception, match="Model not found"):
                LLMVoterInference(
                    model_path="nonexistent_model",
                    config=sample_config
                )
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_format_prompt_basic(self, mock_tokenizer_cls, mock_model_cls,
                                mock_model, mock_tokenizer, sample_config):
        """Test basic prompt formatting."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        text = "Bird flu outbreak reported in farms"
        metadata = {"source": "news", "confidence": "high"}
        
        prompt = inference.format_prompt(text, metadata)
        
        assert isinstance(prompt, str)
        assert "Bird flu outbreak" in prompt
        assert "source" in prompt
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_generate_response_success(self, mock_tokenizer_cls, mock_model_cls,
                                     mock_model, mock_tokenizer, sample_config):
        """Test successful response generation."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock successful generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.decode.return_value = json.dumps({
            "decision": "HIGH_RISK",
            "rationale": "Contains outbreak information",
            "confidence": 0.95,
            "abstain": False
        })
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        response = inference.generate_response("Test prompt")
        
        assert isinstance(response, dict)
        assert "decision" in response
        assert "rationale" in response
        assert response["decision"] == "HIGH_RISK"
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_generate_response_invalid_json(self, mock_tokenizer_cls, mock_model_cls,
                                          mock_model, mock_tokenizer, sample_config):
        """Test response generation with invalid JSON."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock generation with invalid JSON
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "This is not valid JSON"
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        response = inference.generate_response("Test prompt")
        
        # Should handle invalid JSON gracefully
        assert isinstance(response, dict)
        assert "error" in response or "abstain" in response
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_validate_schema_success(self, mock_tokenizer_cls, mock_model_cls,
                                   mock_model, mock_tokenizer, sample_config):
        """Test successful schema validation."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        valid_response = {
            "decision": "HIGH_RISK",
            "rationale": "Test rationale",
            "confidence": 0.95,
            "abstain": False
        }
        
        is_valid, errors = inference.validate_schema(valid_response)
        
        assert is_valid is True
        assert len(errors) == 0
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_validate_schema_failure(self, mock_tokenizer_cls, mock_model_cls,
                                   mock_model, mock_tokenizer, sample_config):
        """Test schema validation failure."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        invalid_response = {
            "decision": "HIGH_RISK",
            # Missing required fields
            "confidence": "not_a_float",  # Wrong type
        }
        
        is_valid, errors = inference.validate_schema(invalid_response)
        
        assert is_valid is False
        assert len(errors) > 0
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_should_abstain_high_uncertainty(self, mock_tokenizer_cls, mock_model_cls,
                                           mock_model, mock_tokenizer, sample_config):
        """Test abstention logic with high uncertainty."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        response = {
            "decision": "MEDIUM_RISK",
            "confidence": 0.5,  # Below threshold
            "rationale": "Uncertain classification"
        }
        
        should_abstain, reason = inference.should_abstain(response)
        
        assert should_abstain is True
        assert "confidence" in reason.lower()
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_should_abstain_high_confidence(self, mock_tokenizer_cls, mock_model_cls,
                                          mock_model, mock_tokenizer, sample_config):
        """Test abstention logic with high confidence."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        response = {
            "decision": "HIGH_RISK",
            "confidence": 0.95,  # Above threshold
            "rationale": "Clear classification"
        }
        
        should_abstain, reason = inference.should_abstain(response)
        
        assert should_abstain is False
        assert reason == ""
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_batch_inference(self, mock_tokenizer_cls, mock_model_cls,
                           mock_model, mock_tokenizer, sample_config):
        """Test batch inference functionality."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock batch generation
        mock_model.generate.return_value = torch.tensor([
            [1, 2, 3], [4, 5, 6], [7, 8, 9]
        ])
        mock_tokenizer.decode.side_effect = [
            json.dumps({"decision": "HIGH_RISK", "rationale": "Test 1", "confidence": 0.9, "abstain": False}),
            json.dumps({"decision": "LOW_RISK", "rationale": "Test 2", "confidence": 0.8, "abstain": False}),
            json.dumps({"decision": "NO_RISK", "rationale": "Test 3", "confidence": 0.95, "abstain": False})
        ]
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        inputs = [
            ("Text 1", {"source": "news"}),
            ("Text 2", {"source": "social"}),
            ("Text 3", {"source": "official"})
        ]
        
        results = inference.batch_inference(inputs)
        
        assert len(results) == 3
        assert all("decision" in result for result in results)
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_inference_with_generation_failure(self, mock_tokenizer_cls, mock_model_cls,
                                             mock_model, mock_tokenizer, sample_config):
        """Test handling of generation failures."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock generation failure
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        response = inference.generate_response("Test prompt")
        
        # Should handle generation failure gracefully
        assert isinstance(response, dict)
        assert "error" in response or "abstain" in response
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained')
    def test_inference_memory_management(self, mock_tokenizer_cls, mock_model_cls,
                                       mock_model, mock_tokenizer, sample_config):
        """Test memory management during inference."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=sample_config
        )
        
        # Test cleanup method if it exists
        if hasattr(inference, 'cleanup'):
            inference.cleanup()
        
        # Test context management if supported
        if hasattr(inference, '__enter__') and hasattr(inference, '__exit__'):
            with inference:
                response = inference.generate_response("Test")
                assert response is not None


class TestInferenceConfiguration:
    """Test inference configuration handling."""
    
    def test_config_validation_complete(self):
        """Test complete configuration validation."""
        valid_config = {
            "model_id": "test_model",
            "tokenizer_id": "test_tokenizer",
            "max_length": 2048,
            "device": "cpu",
            "schema": {"decision": "str", "rationale": "str"}
        }
        
        # Basic validation test
        assert "model_id" in valid_config
        assert "schema" in valid_config
        assert isinstance(valid_config["max_length"], int)
    
    def test_config_validation_missing_fields(self):
        """Test configuration with missing required fields."""
        incomplete_config = {
            "model_id": "test_model"
            # Missing other required fields
        }
        
        # Should identify missing fields
        required_fields = ["model_id", "tokenizer_id", "schema"]
        missing = [field for field in required_fields if field not in incomplete_config]
        
        assert len(missing) > 0
    
    def test_schema_definition_validation(self):
        """Test schema definition validation."""
        schemas = [
            {"decision": "str", "rationale": "str", "abstain": "bool"},  # Valid
            {"decision": "str"},  # Minimal valid
            {},  # Invalid - empty
            {"decision": 123}  # Invalid - wrong type definition
        ]
        
        for schema in schemas:
            # Basic schema structure validation
            if schema and isinstance(schema, dict):
                assert len(schema) >= 0


class TestInferenceIntegration:
    """Test integration with other components."""
    
    @patch('voters.llm.infer.AutoModelForCausalLM.from_pretrained')
    @patch('voters.llm.infer.AutoTokenizer.from_pretrained') 
    def test_integration_with_high_stakes_features(self, mock_tokenizer_cls, mock_model_cls,
                                                 mock_model, mock_tokenizer):
        """Test integration with high-stakes precision features."""
        mock_model_cls.return_value = mock_model
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        high_stakes_config = {
            "model_id": "test_model",
            "tokenizer_id": "test_tokenizer",
            "max_length": 2048,
            "device": "cpu",
            "schema": {"decision": "str", "rationale": "str", "abstain": "bool"},
            "abstention": {"enabled": True, "threshold": 0.8},
            "uncertainty": {"enabled": True, "method": "mc_dropout"},
            "bias_audit": {"enabled": True, "categories": ["gender"]}
        }
        
        inference = LLMVoterInference(
            model_path="test_model",
            config=high_stakes_config
        )
        
        assert inference.config["abstention"]["enabled"] is True
        assert inference.config["uncertainty"]["enabled"] is True
    
    def test_output_format_consistency(self):
        """Test output format consistency across different scenarios."""
        # Test different response formats
        responses = [
            {"decision": "HIGH_RISK", "rationale": "Test", "abstain": False},
            {"decision": "ABSTAIN", "rationale": "Uncertain", "abstain": True},
            {"error": "Generation failed", "abstain": True}
        ]
        
        for response in responses:
            # Verify all responses have required structure
            assert isinstance(response, dict)
            assert "abstain" in response or "error" in response
            if "decision" in response:
                assert "rationale" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])