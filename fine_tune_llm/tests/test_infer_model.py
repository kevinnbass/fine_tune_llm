"""Comprehensive tests for inference pipeline functionality."""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Test imports - Handle both script and module imports
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    
    from infer_model import (
        load_config, load_model, format_prompt, generate_response,
        parse_json_response, main
    )
    INFER_AVAILABLE = True
except ImportError:
    INFER_AVAILABLE = False
    pytest.skip("Inference script not available", allow_module_level=True)


class MockModel:
    """Mock model for testing."""
    def __init__(self):
        self.device = "cpu"
        
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Mock generation."""
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Generate mock tokens
        new_tokens = torch.randint(1, 1000, (batch_size, 20))
        generated = torch.cat([input_ids, new_tokens], dim=1)
        
        class GenerateOutput:
            def __init__(self, sequences):
                self.sequences = sequences
        
        return GenerateOutput(generated)
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.vocab_size = 32000
        
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        
        # Mock tokenization
        input_ids = torch.randint(1, 1000, (len(text), 50))
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, tokens, skip_special_tokens=True):
        """Mock decoding."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        if isinstance(tokens[0], list):
            # Batch decode
            return [self._mock_decode_single(token_list) for token_list in tokens]
        else:
            # Single decode
            return self._mock_decode_single(tokens)
    
    def _mock_decode_single(self, tokens):
        """Mock single sequence decode."""
        # Return mock response based on input length
        if len(tokens) > 100:
            return '{"decision": "relevant", "rationale": "This appears to be relevant content.", "confidence": 0.85, "abstain": false}'
        else:
            return "This is a test response."


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        "model": {
            "base_model": "test/model",
            "lora_adapter": "test/adapter",
            "device": "cpu",
            "torch_dtype": "float32"
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        "structured_output": {
            "enabled": False,
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"}
                }
            }
        },
        "prompt_template": "Text: {text}\n\nResponse:"
    }


@pytest.fixture
def temp_config_file(test_config):
    """Fixture providing temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def mock_model():
    """Fixture providing mock model."""
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return MockTokenizer()


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_config_success(self, temp_config_file, test_config):
        """Test successful config loading."""
        config = load_config(temp_config_file)
        
        assert config == test_config
        assert "model" in config
        assert "generation" in config
    
    def test_load_config_missing_file(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()
            
            with pytest.raises(yaml.YAMLError):
                load_config(f.name)
            
            Path(f.name).unlink()
    
    def test_load_config_empty_file(self):
        """Test handling of empty config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            
            config = load_config(f.name)
            assert config is None or config == {}
            
            Path(f.name).unlink()
    
    def test_load_config_partial_config(self):
        """Test handling of partial configuration."""
        partial_config = {"model": {"base_model": "test"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(partial_config, f)
            f.flush()
            
            config = load_config(f.name)
            assert "model" in config
            assert config["model"]["base_model"] == "test"
            
            Path(f.name).unlink()


class TestLoadModel:
    """Test model loading functionality."""
    
    @patch('infer_model.AutoModelForCausalLM')
    @patch('infer_model.AutoTokenizer')
    def test_load_model_base_only(self, mock_tokenizer_class, mock_model_class, test_config):
        """Test loading base model without LoRA."""
        mock_model_class.from_pretrained.return_value = MockModel()
        mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
        
        # Remove LoRA adapter
        config_no_lora = test_config.copy()
        del config_no_lora["model"]["lora_adapter"]
        
        model, tokenizer = load_model(config_no_lora)
        
        assert isinstance(model, MockModel)
        assert isinstance(tokenizer, MockTokenizer)
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
    
    @patch('infer_model.AutoModelForCausalLM')
    @patch('infer_model.AutoTokenizer')
    @patch('infer_model.PeftModel')
    def test_load_model_with_lora(self, mock_peft_class, mock_tokenizer_class, 
                                  mock_model_class, test_config):
        """Test loading model with LoRA adapter."""
        mock_base_model = MockModel()
        mock_model_class.from_pretrained.return_value = mock_base_model
        mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
        mock_peft_class.from_pretrained.return_value = MockModel()
        
        model, tokenizer = load_model(test_config)
        
        assert isinstance(tokenizer, MockTokenizer)
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_peft_class.from_pretrained.assert_called_once()
    
    @patch('infer_model.AutoModelForCausalLM')
    @patch('infer_model.AutoTokenizer')
    def test_load_model_gpu_device(self, mock_tokenizer_class, mock_model_class, test_config):
        """Test loading model on GPU device."""
        test_config["model"]["device"] = "cuda"
        
        mock_model = MockModel()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
        
        model, tokenizer = load_model(test_config)
        
        assert model.device == "cuda"
    
    @patch('infer_model.AutoModelForCausalLM')
    def test_load_model_invalid_base_model(self, mock_model_class, test_config):
        """Test handling of invalid base model."""
        mock_model_class.from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Model not found"):
            load_model(test_config)
    
    @patch('infer_model.AutoModelForCausalLM')
    @patch('infer_model.AutoTokenizer')
    @patch('infer_model.PeftModel')
    def test_load_model_invalid_lora_adapter(self, mock_peft_class, mock_tokenizer_class,
                                           mock_model_class, test_config):
        """Test handling of invalid LoRA adapter."""
        mock_model_class.from_pretrained.return_value = MockModel()
        mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
        mock_peft_class.from_pretrained.side_effect = Exception("Adapter not found")
        
        with pytest.raises(Exception, match="Adapter not found"):
            load_model(test_config)


class TestFormatPrompt:
    """Test prompt formatting functionality."""
    
    def test_format_prompt_basic(self, test_config):
        """Test basic prompt formatting."""
        text = "What is machine learning?"
        
        prompt = format_prompt(text, test_config)
        
        assert text in prompt
        assert "Text:" in prompt
        assert "Response:" in prompt
    
    def test_format_prompt_with_metadata(self, test_config):
        """Test prompt formatting with metadata."""
        text = "What is machine learning?"
        metadata = {"source": "user", "timestamp": "2024-01-01"}
        
        # Add metadata to template
        test_config["prompt_template"] = "Text: {text}\nMetadata: {metadata}\n\nResponse:"
        
        prompt = format_prompt(text, test_config, metadata=metadata)
        
        assert text in prompt
        assert str(metadata) in prompt
    
    def test_format_prompt_custom_template(self, test_config):
        """Test prompt formatting with custom template."""
        text = "What is machine learning?"
        test_config["prompt_template"] = "Question: {text}\nAnswer:"
        
        prompt = format_prompt(text, test_config)
        
        assert "Question:" in prompt
        assert "Answer:" in prompt
        assert text in prompt
    
    def test_format_prompt_empty_text(self, test_config):
        """Test prompt formatting with empty text."""
        text = ""
        
        prompt = format_prompt(text, test_config)
        
        assert prompt is not None
        assert len(prompt) > 0
    
    def test_format_prompt_special_characters(self, test_config):
        """Test prompt formatting with special characters."""
        text = "Text with \"quotes\" and 'apostrophes' & symbols"
        
        prompt = format_prompt(text, test_config)
        
        assert text in prompt
        assert "\"" in prompt
        assert "'" in prompt
        assert "&" in prompt
    
    def test_format_prompt_unicode(self, test_config):
        """Test prompt formatting with Unicode characters."""
        text = "测试中文 🦠 émoji Arabic: العربية"
        
        prompt = format_prompt(text, test_config)
        
        assert text in prompt
        assert "测试中文" in prompt
        assert "🦠" in prompt
    
    def test_format_prompt_very_long_text(self, test_config):
        """Test prompt formatting with very long text."""
        text = "This is a very long text. " * 1000  # ~25k characters
        
        prompt = format_prompt(text, test_config)
        
        assert prompt is not None
        assert len(prompt) > len(text)  # Should include template


class TestGenerateResponse:
    """Test response generation functionality."""
    
    def test_generate_response_basic(self, test_config, mock_model, mock_tokenizer):
        """Test basic response generation."""
        text = "What is machine learning?"
        
        response = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_response_structured(self, test_config, mock_model, mock_tokenizer):
        """Test structured response generation."""
        test_config["structured_output"]["enabled"] = True
        
        text = "What is machine learning?"
        
        with patch('infer_model.generate_structured') as mock_structured:
            mock_structured.return_value = '{"decision": "relevant", "confidence": 0.8}'
            
            response = generate_response(text, mock_model, mock_tokenizer, test_config)
            
            mock_structured.assert_called_once()
            assert isinstance(response, str)
    
    def test_generate_response_with_metadata(self, test_config, mock_model, mock_tokenizer):
        """Test response generation with metadata."""
        text = "What is machine learning?"
        metadata = {"source": "user"}
        
        response = generate_response(text, mock_model, mock_tokenizer, test_config, metadata=metadata)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_response_custom_generation_params(self, test_config, mock_model, mock_tokenizer):
        """Test response generation with custom parameters."""
        text = "What is machine learning?"
        
        test_config["generation"]["temperature"] = 0.1
        test_config["generation"]["max_new_tokens"] = 50
        
        response = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_response_long_input(self, test_config, mock_model, mock_tokenizer):
        """Test response generation with long input."""
        text = "This is a very long input text. " * 500  # ~15k characters
        
        response = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        assert isinstance(response, str)
        # Should handle long input gracefully
    
    def test_generate_response_empty_input(self, test_config, mock_model, mock_tokenizer):
        """Test response generation with empty input."""
        text = ""
        
        response = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        assert isinstance(response, str)
        # Should handle empty input gracefully
    
    def test_generate_response_generation_failure(self, test_config, mock_model, mock_tokenizer):
        """Test handling of generation failures."""
        text = "What is machine learning?"
        
        # Mock generation failure
        mock_model.generate = Mock(side_effect=Exception("Generation failed"))
        
        with pytest.raises(Exception, match="Generation failed"):
            generate_response(text, mock_model, mock_tokenizer, test_config)


class TestParseJsonResponse:
    """Test JSON response parsing functionality."""
    
    def test_parse_json_response_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"decision": "relevant", "confidence": 0.85, "rationale": "This is relevant"}'
        
        parsed = parse_json_response(response)
        
        assert parsed is not None
        assert parsed["decision"] == "relevant"
        assert parsed["confidence"] == 0.85
        assert parsed["rationale"] == "This is relevant"
    
    def test_parse_json_response_json_in_text(self):
        """Test parsing JSON embedded in text."""
        response = 'Here is my analysis: {"decision": "relevant", "confidence": 0.85} That is my conclusion.'
        
        parsed = parse_json_response(response)
        
        assert parsed is not None
        assert parsed["decision"] == "relevant"
        assert parsed["confidence"] == 0.85
    
    def test_parse_json_response_multiple_json_objects(self):
        """Test parsing response with multiple JSON objects."""
        response = 'First: {"decision": "relevant"} Second: {"confidence": 0.85}'
        
        parsed = parse_json_response(response)
        
        # Should return the first valid JSON object
        assert parsed is not None
        assert parsed["decision"] == "relevant"
    
    def test_parse_json_response_invalid_json(self):
        """Test parsing invalid JSON."""
        response = "This is not JSON at all"
        
        parsed = parse_json_response(response)
        
        assert parsed is None
    
    def test_parse_json_response_malformed_json(self):
        """Test parsing malformed JSON."""
        response = '{"decision": "relevant", "confidence": 0.85'  # Missing closing brace
        
        parsed = parse_json_response(response)
        
        assert parsed is None
    
    def test_parse_json_response_empty_string(self):
        """Test parsing empty string."""
        response = ""
        
        parsed = parse_json_response(response)
        
        assert parsed is None
    
    def test_parse_json_response_none_input(self):
        """Test parsing None input."""
        parsed = parse_json_response(None)
        
        assert parsed is None
    
    def test_parse_json_response_json_with_newlines(self):
        """Test parsing JSON with newlines."""
        response = '''
        {
            "decision": "relevant",
            "confidence": 0.85,
            "rationale": "This is relevant because it discusses the topic."
        }
        '''
        
        parsed = parse_json_response(response)
        
        assert parsed is not None
        assert parsed["decision"] == "relevant"
        assert parsed["confidence"] == 0.85
    
    def test_parse_json_response_json_with_escaped_quotes(self):
        """Test parsing JSON with escaped quotes."""
        response = '{"decision": "relevant", "rationale": "This is \\"very\\" relevant"}'
        
        parsed = parse_json_response(response)
        
        assert parsed is not None
        assert parsed["decision"] == "relevant"
        assert "very" in parsed["rationale"]


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('infer_model.AutoModelForCausalLM')
    @patch('infer_model.AutoTokenizer')
    def test_memory_constraints(self, mock_tokenizer_class, mock_model_class, test_config):
        """Test handling of memory constraints."""
        # Simulate OOM error
        mock_model_class.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            load_model(test_config)
    
    def test_device_switching(self, test_config, mock_model, mock_tokenizer):
        """Test switching between CPU and GPU devices."""
        text = "Test text"
        
        # Test CPU
        test_config["model"]["device"] = "cpu"
        response_cpu = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        # Test GPU (mock)
        test_config["model"]["device"] = "cuda"
        response_gpu = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        assert isinstance(response_cpu, str)
        assert isinstance(response_gpu, str)
    
    def test_concurrent_inference(self, test_config, mock_model, mock_tokenizer):
        """Test concurrent inference requests."""
        import threading
        
        results = []
        errors = []
        
        def run_inference(text):
            try:
                response = generate_response(f"Text {text}", mock_model, mock_tokenizer, test_config)
                results.append(response)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_inference, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        assert all(isinstance(result, str) for result in results)


class TestIntegration:
    """Test integration functionality."""
    
    @patch('infer_model.AutoModelForCausalLM')
    @patch('infer_model.AutoTokenizer')
    def test_end_to_end_inference(self, mock_tokenizer_class, mock_model_class, 
                                  temp_config_file, test_config):
        """Test end-to-end inference pipeline."""
        mock_model_class.from_pretrained.return_value = MockModel()
        mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
        
        # Load config
        config = load_config(temp_config_file)
        assert config is not None
        
        # Load model
        model, tokenizer = load_model(config)
        assert model is not None
        assert tokenizer is not None
        
        # Generate response
        text = "What is machine learning?"
        response = generate_response(text, model, tokenizer, config)
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Try to parse JSON if it looks like JSON
        if response.strip().startswith("{"):
            parsed = parse_json_response(response)
            if parsed:
                assert isinstance(parsed, dict)
    
    def test_structured_output_integration(self, test_config, mock_model, mock_tokenizer):
        """Test structured output integration."""
        test_config["structured_output"]["enabled"] = True
        test_config["structured_output"]["schema"] = {
            "type": "object",
            "properties": {
                "decision": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["decision", "confidence"]
        }
        
        text = "What is machine learning?"
        
        with patch('infer_model.generate_structured') as mock_structured:
            mock_structured.return_value = '{"decision": "relevant", "confidence": 0.8}'
            
            response = generate_response(text, mock_model, mock_tokenizer, test_config)
            parsed = parse_json_response(response)
            
            assert parsed is not None
            assert "decision" in parsed
            assert "confidence" in parsed
            assert isinstance(parsed["confidence"], (int, float))
    
    @patch('sys.argv', ['infer_model.py', '--text', 'test text', '--config', 'test_config.yaml'])
    @patch('infer_model.load_config')
    @patch('infer_model.load_model')
    @patch('infer_model.generate_response')
    def test_main_function_basic(self, mock_generate, mock_load_model, mock_load_config):
        """Test main function with basic arguments."""
        mock_load_config.return_value = {"test": "config"}
        mock_load_model.return_value = (MockModel(), MockTokenizer())
        mock_generate.return_value = "Test response"
        
        # Mock argparse
        with patch('infer_model.argparse.ArgumentParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_args = Mock()
            mock_args.text = "test text"
            mock_args.config = "test_config.yaml"
            mock_args.metadata = None
            mock_args.output_file = None
            mock_parser.parse_args.return_value = mock_args
            
            # Should not raise exception
            try:
                main()
            except SystemExit:
                pass  # main() might call sys.exit()


class TestModularity:
    """Test modularity and configuration toggles."""
    
    def test_structured_output_toggle(self, test_config, mock_model, mock_tokenizer):
        """Test structured output toggle."""
        text = "What is machine learning?"
        
        # Test disabled
        test_config["structured_output"]["enabled"] = False
        response_regular = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        # Test enabled
        test_config["structured_output"]["enabled"] = True
        
        with patch('infer_model.generate_structured') as mock_structured:
            mock_structured.return_value = '{"test": "structured"}'
            response_structured = generate_response(text, mock_model, mock_tokenizer, test_config)
            mock_structured.assert_called_once()
        
        # Both should return strings but might be different
        assert isinstance(response_regular, str)
        assert isinstance(response_structured, str)
    
    def test_device_configuration(self, test_config):
        """Test device configuration options."""
        # Test CPU
        test_config["model"]["device"] = "cpu"
        assert test_config["model"]["device"] == "cpu"
        
        # Test GPU
        test_config["model"]["device"] = "cuda"
        assert test_config["model"]["device"] == "cuda"
        
        # Test specific GPU
        test_config["model"]["device"] = "cuda:1"
        assert test_config["model"]["device"] == "cuda:1"
    
    def test_generation_parameters_toggle(self, test_config, mock_model, mock_tokenizer):
        """Test generation parameters configuration."""
        text = "What is machine learning?"
        
        # Test with sampling
        test_config["generation"]["do_sample"] = True
        test_config["generation"]["temperature"] = 0.8
        
        response_sampling = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        # Test without sampling
        test_config["generation"]["do_sample"] = False
        test_config["generation"]["temperature"] = 0.0
        
        response_deterministic = generate_response(text, mock_model, mock_tokenizer, test_config)
        
        assert isinstance(response_sampling, str)
        assert isinstance(response_deterministic, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])