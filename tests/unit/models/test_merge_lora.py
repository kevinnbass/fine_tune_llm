"""Comprehensive tests for LoRA merging script functionality."""

import pytest
import torch
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Test imports - Handle both script and module imports
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    
    from merge_lora import (
        load_base_model, load_lora_adapter, merge_lora_weights,
        save_merged_model, validate_merge, create_comparison_report
    )
    MERGE_LORA_AVAILABLE = True
except ImportError:
    MERGE_LORA_AVAILABLE = False
    pytest.skip("Merge LoRA script not available", allow_module_level=True)


@pytest.fixture
def temp_model_dir():
    """Fixture providing temporary model directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_base_model():
    """Fixture providing mock base model."""
    model = Mock()
    model.config = Mock()
    model.config.model_type = "glm"
    model.config.vocab_size = 32000
    model.config.hidden_size = 4096
    
    # Mock named parameters
    param_dict = {
        'transformer.embedding.word_embeddings.weight': torch.randn(32000, 4096),
        'transformer.encoder.layers.0.self_attention.query_key_value.weight': torch.randn(4096, 12288),
        'transformer.encoder.layers.0.self_attention.dense.weight': torch.randn(4096, 4096),
        'lm_head.weight': torch.randn(32000, 4096)
    }
    
    model.named_parameters.return_value = param_dict.items()
    model.state_dict.return_value = param_dict
    model.load_state_dict = Mock()
    
    return model


@pytest.fixture
def mock_lora_adapter():
    """Fixture providing mock LoRA adapter."""
    adapter = Mock()
    
    # Mock LoRA weights
    lora_dict = {
        'transformer.encoder.layers.0.self_attention.query_key_value.lora_A': torch.randn(16, 4096),
        'transformer.encoder.layers.0.self_attention.query_key_value.lora_B': torch.randn(12288, 16),
        'transformer.encoder.layers.0.self_attention.dense.lora_A': torch.randn(16, 4096),
        'transformer.encoder.layers.0.self_attention.dense.lora_B': torch.randn(4096, 16)
    }
    
    adapter.state_dict.return_value = lora_dict
    adapter.r = 16
    adapter.alpha = 32
    adapter.scaling = 32 / 16  # alpha / r
    
    return adapter


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        "model": {
            "base_model_id": "ZHIPU-AI/glm-4-9b-chat",
            "model_type": "glm"
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "target_modules": [
                "query_key_value", 
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h"
            ]
        },
        "merge": {
            "method": "linear",
            "validation": {
                "enabled": True,
                "test_prompts": [
                    "What is bird flu?",
                    "How does avian influenza spread?"
                ]
            },
            "output": {
                "save_merged": True,
                "save_comparison": True,
                "precision": "float16"
            }
        }
    }


class TestLoadBaseModel:
    """Test base model loading functionality."""
    
    @patch('merge_lora.AutoModelForCausalLM.from_pretrained')
    @patch('merge_lora.AutoTokenizer.from_pretrained')
    def test_load_base_model_success(self, mock_tokenizer, mock_model, mock_base_model):
        """Test successful base model loading."""
        mock_model.return_value = mock_base_model
        mock_tokenizer.return_value = Mock()
        
        model, tokenizer = load_base_model("test-model", device="cpu")
        
        assert model is not None
        assert tokenizer is not None
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    @patch('merge_lora.AutoModelForCausalLM.from_pretrained')
    def test_load_base_model_failure(self, mock_model):
        """Test base model loading failure."""
        mock_model.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception):
            load_base_model("nonexistent-model")
    
    @patch('merge_lora.AutoModelForCausalLM.from_pretrained')
    @patch('merge_lora.AutoTokenizer.from_pretrained')
    def test_load_base_model_different_precisions(self, mock_tokenizer, mock_model, mock_base_model):
        """Test loading base model with different precisions."""
        mock_model.return_value = mock_base_model
        mock_tokenizer.return_value = Mock()
        
        # Test different torch_dtype options
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            model, tokenizer = load_base_model("test-model", torch_dtype=dtype)
            assert model is not None


class TestLoadLoRAAdapter:
    """Test LoRA adapter loading functionality."""
    
    @patch('merge_lora.PeftModel.from_pretrained')
    def test_load_lora_adapter_success(self, mock_peft, mock_lora_adapter, temp_model_dir):
        """Test successful LoRA adapter loading."""
        mock_peft.return_value = mock_lora_adapter
        
        # Create adapter config
        adapter_config = {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["query_key_value", "dense"]
        }
        
        config_path = Path(temp_model_dir) / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f)
        
        adapter = load_lora_adapter(Mock(), temp_model_dir)
        
        assert adapter is not None
        mock_peft.assert_called_once()
    
    @patch('merge_lora.PeftModel.from_pretrained')
    def test_load_lora_adapter_failure(self, mock_peft):
        """Test LoRA adapter loading failure."""
        mock_peft.side_effect = Exception("Adapter not found")
        
        with pytest.raises(Exception):
            load_lora_adapter(Mock(), "nonexistent-adapter")
    
    def test_load_lora_adapter_missing_config(self, temp_model_dir):
        """Test loading adapter with missing configuration."""
        # No config file in directory
        with pytest.raises(FileNotFoundError):
            load_lora_adapter(Mock(), temp_model_dir)


class TestMergeLoRAWeights:
    """Test LoRA weight merging functionality."""
    
    def test_merge_lora_weights_basic(self, mock_base_model, mock_lora_adapter):
        """Test basic LoRA weight merging."""
        merged_state = merge_lora_weights(mock_base_model, mock_lora_adapter)
        
        assert merged_state is not None
        assert isinstance(merged_state, dict)
        
        # Check that base model weights are present
        for param_name, _ in mock_base_model.named_parameters():
            assert param_name in merged_state
    
    def test_merge_lora_weights_scaling(self, mock_base_model, mock_lora_adapter):
        """Test LoRA weight merging with scaling."""
        # Test different scaling values
        mock_lora_adapter.alpha = 64
        mock_lora_adapter.r = 16
        mock_lora_adapter.scaling = 64 / 16  # 4.0
        
        merged_state = merge_lora_weights(mock_base_model, mock_lora_adapter)
        
        assert merged_state is not None
        # Scaling should affect the merged weights
    
    def test_merge_lora_weights_no_lora_params(self, mock_base_model):
        """Test merging with adapter that has no LoRA parameters."""
        empty_adapter = Mock()
        empty_adapter.state_dict.return_value = {}
        empty_adapter.scaling = 1.0
        
        merged_state = merge_lora_weights(mock_base_model, empty_adapter)
        
        # Should return original base model weights
        assert merged_state is not None
        assert len(merged_state) == len(dict(mock_base_model.named_parameters()))
    
    def test_merge_lora_weights_mismatched_shapes(self, mock_base_model, mock_lora_adapter):
        """Test handling of mismatched tensor shapes."""
        # Create adapter with wrong shapes
        bad_lora_dict = {
            'transformer.encoder.layers.0.self_attention.query_key_value.lora_A': torch.randn(8, 4096),  # Wrong rank
            'transformer.encoder.layers.0.self_attention.query_key_value.lora_B': torch.randn(12288, 8)   # Wrong rank
        }
        
        mock_lora_adapter.state_dict.return_value = bad_lora_dict
        
        # Should handle gracefully
        merged_state = merge_lora_weights(mock_base_model, mock_lora_adapter)
        assert merged_state is not None


class TestSaveMergedModel:
    """Test merged model saving functionality."""
    
    @patch('merge_lora.torch.save')
    def test_save_merged_model_success(self, mock_save, mock_base_model, temp_model_dir):
        """Test successful model saving."""
        merged_state = {'param1': torch.randn(10, 10)}
        tokenizer = Mock()
        
        save_merged_model(mock_base_model, merged_state, tokenizer, temp_model_dir)
        
        # Should save model state dict
        mock_save.assert_called()
        
        # Should save tokenizer
        tokenizer.save_pretrained.assert_called_once()
    
    def test_save_merged_model_create_directory(self, mock_base_model, temp_model_dir):
        """Test model saving with directory creation."""
        nested_dir = Path(temp_model_dir) / "nested" / "model"
        merged_state = {'param1': torch.randn(10, 10)}
        tokenizer = Mock()
        
        with patch('merge_lora.torch.save'):
            save_merged_model(mock_base_model, merged_state, tokenizer, str(nested_dir))
            
            assert nested_dir.exists()
    
    @patch('merge_lora.torch.save')
    def test_save_merged_model_different_formats(self, mock_save, mock_base_model, temp_model_dir):
        """Test saving in different formats."""
        merged_state = {'param1': torch.randn(10, 10)}
        tokenizer = Mock()
        
        # Test different save formats
        for save_format in ['pytorch', 'safetensors']:
            save_merged_model(
                mock_base_model, 
                merged_state, 
                tokenizer, 
                temp_model_dir,
                save_format=save_format
            )
            
            mock_save.assert_called()


class TestValidateMerge:
    """Test merge validation functionality."""
    
    def test_validate_merge_success(self, mock_base_model, temp_model_dir):
        """Test successful merge validation."""
        merged_model = Mock()
        merged_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "Generated response"
        
        test_prompts = ["What is bird flu?", "How does it spread?"]
        
        is_valid, report = validate_merge(mock_base_model, merged_model, tokenizer, test_prompts)
        
        assert isinstance(is_valid, bool)
        assert isinstance(report, dict)
        assert 'validation_results' in report
    
    def test_validate_merge_failure(self, mock_base_model, temp_model_dir):
        """Test merge validation failure."""
        merged_model = Mock()
        merged_model.generate.side_effect = Exception("Generation failed")
        
        tokenizer = Mock()
        test_prompts = ["Test prompt"]
        
        is_valid, report = validate_merge(mock_base_model, merged_model, tokenizer, test_prompts)
        
        assert is_valid is False
        assert 'error' in report
    
    def test_validate_merge_empty_prompts(self, mock_base_model):
        """Test validation with empty prompts."""
        merged_model = Mock()
        tokenizer = Mock()
        
        is_valid, report = validate_merge(mock_base_model, merged_model, tokenizer, [])
        
        # Should handle empty prompts gracefully
        assert isinstance(is_valid, bool)
        assert isinstance(report, dict)
    
    def test_validate_merge_response_comparison(self, mock_base_model):
        """Test validation with response comparison."""
        # Mock different responses for base vs merged
        base_response = "Base model response"
        merged_response = "Merged model response"
        
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3]])
        merged_model = Mock()
        merged_model.generate.return_value = torch.tensor([[4, 5, 6]])
        
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = [base_response, merged_response]
        
        test_prompts = ["Test prompt"]
        
        is_valid, report = validate_merge(mock_base_model, merged_model, tokenizer, test_prompts)
        
        assert 'validation_results' in report
        assert len(report['validation_results']) == 1


class TestCreateComparisonReport:
    """Test comparison report creation."""
    
    def test_create_comparison_report_basic(self, mock_base_model, temp_model_dir):
        """Test basic comparison report creation."""
        merged_model = Mock()
        merged_model.num_parameters.return_value = 9000000000  # 9B parameters
        
        mock_base_model.num_parameters.return_value = 9000000000
        
        tokenizer = Mock()
        validation_report = {
            'validation_results': [
                {
                    'prompt': 'Test',
                    'base_response': 'Base response',
                    'merged_response': 'Merged response',
                    'similarity_score': 0.85
                }
            ]
        }
        
        report = create_comparison_report(
            mock_base_model, 
            merged_model, 
            tokenizer, 
            validation_report
        )
        
        assert isinstance(report, dict)
        assert 'model_comparison' in report
        assert 'validation_summary' in report
    
    def test_create_comparison_report_detailed(self, mock_base_model):
        """Test detailed comparison report."""
        merged_model = Mock()
        tokenizer = Mock()
        
        # Detailed validation report
        validation_report = {
            'validation_results': [
                {
                    'prompt': 'What is bird flu?',
                    'base_response': 'Bird flu is a viral infection.',
                    'merged_response': 'Bird flu, or avian influenza, is a viral infection.',
                    'similarity_score': 0.9,
                    'generation_time_base': 1.2,
                    'generation_time_merged': 1.1
                },
                {
                    'prompt': 'How does it spread?',
                    'base_response': 'It spreads through birds.',
                    'merged_response': 'It primarily spreads through infected birds.',
                    'similarity_score': 0.8,
                    'generation_time_base': 0.8,
                    'generation_time_merged': 0.9
                }
            ]
        }
        
        report = create_comparison_report(mock_base_model, merged_model, tokenizer, validation_report)
        
        assert 'performance_metrics' in report
        assert 'similarity_analysis' in report
    
    def test_create_comparison_report_with_metrics(self, mock_base_model):
        """Test comparison report with performance metrics."""
        merged_model = Mock()
        
        # Mock memory usage
        mock_base_model.get_memory_footprint = Mock(return_value={"total": 18000000000})
        merged_model.get_memory_footprint = Mock(return_value={"total": 18000000000})
        
        tokenizer = Mock()
        validation_report = {'validation_results': []}
        
        report = create_comparison_report(mock_base_model, merged_model, tokenizer, validation_report)
        
        assert 'memory_analysis' in report or 'model_comparison' in report


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_merge_with_different_model_types(self, test_config):
        """Test merging with different model architectures."""
        # Create base model with different architecture
        base_model = Mock()
        base_model.config.model_type = "qwen"
        
        lora_adapter = Mock()
        lora_adapter.state_dict.return_value = {
            'model.embed_tokens.lora_A': torch.randn(16, 4096),
            'model.embed_tokens.lora_B': torch.randn(32000, 16)
        }
        lora_adapter.scaling = 2.0
        
        # Should handle different architectures
        merged_state = merge_lora_weights(base_model, lora_adapter)
        assert merged_state is not None
    
    def test_merge_with_very_large_tensors(self, mock_base_model):
        """Test merging with memory-intensive tensors."""
        # Create adapter with large tensors
        large_adapter = Mock()
        large_lora_dict = {
            'large_layer.lora_A': torch.randn(64, 8192),  # Large rank and dimension
            'large_layer.lora_B': torch.randn(16384, 64)
        }
        
        large_adapter.state_dict.return_value = large_lora_dict
        large_adapter.scaling = 4.0
        
        # Should handle large tensors
        merged_state = merge_lora_weights(mock_base_model, large_adapter)
        assert merged_state is not None
    
    def test_save_model_insufficient_space(self, mock_base_model, temp_model_dir):
        """Test model saving with insufficient disk space."""
        merged_state = {'param1': torch.randn(1000, 1000)}
        tokenizer = Mock()
        
        # Mock disk space error
        with patch('merge_lora.torch.save', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                save_merged_model(mock_base_model, merged_state, tokenizer, temp_model_dir)
    
    def test_validation_with_corrupted_model(self, mock_base_model):
        """Test validation with corrupted merged model."""
        # Create corrupted model that fails randomly
        corrupted_model = Mock()
        corrupted_model.generate.side_effect = [
            torch.tensor([[1, 2, 3]]),  # Success
            RuntimeError("CUDA error"),  # Failure
            torch.tensor([[4, 5, 6]])   # Success again
        ]
        
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "Response"
        
        test_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        is_valid, report = validate_merge(mock_base_model, corrupted_model, tokenizer, test_prompts)
        
        # Should handle partial failures
        assert isinstance(is_valid, bool)
        assert 'errors' in report or 'validation_results' in report


class TestIntegration:
    """Test integration of merge workflow."""
    
    @patch('merge_lora.AutoModelForCausalLM.from_pretrained')
    @patch('merge_lora.AutoTokenizer.from_pretrained')
    @patch('merge_lora.PeftModel.from_pretrained')
    @patch('merge_lora.torch.save')
    def test_full_merge_pipeline(self, mock_save, mock_peft, mock_tokenizer, mock_model, 
                                mock_base_model, mock_lora_adapter, temp_model_dir, test_config):
        """Test complete merge pipeline."""
        # Setup mocks
        mock_model.return_value = mock_base_model
        mock_tokenizer.return_value = Mock()
        mock_peft.return_value = mock_lora_adapter
        
        # Create adapter config
        adapter_config = {"peft_type": "LORA", "r": 16, "lora_alpha": 32}
        config_path = Path(temp_model_dir) / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f)
        
        # Load models
        base_model, tokenizer = load_base_model(test_config["model"]["base_model_id"])
        lora_adapter = load_lora_adapter(base_model, temp_model_dir)
        
        # Merge weights
        merged_state = merge_lora_weights(base_model, lora_adapter)
        
        # Save merged model
        output_dir = Path(temp_model_dir) / "merged"
        save_merged_model(base_model, merged_state, tokenizer, str(output_dir))
        
        # Validate
        merged_model = Mock()
        merged_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        
        test_prompts = test_config["merge"]["validation"]["test_prompts"]
        is_valid, validation_report = validate_merge(base_model, merged_model, tokenizer, test_prompts)
        
        # Create report
        comparison_report = create_comparison_report(base_model, merged_model, tokenizer, validation_report)
        
        assert merged_state is not None
        assert isinstance(is_valid, bool)
        assert isinstance(comparison_report, dict)
        assert output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])