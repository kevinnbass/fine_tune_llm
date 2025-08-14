"""Comprehensive tests for Enhanced LoRA SFT training functionality."""

import pytest
import torch
import yaml
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

# Test imports
try:
    from voters.llm.sft_lora import EnhancedLoRASFTTrainer
    SFT_AVAILABLE = True
except ImportError:
    SFT_AVAILABLE = False
    pytest.skip("SFT LoRA trainer not available", allow_module_level=True)


class MockModel:
    """Mock model for testing."""
    def __init__(self):
        self.parameters = lambda: [torch.randn(10, 10)]
        self.named_parameters = lambda: [("test_param", torch.randn(10, 10))]
        self.train = Mock()
        self.eval = Mock()
        
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
    def print_trainable_parameters(self):
        print("trainable params: 1,000 || all params: 1,000,000 || trainable%: 0.1")


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.vocab_size = 32000
        
    def from_pretrained(self, *args, **kwargs):
        return self
        
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            'input_ids': [[1, 2, 3, 4, 5] for _ in texts],
            'attention_mask': [[1, 1, 1, 1, 1] for _ in texts]
        }
        
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        "selected_model": "test_model",
        "model_options": {
            "test_model": {
                "model_id": "test/model",
                "tokenizer_id": "test/tokenizer", 
                "target_modules": ["q_proj", "v_proj"],
                "chat_template": "generic"
            }
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "method": "lora",
            "quantization": {
                "enabled": False,
                "bits": 4,
                "double_quant": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16"
            }
        },
        "training": {
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "optim": "adamw_torch",
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": False,
            "eval_strategy": "no",
            "eval_steps": 100,
            "save_steps": 100,
            "logging_steps": 10,
            "load_best_model_at_end": False,
            "metric_for_best_model": "loss",
            "greater_is_better": False,
            "save_total_limit": 2,
            "early_stopping": False,
            "early_stopping_patience": 3,
            "scheduler": {
                "type": "cosine",
                "warmup_steps": 10
            },
            "logging": {
                "wandb": False,
                "project_name": "test",
                "tags": ["test"],
                "notes": ""
            },
            "num_proc": 1
        },
        "evaluation": {
            "enabled": False,
            "val_split": 0.1,
            "metrics": ["accuracy"]
        },
        "instruction_format": {
            "system_prompt": "You are a helpful assistant."
        }
    }


@pytest.fixture
def test_dataset():
    """Fixture providing test dataset."""
    return Dataset.from_list([
        {
            "text": "What is machine learning?",
            "output": "Machine learning is a subset of AI.",
            "metadata": {"domain": "tech"}
        },
        {
            "text": "Explain neural networks",
            "output": "Neural networks are computational models.",
            "metadata": {"domain": "tech"}
        }
    ])


@pytest.fixture
def temp_config_file(test_config):
    """Fixture providing temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        yield f.name
    Path(f.name).unlink()


class TestEnhancedLoRASFTTrainer:
    """Test Enhanced LoRA SFT Trainer functionality."""
    
    def test_initialization(self, temp_config_file):
        """Test trainer initialization."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        assert trainer.config is not None
        assert trainer.model_config is not None
        assert trainer.output_dir.name == "llm_lora"
        assert isinstance(trainer.metrics, dict)
    
    def test_initialization_with_evaluation(self, test_config, temp_config_file):
        """Test trainer initialization with evaluation enabled."""
        test_config["evaluation"]["enabled"] = True
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        assert len(trainer.metrics) > 0
    
    def test_quantization_config_disabled(self, temp_config_file):
        """Test quantization config when disabled."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        config = trainer.get_quantization_config()
        assert config is None
    
    def test_quantization_config_4bit(self, test_config, temp_config_file):
        """Test 4-bit quantization config."""
        test_config["lora"]["quantization"]["enabled"] = True
        test_config["lora"]["quantization"]["bits"] = 4
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        config = trainer.get_quantization_config()
        
        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
    
    def test_quantization_config_8bit(self, test_config, temp_config_file):
        """Test 8-bit quantization config."""
        test_config["lora"]["quantization"]["enabled"] = True
        test_config["lora"]["quantization"]["bits"] = 8
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        config = trainer.get_quantization_config()
        
        assert config is not None
        assert config.load_in_8bit is True
    
    def test_quantization_config_invalid_bits(self, test_config, temp_config_file):
        """Test invalid quantization bits."""
        test_config["lora"]["quantization"]["enabled"] = True
        test_config["lora"]["quantization"]["bits"] = 16  # Invalid
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        with pytest.raises(ValueError, match="Unsupported quantization bits"):
            trainer.get_quantization_config()
    
    def test_peft_config_lora(self, temp_config_file):
        """Test standard LoRA PEFT config."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        config = trainer.get_peft_config()
        
        assert config.task_type.value == "CAUSAL_LM"
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
        assert not hasattr(config, 'use_dora') or config.use_dora is False
    
    def test_peft_config_dora(self, test_config, temp_config_file):
        """Test DoRA PEFT config."""
        test_config["lora"]["method"] = "dora"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        config = trainer.get_peft_config()
        
        assert hasattr(config, 'use_dora') and config.use_dora is True
    
    def test_model_specific_prompt_generic(self, temp_config_file):
        """Test generic prompt formatting."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        text = "What is AI?"
        prompt = trainer.get_model_specific_prompt(text)
        
        assert "You are a helpful assistant" in prompt
        assert text in prompt
    
    def test_model_specific_prompt_llama(self, test_config, temp_config_file):
        """Test Llama prompt formatting."""
        test_config["model_options"]["test_model"]["chat_template"] = "llama"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        text = "What is AI?"
        prompt = trainer.get_model_specific_prompt(text)
        
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert text in prompt
    
    def test_model_specific_prompt_mistral(self, test_config, temp_config_file):
        """Test Mistral prompt formatting."""
        test_config["model_options"]["test_model"]["chat_template"] = "mistral"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        text = "What is AI?"
        prompt = trainer.get_model_specific_prompt(text)
        
        assert "<s>[INST]" in prompt
        assert "[/INST]" in prompt
        assert text in prompt
    
    def test_model_specific_prompt_qwen(self, test_config, temp_config_file):
        """Test Qwen prompt formatting."""
        test_config["model_options"]["test_model"]["chat_template"] = "qwen"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        text = "What is AI?"
        prompt = trainer.get_model_specific_prompt(text)
        
        assert "<|im_start|>" in prompt
        assert "<|im_end|>" in prompt
        assert text in prompt
    
    @patch('voters.llm.sft_lora.AutoTokenizer')
    def test_prepare_dataset(self, mock_tokenizer_class, temp_config_file, test_dataset):
        """Test dataset preparation."""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        trainer.tokenizer = mock_tokenizer
        
        # Mock the dataset.map method to avoid actual tokenization
        with patch.object(test_dataset, 'map') as mock_map:
            mock_map.return_value = test_dataset
            
            result = trainer.prepare_dataset(test_dataset)
            
            # Verify map was called with correct parameters
            mock_map.assert_called_once()
            call_args = mock_map.call_args
            assert call_args[1]['batched'] is True
            assert call_args[1]['num_proc'] == 1
            assert call_args[1]['remove_columns'] == test_dataset.column_names
    
    def test_split_dataset_disabled(self, temp_config_file, test_dataset):
        """Test dataset splitting when evaluation disabled."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        train_ds, val_ds = trainer.split_dataset(test_dataset)
        
        assert train_ds == test_dataset
        assert val_ds is None
    
    def test_split_dataset_enabled(self, test_config, temp_config_file, test_dataset):
        """Test dataset splitting when evaluation enabled."""
        test_config["evaluation"]["enabled"] = True
        test_config["evaluation"]["val_split"] = 0.5
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        train_ds, val_ds = trainer.split_dataset(test_dataset)
        
        assert train_ds is not None
        assert val_ds is not None
        assert len(train_ds) + len(val_ds) == len(test_dataset)
    
    def test_compute_metrics(self, temp_config_file):
        """Test metrics computation."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        # Mock evaluation predictions
        predictions = torch.tensor([1.5, 2.0])  # Mock loss values
        labels = torch.tensor([0, 1])  # Mock labels
        
        eval_pred = Mock()
        eval_pred.predictions = predictions
        eval_pred.label_ids = labels
        
        metrics = trainer.compute_metrics((predictions, labels))
        
        assert "perplexity" in metrics
        assert "loss" in metrics
        assert metrics["loss"] == predictions.mean()
        assert metrics["perplexity"] > 0
    
    def test_get_scheduler_cosine(self, temp_config_file):
        """Test cosine scheduler."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)])
        scheduler = trainer.get_scheduler(optimizer, 100)
        
        assert scheduler is not None
    
    def test_get_scheduler_linear(self, test_config, temp_config_file):
        """Test linear scheduler."""
        test_config["training"]["scheduler"]["type"] = "linear"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)])
        scheduler = trainer.get_scheduler(optimizer, 100)
        
        assert scheduler is not None
    
    def test_get_scheduler_none(self, test_config, temp_config_file):
        """Test no scheduler."""
        test_config["training"]["scheduler"]["type"] = "none"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)])
        scheduler = trainer.get_scheduler(optimizer, 100)
        
        assert scheduler is None
    
    def test_get_report_to_default(self, temp_config_file):
        """Test default reporting tools."""
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        report_to = trainer.get_report_to()
        
        assert "tensorboard" in report_to
        assert "wandb" not in report_to  # W&B disabled by default
    
    def test_get_report_to_wandb_enabled(self, test_config, temp_config_file):
        """Test reporting with W&B enabled."""
        test_config["training"]["logging"]["wandb"] = True
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        
        # Mock WANDB_AVAILABLE
        with patch('voters.llm.sft_lora.WANDB_AVAILABLE', True):
            report_to = trainer.get_report_to()
            
            assert "tensorboard" in report_to
            assert "wandb" in report_to
    
    def test_high_stakes_initialization(self, test_config, temp_config_file):
        """Test high-stakes components initialization."""
        test_config["high_stakes"] = {
            "bias_audit": {"enabled": True},
            "procedural": {"enabled": True}, 
            "verifiable": {"enabled": True}
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Mock HIGH_STAKES_AVAILABLE
        with patch('voters.llm.sft_lora.HIGH_STAKES_AVAILABLE', True):
            with patch('voters.llm.sft_lora.BiasAuditor') as mock_bias:
                with patch('voters.llm.sft_lora.ProceduralAlignment') as mock_proc:
                    with patch('voters.llm.sft_lora.VerifiableTraining') as mock_verify:
                        trainer = EnhancedLoRASFTTrainer(temp_config_file)
                        
                        assert 'bias_auditor' in trainer.high_stakes_components
                        assert 'procedural' in trainer.high_stakes_components
                        assert 'verifiable' in trainer.high_stakes_components
                        
                        mock_bias.assert_called_once()
                        mock_proc.assert_called_once()
                        mock_verify.assert_called_once()


class TestConfigurationEdgeCases:
    """Test configuration edge cases and validation."""
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            EnhancedLoRASFTTrainer("nonexistent_config.yaml")
    
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()
            
            with pytest.raises(yaml.YAMLError):
                EnhancedLoRASFTTrainer(f.name)
            
            Path(f.name).unlink()
    
    def test_missing_selected_model(self, test_config, temp_config_file):
        """Test handling of missing selected model."""
        del test_config["selected_model"]
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with pytest.raises(KeyError):
            EnhancedLoRASFTTrainer(temp_config_file)
    
    def test_missing_model_options(self, test_config, temp_config_file):
        """Test handling of missing model options."""
        del test_config["model_options"]
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with pytest.raises(KeyError):
            EnhancedLoRASFTTrainer(temp_config_file)


class TestModularity:
    """Test modularity and toggleability of features."""
    
    def test_quantization_toggle(self, test_config, temp_config_file):
        """Test quantization can be toggled."""
        # Test enabled
        test_config["lora"]["quantization"]["enabled"] = True
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        assert trainer.get_quantization_config() is not None
        
        # Test disabled
        test_config["lora"]["quantization"]["enabled"] = False
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        assert trainer.get_quantization_config() is None
    
    def test_evaluation_toggle(self, test_config, temp_config_file):
        """Test evaluation can be toggled."""
        # Test disabled
        test_config["evaluation"]["enabled"] = False
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        assert len(trainer.metrics) == 0
        
        # Test enabled
        test_config["evaluation"]["enabled"] = True
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        trainer = EnhancedLoRASFTTrainer(temp_config_file)
        assert len(trainer.metrics) > 0
    
    def test_high_stakes_features_toggle(self, test_config, temp_config_file):
        """Test high-stakes features can be toggled individually."""
        test_config["high_stakes"] = {
            "bias_audit": {"enabled": True},
            "procedural": {"enabled": False},
            "verifiable": {"enabled": True}
        }
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with patch('voters.llm.sft_lora.HIGH_STAKES_AVAILABLE', True):
            with patch('voters.llm.sft_lora.BiasAuditor') as mock_bias:
                with patch('voters.llm.sft_lora.ProceduralAlignment') as mock_proc:
                    with patch('voters.llm.sft_lora.VerifiableTraining') as mock_verify:
                        trainer = EnhancedLoRASFTTrainer(temp_config_file)
                        
                        # Only enabled features should be initialized
                        assert 'bias_auditor' in trainer.high_stakes_components
                        assert 'procedural' not in trainer.high_stakes_components
                        assert 'verifiable' in trainer.high_stakes_components
                        
                        mock_bias.assert_called_once()
                        mock_proc.assert_not_called()
                        mock_verify.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])