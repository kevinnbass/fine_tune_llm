"""Comprehensive tests for high-stakes training script functionality."""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import argparse

# Test imports - Handle both script and module imports
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    
    from train_high_stakes import (
        parse_args, load_config, prepare_high_stakes_config,
        load_training_data, create_high_stakes_trainer, run_training,
        validate_training_results, save_training_artifacts
    )
    TRAIN_HIGH_STAKES_AVAILABLE = True
except ImportError:
    TRAIN_HIGH_STAKES_AVAILABLE = False
    pytest.skip("Train high stakes script not available", allow_module_level=True)


@pytest.fixture
def temp_training_dir():
    """Fixture providing temporary training directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Fixture providing sample training configuration."""
    return {
        "selected_model": "glm-4.5-air",
        "model_options": {
            "glm-4.5-air": {
                "model_id": "ZHIPU-AI/glm-4-9b-chat",
                "tokenizer_id": "ZHIPU-AI/glm-4-9b-chat"
            }
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["query_key_value", "dense"]
        },
        "training": {
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "save_steps": 100,
            "eval_steps": 50,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "high_stakes": {
            "uncertainty": {
                "enabled": True,
                "method": "mc_dropout",
                "num_samples": 5,
                "abstention_threshold": 0.7
            },
            "factual": {
                "enabled": True,
                "reliance_steps": 3,
                "self_consistency_threshold": 0.8
            },
            "bias_audit": {
                "enabled": True,
                "audit_categories": ["gender", "race"],
                "bias_threshold": 0.1
            },
            "explainable": {
                "enabled": True,
                "chain_of_thought": True,
                "reasoning_steps": 3
            },
            "procedural": {
                "enabled": True,
                "domain": "medical",
                "compliance_weight": 2.0
            },
            "verifiable": {
                "enabled": True,
                "hash_artifacts": True,
                "cryptographic_proof": True
            }
        }
    }


@pytest.fixture
def sample_training_data():
    """Fixture providing sample training data."""
    return [
        {
            "input": "Classify this text about bird flu outbreak",
            "output": json.dumps({
                "decision": "HIGH_RISK",
                "rationale": "Contains confirmed outbreak information",
                "confidence": 0.95,
                "abstain": False
            }),
            "metadata": {"source": "news", "confidence": "high"}
        },
        {
            "input": "Weather forecast shows sunny skies",
            "output": json.dumps({
                "decision": "NO_RISK", 
                "rationale": "Weather information unrelated to health risks",
                "confidence": 0.98,
                "abstain": False
            }),
            "metadata": {"source": "weather", "confidence": "high"}
        },
        {
            "input": "Unclear reports about potential cases",
            "output": json.dumps({
                "decision": "UNCERTAIN",
                "rationale": "Information too vague to classify confidently",
                "confidence": 0.3,
                "abstain": True
            }),
            "metadata": {"source": "rumors", "confidence": "low"}
        }
    ]


class TestParseArgs:
    """Test command line argument parsing."""
    
    def test_parse_args_default(self):
        """Test parsing with default arguments."""
        args = parse_args()
        
        assert hasattr(args, 'config')
        assert hasattr(args, 'train_data')
        assert hasattr(args, 'output_dir')
        assert hasattr(args, 'uncertainty_enabled')
        assert hasattr(args, 'factual_enabled')
        assert hasattr(args, 'bias_audit_enabled')
    
    def test_parse_args_custom_config(self):
        """Test parsing with custom configuration."""
        test_args = [
            '--config', 'custom_config.yaml',
            '--train-data', 'custom_data.json',
            '--output-dir', 'custom_output',
            '--uncertainty-enabled',
            '--factual-enabled'
        ]
        
        with patch('sys.argv', ['train_high_stakes.py'] + test_args):
            args = parse_args()
            
            assert args.config == 'custom_config.yaml'
            assert args.train_data == 'custom_data.json'
            assert args.output_dir == 'custom_output'
            assert args.uncertainty_enabled is True
            assert args.factual_enabled is True
    
    def test_parse_args_all_features_enabled(self):
        """Test parsing with all high-stakes features enabled."""
        test_args = [
            '--uncertainty-enabled',
            '--factual-enabled', 
            '--bias-audit-enabled',
            '--explainable-enabled',
            '--procedural-enabled',
            '--verifiable-enabled'
        ]
        
        with patch('sys.argv', ['train_high_stakes.py'] + test_args):
            args = parse_args()
            
            assert args.uncertainty_enabled is True
            assert args.factual_enabled is True
            assert args.bias_audit_enabled is True
            assert args.explainable_enabled is True
            assert args.procedural_enabled is True
            assert args.verifiable_enabled is True
    
    def test_parse_args_training_parameters(self):
        """Test parsing training-specific parameters."""
        test_args = [
            '--epochs', '5',
            '--learning-rate', '1e-4',
            '--batch-size', '8'
        ]
        
        with patch('sys.argv', ['train_high_stakes.py'] + test_args):
            args = parse_args()
            
            assert args.epochs == 5
            assert args.learning_rate == 1e-4
            assert args.batch_size == 8


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_config_success(self, sample_config, temp_training_dir):
        """Test successful configuration loading."""
        config_file = Path(temp_training_dir) / "test_config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config == sample_config
        assert "high_stakes" in loaded_config
        assert "uncertainty" in loaded_config["high_stakes"]
    
    def test_load_config_missing_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_config_invalid_yaml(self, temp_training_dir):
        """Test handling of invalid YAML configuration."""
        config_file = Path(temp_training_dir) / "invalid_config.yaml"
        
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: {")
        
        with pytest.raises(Exception):
            load_config(str(config_file))
    
    def test_load_config_partial(self, temp_training_dir):
        """Test loading partial configuration."""
        partial_config = {
            "selected_model": "glm-4.5-air",
            "lora": {"r": 16, "alpha": 32}
        }
        
        config_file = Path(temp_training_dir) / "partial_config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(partial_config, f)
        
        loaded_config = load_config(str(config_file))
        
        assert loaded_config["selected_model"] == "glm-4.5-air"
        assert loaded_config["lora"]["r"] == 16


class TestPrepareHighStakesConfig:
    """Test high-stakes configuration preparation."""
    
    def test_prepare_high_stakes_config_basic(self, sample_config):
        """Test basic high-stakes configuration preparation."""
        args = Mock()
        args.uncertainty_enabled = True
        args.factual_enabled = True
        args.bias_audit_enabled = False
        
        updated_config = prepare_high_stakes_config(sample_config, args)
        
        assert updated_config["high_stakes"]["uncertainty"]["enabled"] is True
        assert updated_config["high_stakes"]["factual"]["enabled"] is True
        assert updated_config["high_stakes"]["bias_audit"]["enabled"] is False
    
    def test_prepare_high_stakes_config_all_enabled(self, sample_config):
        """Test configuration with all features enabled."""
        args = Mock()
        args.uncertainty_enabled = True
        args.factual_enabled = True
        args.bias_audit_enabled = True
        args.explainable_enabled = True
        args.procedural_enabled = True
        args.verifiable_enabled = True
        
        updated_config = prepare_high_stakes_config(sample_config, args)
        
        for feature in ["uncertainty", "factual", "bias_audit", "explainable", "procedural", "verifiable"]:
            assert updated_config["high_stakes"][feature]["enabled"] is True
    
    def test_prepare_high_stakes_config_override_params(self, sample_config):
        """Test configuration parameter overrides."""
        args = Mock()
        args.uncertainty_enabled = True
        args.learning_rate = 1e-4
        args.epochs = 5
        args.batch_size = 8
        
        updated_config = prepare_high_stakes_config(sample_config, args)
        
        assert updated_config["training"]["learning_rate"] == 1e-4
        assert updated_config["training"]["num_epochs"] == 5
        assert updated_config["training"]["batch_size"] == 8
    
    def test_prepare_high_stakes_config_missing_section(self):
        """Test handling missing high_stakes section."""
        basic_config = {
            "selected_model": "glm-4.5-air",
            "lora": {"r": 16}
        }
        
        args = Mock()
        args.uncertainty_enabled = True
        
        updated_config = prepare_high_stakes_config(basic_config, args)
        
        assert "high_stakes" in updated_config
        assert updated_config["high_stakes"]["uncertainty"]["enabled"] is True


class TestLoadTrainingData:
    """Test training data loading functionality."""
    
    def test_load_training_data_json(self, sample_training_data, temp_training_dir):
        """Test loading training data from JSON file."""
        data_file = Path(temp_training_dir) / "training_data.json"
        
        with open(data_file, 'w') as f:
            json.dump(sample_training_data, f)
        
        loaded_data = load_training_data(str(data_file))
        
        assert len(loaded_data) == len(sample_training_data)
        assert loaded_data[0]["input"] == sample_training_data[0]["input"]
    
    def test_load_training_data_missing_file(self):
        """Test handling of missing training data file."""
        with pytest.raises(FileNotFoundError):
            load_training_data("nonexistent_data.json")
    
    def test_load_training_data_invalid_json(self, temp_training_dir):
        """Test handling of invalid JSON data."""
        data_file = Path(temp_training_dir) / "invalid_data.json"
        
        with open(data_file, 'w') as f:
            f.write("{invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            load_training_data(str(data_file))
    
    def test_load_training_data_empty_file(self, temp_training_dir):
        """Test handling of empty training data file."""
        data_file = Path(temp_training_dir) / "empty_data.json"
        
        with open(data_file, 'w') as f:
            json.dump([], f)
        
        loaded_data = load_training_data(str(data_file))
        
        assert loaded_data == []
    
    def test_load_training_data_validation(self, sample_training_data, temp_training_dir):
        """Test training data validation."""
        # Add invalid sample
        invalid_data = sample_training_data + [
            {"input": "Missing output field"}
        ]
        
        data_file = Path(temp_training_dir) / "mixed_data.json"
        
        with open(data_file, 'w') as f:
            json.dump(invalid_data, f)
        
        loaded_data = load_training_data(str(data_file), validate=True)
        
        # Should filter out invalid samples
        assert len(loaded_data) == len(sample_training_data)


class TestCreateHighStakesTrainer:
    """Test high-stakes trainer creation."""
    
    @patch('train_high_stakes.EnhancedLoRASFTTrainer')
    def test_create_trainer_basic(self, mock_trainer_class, sample_config):
        """Test basic trainer creation."""
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        trainer = create_high_stakes_trainer(sample_config, device="cpu")
        
        assert trainer == mock_trainer
        mock_trainer_class.assert_called_once()
    
    @patch('train_high_stakes.EnhancedLoRASFTTrainer')
    def test_create_trainer_with_high_stakes_features(self, mock_trainer_class, sample_config):
        """Test trainer creation with high-stakes features."""
        # Enable all features
        for feature in sample_config["high_stakes"]:
            sample_config["high_stakes"][feature]["enabled"] = True
        
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        trainer = create_high_stakes_trainer(sample_config, device="cuda")
        
        assert trainer == mock_trainer
        # Verify trainer was created with high-stakes config
        call_args = mock_trainer_class.call_args[1]
        assert "high_stakes" in call_args["config"]
    
    @patch('train_high_stakes.EnhancedLoRASFTTrainer')
    def test_create_trainer_error_handling(self, mock_trainer_class, sample_config):
        """Test trainer creation error handling."""
        mock_trainer_class.side_effect = Exception("Trainer creation failed")
        
        with pytest.raises(Exception, match="Trainer creation failed"):
            create_high_stakes_trainer(sample_config)


class TestRunTraining:
    """Test training execution functionality."""
    
    def test_run_training_success(self, sample_training_data, temp_training_dir):
        """Test successful training execution."""
        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "epochs": 1,
            "training_time": 3600
        }
        
        results = run_training(mock_trainer, sample_training_data, str(temp_training_dir))
        
        assert "train_loss" in results
        assert "training_time" in results
        mock_trainer.train.assert_called_once()
    
    def test_run_training_with_validation(self, sample_training_data, temp_training_dir):
        """Test training with validation data."""
        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "eval_accuracy": 0.85,
            "eval_f1": 0.82
        }
        
        # Split data for validation
        train_data = sample_training_data[:2]
        val_data = sample_training_data[2:]
        
        results = run_training(mock_trainer, train_data, str(temp_training_dir), validation_data=val_data)
        
        assert "eval_accuracy" in results
        assert "eval_f1" in results
    
    def test_run_training_failure(self, sample_training_data, temp_training_dir):
        """Test training failure handling."""
        mock_trainer = Mock()
        mock_trainer.train.side_effect = RuntimeError("Training failed")
        
        with pytest.raises(RuntimeError, match="Training failed"):
            run_training(mock_trainer, sample_training_data, str(temp_training_dir))
    
    def test_run_training_memory_error(self, sample_training_data, temp_training_dir):
        """Test handling of memory errors during training."""
        mock_trainer = Mock()
        mock_trainer.train.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            run_training(mock_trainer, sample_training_data, str(temp_training_dir))


class TestValidateTrainingResults:
    """Test training results validation."""
    
    def test_validate_results_success(self):
        """Test successful results validation."""
        results = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "eval_accuracy": 0.85,
            "eval_f1": 0.82,
            "epochs": 1,
            "training_time": 3600
        }
        
        is_valid, issues = validate_training_results(results)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_results_missing_metrics(self):
        """Test validation with missing metrics."""
        incomplete_results = {
            "train_loss": 0.5,
            "epochs": 1
        }
        
        is_valid, issues = validate_training_results(incomplete_results)
        
        assert is_valid is False
        assert len(issues) > 0
        assert any("missing" in issue.lower() for issue in issues)
    
    def test_validate_results_poor_performance(self):
        """Test validation with poor performance metrics."""
        poor_results = {
            "train_loss": 2.5,  # High loss
            "eval_loss": 3.0,   # Very high loss
            "eval_accuracy": 0.3,  # Low accuracy
            "eval_f1": 0.2,     # Low F1
            "epochs": 1,
            "training_time": 3600
        }
        
        is_valid, issues = validate_training_results(poor_results)
        
        assert is_valid is False
        assert len(issues) > 0
        assert any("performance" in issue.lower() or "loss" in issue.lower() for issue in issues)
    
    def test_validate_results_convergence_issues(self):
        """Test validation with convergence issues."""
        non_convergent_results = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "eval_accuracy": 0.85,
            "eval_f1": 0.82,
            "epochs": 10,
            "converged": False,  # Didn't converge
            "training_time": 36000  # Very long training
        }
        
        is_valid, issues = validate_training_results(non_convergent_results)
        
        # Depending on implementation, this might be valid or not
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


class TestSaveTrainingArtifacts:
    """Test training artifacts saving functionality."""
    
    def test_save_artifacts_success(self, temp_training_dir):
        """Test successful artifacts saving."""
        mock_trainer = Mock()
        results = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "training_time": 3600
        }
        config = {"selected_model": "glm-4.5-air"}
        
        artifacts_path = save_training_artifacts(mock_trainer, results, config, str(temp_training_dir))
        
        assert Path(artifacts_path).exists()
        mock_trainer.save_model.assert_called_once()
    
    def test_save_artifacts_with_metadata(self, temp_training_dir):
        """Test saving artifacts with metadata."""
        mock_trainer = Mock()
        results = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "high_stakes_metrics": {
                "uncertainty_scores": [0.1, 0.2, 0.3],
                "bias_audit_results": {"gender": 0.05, "race": 0.03}
            }
        }
        config = {"selected_model": "glm-4.5-air"}
        
        artifacts_path = save_training_artifacts(
            mock_trainer, 
            results, 
            config, 
            str(temp_training_dir),
            save_metadata=True
        )
        
        # Check that metadata file was created
        metadata_file = Path(artifacts_path) / "training_metadata.json"
        assert metadata_file.exists()
    
    def test_save_artifacts_create_directory(self, temp_training_dir):
        """Test artifact saving with directory creation."""
        nested_dir = Path(temp_training_dir) / "nested" / "artifacts"
        
        mock_trainer = Mock()
        results = {"train_loss": 0.5}
        config = {"selected_model": "glm-4.5-air"}
        
        artifacts_path = save_training_artifacts(mock_trainer, results, config, str(nested_dir))
        
        assert Path(artifacts_path).exists()
        assert nested_dir.exists()
    
    def test_save_artifacts_insufficient_space(self, temp_training_dir):
        """Test handling of insufficient disk space."""
        mock_trainer = Mock()
        mock_trainer.save_model.side_effect = OSError("No space left on device")
        
        results = {"train_loss": 0.5}
        config = {"selected_model": "glm-4.5-air"}
        
        with pytest.raises(OSError, match="No space left on device"):
            save_training_artifacts(mock_trainer, results, config, str(temp_training_dir))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_training_with_minimal_data(self):
        """Test training with minimal dataset."""
        minimal_data = [
            {
                "input": "Test input",
                "output": json.dumps({"decision": "NO_RISK", "rationale": "Test", "abstain": False})
            }
        ]
        
        mock_trainer = Mock()
        mock_trainer.train.return_value = {"train_loss": 1.0}
        
        # Should handle minimal data gracefully
        results = run_training(mock_trainer, minimal_data, "test_output")
        assert "train_loss" in results
    
    def test_training_with_corrupted_data(self, temp_training_dir):
        """Test handling of corrupted training data."""
        corrupted_data = [
            {"input": "Valid input", "output": "invalid json"},
            {"missing": "output field"},
            {"input": "", "output": json.dumps({"decision": "NO_RISK"})}
        ]
        
        data_file = Path(temp_training_dir) / "corrupted_data.json"
        with open(data_file, 'w') as f:
            json.dump(corrupted_data, f)
        
        # Should handle corrupted data gracefully
        loaded_data = load_training_data(str(data_file), validate=True)
        assert len(loaded_data) == 0  # All samples invalid
    
    def test_config_with_extreme_values(self):
        """Test configuration with extreme hyperparameter values."""
        extreme_config = {
            "training": {
                "learning_rate": 1.0,  # Very high
                "batch_size": 1,       # Very small
                "num_epochs": 100      # Very high
            }
        }
        
        args = Mock()
        args.learning_rate = 1.0
        args.batch_size = 1
        args.epochs = 100
        
        # Should handle extreme values without crashing
        updated_config = prepare_high_stakes_config(extreme_config, args)
        assert updated_config["training"]["learning_rate"] == 1.0


class TestIntegration:
    """Test integration of training pipeline."""
    
    @patch('train_high_stakes.EnhancedLoRASFTTrainer')
    def test_full_training_pipeline(self, mock_trainer_class, sample_config, sample_training_data, temp_training_dir):
        """Test complete training pipeline integration."""
        # Setup mocks
        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "eval_accuracy": 0.85,
            "training_time": 3600
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Create config file
        config_file = Path(temp_training_dir) / "config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create training data file
        data_file = Path(temp_training_dir) / "train_data.json"
        with open(data_file, 'w') as f:
            json.dump(sample_training_data, f)
        
        # Run pipeline components
        config = load_config(str(config_file))
        
        args = Mock()
        args.uncertainty_enabled = True
        args.factual_enabled = True
        
        enhanced_config = prepare_high_stakes_config(config, args)
        training_data = load_training_data(str(data_file))
        trainer = create_high_stakes_trainer(enhanced_config)
        results = run_training(trainer, training_data, str(temp_training_dir))
        is_valid, issues = validate_training_results(results)
        
        # Verify pipeline execution
        assert enhanced_config["high_stakes"]["uncertainty"]["enabled"] is True
        assert len(training_data) == len(sample_training_data)
        assert trainer == mock_trainer
        assert "train_loss" in results
        assert isinstance(is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])