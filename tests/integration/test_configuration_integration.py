"""
Configuration integration tests for all configuration-related components.

This test module validates configuration integration across all platform components,
including hot-reload, validation, environment overrides, and cross-component consistency.
"""

import pytest
import asyncio
import time
import threading
import os
import tempfile
import json
import yaml
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment
)

# Import platform components
from src.fine_tune_llm.core.events import EventBus, Event, EventType
from src.fine_tune_llm.config.manager import ConfigManager
from src.fine_tune_llm.config.pipeline_config import PipelineConfigManager


class TestConfigurationManagerIntegration:
    """Test configuration manager integration across components."""
    
    def test_hierarchical_configuration_loading(self):
        """Test hierarchical configuration loading and merging."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Create temporary config files
            temp_dir = env.create_temp_directory("config_test")
            
            # Base config
            base_config = {
                "model": {
                    "name": "base-model",
                    "architecture": "transformer",
                    "hidden_size": 768
                },
                "training": {
                    "batch_size": 32,
                    "learning_rate": 2e-4,
                    "epochs": 5
                },
                "paths": {
                    "data_dir": "/data",
                    "model_dir": "/models"
                }
            }
            
            base_config_path = temp_dir / "base.yaml"
            with open(base_config_path, 'w') as f:
                yaml.dump(base_config, f)
            
            # Override config
            override_config = {
                "model": {
                    "name": "fine-tuned-model",
                    "hidden_size": 1024  # Override
                },
                "training": {
                    "learning_rate": 1e-4,  # Override
                    "warmup_steps": 1000    # New key
                },
                "evaluation": {  # New section
                    "batch_size": 64,
                    "metrics": ["accuracy", "f1"]
                }
            }
            
            override_config_path = temp_dir / "override.yaml"
            with open(override_config_path, 'w') as f:
                yaml.dump(override_config, f)
            
            # Load configurations hierarchically
            config_manager.load_from_file(str(base_config_path))
            config_manager.load_from_file(str(override_config_path))
            
            # Verify hierarchical merging
            assert config_manager.get("model.name") == "fine-tuned-model"  # Overridden
            assert config_manager.get("model.architecture") == "transformer"  # From base
            assert config_manager.get("model.hidden_size") == 1024  # Overridden
            assert config_manager.get("training.batch_size") == 32  # From base
            assert config_manager.get("training.learning_rate") == 1e-4  # Overridden
            assert config_manager.get("training.warmup_steps") == 1000  # New
            assert config_manager.get("evaluation.batch_size") == 64  # New section
            assert config_manager.get("paths.data_dir") == "/data"  # From base
    
    def test_environment_variable_override_integration(self):
        """Test environment variable override integration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Set base configuration
            config_manager.set("training.learning_rate", 2e-4)
            config_manager.set("training.batch_size", 32)
            config_manager.set("model.name", "default-model")
            
            # Set environment variables
            env_vars = {
                "TRAINING_LEARNING_RATE": "1e-4",
                "TRAINING_BATCH_SIZE": "64",
                "MODEL_NAME": "env-override-model"
            }
            
            # Apply environment overrides
            original_env = {}
            try:
                for key, value in env_vars.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                # Enable environment override
                config_manager.enable_environment_override()
                
                # Verify overrides
                assert config_manager.get("training.learning_rate") == 1e-4
                assert config_manager.get("training.batch_size") == 64
                assert config_manager.get("model.name") == "env-override-model"
                
            finally:
                # Clean up environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
    
    def test_configuration_validation_integration(self):
        """Test configuration validation integration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Define validation schema
            validation_schema = {
                "model": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "hidden_size": {"type": "integer", "minimum": 64, "maximum": 4096},
                        "architecture": {"type": "string", "enum": ["transformer", "lstm", "cnn"]}
                    },
                    "required": ["name", "architecture"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": 1e-1},
                        "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
                        "epochs": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                    "required": ["learning_rate", "batch_size"]
                }
            }
            
            config_manager.set_validation_schema(validation_schema)
            
            # Test valid configuration
            valid_config = {
                "model": {
                    "name": "test-model",
                    "architecture": "transformer",
                    "hidden_size": 768
                },
                "training": {
                    "learning_rate": 2e-4,
                    "batch_size": 32,
                    "epochs": 5
                }
            }
            
            validation_result = config_manager.validate_config(valid_config)
            assert validation_result["valid"] is True
            assert len(validation_result["errors"]) == 0
            
            # Test invalid configuration
            invalid_config = {
                "model": {
                    "name": "",  # Too short
                    "architecture": "invalid_arch",  # Not in enum
                    "hidden_size": 32  # Too small
                },
                "training": {
                    "learning_rate": 1.0,  # Too large
                    "batch_size": 0  # Too small
                }
            }
            
            validation_result = config_manager.validate_config(invalid_config)
            assert validation_result["valid"] is False
            assert len(validation_result["errors"]) > 0
    
    def test_configuration_hot_reload_integration(self):
        """Test configuration hot reload integration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            event_bus = EventBus()
            
            # Set up hot reload monitoring
            config_changes = []
            
            def config_change_handler(event):
                config_changes.append({
                    "path": event.data["path"],
                    "old_value": event.data["old_value"],
                    "new_value": event.data["new_value"],
                    "timestamp": event.timestamp
                })
            
            event_bus.subscribe(EventType.CONFIGURATION_CHANGED, config_change_handler)
            config_manager.set_event_bus(event_bus)
            
            # Initial configuration
            config_manager.set("model.learning_rate", 2e-4)
            config_manager.set("training.batch_size", 32)
            
            # Enable hot reload
            config_manager.enable_hot_reload()
            
            # Simulate configuration changes
            config_manager.set("model.learning_rate", 1e-4)
            config_manager.set("training.batch_size", 64)
            config_manager.set("training.new_param", "new_value")
            
            # Wait for events to propagate
            time.sleep(0.1)
            
            # Verify events were fired
            assert len(config_changes) >= 3
            
            # Check specific changes
            lr_changes = [c for c in config_changes if c["path"] == "model.learning_rate"]
            assert len(lr_changes) >= 1
            assert lr_changes[0]["old_value"] == 2e-4
            assert lr_changes[0]["new_value"] == 1e-4
            
            batch_changes = [c for c in config_changes if c["path"] == "training.batch_size"]
            assert len(batch_changes) >= 1
            assert batch_changes[0]["old_value"] == 32
            assert batch_changes[0]["new_value"] == 64


class TestPipelineConfigurationIntegration:
    """Test pipeline configuration integration."""
    
    def test_pipeline_configuration_coordination(self):
        """Test coordination between pipeline components through configuration."""
        with mock_dependencies_context() as env:
            pipeline_config = PipelineConfigManager()
            
            # Configure multiple pipeline stages
            pipeline_stages = {
                "data_preprocessing": {
                    "batch_size": 100,
                    "max_length": 512,
                    "tokenizer": "distilbert-base-uncased",
                    "preprocessing_workers": 4
                },
                "training": {
                    "model_name": "distilbert-base-uncased",
                    "learning_rate": 2e-4,
                    "batch_size": 32,
                    "epochs": 5,
                    "gradient_accumulation_steps": 1
                },
                "evaluation": {
                    "batch_size": 64,
                    "metrics": ["accuracy", "f1", "precision", "recall"],
                    "eval_steps": 500
                },
                "inference": {
                    "batch_size": 128,
                    "max_length": 512,
                    "use_cache": True
                }
            }
            
            # Set pipeline configuration
            for stage, config in pipeline_stages.items():
                for key, value in config.items():
                    pipeline_config.set(f"{stage}.{key}", value)
            
            # Test configuration consistency validation
            consistency_check = pipeline_config.validate_consistency([
                ("data_preprocessing.tokenizer", "training.model_name"),
                ("data_preprocessing.max_length", "inference.max_length")
            ])
            
            assert consistency_check["valid"] is True
            
            # Test configuration propagation
            pipeline_config.set("global.max_length", 256)
            pipeline_config.propagate_setting("global.max_length", [
                "data_preprocessing.max_length",
                "inference.max_length"
            ])
            
            assert pipeline_config.get("data_preprocessing.max_length") == 256
            assert pipeline_config.get("inference.max_length") == 256
    
    def test_cross_component_configuration_dependencies(self):
        """Test configuration dependencies across components."""
        with mock_dependencies_context() as env:
            pipeline_config = PipelineConfigManager()
            
            # Set up dependency relationships
            dependencies = {
                "training.model_name": ["evaluation.model_name", "inference.model_name"],
                "training.batch_size": ["data_preprocessing.training_batch_size"],
                "data_preprocessing.tokenizer": ["training.tokenizer", "inference.tokenizer"]
            }
            
            pipeline_config.set_dependencies(dependencies)
            
            # Set primary configuration
            pipeline_config.set("training.model_name", "bert-base-uncased")
            
            # Verify dependent configurations are updated
            assert pipeline_config.get("evaluation.model_name") == "bert-base-uncased"
            assert pipeline_config.get("inference.model_name") == "bert-base-uncased"
            
            # Test batch size dependency
            pipeline_config.set("training.batch_size", 64)
            assert pipeline_config.get("data_preprocessing.training_batch_size") == 64
            
            # Test tokenizer dependency
            pipeline_config.set("data_preprocessing.tokenizer", "roberta-base")
            assert pipeline_config.get("training.tokenizer") == "roberta-base"
            assert pipeline_config.get("inference.tokenizer") == "roberta-base"
    
    def test_dynamic_configuration_updates(self):
        """Test dynamic configuration updates during pipeline execution."""
        with mock_dependencies_context() as env:
            pipeline_config = PipelineConfigManager()
            
            # Set initial configuration
            pipeline_config.set("training.learning_rate", 2e-4)
            pipeline_config.set("training.batch_size", 32)
            pipeline_config.set("training.current_epoch", 0)
            
            # Simulate dynamic learning rate scheduling
            lr_schedule = [
                {"epoch": 0, "lr": 2e-4},
                {"epoch": 1, "lr": 1.5e-4},
                {"epoch": 2, "lr": 1e-4},
                {"epoch": 3, "lr": 5e-5}
            ]
            
            for schedule_item in lr_schedule:
                epoch = schedule_item["epoch"]
                lr = schedule_item["lr"]
                
                # Update epoch
                pipeline_config.set("training.current_epoch", epoch)
                
                # Update learning rate dynamically
                pipeline_config.set("training.learning_rate", lr)
                
                # Verify update was applied
                assert pipeline_config.get("training.learning_rate") == lr
                assert pipeline_config.get("training.current_epoch") == epoch
            
            # Test configuration history
            lr_history = pipeline_config.get_history("training.learning_rate")
            assert len(lr_history) >= len(lr_schedule)
            
            # Verify final state
            assert pipeline_config.get("training.learning_rate") == 5e-5
            assert pipeline_config.get("training.current_epoch") == 3


class TestConfigurationVersioningIntegration:
    """Test configuration versioning and rollback integration."""
    
    def test_configuration_versioning_workflow(self):
        """Test complete configuration versioning workflow."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Enable versioning
            config_manager.enable_versioning()
            
            # Version 1.0 - Initial configuration
            config_v1 = {
                "model": {"name": "distilbert", "hidden_size": 768},
                "training": {"lr": 2e-4, "batch_size": 32}
            }
            
            for path, value in self._flatten_config(config_v1).items():
                config_manager.set(path, value)
            
            version_1 = config_manager.create_version("1.0", "Initial configuration")
            assert version_1["version"] == "1.0"
            assert version_1["description"] == "Initial configuration"
            
            # Version 1.1 - Update learning rate
            config_manager.set("training.lr", 1e-4)
            config_manager.set("training.warmup_steps", 1000)
            
            version_11 = config_manager.create_version("1.1", "Reduced learning rate, added warmup")
            assert version_11["version"] == "1.1"
            
            # Version 2.0 - Major changes
            config_manager.set("model.name", "bert-large")
            config_manager.set("model.hidden_size", 1024)
            config_manager.set("training.batch_size", 16)
            
            version_2 = config_manager.create_version("2.0", "Upgraded to BERT-large")
            assert version_2["version"] == "2.0"
            
            # Test version listing
            versions = config_manager.list_versions()
            assert len(versions) >= 3
            version_numbers = [v["version"] for v in versions]
            assert "1.0" in version_numbers
            assert "1.1" in version_numbers
            assert "2.0" in version_numbers
            
            # Test rollback to version 1.1
            rollback_result = config_manager.rollback_to_version("1.1")
            assert rollback_result["success"] is True
            
            # Verify rollback
            assert config_manager.get("model.name") == "distilbert"
            assert config_manager.get("model.hidden_size") == 768
            assert config_manager.get("training.lr") == 1e-4
            assert config_manager.get("training.warmup_steps") == 1000
            assert config_manager.get("training.batch_size") == 32
    
    def test_configuration_branching_and_merging(self):
        """Test configuration branching and merging."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            config_manager.enable_versioning()
            
            # Create main branch
            config_manager.set("model.name", "base-model")
            config_manager.set("training.lr", 2e-4)
            main_version = config_manager.create_version("main-1.0", "Main branch v1.0")
            
            # Create experiment branch
            exp_branch = config_manager.create_branch("experiment", "main-1.0")
            assert exp_branch["success"] is True
            
            # Switch to experiment branch
            config_manager.switch_branch("experiment")
            
            # Make experimental changes
            config_manager.set("model.name", "experimental-model")
            config_manager.set("training.lr", 5e-4)
            config_manager.set("training.experimental_feature", True)
            
            exp_version = config_manager.create_version("exp-1.0", "Experimental changes")
            
            # Switch back to main
            config_manager.switch_branch("main")
            
            # Verify main branch is unchanged
            assert config_manager.get("model.name") == "base-model"
            assert config_manager.get("training.lr") == 2e-4
            assert config_manager.get("training.experimental_feature") is None
            
            # Test merge
            merge_result = config_manager.merge_branch("experiment", strategy="selective")
            assert merge_result["success"] is True
            
            # Verify selective merge (only non-conflicting changes)
            assert config_manager.get("training.experimental_feature") is True
            # model.name and training.lr should remain from main due to conflicts
    
    def _flatten_config(self, config, prefix=""):
        """Helper to flatten nested configuration."""
        flat = {}
        for key, value in config.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, path))
            else:
                flat[path] = value
        return flat


class TestConfigurationSecurityIntegration:
    """Test configuration security and secret management integration."""
    
    def test_secret_management_integration(self):
        """Test secret management integration with configuration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            secret_manager = env.get_infrastructure('secret_manager')
            
            # Store secrets
            secrets = {
                "huggingface_token": "hf_secret_token_123",
                "openai_api_key": "sk-openai_secret_key_456",
                "database_password": "db_password_789"
            }
            
            for secret_name, secret_value in secrets.items():
                secret_manager.store_secret(secret_name, secret_value)
            
            # Configure with secret references
            config_manager.set("huggingface.token", "${secret:huggingface_token}")
            config_manager.set("openai.api_key", "${secret:openai_api_key}")
            config_manager.set("database.password", "${secret:database_password}")
            config_manager.set("database.host", "localhost")  # Non-secret
            
            # Enable secret resolution
            config_manager.set_secret_manager(secret_manager)
            config_manager.enable_secret_resolution()
            
            # Verify secret resolution
            assert config_manager.get("huggingface.token") == "hf_secret_token_123"
            assert config_manager.get("openai.api_key") == "sk-openai_secret_key_456"
            assert config_manager.get("database.password") == "db_password_789"
            assert config_manager.get("database.host") == "localhost"
            
            # Test secret rotation
            secret_manager.rotate_secret("huggingface_token", "hf_new_token_999")
            
            # Verify rotated secret is resolved
            assert config_manager.get("huggingface.token") == "hf_new_token_999"
    
    def test_configuration_encryption_integration(self):
        """Test configuration encryption integration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Set encryption key
            encryption_key = "test_encryption_key_32_chars_long"
            config_manager.set_encryption_key(encryption_key)
            
            # Set sensitive configuration
            sensitive_config = {
                "api_keys.openai": "sk-sensitive_openai_key",
                "api_keys.huggingface": "hf_sensitive_hf_key",
                "database.connection_string": "postgresql://user:pass@host/db",
                "regular.setting": "not_sensitive_value"
            }
            
            # Mark certain paths as sensitive
            config_manager.mark_sensitive([
                "api_keys.*",
                "database.connection_string"
            ])
            
            for path, value in sensitive_config.items():
                config_manager.set(path, value)
            
            # Test encryption/decryption
            raw_config = config_manager.get_raw_config()
            
            # Sensitive values should be encrypted in raw config
            assert raw_config["api_keys"]["openai"] != "sk-sensitive_openai_key"
            assert raw_config["database"]["connection_string"] != "postgresql://user:pass@host/db"
            
            # Non-sensitive values should remain plain
            assert raw_config["regular"]["setting"] == "not_sensitive_value"
            
            # Decrypted values should be correct
            assert config_manager.get("api_keys.openai") == "sk-sensitive_openai_key"
            assert config_manager.get("database.connection_string") == "postgresql://user:pass@host/db"
            assert config_manager.get("regular.setting") == "not_sensitive_value"
    
    def test_configuration_access_control_integration(self):
        """Test configuration access control integration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Set up role-based access control
            roles = {
                "admin": ["*"],  # Full access
                "trainer": ["training.*", "model.*", "data.*"],
                "viewer": ["training.status", "model.name", "evaluation.*"],
                "api": ["inference.*", "model.name"]
            }
            
            config_manager.set_access_control(roles)
            
            # Set various configurations
            configs = {
                "training.learning_rate": 2e-4,
                "training.status": "running",
                "model.name": "distilbert",
                "model.checkpoint_path": "/models/checkpoint",
                "data.train_path": "/data/train.json",
                "evaluation.accuracy": 0.92,
                "inference.batch_size": 128,
                "system.secret_key": "super_secret",
                "admin.debug_mode": True
            }
            
            for path, value in configs.items():
                config_manager.set(path, value)
            
            # Test admin access (should access everything)
            config_manager.set_current_role("admin")
            assert config_manager.get("system.secret_key") == "super_secret"
            assert config_manager.get("training.learning_rate") == 2e-4
            
            # Test trainer access
            config_manager.set_current_role("trainer")
            assert config_manager.get("training.learning_rate") == 2e-4
            assert config_manager.get("model.name") == "distilbert"
            
            with pytest.raises(PermissionError):
                config_manager.get("system.secret_key")
            
            # Test viewer access
            config_manager.set_current_role("viewer")
            assert config_manager.get("training.status") == "running"
            assert config_manager.get("evaluation.accuracy") == 0.92
            
            with pytest.raises(PermissionError):
                config_manager.get("training.learning_rate")
            
            # Test API access
            config_manager.set_current_role("api")
            assert config_manager.get("inference.batch_size") == 128
            assert config_manager.get("model.name") == "distilbert"
            
            with pytest.raises(PermissionError):
                config_manager.get("training.status")


class TestCrossComponentConfigurationIntegration:
    """Test configuration integration across multiple platform components."""
    
    def test_training_configuration_propagation(self):
        """Test training configuration propagation across components."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            event_bus = EventBus()
            
            # Set up component integration
            components = {
                "trainer": Mock(),
                "data_loader": Mock(),
                "model": Mock(),
                "optimizer": Mock(),
                "scheduler": Mock()
            }
            
            # Subscribe components to configuration changes
            def create_config_handler(component_name, component):
                def handler(event):
                    if event.data["path"].startswith(component_name):
                        component.update_config(event.data["path"], event.data["new_value"])
                return handler
            
            for name, component in components.items():
                handler = create_config_handler(name, component)
                event_bus.subscribe(EventType.CONFIGURATION_CHANGED, handler)
            
            config_manager.set_event_bus(event_bus)
            
            # Set training configuration
            training_config = {
                "trainer.learning_rate": 2e-4,
                "trainer.batch_size": 32,
                "trainer.epochs": 5,
                "data_loader.num_workers": 4,
                "data_loader.shuffle": True,
                "model.hidden_size": 768,
                "model.num_layers": 12,
                "optimizer.type": "adamw",
                "optimizer.weight_decay": 0.01,
                "scheduler.type": "linear",
                "scheduler.warmup_steps": 1000
            }
            
            for path, value in training_config.items():
                config_manager.set(path, value)
            
            # Wait for events to propagate
            time.sleep(0.1)
            
            # Verify components received updates
            components["trainer"].update_config.assert_called()
            components["data_loader"].update_config.assert_called()
            components["model"].update_config.assert_called()
            components["optimizer"].update_config.assert_called()
            components["scheduler"].update_config.assert_called()
            
            # Test configuration consistency
            assert config_manager.get("trainer.batch_size") == 32
            assert config_manager.get("data_loader.shuffle") is True
            assert config_manager.get("model.hidden_size") == 768
    
    def test_inference_configuration_integration(self):
        """Test inference configuration integration."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            
            # Set inference configuration
            inference_config = {
                "inference.model_path": "/models/fine_tuned_model",
                "inference.tokenizer_path": "/models/tokenizer",
                "inference.batch_size": 64,
                "inference.max_length": 512,
                "inference.device": "cuda:0",
                "inference.use_cache": True,
                "inference.temperature": 0.7,
                "inference.top_p": 0.9,
                "inference.top_k": 50,
                "inference.do_sample": True
            }
            
            for path, value in inference_config.items():
                config_manager.set(path, value)
            
            # Test configuration validation for inference
            inference_schema = {
                "inference": {
                    "type": "object",
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1},
                        "max_length": {"type": "integer", "minimum": 1, "maximum": 2048},
                        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                        "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                }
            }
            
            config_manager.set_validation_schema(inference_schema)
            
            # Verify valid configuration
            current_config = {"inference": {}}
            for path, value in inference_config.items():
                key = path.split(".", 1)[1]
                current_config["inference"][key] = value
            
            validation_result = config_manager.validate_config(current_config)
            assert validation_result["valid"] is True
            
            # Test configuration export for inference
            inference_only_config = config_manager.export_subset("inference.*")
            assert len(inference_only_config) == len(inference_config)
            assert inference_only_config["inference.batch_size"] == 64
            assert inference_only_config["inference.temperature"] == 0.7
    
    def test_multi_environment_configuration_integration(self):
        """Test multi-environment configuration integration."""
        with mock_dependencies_context() as env:
            # Create environment-specific configuration managers
            environments = ["development", "staging", "production"]
            config_managers = {}
            
            for environment in environments:
                config_manager = ConfigManager()
                config_manager.set_environment(environment)
                config_managers[environment] = config_manager
            
            # Set base configuration
            base_config = {
                "model.name": "base-model",
                "training.epochs": 5,
                "data.validation_split": 0.2
            }
            
            # Set environment-specific configurations
            env_configs = {
                "development": {
                    "training.learning_rate": 1e-3,  # Higher for faster experimentation
                    "training.batch_size": 8,        # Smaller for limited resources
                    "data.cache_dir": "/tmp/dev_cache",
                    "logging.level": "DEBUG"
                },
                "staging": {
                    "training.learning_rate": 2e-4,  # Production-like
                    "training.batch_size": 32,       # Medium size
                    "data.cache_dir": "/staging/cache",
                    "logging.level": "INFO"
                },
                "production": {
                    "training.learning_rate": 2e-4,  # Stable
                    "training.batch_size": 64,       # Optimized for throughput
                    "data.cache_dir": "/prod/cache",
                    "logging.level": "WARNING"
                }
            }
            
            # Apply configurations
            for environment, config_manager in config_managers.items():
                # Set base config
                for path, value in base_config.items():
                    config_manager.set(path, value)
                
                # Set environment-specific config
                if environment in env_configs:
                    for path, value in env_configs[environment].items():
                        config_manager.set(path, value)
            
            # Verify environment-specific differences
            assert config_managers["development"].get("training.batch_size") == 8
            assert config_managers["staging"].get("training.batch_size") == 32
            assert config_managers["production"].get("training.batch_size") == 64
            
            assert config_managers["development"].get("logging.level") == "DEBUG"
            assert config_managers["staging"].get("logging.level") == "INFO"
            assert config_managers["production"].get("logging.level") == "WARNING"
            
            # Verify common configuration
            for config_manager in config_managers.values():
                assert config_manager.get("model.name") == "base-model"
                assert config_manager.get("training.epochs") == 5
                assert config_manager.get("data.validation_split") == 0.2
            
            # Test configuration promotion
            # Promote staging config to production
            staging_config = config_managers["staging"].export_all()
            production_manager = config_managers["production"]
            
            for path, value in staging_config.items():
                if not path.startswith("data.cache_dir"):  # Don't promote cache dir
                    production_manager.set(path, value)
            
            # Verify promotion (should keep production cache dir)
            assert production_manager.get("training.learning_rate") == 2e-4
            assert production_manager.get("training.batch_size") == 32  # From staging
            assert production_manager.get("data.cache_dir") == "/prod/cache"  # Unchanged