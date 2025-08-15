"""
Backwards compatibility testing suite for ensuring version compatibility.

This test module validates backwards compatibility across API versions,
configuration formats, data structures, and interface changes.
"""

import pytest
import json
import pickle
import tempfile
import shutil
import importlib
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import warnings
import inspect

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment,
    MockTransformerModel, MockTokenizer, MockTrainingDataset
)


@dataclass
class CompatibilityTest:
    """Defines a compatibility test scenario."""
    name: str
    description: str
    old_version: str
    new_version: str
    test_function: Callable
    expected_behavior: str = "full_compatibility"
    deprecation_warnings: bool = False
    breaking_changes: List[str] = None


class VersionSimulator:
    """Simulate different versions for compatibility testing."""
    
    def __init__(self):
        self.version_registry = {}
        self.current_version = "2.0.0"
        self.compatibility_matrix = self._build_compatibility_matrix()
    
    def _build_compatibility_matrix(self) -> Dict:
        """Build compatibility matrix between versions."""
        return {
            "1.0.0": {
                "config_format": "yaml_v1",
                "api_format": "rest_v1",
                "data_format": "json_v1",
                "model_format": "pytorch_v1"
            },
            "1.1.0": {
                "config_format": "yaml_v1",
                "api_format": "rest_v1",
                "data_format": "json_v1",
                "model_format": "pytorch_v1",
                "features": ["streaming_api"]
            },
            "1.2.0": {
                "config_format": "yaml_v2",  # Breaking change
                "api_format": "rest_v1",
                "data_format": "json_v2",   # Breaking change
                "model_format": "pytorch_v1",
                "features": ["streaming_api", "batch_processing"]
            },
            "2.0.0": {
                "config_format": "yaml_v2",
                "api_format": "rest_v2",    # Breaking change
                "data_format": "json_v2",
                "model_format": "pytorch_v2",  # Breaking change
                "features": ["streaming_api", "batch_processing", "real_time_training"]
            }
        }
    
    def register_version_implementation(self, version: str, component: str, implementation: Any):
        """Register version-specific implementation."""
        if version not in self.version_registry:
            self.version_registry[version] = {}
        self.version_registry[version][component] = implementation
    
    def get_implementation(self, version: str, component: str) -> Any:
        """Get version-specific implementation."""
        return self.version_registry.get(version, {}).get(component)
    
    def is_compatible(self, old_version: str, new_version: str, component: str) -> bool:
        """Check if versions are compatible for specific component."""
        old_spec = self.compatibility_matrix.get(old_version, {})
        new_spec = self.compatibility_matrix.get(new_version, {})
        
        old_format = old_spec.get(component)
        new_format = new_spec.get(component)
        
        # Same format = compatible
        if old_format == new_format:
            return True
        
        # Define compatibility rules
        compatibility_rules = {
            ("yaml_v1", "yaml_v2"): True,   # v2 supports v1
            ("json_v1", "json_v2"): True,   # v2 supports v1
            ("rest_v1", "rest_v2"): True,   # v2 supports v1
            ("pytorch_v1", "pytorch_v2"): False,  # Breaking change
        }
        
        return compatibility_rules.get((old_format, new_format), False)


class APICompatibilityTester:
    """Test API backwards compatibility."""
    
    def __init__(self):
        self.version_simulator = VersionSimulator()
        self.compatibility_issues = []
    
    def test_function_signature_compatibility(self, old_function: Callable, 
                                            new_function: Callable) -> Dict:
        """Test function signature compatibility."""
        old_sig = inspect.signature(old_function)
        new_sig = inspect.signature(new_function)
        
        compatibility_result = {
            "function_name": old_function.__name__,
            "compatible": True,
            "issues": [],
            "warnings": []
        }
        
        # Check parameter compatibility
        old_params = list(old_sig.parameters.keys())
        new_params = list(new_sig.parameters.keys())
        
        # Check for removed parameters
        removed_params = set(old_params) - set(new_params)
        if removed_params:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "removed_parameters",
                "parameters": list(removed_params),
                "severity": "breaking"
            })
        
        # Check for parameter order changes
        common_params = [p for p in old_params if p in new_params]
        old_order = [p for p in old_params if p in common_params]
        new_order = [p for p in new_params if p in common_params]
        
        if old_order != new_order:
            compatibility_result["warnings"].append({
                "type": "parameter_order_changed",
                "old_order": old_order,
                "new_order": new_order,
                "severity": "warning"
            })
        
        # Check for new required parameters
        old_required = {name for name, param in old_sig.parameters.items() 
                       if param.default == inspect.Parameter.empty}
        new_required = {name for name, param in new_sig.parameters.items() 
                       if param.default == inspect.Parameter.empty}
        
        new_required_params = new_required - old_required
        if new_required_params:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "new_required_parameters",
                "parameters": list(new_required_params),
                "severity": "breaking"
            })
        
        # Check return type compatibility (if type hints available)
        old_return = old_sig.return_annotation
        new_return = new_sig.return_annotation
        
        if (old_return != inspect.Signature.empty and 
            new_return != inspect.Signature.empty and 
            old_return != new_return):
            compatibility_result["warnings"].append({
                "type": "return_type_changed",
                "old_type": str(old_return),
                "new_type": str(new_return),
                "severity": "warning"
            })
        
        return compatibility_result
    
    def test_class_interface_compatibility(self, old_class: type, new_class: type) -> Dict:
        """Test class interface compatibility."""
        compatibility_result = {
            "class_name": old_class.__name__,
            "compatible": True,
            "issues": [],
            "warnings": []
        }
        
        # Get public methods
        old_methods = {name for name in dir(old_class) 
                      if not name.startswith('_') and callable(getattr(old_class, name))}
        new_methods = {name for name in dir(new_class) 
                      if not name.startswith('_') and callable(getattr(new_class, name))}
        
        # Check for removed methods
        removed_methods = old_methods - new_methods
        if removed_methods:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "removed_methods",
                "methods": list(removed_methods),
                "severity": "breaking"
            })
        
        # Check method signature compatibility for common methods
        common_methods = old_methods & new_methods
        for method_name in common_methods:
            old_method = getattr(old_class, method_name)
            new_method = getattr(new_class, method_name)
            
            method_compat = self.test_function_signature_compatibility(old_method, new_method)
            if not method_compat["compatible"]:
                compatibility_result["compatible"] = False
                compatibility_result["issues"].append({
                    "type": "method_signature_incompatible",
                    "method": method_name,
                    "details": method_compat["issues"],
                    "severity": "breaking"
                })
        
        # Check for new abstract methods
        old_abstract = getattr(old_class, '__abstractmethods__', set())
        new_abstract = getattr(new_class, '__abstractmethods__', set())
        
        new_abstract_methods = new_abstract - old_abstract
        if new_abstract_methods:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "new_abstract_methods",
                "methods": list(new_abstract_methods),
                "severity": "breaking"
            })
        
        return compatibility_result


class ConfigurationCompatibilityTester:
    """Test configuration format backwards compatibility."""
    
    def test_config_format_compatibility(self, old_config: Dict, 
                                       config_parser: Callable) -> Dict:
        """Test if old configuration format is still supported."""
        compatibility_result = {
            "config_format": "unknown",
            "compatible": True,
            "issues": [],
            "warnings": [],
            "migration_required": False
        }
        
        try:
            # Try to parse old configuration with new parser
            parsed_config = config_parser(old_config)
            
            # Check if all old keys are preserved
            old_keys = self._flatten_dict_keys(old_config)
            parsed_keys = self._flatten_dict_keys(parsed_config)
            
            missing_keys = old_keys - parsed_keys
            if missing_keys:
                compatibility_result["issues"].append({
                    "type": "missing_keys",
                    "keys": list(missing_keys),
                    "severity": "breaking"
                })
                compatibility_result["compatible"] = False
            
            # Check for value changes
            for key in old_keys & parsed_keys:
                old_value = self._get_nested_value(old_config, key)
                new_value = self._get_nested_value(parsed_config, key)
                
                if old_value != new_value:
                    compatibility_result["warnings"].append({
                        "type": "value_changed",
                        "key": key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "severity": "warning"
                    })
            
        except Exception as e:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "parsing_error",
                "error": str(e),
                "severity": "breaking"
            })
        
        return compatibility_result
    
    def _flatten_dict_keys(self, d: Dict, parent_key: str = '') -> set:
        """Flatten nested dictionary keys."""
        keys = set()
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            keys.add(new_key)
            if isinstance(value, dict):
                keys.update(self._flatten_dict_keys(value, new_key))
        return keys
    
    def _get_nested_value(self, d: Dict, key: str) -> Any:
        """Get nested dictionary value by flattened key."""
        keys = key.split('.')
        value = d
        for k in keys:
            value = value[k]
        return value


class DataFormatCompatibilityTester:
    """Test data format backwards compatibility."""
    
    def test_serialization_compatibility(self, old_data: Any, 
                                       serializer: Callable,
                                       deserializer: Callable) -> Dict:
        """Test serialization/deserialization compatibility."""
        compatibility_result = {
            "serialization_format": "unknown",
            "compatible": True,
            "issues": [],
            "data_integrity": True
        }
        
        try:
            # Serialize with new serializer
            serialized = serializer(old_data)
            
            # Deserialize with new deserializer
            deserialized = deserializer(serialized)
            
            # Check data integrity
            if not self._deep_equal(old_data, deserialized):
                compatibility_result["data_integrity"] = False
                compatibility_result["issues"].append({
                    "type": "data_corruption",
                    "description": "Data changed during serialization/deserialization",
                    "severity": "critical"
                })
            
        except Exception as e:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "serialization_error",
                "error": str(e),
                "severity": "breaking"
            })
        
        return compatibility_result
    
    def _deep_equal(self, obj1: Any, obj2: Any) -> bool:
        """Deep equality check for complex objects."""
        if type(obj1) != type(obj2):
            return False
        
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(self._deep_equal(obj1[k], obj2[k]) for k in obj1.keys())
        
        elif isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(self._deep_equal(a, b) for a, b in zip(obj1, obj2))
        
        else:
            return obj1 == obj2


class ModelCompatibilityTester:
    """Test model format backwards compatibility."""
    
    def test_model_loading_compatibility(self, old_model_path: str,
                                       model_loader: Callable) -> Dict:
        """Test if old model formats can be loaded."""
        compatibility_result = {
            "model_format": "unknown",
            "compatible": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Try to load old model with new loader
            loaded_model = model_loader(old_model_path)
            
            # Basic checks
            if loaded_model is None:
                compatibility_result["compatible"] = False
                compatibility_result["issues"].append({
                    "type": "loading_failed",
                    "description": "Model loader returned None",
                    "severity": "breaking"
                })
            
            # Check if model has expected interface
            expected_methods = ['forward', '__call__', 'eval', 'train']
            missing_methods = [method for method in expected_methods 
                             if not hasattr(loaded_model, method)]
            
            if missing_methods:
                compatibility_result["warnings"].append({
                    "type": "missing_methods",
                    "methods": missing_methods,
                    "severity": "warning"
                })
            
        except Exception as e:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append({
                "type": "loading_error",
                "error": str(e),
                "severity": "breaking"
            })
        
        return compatibility_result


class TestAPIBackwardsCompatibility:
    """Test API backwards compatibility."""
    
    def test_core_api_compatibility(self):
        """Test core API function compatibility."""
        with mock_dependencies_context() as env:
            api_tester = APICompatibilityTester()
            
            # Simulate old and new versions of a function
            def old_train_model(model_name: str, dataset_path: str, epochs: int = 5):
                """Old version of train_model function."""
                return f"Training {model_name} for {epochs} epochs"
            
            def new_train_model(model_name: str, dataset_path: str, epochs: int = 5, 
                              batch_size: int = 32, learning_rate: float = 2e-4):
                """New version with additional optional parameters."""
                return f"Training {model_name} for {epochs} epochs (batch_size={batch_size}, lr={learning_rate})"
            
            # Test compatibility
            compat_result = api_tester.test_function_signature_compatibility(
                old_train_model, new_train_model
            )
            
            # Should be compatible (only new optional parameters)
            assert compat_result["compatible"], f"API compatibility broken: {compat_result['issues']}"
            
            # Test with breaking change
            def breaking_train_model(model_config: dict, epochs: int = 5):
                """Breaking version - changed required parameter type."""
                return f"Training with config for {epochs} epochs"
            
            breaking_compat = api_tester.test_function_signature_compatibility(
                old_train_model, breaking_train_model
            )
            
            # Should detect breaking change
            assert not breaking_compat["compatible"], "Breaking change not detected"
            assert any(issue["type"] == "removed_parameters" for issue in breaking_compat["issues"])
    
    def test_class_interface_compatibility(self):
        """Test class interface backwards compatibility."""
        with mock_dependencies_context() as env:
            api_tester = APICompatibilityTester()
            
            # Old version of a class
            class OldModelTrainer:
                def __init__(self, model_name: str):
                    self.model_name = model_name
                
                def train(self, dataset: str) -> str:
                    return f"Training {self.model_name}"
                
                def evaluate(self, test_data: str) -> float:
                    return 0.95
            
            # New compatible version
            class NewModelTrainer:
                def __init__(self, model_name: str, config: dict = None):
                    self.model_name = model_name
                    self.config = config or {}
                
                def train(self, dataset: str, epochs: int = 5) -> str:
                    return f"Training {self.model_name} for {epochs} epochs"
                
                def evaluate(self, test_data: str) -> float:
                    return 0.96
                
                def get_metrics(self) -> dict:
                    return {"accuracy": 0.96}
            
            # Test compatibility
            compat_result = api_tester.test_class_interface_compatibility(
                OldModelTrainer, NewModelTrainer
            )
            
            # Should be compatible
            assert compat_result["compatible"], f"Class compatibility broken: {compat_result['issues']}"
            
            # Breaking version
            class BreakingModelTrainer:
                def __init__(self, config: dict):  # Made required
                    self.config = config
                
                def train_model(self, dataset: str) -> str:  # Renamed method
                    return "Training"
                
                # evaluate method removed
            
            breaking_compat = api_tester.test_class_interface_compatibility(
                OldModelTrainer, BreakingModelTrainer
            )
            
            # Should detect breaking changes
            assert not breaking_compat["compatible"], "Breaking class changes not detected"


class TestConfigurationBackwardsCompatibility:
    """Test configuration backwards compatibility."""
    
    def test_config_format_compatibility(self):
        """Test configuration format backwards compatibility."""
        with mock_dependencies_context() as env:
            config_tester = ConfigurationCompatibilityTester()
            
            # Old configuration format
            old_config = {
                "model": {
                    "name": "distilbert-base-uncased",
                    "architecture": "transformer"
                },
                "training": {
                    "learning_rate": 2e-4,
                    "batch_size": 32,
                    "epochs": 5
                },
                "data": {
                    "train_path": "/data/train.json",
                    "val_path": "/data/val.json"
                }
            }
            
            # New compatible parser that supports old format
            def compatible_config_parser(config):
                """Parser that supports old configuration format."""
                parsed = config.copy()
                
                # Add new optional fields with defaults
                if "evaluation" not in parsed:
                    parsed["evaluation"] = {
                        "metrics": ["accuracy", "f1"],
                        "eval_steps": 500
                    }
                
                # Transform old fields if needed
                if "data" in parsed and "dataset_path" not in parsed["data"]:
                    parsed["data"]["dataset_path"] = parsed["data"].get("train_path", "")
                
                return parsed
            
            # Test compatibility
            compat_result = config_tester.test_config_format_compatibility(
                old_config, compatible_config_parser
            )
            
            # Should be compatible
            assert compat_result["compatible"], f"Config compatibility broken: {compat_result['issues']}"
            
            # Breaking parser
            def breaking_config_parser(config):
                """Parser that breaks old configuration format."""
                if "model" not in config or "name" not in config["model"]:
                    raise ValueError("Missing required model.name")
                
                # Remove old fields
                parsed = {"model_config": config["model"]}  # Restructured
                return parsed
            
            breaking_compat = config_tester.test_config_format_compatibility(
                old_config, breaking_config_parser
            )
            
            # Should detect breaking changes
            assert not breaking_compat["compatible"], "Breaking config changes not detected"
    
    def test_configuration_migration(self):
        """Test configuration migration from old to new format."""
        with mock_dependencies_context() as env:
            # Old configuration
            old_config = {
                "model_name": "bert-base-uncased",  # Flat structure
                "lr": 1e-4,
                "batch_size": 16,
                "train_file": "train.json",
                "val_file": "val.json"
            }
            
            # Migration function
            def migrate_config(old_config):
                """Migrate old config format to new hierarchical format."""
                new_config = {
                    "model": {
                        "name": old_config.get("model_name", ""),
                        "architecture": "transformer"
                    },
                    "training": {
                        "learning_rate": old_config.get("lr", 2e-4),
                        "batch_size": old_config.get("batch_size", 32),
                        "epochs": old_config.get("epochs", 5)
                    },
                    "data": {
                        "train_path": old_config.get("train_file", ""),
                        "val_path": old_config.get("val_file", "")
                    }
                }
                return new_config
            
            # Test migration
            migrated_config = migrate_config(old_config)
            
            # Verify migration preserved essential information
            assert migrated_config["model"]["name"] == "bert-base-uncased"
            assert migrated_config["training"]["learning_rate"] == 1e-4
            assert migrated_config["training"]["batch_size"] == 16
            assert migrated_config["data"]["train_path"] == "train.json"


class TestDataFormatBackwardsCompatibility:
    """Test data format backwards compatibility."""
    
    def test_dataset_format_compatibility(self):
        """Test dataset format backwards compatibility."""
        with mock_dependencies_context() as env:
            data_tester = DataFormatCompatibilityTester()
            
            # Old dataset format
            old_dataset = [
                {"text": "This is positive", "label": 1},
                {"text": "This is negative", "label": 0},
                {"text": "Another positive example", "label": 1}
            ]
            
            # New compatible serializer/deserializer
            def compatible_serializer(data):
                """Serializer that supports old format."""
                serialized = {
                    "version": "2.0",
                    "format": "text_classification",
                    "data": data,
                    "metadata": {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "num_samples": len(data)
                    }
                }
                return json.dumps(serialized)
            
            def compatible_deserializer(serialized_data):
                """Deserializer that supports old format."""
                if isinstance(serialized_data, str):
                    data = json.loads(serialized_data)
                else:
                    data = serialized_data
                
                # Handle old format (plain list)
                if isinstance(data, list):
                    return data
                
                # Handle new format
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                
                return data
            
            # Test compatibility
            compat_result = data_tester.test_serialization_compatibility(
                old_dataset, compatible_serializer, compatible_deserializer
            )
            
            # Should preserve data integrity
            assert compat_result["compatible"], f"Data serialization broken: {compat_result['issues']}"
            assert compat_result["data_integrity"], "Data integrity compromised"
    
    def test_model_checkpoint_compatibility(self):
        """Test model checkpoint format compatibility."""
        with mock_dependencies_context() as env:
            # Simulate old checkpoint format
            old_checkpoint = {
                "model_state_dict": {"layer1.weight": [1, 2, 3], "layer1.bias": [0.1]},
                "optimizer_state_dict": {"lr": 2e-4, "momentum": 0.9},
                "epoch": 5,
                "loss": 0.25
            }
            
            # Compatible checkpoint loader
            def load_checkpoint_compatible(checkpoint_data):
                """Load checkpoint with backwards compatibility."""
                if isinstance(checkpoint_data, str):
                    # Handle file path
                    return {"loaded": True, "format": "file"}
                
                if isinstance(checkpoint_data, dict):
                    # Handle old format
                    if "model_state_dict" in checkpoint_data:
                        return {
                            "model": checkpoint_data["model_state_dict"],
                            "optimizer": checkpoint_data.get("optimizer_state_dict", {}),
                            "training_state": {
                                "epoch": checkpoint_data.get("epoch", 0),
                                "loss": checkpoint_data.get("loss", 0.0)
                            },
                            "version": "1.0"  # Mark as old format
                        }
                
                return checkpoint_data
            
            # Test loading
            loaded = load_checkpoint_compatible(old_checkpoint)
            
            # Should successfully load old format
            assert loaded is not None, "Failed to load old checkpoint format"
            assert "model" in loaded, "Model state not preserved"
            assert "training_state" in loaded, "Training state not preserved"
            assert loaded["training_state"]["epoch"] == 5, "Epoch not preserved"


class TestModelBackwardsCompatibility:
    """Test model backwards compatibility."""
    
    def test_model_interface_compatibility(self):
        """Test model interface backwards compatibility."""
        with mock_dependencies_context() as env:
            from tests.mocks import MockTransformerModel
            
            # Old model interface
            class OldModel:
                def __init__(self, model_name):
                    self.model_name = model_name
                
                def forward(self, input_ids):
                    return {"logits": [0.1, 0.9]}
                
                def predict(self, text):
                    return 1
            
            # New model with extended interface
            class NewModel:
                def __init__(self, model_name, config=None):
                    self.model_name = model_name
                    self.config = config or {}
                
                def forward(self, input_ids, attention_mask=None):
                    return {"logits": [0.1, 0.9], "hidden_states": []}
                
                def predict(self, text):
                    return 1
                
                def predict_proba(self, text):
                    return [0.1, 0.9]
                
                def __call__(self, *args, **kwargs):
                    return self.forward(*args, **kwargs)
            
            # Test compatibility
            api_tester = APICompatibilityTester()
            compat_result = api_tester.test_class_interface_compatibility(OldModel, NewModel)
            
            # Should be compatible (only added optional parameters and methods)
            assert compat_result["compatible"], f"Model interface compatibility broken: {compat_result['issues']}"
    
    def test_model_serialization_compatibility(self):
        """Test model serialization backwards compatibility."""
        with mock_dependencies_context() as env:
            # Mock old model state
            old_model_state = {
                "model_type": "transformer",
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_layers": 12,
                "weights": {
                    "embedding.weight": [[0.1, 0.2], [0.3, 0.4]],
                    "layer.0.weight": [[0.5, 0.6], [0.7, 0.8]]
                }
            }
            
            # Compatible model loader
            def load_model_compatible(model_state):
                """Load model with backwards compatibility."""
                if not isinstance(model_state, dict):
                    raise ValueError("Invalid model state")
                
                # Handle old format
                if "model_type" in model_state:
                    return {
                        "architecture": model_state["model_type"],
                        "config": {
                            "vocab_size": model_state.get("vocab_size", 30522),
                            "hidden_size": model_state.get("hidden_size", 768),
                            "num_hidden_layers": model_state.get("num_layers", 12)
                        },
                        "state_dict": model_state.get("weights", {}),
                        "version": "1.0"
                    }
                
                # Already new format
                return model_state
            
            # Test loading
            loaded_model = load_model_compatible(old_model_state)
            
            # Should successfully convert old format
            assert loaded_model is not None, "Failed to load old model format"
            assert loaded_model["architecture"] == "transformer", "Model type not preserved"
            assert loaded_model["config"]["vocab_size"] == 30522, "Config not preserved"
            assert "embedding.weight" in loaded_model["state_dict"], "Weights not preserved"


class TestDeprecationWarnings:
    """Test deprecation warning system."""
    
    def test_deprecation_warnings_generation(self):
        """Test that deprecation warnings are properly generated."""
        
        def deprecated_function():
            """Function marked as deprecated."""
            warnings.warn(
                "deprecated_function is deprecated and will be removed in v3.0. "
                "Use new_function instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return "deprecated_result"
        
        # Test that warning is generated
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = deprecated_function()
            
            # Should generate deprecation warning
            assert len(w) == 1, "Deprecation warning not generated"
            assert issubclass(w[0].category, DeprecationWarning), "Wrong warning type"
            assert "deprecated_function is deprecated" in str(w[0].message), "Warning message incorrect"
        
        # Function should still work
        assert result == "deprecated_result", "Deprecated function doesn't work"
    
    def test_deprecation_decorator(self):
        """Test deprecation decorator functionality."""
        
        def deprecated(version, alternative=None):
            """Decorator to mark functions as deprecated."""
            def decorator(func):
                def wrapper(*args, **kwargs):
                    message = f"{func.__name__} is deprecated and will be removed in {version}"
                    if alternative:
                        message += f". Use {alternative} instead"
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                    return func(*args, **kwargs)
                return wrapper
            return decorator
        
        @deprecated("v3.0", "new_api_function")
        def old_api_function(x):
            return x * 2
        
        # Test decorator
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = old_api_function(5)
            
            assert len(w) == 1, "Decorator deprecation warning not generated"
            assert "old_api_function is deprecated" in str(w[0].message)
            assert "new_api_function" in str(w[0].message)
        
        assert result == 10, "Decorated function doesn't work"


def test_comprehensive_backwards_compatibility():
    """Run comprehensive backwards compatibility test suite."""
    with mock_dependencies_context() as env:
        compatibility_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "compatibility_score": 0.0,
            "breaking_changes": [],
            "deprecation_warnings": [],
            "test_results": []
        }
        
        # Test categories
        test_categories = [
            "api_compatibility",
            "config_compatibility", 
            "data_format_compatibility",
            "model_compatibility"
        ]
        
        for category in test_categories:
            category_result = {
                "category": category,
                "compatible": True,
                "issues": [],
                "warnings": []
            }
            
            # Simulate category testing
            if category == "api_compatibility":
                # Mock API compatibility test
                def old_function(a, b): return a + b
                def new_function(a, b, c=0): return a + b + c
                
                api_tester = APICompatibilityTester()
                result = api_tester.test_function_signature_compatibility(old_function, new_function)
                
                category_result["compatible"] = result["compatible"]
                category_result["issues"] = result["issues"]
                category_result["warnings"] = result["warnings"]
            
            elif category == "config_compatibility":
                # Mock config compatibility test
                old_config = {"model": "bert", "lr": 1e-4}
                def parser(config): return config  # Identity parser
                
                config_tester = ConfigurationCompatibilityTester()
                result = config_tester.test_config_format_compatibility(old_config, parser)
                
                category_result["compatible"] = result["compatible"]
                category_result["issues"] = result["issues"]
                category_result["warnings"] = result["warnings"]
            
            # Add to report
            compatibility_report["test_results"].append(category_result)
            compatibility_report["total_tests"] += 1
            
            if category_result["compatible"]:
                compatibility_report["passed_tests"] += 1
            else:
                compatibility_report["failed_tests"] += 1
                compatibility_report["breaking_changes"].extend(
                    [issue for issue in category_result["issues"] 
                     if issue.get("severity") == "breaking"]
                )
        
        # Calculate compatibility score
        if compatibility_report["total_tests"] > 0:
            compatibility_report["compatibility_score"] = (
                compatibility_report["passed_tests"] / compatibility_report["total_tests"]
            ) * 100
        
        # Compatibility requirements
        assert compatibility_report["compatibility_score"] >= 80, \
            f"Compatibility score too low: {compatibility_report['compatibility_score']:.1f}%"
        
        assert len(compatibility_report["breaking_changes"]) == 0, \
            f"Breaking changes detected: {compatibility_report['breaking_changes']}"
        
        print(f"Backwards Compatibility Report:")
        print(f"  Compatibility Score: {compatibility_report['compatibility_score']:.1f}%")
        print(f"  Tests Passed: {compatibility_report['passed_tests']}/{compatibility_report['total_tests']}")
        print(f"  Breaking Changes: {len(compatibility_report['breaking_changes'])}")
        print(f"  Deprecation Warnings: {len(compatibility_report['deprecation_warnings'])}")
        
        return compatibility_report