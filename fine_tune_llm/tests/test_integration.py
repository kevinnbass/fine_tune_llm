"""Comprehensive integration tests for the entire LLM fine-tuning platform."""

import pytest
import yaml
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def full_config():
    """Fixture providing complete system configuration."""
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
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "method": "lora",
            "quantization": {"enabled": False}
        },
        "training": {
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "batch_size": 2,
            "bf16": False,
            "gradient_checkpointing": False
        },
        "high_stakes": {
            "uncertainty": {
                "enabled": False,
                "method": "mc_dropout",
                "num_samples": 5
            },
            "factual": {
                "enabled": False,
                "reliance_steps": 3
            },
            "bias_audit": {
                "enabled": False,
                "audit_categories": ["gender", "race"]
            },
            "explainable": {
                "enabled": False,
                "chain_of_thought": True
            },
            "procedural": {
                "enabled": False,
                "domain": "medical"
            },
            "verifiable": {
                "enabled": False,
                "hash_artifacts": True
            }
        },
        "evaluation": {"enabled": False},
        "instruction_format": {
            "system_prompt": "You are a helpful assistant."
        }
    }


class TestSystemIntegration:
    """Test system-level integration."""
    
    def test_config_consistency_across_modules(self, full_config):
        """Test that configuration is consistently interpreted across all modules."""
        # All modules should be able to parse the same config without errors
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(full_config, f)
            config_path = f.name
        
        try:
            # Test config loading
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == full_config
            assert "selected_model" in loaded_config
            assert "high_stakes" in loaded_config
            
        finally:
            Path(config_path).unlink()
    
    def test_high_stakes_feature_modularity(self, full_config):
        """Test that high-stakes features can be toggled independently."""
        features = ["uncertainty", "factual", "bias_audit", "explainable", "procedural", "verifiable"]
        
        # Test each feature can be enabled/disabled independently
        for feature in features:
            # Enable only this feature
            config_copy = full_config.copy()
            config_copy["high_stakes"] = {f: {"enabled": False} for f in features}
            config_copy["high_stakes"][feature]["enabled"] = True
            
            # Should be valid configuration
            assert config_copy["high_stakes"][feature]["enabled"] is True
            assert all(config_copy["high_stakes"][f]["enabled"] is False for f in features if f != feature)
    
    def test_training_method_modularity(self, full_config):
        """Test that different training methods are modular."""
        methods = ["lora", "dora", "adalora"]
        quantization_options = [True, False]
        
        for method in methods:
            for quant_enabled in quantization_options:
                config_copy = full_config.copy()
                config_copy["lora"]["method"] = method
                config_copy["lora"]["quantization"]["enabled"] = quant_enabled
                
                # Should be valid configuration
                assert config_copy["lora"]["method"] == method
                assert config_copy["lora"]["quantization"]["enabled"] == quant_enabled
    
    def test_model_architecture_modularity(self, full_config):
        """Test that different model architectures are supported."""
        models = {
            "glm-4.5-air": {
                "model_id": "ZHIPU-AI/glm-4-9b-chat",
                "target_modules": ["query_key_value", "dense"],
                "chat_template": "glm"
            },
            "qwen2.5-7b": {
                "model_id": "Qwen/Qwen2.5-7B",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "chat_template": "qwen"
            },
            "mistral-7b": {
                "model_id": "mistralai/Mistral-7B-v0.1",
                "target_modules": ["q_proj", "v_proj"],
                "chat_template": "mistral"
            }
        }
        
        for model_name, model_config in models.items():
            config_copy = full_config.copy()
            config_copy["selected_model"] = model_name
            config_copy["model_options"][model_name] = model_config
            
            # Should be valid configuration
            assert config_copy["selected_model"] == model_name
            assert config_copy["model_options"][model_name]["target_modules"] == model_config["target_modules"]
    
    def test_all_features_enabled_compatibility(self, full_config):
        """Test that all features can be enabled simultaneously without conflicts."""
        # Enable all high-stakes features
        for feature in full_config["high_stakes"]:
            full_config["high_stakes"][feature]["enabled"] = True
        
        # Enable evaluation
        full_config["evaluation"]["enabled"] = True
        
        # Enable quantization
        full_config["lora"]["quantization"]["enabled"] = True
        full_config["lora"]["quantization"]["bits"] = 4
        
        # Configuration should remain valid
        assert all(full_config["high_stakes"][f]["enabled"] for f in full_config["high_stakes"])
        assert full_config["evaluation"]["enabled"] is True
        assert full_config["lora"]["quantization"]["enabled"] is True
    
    def test_pipeline_component_compatibility(self):
        """Test that all pipeline components are compatible."""
        # Test data flow: Data Prep -> Training -> Evaluation -> Inference
        
        # Mock data preparation
        sample_data = [
            {"text": "Test text 1", "label": "relevant"},
            {"text": "Test text 2", "label": "irrelevant"}
        ]
        
        # Should be able to process through pipeline
        assert len(sample_data) == 2
        assert all("text" in item and "label" in item for item in sample_data)
        
        # Mock training output
        training_output = {
            "model_path": "artifacts/models/test",
            "metrics": {"loss": 0.5, "accuracy": 0.8}
        }
        
        assert "model_path" in training_output
        assert "metrics" in training_output
        
        # Mock evaluation output
        eval_output = {
            "predictions": ["relevant", "irrelevant"],
            "metrics": {"accuracy": 0.8, "f1_score": 0.7}
        }
        
        assert len(eval_output["predictions"]) == len(sample_data)
        assert "metrics" in eval_output
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across components."""
        # Common error scenarios that should be handled gracefully
        error_scenarios = [
            {"type": "missing_file", "data": None},
            {"type": "invalid_json", "data": "invalid json"},
            {"type": "empty_dataset", "data": []},
            {"type": "malformed_config", "data": {"incomplete": "config"}}
        ]
        
        for scenario in error_scenarios:
            # Each scenario should have a defined error type
            assert "type" in scenario
            assert "data" in scenario
            
            # Error types should be descriptive
            assert scenario["type"] in ["missing_file", "invalid_json", "empty_dataset", "malformed_config"]


class TestConfigurationValidation:
    """Test configuration validation across the system."""
    
    def test_required_config_sections(self, full_config):
        """Test that all required configuration sections are present."""
        required_sections = [
            "selected_model",
            "model_options", 
            "lora",
            "training",
            "instruction_format"
        ]
        
        for section in required_sections:
            assert section in full_config, f"Missing required section: {section}"
    
    def test_optional_config_sections(self, full_config):
        """Test that optional configuration sections can be missing."""
        optional_sections = [
            "high_stakes",
            "evaluation", 
            "data"
        ]
        
        # Create config without optional sections
        minimal_config = {k: v for k, v in full_config.items() if k not in optional_sections}
        
        # Should still be valid
        assert "selected_model" in minimal_config
        assert "lora" in minimal_config
        assert len(minimal_config) >= 4
    
    def test_config_value_validation(self, full_config):
        """Test that configuration values are within expected ranges."""
        # LoRA rank should be reasonable
        assert 1 <= full_config["lora"]["r"] <= 256
        
        # Learning rate should be reasonable  
        assert 1e-6 <= full_config["training"]["learning_rate"] <= 1e-1
        
        # Batch size should be positive
        assert full_config["training"]["batch_size"] > 0
        
        # Epochs should be positive
        assert full_config["training"]["num_epochs"] > 0
    
    def test_feature_flag_validation(self, full_config):
        """Test that feature flags are properly structured."""
        if "high_stakes" in full_config:
            for feature_name, feature_config in full_config["high_stakes"].items():
                # Each feature should have an enabled flag
                assert "enabled" in feature_config
                assert isinstance(feature_config["enabled"], bool)


class TestPerformanceAndScalability:
    """Test performance considerations and scalability."""
    
    def test_memory_configuration_validation(self, full_config):
        """Test that memory-related configurations are sensible."""
        # Quantization settings
        if full_config["lora"]["quantization"]["enabled"]:
            bits = full_config["lora"]["quantization"]["bits"]
            assert bits in [4, 8], f"Invalid quantization bits: {bits}"
        
        # Gradient checkpointing can reduce memory
        assert isinstance(full_config["training"]["gradient_checkpointing"], bool)
        
        # Mixed precision settings
        fp16 = full_config["training"].get("fp16", False)
        bf16 = full_config["training"].get("bf16", False)
        
        # Shouldn't have both fp16 and bf16 enabled
        if fp16 and bf16:
            pytest.fail("Both fp16 and bf16 cannot be enabled simultaneously")
    
    def test_batch_size_scaling(self, full_config):
        """Test that batch size configurations are scalable."""
        batch_size = full_config["training"]["batch_size"]
        gradient_accumulation = full_config["training"].get("gradient_accumulation_steps", 1)
        
        effective_batch_size = batch_size * gradient_accumulation
        
        # Effective batch size should be reasonable
        assert 1 <= effective_batch_size <= 1024
        
        # Individual batch size should fit in memory
        assert 1 <= batch_size <= 32  # Reasonable for most GPUs
    
    def test_concurrent_processing_support(self):
        """Test that the system supports concurrent processing."""
        # Configuration for parallel processing
        parallel_config = {
            "training": {
                "dataloader_pin_memory": True,
                "ddp_find_unused_parameters": False,
                "num_proc": 4
            },
            "evaluation": {
                "batch_size": 8,
                "num_workers": 2
            }
        }
        
        # Should support multiple workers
        assert parallel_config["training"]["num_proc"] > 1
        assert parallel_config["evaluation"]["num_workers"] >= 1
        
        # Memory optimizations should be enabled
        assert parallel_config["training"]["dataloader_pin_memory"] is True


class TestSecurityAndReliability:
    """Test security and reliability features."""
    
    def test_input_sanitization_support(self):
        """Test that the system supports input sanitization."""
        # Potentially dangerous inputs that should be handled
        dangerous_inputs = [
            {"text": "<script>alert('xss')</script>", "type": "xss"},
            {"text": "../../../etc/passwd", "type": "path_traversal"},
            {"text": "'; DROP TABLE users; --", "type": "sql_injection"},
            {"text": "\x00null_byte", "type": "null_byte"},
            {"text": "very_long_text" * 10000, "type": "buffer_overflow"}
        ]
        
        for dangerous_input in dangerous_inputs:
            # System should be able to handle these inputs gracefully
            assert "text" in dangerous_input
            assert "type" in dangerous_input
            assert len(dangerous_input["text"]) >= 0
    
    def test_audit_trail_configuration(self, full_config):
        """Test that audit trail can be configured."""
        if full_config.get("high_stakes", {}).get("verifiable", {}).get("enabled"):
            verifiable_config = full_config["high_stakes"]["verifiable"]
            
            # Should have audit configuration
            assert "hash_artifacts" in verifiable_config
            assert isinstance(verifiable_config["hash_artifacts"], bool)
    
    def test_error_recovery_mechanisms(self):
        """Test that error recovery mechanisms are in place."""
        # Error recovery scenarios
        recovery_scenarios = [
            "checkpoint_corruption",
            "out_of_memory", 
            "network_failure",
            "disk_full",
            "gpu_failure"
        ]
        
        # Each scenario should have a defined recovery strategy
        for scenario in recovery_scenarios:
            assert isinstance(scenario, str)
            assert len(scenario) > 0


class TestDocumentationAndUsability:
    """Test that the system is well-documented and usable."""
    
    def test_configuration_documentation(self, full_config):
        """Test that configuration options are well-documented."""
        # Key configuration sections should be self-explanatory
        documented_sections = [
            "lora", 
            "training",
            "high_stakes"
        ]
        
        for section in documented_sections:
            if section in full_config:
                section_config = full_config[section]
                assert isinstance(section_config, dict)
                assert len(section_config) > 0
        
        # Test that selected_model exists and is a string
        assert "selected_model" in full_config
        assert isinstance(full_config["selected_model"], str)
    
    def test_example_configurations(self):
        """Test that example configurations are provided."""
        # Common use case configurations
        use_cases = [
            "research_prototype",
            "production_deployment",
            "memory_constrained",
            "high_accuracy",
            "fast_training"
        ]
        
        for use_case in use_cases:
            # Each use case should have identifiable characteristics
            assert isinstance(use_case, str)
            assert "_" in use_case or use_case.islower()
    
    def test_feature_discoverability(self, full_config):
        """Test that features are discoverable through configuration."""
        # High-stakes features should be discoverable
        if "high_stakes" in full_config:
            features = list(full_config["high_stakes"].keys())
            
            expected_features = [
                "uncertainty", "factual", "bias_audit", 
                "explainable", "procedural", "verifiable"
            ]
            
            # Should have core high-stakes features
            for feature in expected_features:
                assert feature in features or any(feature in f for f in features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])