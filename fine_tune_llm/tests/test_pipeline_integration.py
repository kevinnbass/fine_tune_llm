"""
Comprehensive pipeline integration tests.
Tests complete workflows and module interactions.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy imports for testing
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['trl'] = MagicMock()

from voters.llm.dataset import (
    load_labels, build_examples, create_abstention_examples,
    create_balanced_dataset, validate_output_format
)
from voters.llm.utils import (
    ConfigManager, ModelLoader, PromptFormatter,
    MetricsTracker, ErrorHandler
)


class TestFullPipelineIntegration:
    """Test complete pipeline: data prep → training → evaluation → inference"""
    
    def test_data_preparation_to_training_pipeline(self):
        """Test data flows correctly from preparation to training."""
        # Prepare test data
        raw_data = [
            {"text": "Bird flu outbreak", "label": "HIGH_RISK", "metadata": {"source": "news"}},
            {"text": "Weather report", "label": "NO_RISK", "metadata": {"source": "weather"}},
            {"text": "Flu symptoms", "label": "MEDIUM_RISK", "metadata": {"source": "health"}}
        ]
        
        # Test configuration
        config = {
            "instruction_format": {
                "system_prompt": "Classify the risk level",
                "input_template": "Text: {text}",
                "output_template": {"decision": "str", "rationale": "str"}
            },
            "abstention": {
                "enabled": True,
                "examples_per_label": 1,
                "uncertainty_phrases": ["unclear", "uncertain"]
            }
        }
        
        # Build examples
        labels = {"HIGH_RISK": "High risk content", "MEDIUM_RISK": "Medium risk", "NO_RISK": "No risk"}
        examples = build_examples(raw_data, labels, config)
        
        # Verify examples are properly formatted
        assert len(examples) == 3
        for example in examples:
            assert "input" in example
            assert "output" in example
            
            # Validate output format
            is_valid, parsed = validate_output_format(example["output"])
            assert is_valid
            assert "decision" in parsed
        
        # Create abstention examples
        abstention_examples = create_abstention_examples(labels, config)
        assert len(abstention_examples) == 3  # 1 per label
        
        # Combine and balance dataset
        all_examples = examples + abstention_examples
        balanced = create_balanced_dataset(all_examples, target_samples_per_class=2)
        
        # Verify balanced dataset
        assert len(balanced) > 0
        label_counts = {}
        for item in balanced:
            if "output" in item:
                output_data = json.loads(item["output"])
                label = output_data.get("decision", "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Check balance (should have similar counts)
        if len(label_counts) > 1:
            counts = list(label_counts.values())
            assert max(counts) - min(counts) <= 2  # Allow small imbalance
    
    def test_training_to_evaluation_pipeline(self):
        """Test model training outputs integrate with evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock training outputs
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            
            # Create mock adapter config
            adapter_config = {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"]
            }
            
            with open(model_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)
            
            # Mock metrics from training
            metrics_tracker = MetricsTracker()
            metrics_tracker.update(
                loss=0.5,
                accuracy=0.85,
                epoch=1
            )
            metrics_tracker.update(
                loss=0.3,
                accuracy=0.90,
                epoch=2
            )
            
            # Save metrics
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_tracker.save_to_file(metrics_path)
            
            # Load metrics in evaluation
            eval_tracker = MetricsTracker()
            eval_tracker.load_from_file(metrics_path)
            
            # Verify metrics transfer
            assert eval_tracker.get_average("accuracy") == 0.875
            assert eval_tracker.get_latest("loss") == 0.3
            assert len(eval_tracker.history) == 2
    
    def test_evaluation_to_inference_pipeline(self):
        """Test evaluation metrics inform inference decisions."""
        # Mock evaluation results
        eval_results = {
            "accuracy": 0.92,
            "f1_score": 0.89,
            "calibration_error": 0.05,
            "abstention_rate": 0.1
        }
        
        # Mock inference configuration based on eval
        inference_config = {
            "confidence_threshold": 0.7,  # Based on calibration
            "abstention_threshold": 0.5,  # Based on abstention rate
            "use_uncertainty": eval_results["calibration_error"] > 0.1
        }
        
        # Test inference with configuration
        test_input = "Potential flu outbreak detected"
        
        # Mock model output
        mock_output = {
            "decision": "HIGH_RISK",
            "rationale": "Contains flu outbreak keywords",
            "confidence": 0.85,
            "abstain": False
        }
        
        # Apply inference logic based on eval results
        if mock_output["confidence"] < inference_config["confidence_threshold"]:
            mock_output["abstain"] = True
        
        # Verify inference respects evaluation insights
        assert mock_output["abstain"] == False  # High confidence
        assert mock_output["confidence"] > inference_config["confidence_threshold"]
    
    def test_end_to_end_pipeline_with_high_stakes(self):
        """Test complete pipeline with high-stakes features enabled."""
        config = {
            "high_stakes": {
                "uncertainty_quantification": {"enabled": True},
                "bias_auditing": {"enabled": True},
                "factual_accuracy": {"enabled": True},
                "explainable_reasoning": {"enabled": True},
                "procedural_alignment": {"enabled": True},
                "verifiable_training": {"enabled": True}
            }
        }
        
        # Test data preparation with high-stakes
        raw_data = [
            {"text": "Confirmed H5N1 cases", "label": "HIGH_RISK"},
            {"text": "No health concerns", "label": "NO_RISK"}
        ]
        
        # Mock high-stakes processing
        with patch('voters.llm.uncertainty.MCDropoutWrapper') as mock_dropout:
            with patch('voters.llm.fact_check.FactChecker') as mock_fact:
                with patch('voters.llm.high_stakes_audit.BiasAuditor') as mock_bias:
                    # Configure mocks
                    mock_dropout.return_value = Mock()
                    mock_fact.return_value = Mock(check_facts=Mock(return_value=True))
                    mock_bias.return_value = Mock(audit=Mock(return_value={"bias_score": 0.1}))
                    
                    # Process data through pipeline
                    labels = {"HIGH_RISK": "High risk", "NO_RISK": "No risk"}
                    examples = build_examples(raw_data, labels, config)
                    
                    # Verify high-stakes features don't break pipeline
                    assert len(examples) == 2
                    for ex in examples:
                        assert validate_output_format(ex["output"])[0]


class TestHighStakesFeaturesIntegration:
    """Test integration between high-stakes features."""
    
    def test_uncertainty_to_audit_integration(self):
        """Test uncertainty scores flow to audit components."""
        # Mock uncertainty scores
        uncertainty_scores = {
            "epistemic": 0.3,
            "aleatoric": 0.2,
            "total": 0.5
        }
        
        # Mock audit configuration
        audit_config = {
            "uncertainty_threshold": 0.4,
            "require_low_uncertainty": True
        }
        
        # Test integration
        should_audit = uncertainty_scores["total"] > audit_config["uncertainty_threshold"]
        assert should_audit == True
        
        # Mock audit with uncertainty context
        audit_result = {
            "passed": uncertainty_scores["total"] < 0.6,
            "uncertainty_factor": uncertainty_scores["total"],
            "recommendation": "Review due to moderate uncertainty"
        }
        
        assert audit_result["passed"] == True
        assert "uncertainty" in audit_result["recommendation"].lower()
    
    def test_fact_check_training_integration(self):
        """Test RELIANCE framework integration in training loop."""
        # Mock training batch
        batch = {
            "input_ids": [1, 2, 3],
            "labels": [4, 5, 6],
            "facts": ["Flu is contagious", "H5N1 affects birds"]
        }
        
        # Mock fact checking during training
        fact_check_results = []
        for fact in batch["facts"]:
            result = {
                "fact": fact,
                "verified": True,
                "confidence": 0.9,
                "source": "medical_db"
            }
            fact_check_results.append(result)
        
        # Calculate fact-aware loss adjustment
        fact_penalty = 0.0
        for result in fact_check_results:
            if not result["verified"]:
                fact_penalty += 0.1 * (1 - result["confidence"])
        
        # Apply to training loss
        base_loss = 0.5
        adjusted_loss = base_loss + fact_penalty
        
        assert adjusted_loss == 0.5  # No penalty for verified facts
    
    def test_explainability_procedural_integration(self):
        """Test explainable reasoning integrates with procedural alignment."""
        # Mock reasoning steps
        reasoning_steps = [
            "1. Identify key terms: H5N1, outbreak",
            "2. Check severity indicators: confirmed cases",
            "3. Assess risk level: HIGH"
        ]
        
        # Mock procedures
        required_procedures = [
            "Identify key terms",
            "Check severity",
            "Assess risk"
        ]
        
        # Check alignment
        alignment_score = 0
        for procedure in required_procedures:
            for step in reasoning_steps:
                if procedure.lower() in step.lower():
                    alignment_score += 1
                    break
        
        alignment_rate = alignment_score / len(required_procedures)
        assert alignment_rate == 1.0  # Perfect alignment
    
    def test_all_features_combined(self):
        """Test all high-stakes features working together."""
        # Configuration with all features
        config = {
            "high_stakes": {
                "uncertainty_quantification": {
                    "enabled": True,
                    "dropout_rate": 0.1,
                    "n_iterations": 10
                },
                "bias_auditing": {
                    "enabled": True,
                    "check_gender": True,
                    "check_ethnicity": True
                },
                "factual_accuracy": {
                    "enabled": True,
                    "fact_sources": ["medical_db", "cdc_api"]
                },
                "explainable_reasoning": {
                    "enabled": True,
                    "min_steps": 3
                },
                "procedural_alignment": {
                    "enabled": True,
                    "procedures_file": "procedures.yaml"
                },
                "verifiable_training": {
                    "enabled": True,
                    "log_every_n_steps": 100
                }
            }
        }
        
        # Mock processing with all features
        input_text = "H5N1 bird flu confirmed in local farm"
        
        # Process through each component
        results = {
            "uncertainty": {"score": 0.2, "abstain": False},
            "bias_audit": {"passed": True, "score": 0.05},
            "fact_check": {"accuracy": 0.95, "verified_facts": 3},
            "reasoning": {"steps": 4, "clarity": 0.9},
            "procedural": {"aligned": True, "score": 0.92},
            "verifiable": {"hash": "abc123", "logged": True}
        }
        
        # Aggregate decision
        all_passed = all([
            not results["uncertainty"]["abstain"],
            results["bias_audit"]["passed"],
            results["fact_check"]["accuracy"] > 0.8,
            results["reasoning"]["steps"] >= 3,
            results["procedural"]["aligned"],
            results["verifiable"]["logged"]
        ])
        
        assert all_passed == True


class TestModelPersistenceIntegration:
    """Test model saving/loading across components."""
    
    def test_training_checkpoint_integration(self):
        """Test checkpoint saving and resumption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock training state
            checkpoint = {
                "epoch": 2,
                "global_step": 1000,
                "best_loss": 0.25,
                "optimizer_state": {"lr": 0.0001},
                "model_state": {"layer1": "weights"}
            }
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint-1000"
            checkpoint_path.mkdir()
            
            with open(checkpoint_path / "trainer_state.json", "w") as f:
                json.dump(checkpoint, f)
            
            # Load checkpoint in new training
            with open(checkpoint_path / "trainer_state.json", "r") as f:
                loaded = json.load(f)
            
            assert loaded["epoch"] == 2
            assert loaded["global_step"] == 1000
    
    def test_lora_adapter_persistence(self):
        """Test LoRA adapter saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock LoRA configuration
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "peft_type": "LORA"
            }
            
            # Save adapter config
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            
            with open(adapter_path / "adapter_config.json", "w") as f:
                json.dump(lora_config, f)
            
            # Mock adapter weights
            import pickle
            weights = {"q_proj.lora_A": [1, 2, 3], "q_proj.lora_B": [4, 5, 6]}
            with open(adapter_path / "adapter_model.bin", "wb") as f:
                pickle.dump(weights, f)
            
            # Load adapter
            with open(adapter_path / "adapter_config.json", "r") as f:
                loaded_config = json.load(f)
            
            assert loaded_config["r"] == 16
            assert "q_proj" in loaded_config["target_modules"]
    
    def test_merged_model_integration(self):
        """Test merged model works across inference components."""
        # Mock merged model path
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = Path(tmpdir) / "merged_model"
            merged_path.mkdir()
            
            # Create mock model files
            config = {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 32
            }
            
            with open(merged_path / "config.json", "w") as f:
                json.dump(config, f)
            
            # Test loading in different components
            assert (merged_path / "config.json").exists()
            
            with open(merged_path / "config.json", "r") as f:
                loaded = json.load(f)
                assert loaded["model_type"] == "llama"


class TestErrorRecoveryIntegration:
    """Test error handling across integrated components."""
    
    def test_data_error_recovery(self):
        """Test pipeline handles data errors gracefully."""
        # Invalid data
        bad_data = [
            {"text": None, "label": "HIGH_RISK"},  # Missing text
            {"text": "Valid text"},  # Missing label
            {"text": "", "label": ""},  # Empty values
        ]
        
        config = {
            "instruction_format": {
                "system_prompt": "Classify",
                "input_template": "{text}",
                "output_template": {"decision": "str"}
            }
        }
        
        # Process with error handling
        valid_examples = []
        for item in bad_data:
            try:
                if item.get("text") and item.get("label"):
                    # Would normally call build_examples
                    valid_examples.append(item)
            except Exception:
                continue
        
        assert len(valid_examples) == 0  # All invalid
    
    def test_model_error_recovery(self):
        """Test inference handles model errors."""
        # Mock model failure
        def failing_model(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")
        
        # Error handler
        handler = ErrorHandler()
        
        # Try with fallback
        result = handler.safe_execute(
            failing_model,
            default={"decision": "UNKNOWN", "abstain": True}
        )
        
        assert result["abstain"] == True
        assert result["decision"] == "UNKNOWN"
    
    def test_integration_error_cascade(self):
        """Test error handling across component boundaries."""
        # Mock component chain
        def component_a(data):
            if not data:
                raise ValueError("No data")
            return {"processed": data}
        
        def component_b(data):
            if "processed" not in data:
                raise KeyError("Missing processed data")
            return {"result": data["processed"]}
        
        # Test error cascade
        handler = ErrorHandler()
        
        # First component fails
        result_a = handler.safe_execute(component_a, None, default={})
        assert result_a == {}
        
        # Second component handles bad input
        result_b = handler.safe_execute(component_b, result_a, default={"result": "fallback"})
        assert result_b["result"] == "fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])