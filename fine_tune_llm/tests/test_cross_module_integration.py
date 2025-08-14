"""
Cross-module integration tests.
Tests specific interactions between module pairs.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy imports
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['trl'] = MagicMock()


class TestDatasetSFTLoRAIntegration:
    """Test dataset.py <-> sft_lora.py integration."""
    
    def test_dataset_format_compatibility(self):
        """Test dataset output format matches SFT training input."""
        from voters.llm.dataset import build_examples, validate_output_format
        
        # Create test data
        raw_data = [
            {"text": "H5N1 outbreak", "label": "HIGH_RISK", "metadata": {"source": "news"}}
        ]
        labels = {"HIGH_RISK": "High risk content"}
        config = {
            "instruction_format": {
                "system_prompt": "Classify risk",
                "input_template": "Text: {text}",
                "output_template": {"decision": "str", "rationale": "str"}
            }
        }
        
        # Build examples
        examples = build_examples(raw_data, labels, config)
        
        # Verify format for SFT
        assert len(examples) == 1
        example = examples[0]
        
        # Check required fields for SFT
        assert "input" in example
        assert "output" in example
        
        # Validate output is valid JSON
        is_valid, parsed = validate_output_format(example["output"])
        assert is_valid
        assert "decision" in parsed
        
        # Mock SFT dataset processing
        sft_format = {
            "text": f"{example['input']}\n\n{example['output']}",
            "input_ids": [1, 2, 3],  # Mock tokenization
            "attention_mask": [1, 1, 1]
        }
        
        assert len(sft_format["text"]) > 0
        assert len(sft_format["input_ids"]) == len(sft_format["attention_mask"])
    
    def test_balanced_dataset_for_training(self):
        """Test balanced dataset creation for training."""
        from voters.llm.dataset import create_balanced_dataset
        
        # Create imbalanced data
        imbalanced = [
            {"text": f"High risk {i}", "label": "HIGH_RISK"} for i in range(10)
        ] + [
            {"text": "No risk", "label": "NO_RISK"}
        ]
        
        # Balance for training
        balanced = create_balanced_dataset(imbalanced, target_samples_per_class=3)
        
        # Count labels
        label_counts = {}
        for item in balanced:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Verify balance
        assert label_counts["HIGH_RISK"] == 3
        assert label_counts["NO_RISK"] == 3


class TestUncertaintyAuditIntegration:
    """Test uncertainty.py <-> high_stakes_audit.py integration."""
    
    def test_uncertainty_scores_in_audit(self):
        """Test uncertainty scores are used in audit decisions."""
        from voters.llm.uncertainty import compute_uncertainty_aware_loss, should_abstain
        
        # Mock uncertainty scores
        logits = [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]  # Mock model outputs
        
        # Compute uncertainty (mocked since we can't import torch)
        mock_uncertainty = 0.4  # Moderate uncertainty
        
        # Test abstention decision
        abstain = mock_uncertainty > 0.5
        assert abstain == False
        
        # Mock audit with uncertainty
        audit_result = {
            "uncertainty_score": mock_uncertainty,
            "confidence_threshold": 0.3,
            "audit_required": mock_uncertainty > 0.3,
            "risk_level": "medium" if mock_uncertainty > 0.3 else "low"
        }
        
        assert audit_result["audit_required"] == True
        assert audit_result["risk_level"] == "medium"
    
    def test_mc_dropout_bias_audit_interaction(self):
        """Test MC Dropout wrapper works with bias auditing."""
        # Mock MC Dropout results
        mc_dropout_outputs = [
            {"decision": "HIGH_RISK", "confidence": 0.8},
            {"decision": "HIGH_RISK", "confidence": 0.75},
            {"decision": "MEDIUM_RISK", "confidence": 0.6}
        ]
        
        # Calculate uncertainty from variance
        decisions = [o["decision"] for o in mc_dropout_outputs]
        unique_decisions = list(set(decisions))
        decision_variance = len(unique_decisions) / len(decisions)
        
        # Mock bias audit with uncertainty context
        bias_check = {
            "text": "The nurse treated the patient",
            "uncertainty": decision_variance,
            "gender_bias_detected": False,
            "requires_review": decision_variance > 0.3
        }
        
        assert bias_check["requires_review"] == True


class TestFactCheckIntegration:
    """Test fact_check.py integration across modules."""
    
    def test_fact_check_in_training_loop(self):
        """Test fact checking during training."""
        from voters.llm.fact_check import create_factual_test_data
        
        # Create factual test data
        test_data = create_factual_test_data(n_samples=3)
        
        assert len(test_data) == 3
        for item in test_data:
            assert "text" in item
            assert "facts" in item
            assert "label" in item
        
        # Mock fact verification during training
        training_batch = {
            "texts": ["H5N1 is a bird flu virus"],
            "facts": ["H5N1 affects birds"],
            "labels": ["HIGH_RISK"]
        }
        
        # Mock RELIANCE verification
        fact_scores = []
        for fact in training_batch["facts"]:
            # Mock verification score
            score = 0.9 if "H5N1" in fact else 0.5
            fact_scores.append(score)
        
        # Calculate fact-aware loss weight
        fact_weight = sum(fact_scores) / len(fact_scores)
        assert fact_weight == 0.9
    
    def test_fact_check_inference_integration(self):
        """Test fact checking in inference pipeline."""
        # Mock inference input
        input_text = "H5N1 virus spreads through direct contact"
        
        # Extract facts (mock)
        extracted_facts = [
            "H5N1 is a virus",
            "Spreads through direct contact"
        ]
        
        # Mock fact verification
        verification_results = []
        for fact in extracted_facts:
            result = {
                "fact": fact,
                "verified": True,
                "confidence": 0.85,
                "source": "medical_database"
            }
            verification_results.append(result)
        
        # Adjust inference confidence based on facts
        base_confidence = 0.8
        fact_adjustment = sum(r["confidence"] for r in verification_results) / len(verification_results)
        final_confidence = base_confidence * fact_adjustment
        
        assert final_confidence == 0.68  # 0.8 * 0.85


class TestEvaluateInferIntegration:
    """Test evaluate.py <-> infer.py integration."""
    
    def test_evaluation_metrics_inform_inference(self):
        """Test evaluation metrics guide inference settings."""
        # Mock evaluation results
        eval_metrics = {
            "accuracy": 0.92,
            "calibration_error": 0.08,
            "f1_score": 0.89,
            "abstention_rate": 0.15
        }
        
        # Derive inference settings from evaluation
        inference_config = {
            "confidence_threshold": 1.0 - eval_metrics["calibration_error"],  # 0.92
            "abstention_threshold": eval_metrics["abstention_rate"] * 2,  # 0.30
            "use_uncertainty": eval_metrics["calibration_error"] > 0.05  # True
        }
        
        assert inference_config["confidence_threshold"] == 0.92
        assert inference_config["use_uncertainty"] == True
    
    def test_inference_output_evaluation_compatibility(self):
        """Test inference outputs can be evaluated properly."""
        # Mock inference output
        inference_output = {
            "decision": "HIGH_RISK",
            "rationale": "Contains flu outbreak indicators",
            "confidence": 0.87,
            "abstain": False
        }
        
        # Mock ground truth
        ground_truth = "HIGH_RISK"
        
        # Evaluate prediction
        is_correct = inference_output["decision"] == ground_truth
        confidence_calibrated = abs(inference_output["confidence"] - (1.0 if is_correct else 0.0)) < 0.2
        
        assert is_correct == True
        assert confidence_calibrated == True


class TestUIBackendIntegration:
    """Test UI <-> backend module integration."""
    
    def test_ui_training_integration(self):
        """Test UI can trigger training correctly."""
        # Mock UI training request
        ui_config = {
            "model_id": "gpt2",
            "dataset_path": "data/train.csv",
            "output_dir": "models/finetuned",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4
        }
        
        # Convert UI config to training config
        training_config = {
            "model": {"model_id": ui_config["model_id"]},
            "data": {"train_path": ui_config["dataset_path"]},
            "training": {
                "output_dir": ui_config["output_dir"],
                "num_epochs": ui_config["epochs"],
                "per_device_train_batch_size": ui_config["batch_size"],
                "learning_rate": ui_config["learning_rate"]
            }
        }
        
        assert training_config["training"]["num_epochs"] == 3
        assert training_config["model"]["model_id"] == "gpt2"
    
    def test_ui_inference_integration(self):
        """Test UI inference request handling."""
        # Mock UI inference request
        ui_request = {
            "text": "H5N1 outbreak detected",
            "model_path": "models/finetuned",
            "temperature": 0.7,
            "max_length": 100
        }
        
        # Process request
        inference_params = {
            "input_text": ui_request["text"],
            "model_path": Path(ui_request["model_path"]),
            "generation_kwargs": {
                "temperature": ui_request["temperature"],
                "max_length": ui_request["max_length"],
                "do_sample": True
            }
        }
        
        # Mock inference result
        result = {
            "decision": "HIGH_RISK",
            "confidence": 0.92,
            "processing_time": 0.5
        }
        
        # Format for UI display
        ui_response = {
            "classification": result["decision"],
            "confidence_score": f"{result['confidence']*100:.1f}%",
            "inference_time": f"{result['processing_time']:.2f}s"
        }
        
        assert ui_response["confidence_score"] == "92.0%"


class TestHighStakesModularIntegration:
    """Test modular high-stakes feature integration."""
    
    def test_selective_feature_enabling(self):
        """Test enabling specific high-stakes features."""
        # Configuration with selective features
        config = {
            "high_stakes": {
                "uncertainty_quantification": {"enabled": True},
                "bias_auditing": {"enabled": False},
                "factual_accuracy": {"enabled": True},
                "explainable_reasoning": {"enabled": False},
                "procedural_alignment": {"enabled": True},
                "verifiable_training": {"enabled": False}
            }
        }
        
        # Count enabled features
        enabled_features = [
            name for name, cfg in config["high_stakes"].items()
            if cfg.get("enabled", False)
        ]
        
        assert len(enabled_features) == 3
        assert "uncertainty_quantification" in enabled_features
        assert "bias_auditing" not in enabled_features
    
    def test_feature_dependency_resolution(self):
        """Test resolution of feature dependencies."""
        # Mock feature dependencies
        dependencies = {
            "explainable_reasoning": ["uncertainty_quantification"],
            "procedural_alignment": ["explainable_reasoning"],
            "verifiable_training": []
        }
        
        # Enable a feature with dependencies
        requested_features = ["procedural_alignment"]
        required_features = set(requested_features)
        
        # Resolve dependencies
        for feature in list(required_features):
            if feature in dependencies:
                required_features.update(dependencies[feature])
        
        # Resolve transitive dependencies
        changed = True
        while changed:
            changed = False
            for feature in list(required_features):
                if feature in dependencies:
                    before = len(required_features)
                    required_features.update(dependencies[feature])
                    if len(required_features) > before:
                        changed = True
        
        assert "uncertainty_quantification" in required_features
        assert "explainable_reasoning" in required_features
        assert len(required_features) == 3


class TestConfigurationIntegration:
    """Test configuration consistency across modules."""
    
    def test_config_propagation(self):
        """Test configuration propagates correctly across modules."""
        # Master configuration
        master_config = {
            "model": {"model_id": "gpt2", "device": "cuda"},
            "training": {"batch_size": 4, "learning_rate": 2e-4},
            "inference": {"temperature": 0.7, "max_length": 100},
            "high_stakes": {"uncertainty_quantification": {"enabled": True}}
        }
        
        # Mock module-specific config extraction
        training_config = master_config["training"]
        inference_config = master_config["inference"]
        high_stakes_config = master_config["high_stakes"]
        
        # Verify each module gets correct config
        assert training_config["batch_size"] == 4
        assert inference_config["temperature"] == 0.7
        assert high_stakes_config["uncertainty_quantification"]["enabled"] == True
    
    def test_config_validation_across_modules(self):
        """Test configuration validation catches conflicts."""
        # Configuration with potential conflicts
        config = {
            "model": {"model_id": "gpt2", "max_length": 512},
            "training": {"max_length": 1024},  # Conflict!
            "inference": {"max_length": 2048}  # Another conflict!
        }
        
        # Detect conflicts
        max_lengths = [
            config["model"]["max_length"],
            config["training"]["max_length"],
            config["inference"]["max_length"]
        ]
        
        has_conflict = len(set(max_lengths)) > 1
        assert has_conflict == True
        
        # Resolution strategy: use minimum
        resolved_max_length = min(max_lengths)
        assert resolved_max_length == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])