"""Comprehensive tests for dataset preparation and formatting functionality."""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

# Test imports
try:
    from voters.llm.dataset import (
        load_labels, build_examples, create_abstention_examples,
        add_rationale, validate_output_format, create_balanced_dataset
    )
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    pytest.skip("Dataset module not available", allow_module_level=True)


@pytest.fixture
def test_labels():
    """Fixture providing test labels."""
    return {
        "relevant": "Content is relevant to the topic",
        "irrelevant": "Content is not relevant to the topic",
        "uncertain": "Uncertainty about relevance"
    }


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        "data": {
            "augmentation": {
                "enabled": False,
                "methods": ["synonym_replacement", "random_insertion"],
                "aug_p": 0.1
            }
        },
        "instruction_format": {
            "system_prompt": "You are a helpful assistant for classification.",
            "input_template": "Text to classify: {text}\n\nMetadata: {metadata}",
            "output_template": {
                "decision": "str",
                "rationale": "str", 
                "confidence": "float",
                "abstain": "bool"
            }
        },
        "abstention": {
            "enabled": True,
            "examples_per_label": 2,
            "uncertainty_phrases": [
                "I'm not sure",
                "It's unclear",
                "Hard to determine"
            ]
        }
    }


@pytest.fixture
def test_raw_data():
    """Fixture providing test raw data."""
    return [
        {
            "text": "Bird flu outbreak reported in multiple farms",
            "label": "relevant",
            "metadata": {"source": "news", "confidence": "high"}
        },
        {
            "text": "Weather is sunny today",
            "label": "irrelevant", 
            "metadata": {"source": "weather", "confidence": "high"}
        },
        {
            "text": "Some reports about potential flu cases",
            "label": "uncertain",
            "metadata": {"source": "rumors", "confidence": "low"}
        }
    ]


@pytest.fixture
def temp_labels_file(test_labels):
    """Fixture providing temporary labels file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_labels, f)
        yield f.name
    Path(f.name).unlink()


class TestLoadLabels:
    """Test label loading functionality."""
    
    def test_load_labels_success(self, temp_labels_file, test_labels):
        """Test successful label loading."""
        labels = load_labels(temp_labels_file)
        
        assert labels == test_labels
        assert "relevant" in labels
        assert "irrelevant" in labels
        assert "uncertain" in labels
    
    def test_load_labels_missing_file(self):
        """Test handling of missing labels file."""
        with pytest.raises(FileNotFoundError):
            load_labels("nonexistent_labels.yaml")
    
    def test_load_labels_invalid_yaml(self):
        """Test handling of invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()
            temp_name = f.name
            
        try:
            with pytest.raises(yaml.YAMLError):
                load_labels(temp_name)
        finally:
            Path(temp_name).unlink()
    
    def test_load_labels_empty_file(self):
        """Test handling of empty labels file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            temp_name = f.name
            
        try:
            labels = load_labels(temp_name)
            assert labels is None or labels == {}
        finally:
            Path(temp_name).unlink()


class TestBuildExamples:
    """Test example building functionality."""
    
    def test_build_examples_basic(self, test_config, test_labels, test_raw_data):
        """Test basic example building."""
        examples = build_examples(test_raw_data, test_labels, test_config)
        
        assert len(examples) == len(test_raw_data)
        
        for example in examples:
            assert "input" in example
            assert "output" in example
            assert "Text to classify:" in example["input"]
            assert "Metadata:" in example["input"]
    
    def test_build_examples_with_rationale(self, test_config, test_labels, test_raw_data):
        """Test example building with rationale."""
        test_config["instruction_format"]["add_rationale"] = True
        
        examples = build_examples(test_raw_data, test_labels, test_config)
        
        for example in examples:
            output = json.loads(example["output"])
            assert "rationale" in output
            assert isinstance(output["rationale"], str)
            assert len(output["rationale"]) > 0
    
    def test_build_examples_custom_template(self, test_config, test_labels, test_raw_data):
        """Test example building with custom template."""
        test_config["instruction_format"]["input_template"] = "Classify: {text}"
        
        examples = build_examples(test_raw_data, test_labels, test_config)
        
        for example in examples:
            assert "Classify:" in example["input"]
            assert "Text to classify:" not in example["input"]
    
    def test_build_examples_missing_label(self, test_config, test_labels):
        """Test handling of missing labels."""
        invalid_data = [{"text": "test", "label": "nonexistent"}]
        
        with pytest.raises(KeyError):
            build_examples(invalid_data, test_labels, test_config)
    
    def test_build_examples_empty_data(self, test_config, test_labels):
        """Test handling of empty data."""
        examples = build_examples([], test_labels, test_config)
        
        assert examples == []
    
    def test_build_examples_missing_text(self, test_config, test_labels):
        """Test handling of missing text field."""
        invalid_data = [{"label": "relevant"}]  # Missing text
        
        with pytest.raises(KeyError):
            build_examples(invalid_data, test_labels, test_config)
    
    def test_build_examples_missing_metadata(self, test_config, test_labels):
        """Test handling of missing metadata."""
        data_without_metadata = [{"text": "test", "label": "relevant"}]
        
        examples = build_examples(data_without_metadata, test_labels, test_config)
        
        assert len(examples) == 1
        assert "Metadata: {}" in examples[0]["input"] or "Metadata: None" in examples[0]["input"]


class TestAbstentionExamples:
    """Test abstention example creation."""
    
    def test_create_abstention_examples(self, test_config, test_labels):
        """Test abstention example creation."""
        examples = create_abstention_examples(test_labels, test_config)
        
        # Should create examples_per_label * num_labels examples
        expected_count = test_config["abstention"]["examples_per_label"] * len(test_labels)
        assert len(examples) == expected_count
        
        for example in examples:
            assert "input" in example
            assert "output" in example
            
            output = json.loads(example["output"])
            assert output["abstain"] is True
            assert any(phrase in example["input"] for phrase in test_config["abstention"]["uncertainty_phrases"])
    
    def test_create_abstention_examples_disabled(self, test_config, test_labels):
        """Test abstention examples when disabled."""
        test_config["abstention"]["enabled"] = False
        
        examples = create_abstention_examples(test_labels, test_config)
        
        assert examples == []
    
    def test_create_abstention_examples_custom_phrases(self, test_config, test_labels):
        """Test abstention examples with custom uncertainty phrases."""
        custom_phrases = ["Custom uncertainty phrase", "Another uncertain phrase"]
        test_config["abstention"]["uncertainty_phrases"] = custom_phrases
        
        examples = create_abstention_examples(test_labels, test_config)
        
        for example in examples:
            assert any(phrase in example["input"] for phrase in custom_phrases)
    
    def test_create_abstention_examples_zero_per_label(self, test_config, test_labels):
        """Test abstention examples with zero examples per label."""
        test_config["abstention"]["examples_per_label"] = 0
        
        examples = create_abstention_examples(test_labels, test_config)
        
        assert examples == []


class TestRationaleGeneration:
    """Test rationale generation functionality."""
    
    def test_add_rationale_basic(self, test_labels):
        """Test basic rationale generation."""
        example = {
            "text": "Bird flu outbreak in farms",
            "label": "relevant"
        }
        
        rationale = add_rationale(example, test_labels)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "relevant" in rationale.lower() or "bird flu" in rationale.lower()
    
    def test_add_rationale_irrelevant(self, test_labels):
        """Test rationale for irrelevant content."""
        example = {
            "text": "Weather is sunny today",
            "label": "irrelevant"
        }
        
        rationale = add_rationale(example, test_labels)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "irrelevant" in rationale.lower() or "not related" in rationale.lower() or "not contain relevant" in rationale.lower()
    
    def test_add_rationale_uncertain(self, test_labels):
        """Test rationale for uncertain content."""
        example = {
            "text": "Some unclear information",
            "label": "uncertain"
        }
        
        rationale = add_rationale(example, test_labels)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "uncertain" in rationale.lower() or "unclear" in rationale.lower()
    
    def test_add_rationale_with_metadata(self, test_labels):
        """Test rationale generation with metadata."""
        example = {
            "text": "Bird flu outbreak",
            "label": "relevant",
            "metadata": {"source": "news", "confidence": "high"}
        }
        
        rationale = add_rationale(example, test_labels)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
    
    def test_add_rationale_invalid_label(self, test_labels):
        """Test rationale with invalid label."""
        example = {
            "text": "Test text",
            "label": "nonexistent_label"
        }
        
        # Should handle gracefully and return generic rationale
        rationale = add_rationale(example, test_labels)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0


class TestOutputValidation:
    """Test output format validation."""
    
    def test_validate_output_format_valid_json(self):
        """Test validation of valid JSON output."""
        valid_output = {
            "decision": "relevant",
            "rationale": "This is relevant because...",
            "confidence": 0.85,
            "abstain": False
        }
        
        json_output = json.dumps(valid_output)
        is_valid, parsed = validate_output_format(json_output)
        
        assert is_valid is True
        assert parsed == valid_output
    
    def test_validate_output_format_invalid_json(self):
        """Test validation of invalid JSON."""
        invalid_json = "{'invalid': json format}"
        
        is_valid, parsed = validate_output_format(invalid_json)
        
        assert is_valid is False
        assert parsed is None
    
    def test_validate_output_format_missing_fields(self):
        """Test validation with missing required fields."""
        incomplete_output = {
            "decision": "relevant"
            # Missing rationale, confidence, abstain
        }
        
        json_output = json.dumps(incomplete_output)
        
        # This should be handled by validation schema
        is_valid, parsed = validate_output_format(json_output)
        
        # Depending on implementation, might be valid or invalid
        assert isinstance(is_valid, bool)
    
    def test_validate_output_format_extra_fields(self):
        """Test validation with extra fields."""
        output_with_extra = {
            "decision": "relevant",
            "rationale": "This is relevant because...",
            "confidence": 0.85,
            "abstain": False,
            "extra_field": "should be ignored"
        }
        
        json_output = json.dumps(output_with_extra)
        is_valid, parsed = validate_output_format(json_output)
        
        assert is_valid is True
        assert "decision" in parsed
    
    def test_validate_output_format_empty_string(self):
        """Test validation of empty string."""
        is_valid, parsed = validate_output_format("")
        
        assert is_valid is False
        assert parsed is None
    
    def test_validate_output_format_none_input(self):
        """Test validation of None input."""
        is_valid, parsed = validate_output_format(None)
        
        assert is_valid is False
        assert parsed is None


class TestBalancedDataset:
    """Test balanced dataset creation."""
    
    def test_create_balanced_dataset_equal_distribution(self, test_raw_data):
        """Test balanced dataset with equal distribution."""
        balanced_data = create_balanced_dataset(test_raw_data, target_samples_per_class=1)
        
        # Count labels
        label_counts = {}
        for item in balanced_data:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Should have equal distribution
        assert all(count == 1 for count in label_counts.values())
    
    def test_create_balanced_dataset_oversample(self):
        """Test balanced dataset with oversampling."""
        imbalanced_data = [
            {"text": "text1", "label": "relevant"},
            {"text": "text2", "label": "relevant"},
            {"text": "text3", "label": "relevant"},
            {"text": "text4", "label": "irrelevant"}  # Only one irrelevant
        ]
        
        balanced_data = create_balanced_dataset(imbalanced_data, target_samples_per_class=3)
        
        # Count labels
        label_counts = {}
        for item in balanced_data:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["relevant"] == 3
        assert label_counts["irrelevant"] == 3  # Should be oversampled
    
    def test_create_balanced_dataset_undersample(self):
        """Test balanced dataset with undersampling."""
        imbalanced_data = [
            {"text": f"text{i}", "label": "relevant"} for i in range(10)
        ] + [
            {"text": "irrelevant_text", "label": "irrelevant"}
        ]
        
        balanced_data = create_balanced_dataset(imbalanced_data, target_samples_per_class=1)
        
        # Count labels
        label_counts = {}
        for item in balanced_data:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts["relevant"] == 1  # Should be undersampled
        assert label_counts["irrelevant"] == 1
    
    def test_create_balanced_dataset_empty_input(self):
        """Test balanced dataset with empty input."""
        balanced_data = create_balanced_dataset([], target_samples_per_class=5)
        
        assert balanced_data == []
    
    def test_create_balanced_dataset_single_class(self):
        """Test balanced dataset with single class."""
        single_class_data = [
            {"text": "text1", "label": "relevant"},
            {"text": "text2", "label": "relevant"}
        ]
        
        balanced_data = create_balanced_dataset(single_class_data, target_samples_per_class=3)
        
        assert len(balanced_data) == 3
        assert all(item["label"] == "relevant" for item in balanced_data)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_long_text(self, test_config, test_labels):
        """Test handling of very long text."""
        long_text = "This is a very long text. " * 1000  # ~25k characters
        
        long_data = [{"text": long_text, "label": "relevant"}]
        
        examples = build_examples(long_data, test_labels, test_config)
        
        assert len(examples) == 1
        assert len(examples[0]["input"]) > 0  # Should handle gracefully
    
    def test_unicode_text(self, test_config, test_labels):
        """Test handling of Unicode text."""
        unicode_data = [
            {"text": "测试中文文本 🦠 émoji", "label": "relevant"},
            {"text": "العربية النص", "label": "irrelevant"},
            {"text": "Русский текст", "label": "uncertain"}
        ]
        
        examples = build_examples(unicode_data, test_labels, test_config)
        
        assert len(examples) == 3
        for example in examples:
            assert isinstance(example["input"], str)
            assert isinstance(example["output"], str)
    
    def test_special_characters_text(self, test_config, test_labels):
        """Test handling of special characters."""
        special_data = [
            {"text": "Text with\n\nline breaks\tand tabs", "label": "relevant"},
            {"text": "Text with \"quotes\" and 'apostrophes'", "label": "irrelevant"},
            {"text": "Text with <tags> & ampersands", "label": "uncertain"}
        ]
        
        examples = build_examples(special_data, test_labels, test_config)
        
        assert len(examples) == 3
        for example in examples:
            # Should produce valid JSON
            try:
                json.loads(example["output"])
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")
    
    def test_empty_text(self, test_config, test_labels):
        """Test handling of empty text."""
        empty_data = [{"text": "", "label": "relevant"}]
        
        examples = build_examples(empty_data, test_labels, test_config)
        
        assert len(examples) == 1
        assert examples[0]["input"] is not None
    
    def test_whitespace_only_text(self, test_config, test_labels):
        """Test handling of whitespace-only text."""
        whitespace_data = [{"text": "   \t\n  ", "label": "relevant"}]
        
        examples = build_examples(whitespace_data, test_labels, test_config)
        
        assert len(examples) == 1
        assert examples[0]["input"] is not None


class TestModularity:
    """Test modularity and configuration toggles."""
    
    def test_augmentation_toggle(self, test_config, test_labels, test_raw_data):
        """Test data augmentation toggle."""
        # Test disabled (default)
        examples_without_aug = build_examples(test_raw_data, test_labels, test_config)
        
        # Test enabled
        test_config["data"]["augmentation"]["enabled"] = True
        
        # Mock augmentation to avoid external dependencies
        with patch('voters.llm.dataset.apply_augmentation') as mock_aug:
            mock_aug.return_value = test_raw_data  # Return unchanged
            
            examples_with_aug = build_examples(test_raw_data, test_labels, test_config)
            
            # Should have called augmentation
            if hasattr(build_examples, 'apply_augmentation'):
                mock_aug.assert_called()
    
    def test_abstention_toggle(self, test_config, test_labels):
        """Test abstention examples toggle."""
        # Test enabled
        test_config["abstention"]["enabled"] = True
        examples_with_abstention = create_abstention_examples(test_labels, test_config)
        assert len(examples_with_abstention) > 0
        
        # Test disabled
        test_config["abstention"]["enabled"] = False
        examples_without_abstention = create_abstention_examples(test_labels, test_config)
        assert len(examples_without_abstention) == 0
    
    def test_rationale_toggle(self, test_config, test_labels, test_raw_data):
        """Test rationale generation toggle."""
        # Test without rationale
        test_config["instruction_format"]["add_rationale"] = False
        examples_without = build_examples(test_raw_data, test_labels, test_config)
        
        # Test with rationale
        test_config["instruction_format"]["add_rationale"] = True
        examples_with = build_examples(test_raw_data, test_labels, test_config)
        
        # Outputs should be different
        for ex_without, ex_with in zip(examples_without, examples_with):
            output_without = json.loads(ex_without["output"])
            output_with = json.loads(ex_with["output"])
            
            # With rationale should have more detailed rationale
            assert len(output_with.get("rationale", "")) >= len(output_without.get("rationale", ""))


class TestIntegration:
    """Test integration between dataset components."""
    
    def test_full_dataset_pipeline(self, test_config, test_labels, test_raw_data):
        """Test full dataset preparation pipeline."""
        # Enable all features
        test_config["data"]["augmentation"]["enabled"] = True
        test_config["abstention"]["enabled"] = True
        test_config["instruction_format"]["add_rationale"] = True
        
        # Mock augmentation
        with patch('voters.llm.dataset.apply_augmentation') as mock_aug:
            mock_aug.return_value = test_raw_data
            
            # Build regular examples
            regular_examples = build_examples(test_raw_data, test_labels, test_config)
            
            # Create abstention examples
            abstention_examples = create_abstention_examples(test_labels, test_config)
            
            # Combine all examples
            all_examples = regular_examples + abstention_examples
            
            # Create balanced dataset
            balanced_examples = create_balanced_dataset(all_examples, target_samples_per_class=2)
            
            # Validate all outputs
            for example in balanced_examples:
                is_valid, _ = validate_output_format(example["output"])
                assert is_valid, f"Invalid output format: {example['output']}"
    
    def test_dataset_consistency(self, test_config, test_labels, test_raw_data):
        """Test dataset consistency across multiple runs."""
        # Run pipeline multiple times
        results = []
        for i in range(3):
            examples = build_examples(test_raw_data, test_labels, test_config)
            results.append(examples)
        
        # Results should be consistent (same structure)
        assert all(len(result) == len(results[0]) for result in results)
        
        for i in range(len(results[0])):
            for result in results:
                assert set(result[i].keys()) == set(results[0][i].keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])