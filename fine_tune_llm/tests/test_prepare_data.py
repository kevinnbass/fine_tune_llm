"""Comprehensive tests for data preparation script functionality."""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Test imports - Handle both script and module imports
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    
    from prepare_data import (
        load_raw_data, validate_data, augment_data, 
        create_instruction_format, save_processed_data
    )
    PREPARE_DATA_AVAILABLE = True
except ImportError:
    PREPARE_DATA_AVAILABLE = False
    pytest.skip("Prepare data script not available", allow_module_level=True)


@pytest.fixture
def sample_raw_data():
    """Fixture providing sample raw data."""
    return [
        {
            "text": "Bird flu outbreak reported in multiple farms",
            "label": "relevant",
            "metadata": {"source": "news", "confidence": "high"}
        },
        {
            "text": "Weather forecast shows sunny skies",
            "label": "irrelevant",
            "metadata": {"source": "weather", "confidence": "high"}
        },
        {
            "text": "Unclear reports about potential cases",
            "label": "uncertain", 
            "metadata": {"source": "rumors", "confidence": "low"}
        }
    ]


@pytest.fixture
def temp_data_file(sample_raw_data):
    """Fixture providing temporary data file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_raw_data, f)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        "data": {
            "input_format": "json",
            "text_column": "text",
            "label_column": "label",
            "validation": {
                "min_length": 10,
                "max_length": 1000,
                "required_fields": ["text", "label"]
            },
            "augmentation": {
                "enabled": True,
                "methods": ["synonym_replacement", "random_insertion"],
                "aug_p": 0.1,
                "num_aug_per_example": 1
            }
        },
        "output": {
            "format": "instruction",
            "train_split": 0.8,
            "val_split": 0.2,
            "save_path": "processed_data"
        }
    }


class TestLoadRawData:
    """Test raw data loading functionality."""
    
    def test_load_json_data(self, temp_data_file, sample_raw_data):
        """Test loading JSON data."""
        data = load_raw_data(temp_data_file, format="json")
        
        assert len(data) == len(sample_raw_data)
        assert data[0]["text"] == sample_raw_data[0]["text"]
        assert data[0]["label"] == sample_raw_data[0]["label"]
    
    def test_load_csv_data(self, sample_raw_data):
        """Test loading CSV data."""
        # Create temporary CSV file
        df = pd.DataFrame(sample_raw_data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_file = f.name
        
        try:
            data = load_raw_data(csv_file, format="csv")
            
            assert len(data) >= len(sample_raw_data)
            assert "text" in data[0]
            assert "label" in data[0]
            
        finally:
            Path(csv_file).unlink()
    
    def test_load_missing_file(self):
        """Test handling of missing data file."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.json", format="json")
    
    def test_load_invalid_json(self):
        """Test handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json content}")
            f.flush()
            
            with pytest.raises(json.JSONDecodeError):
                load_raw_data(f.name, format="json")
            
            Path(f.name).unlink()
    
    def test_load_empty_file(self):
        """Test handling of empty data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("[]")
            f.flush()
            
            data = load_raw_data(f.name, format="json")
            assert data == []
            
            Path(f.name).unlink()


class TestValidateData:
    """Test data validation functionality."""
    
    def test_validate_complete_data(self, sample_raw_data, test_config):
        """Test validation with complete data."""
        valid_data, errors = validate_data(sample_raw_data, test_config)
        
        assert len(valid_data) == len(sample_raw_data)
        assert len(errors) == 0
    
    def test_validate_missing_fields(self, test_config):
        """Test validation with missing required fields."""
        invalid_data = [
            {"text": "Valid text", "label": "relevant"},
            {"text": "Missing label"},  # Missing label
            {"label": "relevant"}  # Missing text
        ]
        
        valid_data, errors = validate_data(invalid_data, test_config)
        
        assert len(valid_data) == 1  # Only first item is valid
        assert len(errors) == 2      # Two validation errors
    
    def test_validate_text_length(self, test_config):
        """Test text length validation."""
        invalid_data = [
            {"text": "Short", "label": "relevant"},  # Too short
            {"text": "x" * 2000, "label": "relevant"},  # Too long
            {"text": "Valid length text", "label": "relevant"}  # Valid
        ]
        
        valid_data, errors = validate_data(invalid_data, test_config)
        
        assert len(valid_data) == 1  # Only third item is valid
        assert len(errors) == 2      # Two length errors
    
    def test_validate_empty_data(self, test_config):
        """Test validation with empty dataset."""
        valid_data, errors = validate_data([], test_config)
        
        assert valid_data == []
        assert len(errors) == 0
    
    def test_validate_no_config(self, sample_raw_data):
        """Test validation without configuration."""
        valid_data, errors = validate_data(sample_raw_data, {})
        
        # Should return all data if no validation config
        assert len(valid_data) == len(sample_raw_data)
        assert len(errors) == 0


class TestAugmentData:
    """Test data augmentation functionality."""
    
    @patch('prepare_data.nlpaug')
    def test_augment_data_enabled(self, mock_nlpaug, sample_raw_data, test_config):
        """Test data augmentation when enabled."""
        # Mock augmentation
        mock_nlpaug.text.SynonymAug.return_value.augment.return_value = "Augmented text"
        
        augmented_data = augment_data(sample_raw_data, test_config)
        
        # Should have more data due to augmentation
        assert len(augmented_data) > len(sample_raw_data)
    
    def test_augment_data_disabled(self, sample_raw_data, test_config):
        """Test data augmentation when disabled."""
        test_config["data"]["augmentation"]["enabled"] = False
        
        augmented_data = augment_data(sample_raw_data, test_config)
        
        # Should return original data unchanged
        assert len(augmented_data) == len(sample_raw_data)
        assert augmented_data == sample_raw_data
    
    def test_augment_empty_data(self, test_config):
        """Test augmentation with empty data."""
        augmented_data = augment_data([], test_config)
        
        assert augmented_data == []
    
    @patch('prepare_data.nlpaug')
    def test_augment_with_different_methods(self, mock_nlpaug, sample_raw_data, test_config):
        """Test augmentation with different methods."""
        test_config["data"]["augmentation"]["methods"] = ["random_insertion", "random_deletion"]
        
        augmented_data = augment_data(sample_raw_data, test_config)
        
        # Should handle different augmentation methods
        assert len(augmented_data) >= len(sample_raw_data)


class TestCreateInstructionFormat:
    """Test instruction format creation."""
    
    def test_create_instruction_basic(self, sample_raw_data):
        """Test basic instruction format creation."""
        template = "Text: {text}\nLabel: {label}\nResponse:"
        
        formatted_data = create_instruction_format(sample_raw_data, template)
        
        assert len(formatted_data) == len(sample_raw_data)
        assert "Text:" in formatted_data[0]["input"]
        assert "Label:" in formatted_data[0]["input"] 
        assert "output" in formatted_data[0]
    
    def test_create_instruction_with_metadata(self, sample_raw_data):
        """Test instruction format with metadata."""
        template = "Text: {text}\nMetadata: {metadata}\nClassify:"
        
        formatted_data = create_instruction_format(sample_raw_data, template)
        
        assert "Metadata:" in formatted_data[0]["input"]
    
    def test_create_instruction_custom_template(self, sample_raw_data):
        """Test custom instruction template."""
        template = "Question: Is this relevant?\nText: {text}\nAnswer:"
        
        formatted_data = create_instruction_format(sample_raw_data, template)
        
        assert "Question:" in formatted_data[0]["input"]
        assert "Answer:" in formatted_data[0]["input"]
    
    def test_create_instruction_missing_placeholder(self, sample_raw_data):
        """Test instruction format with missing placeholder."""
        template = "Text: {nonexistent_field}\nResponse:"
        
        # Should handle missing placeholders gracefully
        formatted_data = create_instruction_format(sample_raw_data, template)
        
        assert len(formatted_data) == len(sample_raw_data)
    
    def test_create_instruction_empty_data(self):
        """Test instruction format with empty data."""
        template = "Text: {text}\nResponse:"
        
        formatted_data = create_instruction_format([], template)
        
        assert formatted_data == []


class TestSaveProcessedData:
    """Test processed data saving."""
    
    def test_save_json_data(self, sample_raw_data):
        """Test saving data as JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "processed.json"
            
            save_processed_data(sample_raw_data, str(output_path), format="json")
            
            assert output_path.exists()
            
            # Verify saved data
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) == len(sample_raw_data)
            assert saved_data[0]["text"] == sample_raw_data[0]["text"]
    
    def test_save_csv_data(self, sample_raw_data):
        """Test saving data as CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "processed.csv"
            
            save_processed_data(sample_raw_data, str(output_path), format="csv")
            
            assert output_path.exists()
            
            # Verify saved data
            df = pd.read_csv(output_path)
            assert len(df) == len(sample_raw_data)
    
    def test_save_split_data(self, sample_raw_data):
        """Test saving train/validation splits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "processed"
            
            save_processed_data(sample_raw_data, str(base_path), 
                              format="json", split_data=True, train_ratio=0.8)
            
            train_path = Path(str(base_path) + "_train.json")
            val_path = Path(str(base_path) + "_val.json")
            
            assert train_path.exists()
            assert val_path.exists()
            
            # Verify splits
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            
            assert len(train_data) + len(val_data) == len(sample_raw_data)
            assert len(train_data) > len(val_data)  # 80/20 split
    
    def test_save_empty_data(self):
        """Test saving empty data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty.json"
            
            save_processed_data([], str(output_path), format="json")
            
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == []
    
    def test_save_create_directory(self, sample_raw_data):
        """Test saving with directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "dir" / "processed.json"
            
            save_processed_data(sample_raw_data, str(nested_path), format="json")
            
            assert nested_path.exists()
            assert nested_path.parent.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_unicode_text_processing(self, test_config):
        """Test processing Unicode text."""
        unicode_data = [
            {"text": "测试中文文本", "label": "relevant"},
            {"text": "العربية النص", "label": "irrelevant"},
            {"text": "Русский текст 🦠", "label": "relevant"}
        ]
        
        valid_data, errors = validate_data(unicode_data, test_config)
        
        assert len(valid_data) == len(unicode_data)
        assert len(errors) == 0
    
    def test_very_large_dataset(self, test_config):
        """Test processing large dataset."""
        large_data = [
            {"text": f"Sample text {i}", "label": "relevant"}
            for i in range(1000)
        ]
        
        valid_data, errors = validate_data(large_data, test_config)
        
        assert len(valid_data) == len(large_data)
        assert len(errors) == 0
    
    def test_special_characters_in_text(self, test_config):
        """Test handling special characters."""
        special_data = [
            {"text": "Text with \"quotes\" and 'apostrophes'", "label": "relevant"},
            {"text": "Text with <tags> & ampersands", "label": "relevant"},
            {"text": "Text with\nnewlines\tand\ttabs", "label": "relevant"}
        ]
        
        valid_data, errors = validate_data(special_data, test_config)
        
        assert len(valid_data) == len(special_data)
        
        # Test instruction formatting preserves special characters
        template = "Text: {text}\nResponse:"
        formatted = create_instruction_format(valid_data, template)
        
        assert "\"quotes\"" in formatted[0]["input"]
        assert "&" in formatted[1]["input"]
        assert "\n" in formatted[2]["input"]
    
    def test_malformed_metadata(self, test_config):
        """Test handling malformed metadata."""
        malformed_data = [
            {"text": "Valid text", "label": "relevant", "metadata": "not_a_dict"},
            {"text": "Another text", "label": "relevant", "metadata": None},
            {"text": "Third text", "label": "relevant"}  # No metadata
        ]
        
        valid_data, errors = validate_data(malformed_data, test_config)
        
        # Should still be valid despite malformed metadata
        assert len(valid_data) == len(malformed_data)


class TestIntegration:
    """Test integration of data preparation pipeline."""
    
    def test_full_pipeline(self, temp_data_file, test_config):
        """Test complete data preparation pipeline."""
        # Load raw data
        raw_data = load_raw_data(temp_data_file, format="json")
        
        # Validate data
        valid_data, errors = validate_data(raw_data, test_config)
        assert len(errors) == 0
        
        # Augment data (mock augmentation)
        with patch('prepare_data.nlpaug'):
            augmented_data = augment_data(valid_data, test_config)
        
        # Create instruction format
        template = "Classify the text: {text}\nLabel:"
        formatted_data = create_instruction_format(augmented_data, template)
        
        # Save processed data
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "final.json"
            save_processed_data(formatted_data, str(output_path), format="json")
            
            assert output_path.exists()
            
            # Verify final output
            with open(output_path, 'r') as f:
                final_data = json.load(f)
            
            assert len(final_data) > 0
            assert "input" in final_data[0]
            assert "output" in final_data[0]
    
    def test_pipeline_error_recovery(self, test_config):
        """Test pipeline error recovery."""
        # Create problematic data
        problematic_data = [
            {"text": "Valid item", "label": "relevant"},
            {"text": "x", "label": "relevant"},  # Too short
            {"missing": "field"}  # Missing required fields
        ]
        
        # Pipeline should continue despite errors
        valid_data, errors = validate_data(problematic_data, test_config)
        
        assert len(valid_data) == 1  # Only valid item remains
        assert len(errors) == 2      # Two items had errors
        
        # Should be able to continue with valid data
        template = "Text: {text}\nResponse:"
        formatted_data = create_instruction_format(valid_data, template)
        
        assert len(formatted_data) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])