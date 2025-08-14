"""Comprehensive tests for evaluation pipeline functionality."""

import pytest
import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Test imports
try:
    from voters.llm.evaluate import (
        LLMEvaluator, load_test_data, compute_metrics,
        create_visualizations, generate_report
    )
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    pytest.skip("Evaluate module not available", allow_module_level=True)


class MockModel:
    """Mock model for testing."""
    def __init__(self):
        self.device = "cpu"
        
    def eval(self):
        return self
        
    def to(self, device):
        self.device = device
        return self
        
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Mock generation that returns JSON-like responses."""
        batch_size = input_ids.size(0)
        
        # Generate different responses based on input
        responses = []
        for i in range(batch_size):
            # Simulate different prediction outcomes
            if i % 3 == 0:
                response = '{"decision": "relevant", "confidence": 0.8, "rationale": "Contains relevant information", "abstain": false}'
            elif i % 3 == 1:
                response = '{"decision": "irrelevant", "confidence": 0.9, "rationale": "Not related to topic", "abstain": false}'
            else:
                response = '{"decision": "relevant", "confidence": 0.4, "rationale": "Uncertain about relevance", "abstain": true}'
            
            # Mock tokenization of response
            response_tokens = torch.randint(1, 1000, (1, len(response.split())))
            responses.append(response_tokens)
        
        sequences = torch.cat(responses, dim=0)
        
        class GenerateOutput:
            def __init__(self, sequences):
                self.sequences = sequences
        
        return GenerateOutput(sequences)


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        
        # Mock tokenization
        input_ids = torch.randint(1, 1000, (len(text), 50))
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, tokens, skip_special_tokens=True):
        """Mock decoding that returns realistic JSON responses."""
        if isinstance(tokens, torch.Tensor):
            if len(tokens.shape) == 2:
                # Batch decode
                return [self._get_mock_response(i) for i in range(tokens.size(0))]
            else:
                # Single decode
                return self._get_mock_response(0)
        else:
            return self._get_mock_response(0)
    
    def _get_mock_response(self, idx):
        """Generate mock JSON response based on index."""
        responses = [
            '{"decision": "relevant", "confidence": 0.85, "rationale": "This text contains relevant information about the topic.", "abstain": false}',
            '{"decision": "irrelevant", "confidence": 0.92, "rationale": "This text is not related to the topic.", "abstain": false}',
            '{"decision": "relevant", "confidence": 0.45, "rationale": "Uncertain about the relevance.", "abstain": true}',
            'Not a valid JSON response',  # Test invalid response handling
        ]
        return responses[idx % len(responses)]


@pytest.fixture
def test_data():
    """Fixture providing test evaluation data."""
    return [
        {
            "text": "Bird flu outbreak reported in multiple farms across the region",
            "label": "relevant",
            "expected_confidence": 0.8,
            "metadata": {"source": "news", "domain": "health"}
        },
        {
            "text": "Weather forecast shows sunny skies for the weekend",
            "label": "irrelevant", 
            "expected_confidence": 0.9,
            "metadata": {"source": "weather", "domain": "weather"}
        },
        {
            "text": "Some reports suggest possible flu cases but details unclear",
            "label": "relevant",
            "expected_confidence": 0.4,
            "metadata": {"source": "rumors", "domain": "health"}
        },
        {
            "text": "Economic impact of recent events on agriculture sector",
            "label": "relevant",
            "expected_confidence": 0.7,
            "metadata": {"source": "economic", "domain": "economics"}
        },
        {
            "text": "Random text that should be irrelevant",
            "label": "irrelevant",
            "expected_confidence": 0.85,
            "metadata": {"source": "random", "domain": "misc"}
        }
    ]


@pytest.fixture
def mock_model():
    """Fixture providing mock model."""
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def temp_test_data_file(test_data):
    """Fixture providing temporary test data file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        yield f.name
    Path(f.name).unlink()


class TestLoadTestData:
    """Test test data loading functionality."""
    
    def test_load_test_data_success(self, temp_test_data_file, test_data):
        """Test successful test data loading."""
        data = load_test_data(temp_test_data_file)
        
        assert len(data) == len(test_data)
        assert data[0]["text"] == test_data[0]["text"]
        assert data[0]["label"] == test_data[0]["label"]
    
    def test_load_test_data_missing_file(self):
        """Test handling of missing test data file."""
        with pytest.raises(FileNotFoundError):
            load_test_data("nonexistent_data.json")
    
    def test_load_test_data_invalid_json(self):
        """Test handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json content}")
            f.flush()
            
            with pytest.raises(json.JSONDecodeError):
                load_test_data(f.name)
            
            Path(f.name).unlink()
    
    def test_load_test_data_empty_file(self):
        """Test handling of empty test data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("[]")
            f.flush()
            
            data = load_test_data(f.name)
            assert data == []
            
            Path(f.name).unlink()
    
    def test_load_test_data_missing_required_fields(self):
        """Test handling of data with missing required fields."""
        incomplete_data = [{"text": "test"}]  # Missing label
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_data, f)
            f.flush()
            
            data = load_test_data(f.name)
            assert len(data) == 1
            # Should load successfully, validation happens during evaluation
            
            Path(f.name).unlink()


class TestLLMEvaluator:
    """Test LLM Evaluator functionality."""
    
    def test_evaluator_initialization(self, mock_model, mock_tokenizer):
        """Test evaluator initialization."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        assert evaluator.model == mock_model
        assert evaluator.tokenizer == mock_tokenizer
        assert evaluator.predictions == []
        assert evaluator.results == {}
    
    def test_evaluate_single_text(self, mock_model, mock_tokenizer):
        """Test single text evaluation."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        text = "Bird flu outbreak in farms"
        result = evaluator.evaluate_single(text)
        
        assert isinstance(result, dict)
        assert "raw_response" in result
        assert isinstance(result["raw_response"], str)
    
    def test_evaluate_single_with_parsing(self, mock_model, mock_tokenizer):
        """Test single text evaluation with JSON parsing."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        text = "Bird flu outbreak in farms"
        result = evaluator.evaluate_single(text, parse_json=True)
        
        assert isinstance(result, dict)
        assert "raw_response" in result
        
        # If JSON parsing successful, should have structured data
        if "parsed_response" in result:
            assert isinstance(result["parsed_response"], dict)
    
    def test_evaluate_dataset(self, mock_model, mock_tokenizer, test_data):
        """Test dataset evaluation."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        results = evaluator.evaluate_dataset(test_data[:3])  # Use subset for speed
        
        assert len(results) == 3
        assert all("text" in result for result in results)
        assert all("label" in result for result in results)
        assert all("prediction" in result for result in results)
    
    def test_evaluate_dataset_with_batch_processing(self, mock_model, mock_tokenizer, test_data):
        """Test dataset evaluation with batch processing."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        results = evaluator.evaluate_dataset(test_data, batch_size=2)
        
        assert len(results) == len(test_data)
        assert all("prediction" in result for result in results)
    
    def test_evaluate_empty_dataset(self, mock_model, mock_tokenizer):
        """Test evaluation with empty dataset."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        results = evaluator.evaluate_dataset([])
        
        assert results == []
    
    def test_evaluate_with_invalid_responses(self, mock_model, mock_tokenizer):
        """Test evaluation handling invalid model responses."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        # Mock tokenizer to return invalid JSON
        mock_tokenizer.decode = lambda x, **kwargs: "Invalid response format"
        
        text = "Test text"
        result = evaluator.evaluate_single(text, parse_json=True)
        
        assert "raw_response" in result
        assert result.get("parsed_response") is None or result.get("parse_error") is not None
    
    def test_evaluate_with_abstention(self, mock_model, mock_tokenizer):
        """Test evaluation with model abstention."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        # Mock response with abstention
        mock_response = '{"decision": "relevant", "confidence": 0.3, "abstain": true}'
        mock_tokenizer.decode = lambda x, **kwargs: mock_response
        
        text = "Uncertain text"
        result = evaluator.evaluate_single(text, parse_json=True)
        
        if "parsed_response" in result:
            assert result["parsed_response"].get("abstain") is True


class TestComputeMetrics:
    """Test metrics computation functionality."""
    
    def test_compute_basic_metrics(self):
        """Test basic metrics computation."""
        predictions = ["relevant", "irrelevant", "relevant", "irrelevant"]
        labels = ["relevant", "irrelevant", "relevant", "relevant"]
        
        metrics = compute_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = ["relevant", "irrelevant", "relevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        
        metrics = compute_metrics(predictions, labels)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
    
    def test_compute_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""
        predictions = ["irrelevant", "relevant", "irrelevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        
        metrics = compute_metrics(predictions, labels)
        
        assert metrics["accuracy"] == 0.0
    
    def test_compute_metrics_with_confidence(self):
        """Test metrics computation with confidence scores."""
        predictions = ["relevant", "irrelevant", "relevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        confidences = [0.9, 0.8, 0.7]
        
        metrics = compute_metrics(predictions, labels, confidences=confidences)
        
        assert "accuracy" in metrics
        assert "avg_confidence" in metrics
        assert "confidence_std" in metrics
        assert metrics["avg_confidence"] == np.mean(confidences)
    
    def test_compute_metrics_with_abstentions(self):
        """Test metrics computation with abstentions."""
        predictions = ["relevant", "abstain", "irrelevant", "relevant"]
        labels = ["relevant", "relevant", "irrelevant", "relevant"]
        
        metrics = compute_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "abstention_rate" in metrics
        assert metrics["abstention_rate"] == 0.25  # 1 out of 4 abstained
    
    def test_compute_metrics_empty_inputs(self):
        """Test metrics with empty inputs."""
        metrics = compute_metrics([], [])
        
        # Should handle gracefully
        assert isinstance(metrics, dict)
    
    def test_compute_metrics_mismatched_lengths(self):
        """Test metrics with mismatched input lengths."""
        predictions = ["relevant", "irrelevant"]
        labels = ["relevant"]  # Different length
        
        with pytest.raises(ValueError):
            compute_metrics(predictions, labels)
    
    def test_compute_metrics_multiclass(self):
        """Test metrics with multiple classes."""
        predictions = ["class1", "class2", "class3", "class1", "class2"]
        labels = ["class1", "class2", "class1", "class1", "class3"]
        
        metrics = compute_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics
        assert "macro_f1" in metrics
    
    def test_compute_calibration_metrics(self):
        """Test confidence calibration metrics."""
        predictions = ["relevant", "relevant", "irrelevant", "irrelevant"]
        labels = ["relevant", "irrelevant", "irrelevant", "irrelevant"]
        confidences = [0.9, 0.8, 0.7, 0.6]
        
        metrics = compute_metrics(predictions, labels, confidences=confidences)
        
        if "calibration_error" in metrics:
            assert isinstance(metrics["calibration_error"], float)
            assert metrics["calibration_error"] >= 0


class TestCreateVisualizations:
    """Test visualization creation functionality."""
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_confusion_matrix(self, mock_show, mock_savefig):
        """Test confusion matrix creation."""
        predictions = ["relevant", "irrelevant", "relevant", "irrelevant"]
        labels = ["relevant", "irrelevant", "relevant", "relevant"]
        
        create_visualizations(predictions, labels, save_dir="test_dir")
        
        # Should have called savefig for confusion matrix
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_confidence_histogram(self, mock_show, mock_savefig):
        """Test confidence histogram creation."""
        predictions = ["relevant", "irrelevant", "relevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        confidences = [0.9, 0.8, 0.7]
        
        create_visualizations(predictions, labels, confidences=confidences, save_dir="test_dir")
        
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_calibration_plot(self, mock_show, mock_savefig):
        """Test calibration plot creation."""
        predictions = ["relevant"] * 10 + ["irrelevant"] * 10
        labels = ["relevant"] * 8 + ["irrelevant"] * 2 + ["relevant"] * 2 + ["irrelevant"] * 8
        confidences = [0.9] * 5 + [0.7] * 5 + [0.8] * 5 + [0.6] * 5
        
        create_visualizations(predictions, labels, confidences=confidences, save_dir="test_dir")
        
        mock_savefig.assert_called()
    
    def test_create_visualizations_no_save_dir(self):
        """Test visualization creation without save directory."""
        predictions = ["relevant", "irrelevant"]
        labels = ["relevant", "relevant"]
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.show'):
                create_visualizations(predictions, labels)
                # Should still work without errors
    
    def test_create_visualizations_empty_data(self):
        """Test visualization with empty data."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.show'):
                create_visualizations([], [])
                # Should handle gracefully


class TestGenerateReport:
    """Test report generation functionality."""
    
    def test_generate_report_basic(self):
        """Test basic report generation."""
        predictions = ["relevant", "irrelevant", "relevant", "irrelevant"]
        labels = ["relevant", "irrelevant", "relevant", "relevant"]
        
        report = generate_report(predictions, labels)
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "metrics" in report
        assert "timestamp" in report
    
    def test_generate_report_with_confidence(self):
        """Test report generation with confidence scores."""
        predictions = ["relevant", "irrelevant", "relevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        confidences = [0.9, 0.8, 0.7]
        
        report = generate_report(predictions, labels, confidences=confidences)
        
        assert "confidence_stats" in report
        assert "avg_confidence" in report["confidence_stats"]
    
    def test_generate_report_with_metadata(self):
        """Test report generation with additional metadata."""
        predictions = ["relevant", "irrelevant"]
        labels = ["relevant", "relevant"]
        metadata = {"model_name": "test_model", "dataset": "test_data"}
        
        report = generate_report(predictions, labels, metadata=metadata)
        
        assert "metadata" in report
        assert report["metadata"]["model_name"] == "test_model"
    
    def test_generate_report_with_errors(self):
        """Test report generation with error tracking."""
        predictions = ["relevant", "error", "irrelevant"]
        labels = ["relevant", "relevant", "irrelevant"] 
        errors = [None, "JSON parse error", None]
        
        report = generate_report(predictions, labels, errors=errors)
        
        assert "error_stats" in report
        assert "error_rate" in report["error_stats"]
    
    def test_generate_report_save_to_file(self):
        """Test saving report to file."""
        predictions = ["relevant", "irrelevant"]
        labels = ["relevant", "relevant"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report = generate_report(predictions, labels, save_path=f.name)
            
            # Should save to file
            assert Path(f.name).exists()
            
            # Load and verify
            with open(f.name, 'r') as rf:
                saved_report = json.load(rf)
                assert saved_report["metrics"]["accuracy"] == report["metrics"]["accuracy"]
            
            Path(f.name).unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_memory_efficient_evaluation(self, mock_model, mock_tokenizer):
        """Test memory efficient evaluation with large dataset."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        # Create large dataset
        large_dataset = [{"text": f"Text {i}", "label": "relevant"} for i in range(100)]
        
        # Evaluate with small batch size
        results = evaluator.evaluate_dataset(large_dataset, batch_size=5)
        
        assert len(results) == 100
        assert all("prediction" in result for result in results)
    
    def test_unicode_text_evaluation(self, mock_model, mock_tokenizer):
        """Test evaluation with Unicode text."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        unicode_data = [
            {"text": "测试中文文本", "label": "relevant"},
            {"text": "العربية النص", "label": "irrelevant"},
            {"text": "Русский текст 🦠", "label": "relevant"}
        ]
        
        results = evaluator.evaluate_dataset(unicode_data)
        
        assert len(results) == 3
        assert all("prediction" in result for result in results)
    
    def test_very_long_text_evaluation(self, mock_model, mock_tokenizer):
        """Test evaluation with very long text."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        long_text = "This is a very long text. " * 1000  # ~25k characters
        long_data = [{"text": long_text, "label": "relevant"}]
        
        results = evaluator.evaluate_dataset(long_data)
        
        assert len(results) == 1
        assert "prediction" in results[0]
    
    def test_evaluation_with_timeouts(self, mock_model, mock_tokenizer):
        """Test evaluation with generation timeouts."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        # Mock timeout error
        mock_model.generate = Mock(side_effect=TimeoutError("Generation timeout"))
        
        data = [{"text": "Test text", "label": "relevant"}]
        results = evaluator.evaluate_dataset(data)
        
        assert len(results) == 1
        assert "error" in results[0] or results[0]["prediction"] == "error"
    
    def test_concurrent_evaluation(self, mock_model, mock_tokenizer, test_data):
        """Test concurrent evaluation (thread safety)."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        import threading
        results_list = []
        errors = []
        
        def evaluate_subset(data_subset):
            try:
                results = evaluator.evaluate_dataset(data_subset)
                results_list.extend(results)
            except Exception as e:
                errors.append(e)
        
        # Split data and run concurrently
        mid = len(test_data) // 2
        thread1 = threading.Thread(target=evaluate_subset, args=(test_data[:mid],))
        thread2 = threading.Thread(target=evaluate_subset, args=(test_data[mid:],))
        
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        
        assert len(errors) == 0
        assert len(results_list) == len(test_data)


class TestIntegration:
    """Test integration functionality."""
    
    def test_full_evaluation_pipeline(self, mock_model, mock_tokenizer, test_data):
        """Test complete evaluation pipeline."""
        # Initialize evaluator
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        # Evaluate dataset
        results = evaluator.evaluate_dataset(test_data)
        
        # Extract predictions and labels
        predictions = [result.get("prediction", "error") for result in results]
        labels = [result["label"] for result in results]
        confidences = []
        
        for result in results:
            if "parsed_response" in result and result["parsed_response"]:
                conf = result["parsed_response"].get("confidence", 0.5)
                confidences.append(conf)
            else:
                confidences.append(0.5)  # Default confidence
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, confidences=confidences)
        
        # Generate report
        report = generate_report(predictions, labels, confidences=confidences, 
                               metadata={"test_run": True})
        
        # Create visualizations (mock)
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.show'):
                create_visualizations(predictions, labels, confidences=confidences)
        
        # Verify pipeline
        assert len(results) == len(test_data)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert isinstance(report, dict)
        assert "summary" in report
    
    def test_evaluation_with_model_loading(self, test_data):
        """Test evaluation with actual model loading simulation."""
        with patch('voters.llm.evaluate.AutoModelForCausalLM') as mock_model_class:
            with patch('voters.llm.evaluate.AutoTokenizer') as mock_tokenizer_class:
                mock_model_class.from_pretrained.return_value = MockModel()
                mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
                
                # Simulate loading model and tokenizer
                model = mock_model_class.from_pretrained("test/model")
                tokenizer = mock_tokenizer_class.from_pretrained("test/tokenizer")
                
                # Evaluate
                evaluator = LLMEvaluator(model, tokenizer)
                results = evaluator.evaluate_dataset(test_data[:3])
                
                assert len(results) == 3
                assert all("prediction" in result for result in results)


class TestModularity:
    """Test modularity and configuration options."""
    
    def test_metric_selection(self):
        """Test selective metric computation."""
        predictions = ["relevant", "irrelevant", "relevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        
        # Test with specific metrics
        metrics = compute_metrics(predictions, labels, metrics=["accuracy", "f1_score"])
        
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        # Other metrics might or might not be included depending on implementation
    
    def test_visualization_selection(self):
        """Test selective visualization creation."""
        predictions = ["relevant", "irrelevant", "relevant"]
        labels = ["relevant", "irrelevant", "relevant"]
        confidences = [0.9, 0.8, 0.7]
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.show'):
                # Test creating only specific visualizations
                create_visualizations(predictions, labels, confidences=confidences,
                                    plots=["confusion_matrix", "confidence_histogram"])
                
                # Should have made some plots
                assert mock_savefig.called
    
    def test_report_customization(self):
        """Test report customization options."""
        predictions = ["relevant", "irrelevant"]
        labels = ["relevant", "relevant"]
        
        # Test with custom sections
        report = generate_report(predictions, labels, 
                               include_sections=["metrics", "summary"])
        
        assert "metrics" in report
        assert "summary" in report
    
    def test_batch_size_configuration(self, mock_model, mock_tokenizer, test_data):
        """Test configurable batch size."""
        evaluator = LLMEvaluator(mock_model, mock_tokenizer)
        
        # Test different batch sizes
        for batch_size in [1, 2, 5]:
            results = evaluator.evaluate_dataset(test_data[:5], batch_size=batch_size)
            assert len(results) == 5
            assert all("prediction" in result for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])