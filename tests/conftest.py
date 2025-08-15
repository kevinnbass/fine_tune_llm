"""
Global pytest configuration and fixtures.

This module provides shared fixtures and configuration for all tests
to support comprehensive testing with 100% line coverage.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional, Generator
import logging
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import core modules for testing
from src.fine_tune_llm.core.events import EventBus, Event, EventType
from src.fine_tune_llm.config.manager import ConfigManager
from src.fine_tune_llm.utils.logging import get_centralized_logger


# Configure logging for tests
logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="fine_tune_llm_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "name": "test-model",
            "type": "llm",
            "parameters": {
                "max_length": 512,
                "temperature": 0.7
            }
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 2e-4,
            "epochs": 1,
            "save_steps": 100
        },
        "data": {
            "train_file": "train.json",
            "eval_file": "eval.json",
            "max_samples": 1000
        },
        "output": {
            "dir": "outputs",
            "save_model": True,
            "save_logs": True
        }
    }


@pytest.fixture
def temp_config_file(test_data_dir, sample_config):
    """Create temporary configuration file."""
    config_file = test_data_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    mock_manager = Mock(spec=ConfigManager)
    mock_manager.get.return_value = "test_value"
    mock_manager.set.return_value = True
    mock_manager.validate.return_value = True
    mock_manager.load_from_file.return_value = True
    mock_manager.save_to_file.return_value = True
    return mock_manager


@pytest.fixture
def mock_event_bus():
    """Mock event bus."""
    mock_bus = Mock(spec=EventBus)
    mock_bus.publish.return_value = None
    mock_bus.subscribe.return_value = True
    mock_bus.unsubscribe.return_value = True
    return mock_bus


@pytest.fixture
def sample_event():
    """Sample event for testing."""
    return Event(
        type=EventType.TRAINING_STARTED,
        data={"model": "test-model", "timestamp": datetime.now(timezone.utc).isoformat()},
        source="test"
    )


@pytest.fixture
def sample_training_data():
    """Sample training data."""
    return [
        {"input": "What is AI?", "output": "AI is artificial intelligence."},
        {"input": "Explain machine learning", "output": "Machine learning is a subset of AI."},
        {"input": "What is deep learning?", "output": "Deep learning uses neural networks."}
    ]


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3'],
        'label': [0, 1, 0],
        'confidence': [0.95, 0.87, 0.92]
    })


@pytest.fixture
def sample_numpy_array():
    """Sample numpy array for testing."""
    return np.random.random((10, 5))


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.8, 0.2])
    mock_model.fit.return_value = None
    mock_model.save.return_value = True
    mock_model.load.return_value = True
    mock_model.parameters = {"param1": "value1"}
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "decoded text"
    mock_tokenizer.vocab_size = 50000
    return mock_tokenizer


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 100
    mock_dataset.__getitem__.return_value = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [1]
    }
    return mock_dataset


@pytest.fixture
def mock_trainer():
    """Mock trainer for testing."""
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {"loss": 0.5, "accuracy": 0.95}
    mock_trainer.evaluate.return_value = {"eval_loss": 0.3, "eval_accuracy": 0.97}
    mock_trainer.save_model.return_value = True
    return mock_trainer


@pytest.fixture
def mock_file_system(test_data_dir):
    """Mock file system operations."""
    # Create some test files
    (test_data_dir / "models").mkdir(exist_ok=True)
    (test_data_dir / "data").mkdir(exist_ok=True)
    (test_data_dir / "outputs").mkdir(exist_ok=True)
    
    # Create test files
    test_model_file = test_data_dir / "models" / "test_model.bin"
    test_model_file.write_text("mock model data")
    
    test_data_file = test_data_dir / "data" / "test_data.json"
    with open(test_data_file, 'w') as f:
        json.dump({"test": "data"}, f)
    
    return test_data_dir


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=1):
            yield True


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.cuda.device_count', return_value=0):
            yield False


@pytest.fixture
def mock_network_available():
    """Mock network connectivity."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok"}
        yield True


@pytest.fixture
def mock_network_unavailable():
    """Mock network unavailability."""
    with patch('requests.get', side_effect=ConnectionError("Network unavailable")):
        yield False


@pytest.fixture
def mock_huggingface_model():
    """Mock Hugging Face model."""
    with patch('transformers.AutoModel.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            yield mock_model, mock_tokenizer


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API."""
    with patch('openai.ChatCompletion.create') as mock_api:
        mock_api.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Mock OpenAI response"
                    }
                }
            ]
        }
        yield mock_api


@pytest.fixture
def sample_metrics():
    """Sample metrics data."""
    return {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.89,
        "f1": 0.90,
        "loss": 0.05,
        "perplexity": 1.2
    }


@pytest.fixture
def mock_database():
    """Mock database connection."""
    mock_db = MagicMock()
    mock_db.connect.return_value = True
    mock_db.execute.return_value = True
    mock_db.fetchall.return_value = [{"id": 1, "data": "test"}]
    mock_db.commit.return_value = True
    mock_db.close.return_value = True
    return mock_db


@pytest.fixture
def mock_cache():
    """Mock cache system."""
    cache_data = {}
    
    mock_cache = MagicMock()
    mock_cache.get.side_effect = lambda key: cache_data.get(key)
    mock_cache.set.side_effect = lambda key, value: cache_data.update({key: value})
    mock_cache.delete.side_effect = lambda key: cache_data.pop(key, None)
    mock_cache.clear.side_effect = lambda: cache_data.clear()
    
    return mock_cache


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock_logger = Mock()
    mock_logger.debug.return_value = None
    mock_logger.info.return_value = None
    mock_logger.warning.return_value = None
    mock_logger.error.return_value = None
    mock_logger.critical.return_value = None
    return mock_logger


@pytest.fixture
def performance_timer():
    """Performance timing fixture."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return Timer()


@pytest.fixture
def memory_profiler():
    """Memory profiling fixture."""
    import psutil
    import os
    
    class MemoryProfiler:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
            self.end_memory = None
        
        def start(self):
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            self.end_memory = self.process.memory_info().rss
            return self.memory_used
        
        @property
        def memory_used(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return 0
    
    return MemoryProfiler()


@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch):
    """Isolate each test by clearing global state."""
    # Clear environment variables that might affect tests
    env_vars_to_clear = [
        'CUDA_VISIBLE_DEVICES',
        'HF_HOME',
        'TRANSFORMERS_CACHE',
        'TORCH_HOME'
    ]
    
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    
    # Set test-specific environment
    monkeypatch.setenv('TESTING', 'true')
    monkeypatch.setenv('LOG_LEVEL', 'CRITICAL')


@pytest.fixture
def error_conditions():
    """Fixture providing various error conditions for testing."""
    return {
        'connection_error': ConnectionError("Connection failed"),
        'timeout_error': TimeoutError("Operation timed out"),
        'file_not_found': FileNotFoundError("File not found"),
        'permission_error': PermissionError("Permission denied"),
        'value_error': ValueError("Invalid value"),
        'type_error': TypeError("Invalid type"),
        'key_error': KeyError("Key not found"),
        'index_error': IndexError("Index out of range"),
        'memory_error': MemoryError("Out of memory"),
        'runtime_error': RuntimeError("Runtime error")
    }


# Pytest hooks for custom behavior

def pytest_configure(config):
    """Configure pytest for our testing needs."""
    # Ensure test reports directory exists
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test path and name
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add markers based on test names
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        elif "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        elif "gpu" in item.name:
            item.add_marker(pytest.mark.gpu)
        elif "network" in item.name:
            item.add_marker(pytest.mark.network)


def pytest_runtest_setup(item):
    """Setup for each test."""
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip network tests if network not available
    if item.get_closest_marker("network"):
        try:
            import requests
            requests.get("http://httpbin.org/get", timeout=5)
        except:
            pytest.skip("Network not available")


def pytest_runtest_teardown(item):
    """Cleanup after each test."""
    # Clear any global state
    import gc
    gc.collect()


# Helper functions for tests

def assert_approximate_equal(actual, expected, tolerance=1e-6):
    """Assert that two values are approximately equal."""
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a - e) < tolerance
    else:
        assert abs(actual - expected) < tolerance


def create_temp_file(directory: Path, filename: str, content: str = "") -> Path:
    """Create a temporary file with content."""
    file_path = directory / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def create_mock_response(status_code: int = 200, json_data: Dict = None, text: str = ""):
    """Create a mock HTTP response."""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data or {}
    mock_response.text = text
    mock_response.raise_for_status.return_value = None
    return mock_response