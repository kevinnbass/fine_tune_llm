"""Pytest configuration and fixtures."""

import pytest
import asyncio
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "H5N1 bird flu outbreak reported in multiple farms across the region",
        "Scientists study avian influenza transmission patterns",
        "Weather forecast shows sunny skies for the weekend",
        "Recipe for delicious chicken soup with vegetables",
        "Breaking news: new flu strain discovered in poultry",
        "The quick brown fox jumps over the lazy dog",
        "WHO reports increased surveillance for bird flu cases",
        "Not related to bird flu at all - just general content"
    ]


@pytest.fixture
def sample_labels():
    """Sample labels corresponding to sample texts."""
    return [
        "HIGH_RISK",
        "MEDIUM_RISK", 
        "NO_RISK",
        "NO_RISK",
        "HIGH_RISK",
        "NO_RISK",
        "MEDIUM_RISK",
        "NO_RISK"
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return [
        {"source": "news", "date": "2023-01-01"},
        {"source": "research", "date": "2023-01-02"},
        {"source": "weather", "date": "2023-01-03"},
        {"source": "recipe", "date": "2023-01-04"},
        {"source": "news", "date": "2023-01-05"},
        {"source": "test", "date": "2023-01-06"},
        {"source": "who", "date": "2023-01-07"},
        {"source": "general", "date": "2023-01-08"}
    ]


@pytest.fixture
def sample_voter_outputs():
    """Sample voter outputs for testing."""
    return {
        'regex_dsl': {
            'probs': {'HIGH_RISK': 0.8, 'MEDIUM_RISK': 0.1, 'LOW_RISK': 0.05, 'NO_RISK': 0.05},
            'abstain': False,
            'decision': 'HIGH_RISK',
            'latency_ms': 5.0,
            'cost_cents': 0.0001,
            'model_id': 'regex_dsl'
        },
        'tfidf_lr': {
            'probs': {'HIGH_RISK': 0.7, 'MEDIUM_RISK': 0.2, 'LOW_RISK': 0.05, 'NO_RISK': 0.05},
            'abstain': False,
            'decision': 'HIGH_RISK',
            'latency_ms': 15.0,
            'cost_cents': 0.001,
            'model_id': 'tfidf_lr'
        },
        'llm_lora': {
            'probs': {'HIGH_RISK': 0.6, 'MEDIUM_RISK': 0.3, 'LOW_RISK': 0.05, 'NO_RISK': 0.05},
            'abstain': False,
            'decision': 'HIGH_RISK',
            'latency_ms': 500.0,
            'cost_cents': 0.05,
            'model_id': 'llm_lora'
        }
    }


@pytest.fixture
def temp_config_dir():
    """Temporary directory with config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Create sample configs
        labels_config = {
            'labels': [
                {'name': 'HIGH_RISK', 'harm_weight': 10.0},
                {'name': 'MEDIUM_RISK', 'harm_weight': 5.0},
                {'name': 'LOW_RISK', 'harm_weight': 1.0},
                {'name': 'NO_RISK', 'harm_weight': 0.1}
            ]
        }
        
        voters_config = {
            'voters': {
                'regex_dsl': {'enabled': True, 'cost_cents': 0.0001},
                'tfidf_lr': {'enabled': True, 'cost_cents': 0.001},
                'llm_lora': {'enabled': True, 'cost_cents': 0.05}
            }
        }
        
        import yaml
        
        with open(config_dir / 'labels.yaml', 'w') as f:
            yaml.dump(labels_config, f)
        
        with open(config_dir / 'voters.yaml', 'w') as f:
            yaml.dump(voters_config, f)
        
        yield str(config_dir)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.dbsize = AsyncMock(return_value=100)
    mock_redis.keys = AsyncMock(return_value=[])
    return mock_redis


@pytest.fixture
def mock_model():
    """Mock trained model for testing."""
    mock = Mock()
    mock.predict = Mock(return_value=np.array([0, 1, 2, 3]))
    mock.predict_proba = Mock(return_value=np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.15, 0.05],
        [0.05, 0.1, 0.8, 0.05],
        [0.05, 0.05, 0.1, 0.8]
    ]))
    return mock


@pytest.fixture
def sample_lf_votes():
    """Sample labeling function votes."""
    return np.array([
        [0, 1, -1, 0, -1],  # HIGH_RISK sample
        [1, 1, 0, -1, 1],   # MEDIUM_RISK sample
        [-1, -1, 2, 2, -1], # LOW_RISK sample
        [-1, -1, -1, 3, 3]  # NO_RISK sample
    ])


@pytest.fixture
def sample_probabilities():
    """Sample probability distributions."""
    return np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.15, 0.05],
        [0.05, 0.1, 0.8, 0.05],
        [0.05, 0.05, 0.1, 0.8]
    ])


@pytest.fixture
def sample_features():
    """Sample feature matrix for testing."""
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing."""
    return pd.DataFrame({
        'id': range(8),
        'text': [
            "H5N1 bird flu outbreak",
            "Avian influenza study",
            "Weather forecast",
            "Chicken recipe",
            "New flu strain",
            "Random text",
            "WHO surveillance",
            "General content"
        ],
        'label': [
            "HIGH_RISK", "MEDIUM_RISK", "NO_RISK", "NO_RISK",
            "HIGH_RISK", "NO_RISK", "MEDIUM_RISK", "NO_RISK"
        ],
        'metadata': [
            '{"source": "news"}', '{"source": "research"}',
            '{"source": "weather"}', '{"source": "recipe"}',
            '{"source": "news"}', '{"source": "test"}',
            '{"source": "who"}', '{"source": "general"}'
        ]
    })


@pytest.fixture
def mock_ensemble_system():
    """Mock ensemble system for testing."""
    mock = Mock()
    mock.voters = {
        'regex_dsl': Mock(),
        'tfidf_lr': Mock(),
        'llm_lora': Mock()
    }
    mock.predict_with_cascade = Mock(return_value={
        'decision': 'HIGH_RISK',
        'confidence': 0.8,
        'abstain': False,
        'tier': 1
    })
    return mock


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)


# Async test utilities
def async_test(func):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper


# Test data generators
def generate_test_texts(n: int = 100, seed: int = 42) -> List[str]:
    """Generate test texts with known patterns."""
    np.random.seed(seed)
    
    texts = []
    patterns = [
        "H5N1 outbreak in {} with {} cases reported",
        "Avian influenza detected in {} poultry farm",
        "Weather forecast for {} shows {} conditions",
        "Recipe for {} with {} ingredients",
        "News report about {} incident in {}"
    ]
    
    locations = ["California", "Texas", "New York", "Florida"]
    values = ["123", "456", "sunny", "rainy", "delicious", "healthy"]
    
    for i in range(n):
        pattern = np.random.choice(patterns)
        location = np.random.choice(locations)
        value = np.random.choice(values)
        text = pattern.format(location, value)
        texts.append(text)
    
    return texts


def generate_drift_data(
    reference_size: int = 1000,
    drift_size: int = 100,
    drift_magnitude: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for drift testing."""
    np.random.seed(42)
    
    # Reference data
    reference = np.random.normal(0, 1, (reference_size, 10))
    
    # Drifted data
    drift_mean = np.random.normal(drift_magnitude, 0.1, 10)
    drifted = np.random.normal(drift_mean, 1, (drift_size, 10))
    
    return reference, drifted