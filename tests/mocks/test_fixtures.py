"""
Test fixtures and utility functions for mock management.

This module provides comprehensive test fixtures for setting up and managing
mock environments, dependency injection, and isolated testing contexts.
"""

import pytest
import tempfile
import shutil
import threading
import contextlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, ContextManager
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from .external_services import (
    MockHuggingFaceAPI, MockOpenAIAPI, MockDatabaseConnection,
    MockFileSystem, MockNetworkService, MockCacheService,
    MockMLFlowTracker, MockWandBTracker, MockS3Service, MockGCSService
)
from .model_mocks import (
    MockTransformerModel, MockTokenizer, MockTrainingDataset,
    MockDataLoader, MockOptimizer, MockScheduler, MockTrainer, MockPredictor
)
from .infrastructure_mocks import (
    MockGPUManager, MockResourceMonitor, MockConfigValidator,
    MockSecretManager, MockEmailService, MockSlackNotifier,
    MockPrometheusCollector, MockGrafanaClient
)


class MockEnvironment:
    """Comprehensive mock environment for testing."""
    
    def __init__(self):
        self.external_services = {}
        self.model_components = {}
        self.infrastructure = {}
        self.temp_directories = []
        self.active_patches = []
        self.cleanup_callbacks = []
        
    def setup(self):
        """Set up the mock environment."""
        self._setup_external_services()
        self._setup_model_components()
        self._setup_infrastructure()
        self._setup_patches()
        
    def cleanup(self):
        """Clean up the mock environment."""
        # Stop active patches
        for patcher in self.active_patches:
            try:
                patcher.stop()
            except Exception:
                pass
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception:
                pass
        
        # Clean up temporary directories
        for temp_dir in self.temp_directories:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
        
        # Clean up infrastructure components
        for component in self.infrastructure.values():
            if hasattr(component, 'stop_monitoring'):
                try:
                    component.stop_monitoring()
                except Exception:
                    pass
    
    def _setup_external_services(self):
        """Set up external service mocks."""
        self.external_services.update({
            'huggingface': MockHuggingFaceAPI(),
            'openai': MockOpenAIAPI(),
            'database': MockDatabaseConnection(),
            'filesystem': MockFileSystem(),
            'network': MockNetworkService(),
            'cache': MockCacheService(),
            'mlflow': MockMLFlowTracker(),
            'wandb': MockWandBTracker(),
            's3': MockS3Service(),
            'gcs': MockGCSService()
        })
        
        # Disable network delays for faster testing
        self.external_services['huggingface'].disable_network_delay()
    
    def _setup_model_components(self):
        """Set up model component mocks."""
        self.model_components.update({
            'tokenizer': MockTokenizer(),
            'model': MockTransformerModel('test-model'),
            'dataset': MockTrainingDataset('test-dataset'),
            'optimizer': None,  # Will be created when needed
            'scheduler': None,  # Will be created when needed
            'trainer': None,    # Will be created when needed
            'predictor': None   # Will be created when needed
        })
    
    def _setup_infrastructure(self):
        """Set up infrastructure component mocks."""
        self.infrastructure.update({
            'gpu_manager': MockGPUManager(num_gpus=2),
            'resource_monitor': MockResourceMonitor(),
            'config_validator': MockConfigValidator(),
            'secret_manager': MockSecretManager(),
            'email_service': MockEmailService(),
            'slack_notifier': MockSlackNotifier(),
            'prometheus_collector': MockPrometheusCollector(),
            'grafana_client': MockGrafanaClient()
        })
    
    def _setup_patches(self):
        """Set up automatic patches for common libraries."""
        patches = [
            # HuggingFace patches
            patch('transformers.AutoModel.from_pretrained', 
                  side_effect=self.external_services['huggingface'].from_pretrained),
            patch('transformers.AutoTokenizer.from_pretrained',
                  return_value=self.model_components['tokenizer']),
            patch('datasets.load_dataset',
                  side_effect=self.external_services['huggingface'].load_dataset),
            
            # OpenAI patches
            patch('openai.ChatCompletion.create',
                  side_effect=self.external_services['openai'].chat_completions_create),
            patch('openai.Embedding.create',
                  side_effect=self.external_services['openai'].embeddings_create),
            
            # Network patches
            patch('requests.get', side_effect=self.external_services['network'].get),
            patch('requests.post', side_effect=self.external_services['network'].post),
            patch('requests.put', side_effect=self.external_services['network'].put),
            patch('requests.delete', side_effect=self.external_services['network'].delete),
            
            # GPU patches
            patch('torch.cuda.is_available', return_value=True),
            patch('torch.cuda.device_count', return_value=2),
            patch('torch.cuda.get_device_properties', 
                  return_value=Mock(name='Mock GPU', total_memory=8589934592)),
            
            # File system patches for cloud storage
            patch('boto3.client', return_value=self.external_services['s3']),
            patch('google.cloud.storage.Client', return_value=self.external_services['gcs'])
        ]
        
        # Start all patches
        for patcher in patches:
            try:
                patcher.start()
                self.active_patches.append(patcher)
            except Exception as e:
                print(f"Warning: Failed to start patch {patcher}: {e}")
    
    def create_temp_directory(self, prefix: str = "test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_directories.append(temp_dir)
        return temp_dir
    
    def add_cleanup_callback(self, callback):
        """Add cleanup callback function."""
        self.cleanup_callbacks.append(callback)
    
    def get_service(self, service_name: str):
        """Get external service mock."""
        return self.external_services.get(service_name)
    
    def get_model_component(self, component_name: str):
        """Get model component mock."""
        return self.model_components.get(component_name)
    
    def get_infrastructure(self, infrastructure_name: str):
        """Get infrastructure component mock."""
        return self.infrastructure.get(infrastructure_name)
    
    def simulate_failure(self, component_name: str, failure_rate: float = 1.0):
        """Simulate failures in a component."""
        if component_name in self.external_services:
            component = self.external_services[component_name]
            if hasattr(component, 'set_failure_rate'):
                component.set_failure_rate(failure_rate)
        elif component_name in self.infrastructure:
            component = self.infrastructure[component_name]
            if hasattr(component, 'set_failure_rate'):
                component.set_failure_rate(failure_rate)
    
    def enable_monitoring(self):
        """Enable monitoring for infrastructure components."""
        if 'resource_monitor' in self.infrastructure:
            self.infrastructure['resource_monitor'].start_monitoring(interval_seconds=0.1)
        if 'gpu_manager' in self.infrastructure:
            self.infrastructure['gpu_manager'].start_monitoring(interval_seconds=0.1)


# Global mock environment instance
_mock_environment = None


def create_mock_environment() -> MockEnvironment:
    """Create a new mock environment."""
    return MockEnvironment()


def setup_external_mocks(environment: Optional[MockEnvironment] = None) -> MockEnvironment:
    """Set up external service mocks."""
    if environment is None:
        environment = create_mock_environment()
    
    environment.setup()
    return environment


def teardown_external_mocks(environment: MockEnvironment):
    """Tear down external service mocks."""
    environment.cleanup()


@contextlib.contextmanager
def mock_dependencies_context(**kwargs) -> ContextManager[MockEnvironment]:
    """Context manager for mock dependencies."""
    environment = create_mock_environment()
    
    try:
        environment.setup()
        
        # Apply any custom configurations
        for key, value in kwargs.items():
            if hasattr(environment, key):
                setattr(environment, key, value)
        
        yield environment
        
    finally:
        environment.cleanup()


@contextlib.contextmanager
def isolated_test_environment(temp_dir: bool = True, 
                            mock_network: bool = True,
                            mock_gpu: bool = True,
                            mock_external_apis: bool = True) -> ContextManager[Dict[str, Any]]:
    """Create isolated test environment with configurable mocking."""
    
    test_env = {
        'temp_dir': None,
        'mocks': {},
        'patches': []
    }
    
    try:
        # Create temporary directory if requested
        if temp_dir:
            test_env['temp_dir'] = Path(tempfile.mkdtemp(prefix="isolated_test_"))
        
        # Set up network mocking
        if mock_network:
            network_mock = MockNetworkService()
            test_env['mocks']['network'] = network_mock
            
            network_patches = [
                patch('requests.get', side_effect=network_mock.get),
                patch('requests.post', side_effect=network_mock.post),
                patch('urllib.request.urlopen', return_value=Mock())
            ]
            
            for patcher in network_patches:
                patcher.start()
                test_env['patches'].append(patcher)
        
        # Set up GPU mocking
        if mock_gpu:
            gpu_mock = MockGPUManager(num_gpus=1)
            test_env['mocks']['gpu'] = gpu_mock
            
            gpu_patches = [
                patch('torch.cuda.is_available', return_value=True),
                patch('torch.cuda.device_count', return_value=1),
                patch('torch.cuda.current_device', return_value=0)
            ]
            
            for patcher in gpu_patches:
                patcher.start()
                test_env['patches'].append(patcher)
        
        # Set up external API mocking
        if mock_external_apis:
            api_mocks = {
                'huggingface': MockHuggingFaceAPI(),
                'openai': MockOpenAIAPI()
            }
            test_env['mocks'].update(api_mocks)
            
            api_patches = [
                patch('transformers.AutoModel.from_pretrained',
                      side_effect=api_mocks['huggingface'].from_pretrained),
                patch('openai.ChatCompletion.create',
                      side_effect=api_mocks['openai'].chat_completions_create)
            ]
            
            for patcher in api_patches:
                patcher.start()
                test_env['patches'].append(patcher)
        
        yield test_env
        
    finally:
        # Clean up patches
        for patcher in test_env['patches']:
            try:
                patcher.stop()
            except Exception:
                pass
        
        # Clean up temporary directory
        if test_env['temp_dir'] and test_env['temp_dir'].exists():
            try:
                shutil.rmtree(test_env['temp_dir'])
            except Exception:
                pass
        
        # Clean up mocks with monitoring
        for mock_obj in test_env['mocks'].values():
            if hasattr(mock_obj, 'stop_monitoring'):
                try:
                    mock_obj.stop_monitoring()
                except Exception:
                    pass


# Pytest fixtures

@pytest.fixture(scope="session")
def global_mock_environment():
    """Session-scoped mock environment."""
    global _mock_environment
    if _mock_environment is None:
        _mock_environment = create_mock_environment()
        _mock_environment.setup()
    
    yield _mock_environment
    
    # Cleanup is handled by pytest session teardown


@pytest.fixture
def mock_environment():
    """Function-scoped mock environment."""
    environment = create_mock_environment()
    environment.setup()
    
    yield environment
    
    environment.cleanup()


@pytest.fixture
def mock_huggingface():
    """Mock HuggingFace API."""
    mock_api = MockHuggingFaceAPI()
    mock_api.disable_network_delay()
    return mock_api


@pytest.fixture
def mock_openai():
    """Mock OpenAI API."""
    return MockOpenAIAPI()


@pytest.fixture
def mock_database():
    """Mock database connection."""
    db = MockDatabaseConnection()
    db.connect()
    
    yield db
    
    db.disconnect()


@pytest.fixture
def mock_filesystem():
    """Mock file system."""
    fs = MockFileSystem()
    temp_workspace = fs.create_temp_workspace()
    
    yield fs
    
    fs.cleanup_temp_workspace()


@pytest.fixture
def mock_gpu_manager():
    """Mock GPU manager."""
    gpu_manager = MockGPUManager(num_gpus=2)
    gpu_manager.start_monitoring(interval_seconds=0.1)
    
    yield gpu_manager
    
    gpu_manager.stop_monitoring()


@pytest.fixture
def mock_resource_monitor():
    """Mock resource monitor."""
    monitor = MockResourceMonitor()
    monitor.start_monitoring(interval_seconds=0.1)
    
    yield monitor
    
    monitor.stop_monitoring()


@pytest.fixture
def mock_cache():
    """Mock cache service."""
    return MockCacheService()


@pytest.fixture
def mock_secret_manager():
    """Mock secret manager."""
    return MockSecretManager()


@pytest.fixture
def mock_email_service():
    """Mock email service."""
    return MockEmailService()


@pytest.fixture
def mock_slack_notifier():
    """Mock Slack notifier."""
    return MockSlackNotifier()


@pytest.fixture
def mock_prometheus():
    """Mock Prometheus collector."""
    return MockPrometheusCollector()


@pytest.fixture
def mock_model_components():
    """Mock model components."""
    tokenizer = MockTokenizer()
    model = MockTransformerModel('test-model')
    dataset = MockTrainingDataset('test-dataset', tokenizer=tokenizer)
    
    return {
        'tokenizer': tokenizer,
        'model': model,
        'dataset': dataset
    }


@pytest.fixture
def mock_training_setup(mock_model_components):
    """Complete mock training setup."""
    tokenizer = mock_model_components['tokenizer']
    model = mock_model_components['model']
    dataset = mock_model_components['dataset']
    
    # Create data loader
    dataloader = MockDataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create optimizer and scheduler
    optimizer = MockOptimizer(model.parameters(), lr=2e-4)
    scheduler = MockScheduler(optimizer)
    
    # Create trainer
    trainer = MockTrainer(
        model=model,
        train_dataloader=dataloader,
        eval_dataloader=dataloader,  # Using same for simplicity
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'dataset': dataset,
        'dataloader': dataloader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'trainer': trainer
    }


@pytest.fixture
def mock_inference_setup(mock_model_components):
    """Mock inference setup."""
    model = mock_model_components['model']
    tokenizer = mock_model_components['tokenizer']
    
    predictor = MockPredictor(model, tokenizer)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'predictor': predictor
    }


@pytest.fixture
def temp_workspace():
    """Temporary workspace for testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_workspace_"))
    
    # Create common subdirectories
    (temp_dir / "data").mkdir()
    (temp_dir / "models").mkdir()
    (temp_dir / "outputs").mkdir()
    (temp_dir / "logs").mkdir()
    
    yield temp_dir
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return [
        {"text": "This is a positive example", "label": 1},
        {"text": "This is a negative example", "label": 0},
        {"text": "Another positive case", "label": 1},
        {"text": "Another negative case", "label": 0},
        {"text": "Neutral example", "label": 2}
    ]


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "name": "test-model",
            "architecture": "transformer",
            "vocab_size": 50000,
            "hidden_size": 768
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 2e-4,
            "epochs": 3,
            "max_length": 512
        },
        "evaluation": {
            "batch_size": 8,
            "metrics": ["accuracy", "f1", "precision", "recall"]
        },
        "paths": {
            "data_dir": "data",
            "model_dir": "models",
            "output_dir": "outputs"
        }
    }


# Utility functions for test setup

def setup_mock_training_environment() -> Dict[str, Any]:
    """Set up complete mock training environment."""
    with mock_dependencies_context() as env:
        # Get components
        tokenizer = env.get_model_component('tokenizer')
        model = env.get_model_component('model')
        dataset = env.get_model_component('dataset')
        
        # Set up training components
        dataloader = MockDataLoader(dataset, batch_size=4)
        optimizer = MockOptimizer(model.parameters(), lr=2e-4)
        scheduler = MockScheduler(optimizer)
        trainer = MockTrainer(model, dataloader, optimizer=optimizer, scheduler=scheduler)
        
        return {
            'environment': env,
            'model': model,
            'tokenizer': tokenizer,
            'dataset': dataset,
            'dataloader': dataloader,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'trainer': trainer
        }


def setup_mock_inference_environment() -> Dict[str, Any]:
    """Set up complete mock inference environment."""
    with mock_dependencies_context() as env:
        model = env.get_model_component('model')
        tokenizer = env.get_model_component('tokenizer')
        predictor = MockPredictor(model, tokenizer)
        
        return {
            'environment': env,
            'model': model,
            'tokenizer': tokenizer,
            'predictor': predictor
        }


def create_mock_training_data(size: int = 100, tokenizer: Optional[MockTokenizer] = None) -> MockTrainingDataset:
    """Create mock training data."""
    if tokenizer is None:
        tokenizer = MockTokenizer()
    
    return MockTrainingDataset(f"mock_dataset_{size}", size=size, tokenizer=tokenizer)


def configure_mock_failures(environment: MockEnvironment, failure_configs: Dict[str, float]):
    """Configure failure rates for mock components."""
    for component_name, failure_rate in failure_configs.items():
        environment.simulate_failure(component_name, failure_rate)


def wait_for_mock_operations(environment: MockEnvironment, timeout_seconds: int = 10):
    """Wait for mock operations to complete."""
    import time
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        # Check if any async operations are running
        all_idle = True
        
        # Check infrastructure monitoring
        for component in environment.infrastructure.values():
            if hasattr(component, 'monitoring') and component.monitoring:
                # Give some time for operations to complete
                time.sleep(0.1)
                all_idle = False
                break
        
        if all_idle:
            break
        
        time.sleep(0.1)


# Context managers for specific mock scenarios

@contextlib.contextmanager
def mock_training_failure_scenario():
    """Context manager for training failure scenarios."""
    with mock_dependencies_context() as env:
        # Configure various failure scenarios
        env.simulate_failure('huggingface', 0.3)  # 30% API failure rate
        env.simulate_failure('openai', 0.2)       # 20% API failure rate
        
        # Simulate GPU issues
        gpu_manager = env.get_infrastructure('gpu_manager')
        if gpu_manager:
            gpu_manager.set_gpu_availability(1, False)  # Disable one GPU
        
        yield env


@contextlib.contextmanager
def mock_network_isolation():
    """Context manager for network isolation testing."""
    with mock_dependencies_context() as env:
        # Set high failure rates for network operations
        env.simulate_failure('huggingface', 1.0)
        env.simulate_failure('openai', 1.0)
        env.simulate_failure('network', 1.0)
        
        yield env


@contextlib.contextmanager
def mock_resource_constrained_environment():
    """Context manager for resource-constrained testing."""
    with mock_dependencies_context() as env:
        # Simulate resource constraints
        gpu_manager = env.get_infrastructure('gpu_manager')
        if gpu_manager:
            # Reduce available GPU memory
            for gpu in gpu_manager.gpus.values():
                gpu['memory_total'] = 2048  # Reduce to 2GB
                gpu['memory_used'] = 1536   # 75% used
                gpu['memory_free'] = 512    # Only 512MB free
        
        resource_monitor = env.get_infrastructure('resource_monitor')
        if resource_monitor:
            # Set stricter thresholds
            resource_monitor.set_thresholds({
                'cpu_percent': 50,
                'memory_percent': 60,
                'disk_percent': 70
            })
        
        yield env