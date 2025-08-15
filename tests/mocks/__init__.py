"""
Mock Framework for External Dependencies.

This module provides comprehensive mocking for all external dependencies
including HuggingFace models, OpenAI APIs, databases, file systems, and more.
"""

from .external_services import (
    MockHuggingFaceAPI,
    MockOpenAIAPI,
    MockDatabaseConnection,
    MockFileSystem,
    MockNetworkService,
    MockCacheService,
    MockMLFlowTracker,
    MockWandBTracker,
    MockS3Service,
    MockGCSService,
    MockDockerService,
    MockKubernetesService
)

from .model_mocks import (
    MockTransformerModel,
    MockTokenizer,
    MockTrainingDataset,
    MockDataLoader,
    MockOptimizer,
    MockScheduler,
    MockTrainer,
    MockPredictor
)

from .infrastructure_mocks import (
    MockGPUManager,
    MockResourceMonitor,
    MockConfigValidator,
    MockSecretManager,
    MockEmailService,
    MockSlackNotifier,
    MockPrometheusCollector,
    MockGrafanaClient
)

from .test_fixtures import (
    create_mock_environment,
    setup_external_mocks,
    teardown_external_mocks,
    mock_dependencies_context,
    isolated_test_environment
)

__all__ = [
    # External Services
    'MockHuggingFaceAPI',
    'MockOpenAIAPI',
    'MockDatabaseConnection',
    'MockFileSystem',
    'MockNetworkService',
    'MockCacheService',
    'MockMLFlowTracker',
    'MockWandBTracker',
    'MockS3Service',
    'MockGCSService',
    'MockDockerService',
    'MockKubernetesService',
    
    # Model Mocks
    'MockTransformerModel',
    'MockTokenizer',
    'MockTrainingDataset',
    'MockDataLoader',
    'MockOptimizer',
    'MockScheduler',
    'MockTrainer',
    'MockPredictor',
    
    # Infrastructure Mocks
    'MockGPUManager',
    'MockResourceMonitor',
    'MockConfigValidator',
    'MockSecretManager',
    'MockEmailService',
    'MockSlackNotifier',
    'MockPrometheusCollector',
    'MockGrafanaClient',
    
    # Test Fixtures
    'create_mock_environment',
    'setup_external_mocks',
    'teardown_external_mocks',
    'mock_dependencies_context',
    'isolated_test_environment'
]