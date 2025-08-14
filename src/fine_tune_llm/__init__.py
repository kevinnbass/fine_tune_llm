"""
Fine-tune LLM: Advanced LoRA Fine-tuning Library

A comprehensive library for fine-tuning large language models with LoRA,
featuring calibration-aware training, conformal prediction, risk control,
and comprehensive evaluation metrics.
"""

__version__ = "2.0.0"
__author__ = "Fine-tune LLM Team"
__email__ = "support@finetunellm.ai"

# Core imports - Public API
from .config import ConfigManager, ValidationError
from .models import ModelFactory, ModelManager
from .training import TrainerFactory, CalibratedTrainer
from .inference import InferenceEngine, ConformalPredictor
from .evaluation import MetricsEngine, HighStakesAuditor
from .data import DataProcessor, DataValidator
from .monitoring import Dashboard, MetricsCollector
from .services import (
    ModelService,
    TrainingService,
    InferenceService,
    ConfigService,
    MonitoringService
)

# Exception classes
from .core.exceptions import (
    FineTuneLLMError,
    ConfigurationError,
    ModelError,
    TrainingError,
    InferenceError,
    DataError,
    IntegrationError,
    SystemError
)

# Main factory for convenience
from .core.factory import ComponentFactory

# Public API
__all__ = [
    # Core classes
    "ConfigManager",
    "ModelFactory",
    "ModelManager", 
    "TrainerFactory",
    "CalibratedTrainer",
    "InferenceEngine",
    "ConformalPredictor",
    "MetricsEngine",
    "HighStakesAuditor",
    "DataProcessor",
    "DataValidator",
    "Dashboard",
    "MetricsCollector",
    
    # Services
    "ModelService",
    "TrainingService", 
    "InferenceService",
    "ConfigService",
    "MonitoringService",
    
    # Exceptions
    "FineTuneLLMError",
    "ConfigurationError",
    "ModelError",
    "TrainingError",
    "InferenceError", 
    "DataError",
    "IntegrationError",
    "SystemError",
    "ValidationError",
    
    # Factory
    "ComponentFactory",
    
    # Package info
    "__version__",
    "__author__",
    "__email__"
]

# Package configuration
import logging

# Set up default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version info
def get_version():
    """Get package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "fine-tune-llm",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Advanced LoRA Fine-tuning Library with Calibration and Risk Control"
    }