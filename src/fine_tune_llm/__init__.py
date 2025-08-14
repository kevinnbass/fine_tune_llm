"""
Fine-tune LLM: Advanced LoRA Fine-tuning Library

A comprehensive library for fine-tuning large language models with LoRA,
featuring calibration-aware training, conformal prediction, risk control,
and comprehensive evaluation metrics.
"""

from ._version import __version__

__author__ = "Fine-tune LLM Team"
__email__ = "support@finetunellm.ai"

# Core imports - Public API (lazy loading for better performance)
def __getattr__(name: str):
    """Lazy import for public API components."""
    if name == "ConfigManager":
        from .config import ConfigManager
        return ConfigManager
    elif name == "ValidationError":
        from .config import ValidationError
        return ValidationError
    elif name == "ModelFactory":
        from .models import ModelFactory
        return ModelFactory
    elif name == "ModelManager":
        from .models import ModelManager
        return ModelManager
    elif name == "TrainerFactory":
        from .training import TrainerFactory
        return TrainerFactory
    elif name == "CalibratedTrainer":
        from .training import CalibratedTrainer
        return CalibratedTrainer
    elif name == "InferenceEngine":
        from .inference import InferenceEngine
        return InferenceEngine
    elif name == "ConformalPredictor":
        from .inference import ConformalPredictor
        return ConformalPredictor
    elif name == "MetricsEngine":
        from .evaluation import MetricsEngine
        return MetricsEngine
    elif name == "HighStakesAuditor":
        from .evaluation import HighStakesAuditor
        return HighStakesAuditor
    elif name == "DataProcessor":
        from .data import DataProcessor
        return DataProcessor
    elif name == "DataValidator":
        from .data import DataValidator
        return DataValidator
    elif name == "Dashboard":
        from .monitoring import Dashboard
        return Dashboard
    elif name == "MetricsCollector":
        from .monitoring import MetricsCollector
        return MetricsCollector
    elif name == "ModelService":
        from .services import ModelService
        return ModelService
    elif name == "TrainingService":
        from .services import TrainingService
        return TrainingService
    elif name == "InferenceService":
        from .services import InferenceService
        return InferenceService
    elif name == "ConfigService":
        from .services import ConfigService
        return ConfigService
    elif name == "MonitoringService":
        from .services import MonitoringService
        return MonitoringService
    elif name == "ComponentFactory":
        from .core.factory import ComponentFactory
        return ComponentFactory
    else:
        # Try exception imports
        try:
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
            exceptions_map = {
                "FineTuneLLMError": FineTuneLLMError,
                "ConfigurationError": ConfigurationError,
                "ModelError": ModelError,
                "TrainingError": TrainingError,
                "InferenceError": InferenceError,
                "DataError": DataError,
                "IntegrationError": IntegrationError,
                "SystemError": SystemError
            }
            if name in exceptions_map:
                return exceptions_map[name]
        except ImportError:
            pass
        
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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