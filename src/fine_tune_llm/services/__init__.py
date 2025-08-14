"""
Services layer for business logic and orchestration.

This package provides high-level service classes that orchestrate
the various components of the fine-tuning platform.
"""

from .base import BaseService
from .model_service import ModelService
from .training_service import TrainingService
from .inference_service import InferenceService
from .data_service import DataService
from .evaluation_service import EvaluationService

__all__ = [
    'BaseService',
    'ModelService',
    'TrainingService', 
    'InferenceService',
    'DataService',
    'EvaluationService',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Service layer for LLM fine-tuning platform'