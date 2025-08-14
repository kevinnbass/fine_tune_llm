"""
Training package for model training and fine-tuning.

This package provides comprehensive training capabilities including
trainers, callbacks, strategies, and loss functions.
"""

from .trainers import BaseTrainer, CalibratedTrainer, EnhancedLoRASFTTrainer, create_trainer
from .callbacks import (
    BaseTrainingCallback,
    CalibrationMonitorCallback,
    MetricsAggregatorCallback, 
    EarlyStoppingCallback,
    ProgressCallback,
    ResourceMonitorCallback,
    create_callback
)

__all__ = [
    # Trainers
    'BaseTrainer',
    'CalibratedTrainer',
    'EnhancedLoRASFTTrainer',
    'create_trainer',
    
    # Callbacks
    'BaseTrainingCallback',
    'CalibrationMonitorCallback',
    'MetricsAggregatorCallback',
    'EarlyStoppingCallback', 
    'ProgressCallback',
    'ResourceMonitorCallback',
    'create_callback',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Comprehensive training framework for LLM fine-tuning'