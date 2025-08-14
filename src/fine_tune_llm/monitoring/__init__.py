"""
Monitoring package for training and model monitoring.

This package provides comprehensive monitoring capabilities including
metrics collection, dashboards, alerting, and visualization.
"""

from .collectors import MetricsCollector, TrainingMetrics
from .dashboards import DashboardRenderer

__all__ = [
    'MetricsCollector',
    'TrainingMetrics', 
    'DashboardRenderer',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Comprehensive monitoring framework for LLM training'