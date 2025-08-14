"""
Metrics collectors for monitoring training processes.

This package provides comprehensive metrics collection capabilities
including automatic monitoring, file watching, and real-time updates.
"""

from .metrics_collector import MetricsCollector, TrainingMetrics

__all__ = [
    'MetricsCollector',
    'TrainingMetrics',
]

# Version information
__version__ = '2.0.0'