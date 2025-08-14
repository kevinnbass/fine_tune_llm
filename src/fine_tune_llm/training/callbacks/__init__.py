"""
Training callbacks for monitoring and control.

This package provides comprehensive callback implementations for
training monitoring, calibration adjustment, and advanced metrics.
"""

from .monitoring import (
    BaseTrainingCallback,
    CalibrationMonitorCallback,
    MetricsAggregatorCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    ResourceMonitorCallback,
    create_callback
)

__all__ = [
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