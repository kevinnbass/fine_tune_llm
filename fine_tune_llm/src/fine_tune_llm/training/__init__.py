"""
Training System

Advanced training pipelines with calibration-aware training,
abstention-aware loss functions, and real-time monitoring.
"""

from .factory import TrainerFactory
from .calibrated import CalibratedTrainer
from .base import BaseTrainer
from .callbacks import CalibrationMonitorCallback, AdvancedMetricsCallback
from .loss import AbstentionAwareLoss, CalibrationLoss

__all__ = [
    "TrainerFactory",
    "CalibratedTrainer",
    "BaseTrainer",
    "CalibrationMonitorCallback",
    "AdvancedMetricsCallback",
    "AbstentionAwareLoss",
    "CalibrationLoss",
]