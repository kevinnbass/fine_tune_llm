"""
Training system for fine-tune LLM library.

Provides comprehensive training capabilities with LoRA, calibration-aware
training, conformal prediction, and advanced monitoring.
"""

# Public API - Core training components
from .factory import TrainerFactory

# Public API - Trainers (explicit imports for hierarchical structure)
from .trainers.base import BaseTrainer
from .trainers.calibrated import CalibratedTrainer
from .trainers.lora import LoRATrainer

# Public API - Training strategies
from .strategies.base import BaseStrategy
from .strategies.lora import LoRAStrategy

# Public API - Callbacks (explicit imports)
from .callbacks.base import BaseTrainingCallback
from .callbacks.calibration import CalibrationMonitorCallback
from .callbacks.metrics import MetricsAggregatorCallback
from .callbacks.early_stopping import EarlyStoppingCallback
from .callbacks.progress import ProgressCallback
from .callbacks.resource import ResourceMonitorCallback

# Public API - Loss functions
try:
    from .losses.base import BaseLoss
    from .losses.abstention import AbstentionLoss
    from .losses.calibration import CalibrationLoss
except ImportError:
    # Graceful fallback if loss modules not implemented yet
    BaseLoss = None
    AbstentionLoss = None
    CalibrationLoss = None

__all__ = [
    # Core training
    "TrainerFactory",
    "BaseTrainer",
    "CalibratedTrainer",
    "LoRATrainer",
    
    # Training strategies
    "BaseStrategy",
    "LoRAStrategy",
    
    # Callbacks
    "BaseTrainingCallback",
    "CalibrationMonitorCallback",
    "MetricsAggregatorCallback",
    "EarlyStoppingCallback", 
    "ProgressCallback",
    "ResourceMonitorCallback",
    
    # Loss functions (if available)
    "BaseLoss",
    "AbstentionLoss",
    "CalibrationLoss"
]

# Remove None values from exports
__all__ = [name for name in __all__ if globals().get(name) is not None]

# Private imports for internal use
try:
    from . import _internals
except ImportError:
    # Create placeholder if internals not implemented
    pass