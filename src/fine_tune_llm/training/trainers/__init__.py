"""
Trainer modules for model training.

This package provides various trainer implementations including
base trainers, calibrated trainers, and specialized training strategies.
"""

from .base import BaseTrainer
from .calibrated import CalibratedTrainer
from .lora_sft import EnhancedLoRASFTTrainer

__all__ = [
    'BaseTrainer',
    'CalibratedTrainer',
    'EnhancedLoRASFTTrainer',
]

# Version information
__version__ = '2.0.0'

# Factory function for creating trainers
def create_trainer(trainer_type: str = 'lora_sft', **kwargs) -> BaseTrainer:
    """
    Factory function for creating trainers.
    
    Args:
        trainer_type: Type of trainer to create
        **kwargs: Arguments to pass to trainer
        
    Returns:
        Trainer instance
    """
    if trainer_type == 'lora_sft':
        return EnhancedLoRASFTTrainer(**kwargs)
    elif trainer_type == 'calibrated':
        return CalibratedTrainer(**kwargs)
    elif trainer_type == 'base':
        return BaseTrainer(**kwargs)  # Note: This is abstract, would need concrete implementation
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")