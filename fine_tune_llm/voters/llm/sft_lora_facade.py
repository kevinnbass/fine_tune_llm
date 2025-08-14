"""
Facade pattern for backward compatibility with original sft_lora.py.

This module provides a compatibility layer that maintains the original API
while delegating to the new decomposed components.
"""

import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Import new decomposed components
from src.fine_tune_llm.training import (
    CalibratedTrainer as NewCalibratedTrainer,
    CalibrationMonitorCallback as NewCalibrationMonitorCallback
)
from src.fine_tune_llm.training.trainers.lora_sft import EnhancedLoRASFTTrainer as NewEnhancedLoRASFTTrainer

# Deprecation warning
warnings.warn(
    "sft_lora.py has been decomposed into multiple modules. "
    "Please update your imports to use the new modular components from "
    "src.fine_tune_llm.training/. This facade will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)


class CalibratedTrainer:
    """
    Backward compatibility facade for CalibratedTrainer.
    
    This class maintains the original API while delegating to new components.
    """
    
    def __init__(self, 
                 metrics_aggregator=None, 
                 conformal_predictor=None,
                 calibration_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize with backward compatibility."""
        self._trainer = NewCalibratedTrainer(
            metrics_aggregator=metrics_aggregator,
            conformal_predictor=conformal_predictor,
            calibration_config=calibration_config,
            **kwargs
        )
        
        # Expose original attributes
        self.metrics_aggregator = self._trainer.metrics_aggregator
        self.conformal_predictor = self._trainer.conformal_predictor
        self.calibration_config = self._trainer.calibration_config
        
        # Calibration monitoring
        self.ece_threshold = self._trainer.ece_threshold
        self.mce_threshold = self._trainer.mce_threshold
        self.adjustment_threshold = self._trainer.adjustment_threshold
        self.lr_reduction_factor = self._trainer.lr_reduction_factor
        
        # Abstention configuration
        self.abstention_config = self._trainer.abstention_config
        self.enable_abstention = self._trainer.enable_abstention
        self.confidence_threshold = self._trainer.confidence_threshold
        self.abstention_penalty = self._trainer.abstention_penalty
        self.uncertainty_weight = self._trainer.uncertainty_weight
        
        # Advanced features
        self.enable_advanced_metrics = self._trainer.enable_advanced_metrics
        self.conformal_prediction = self._trainer.conformal_prediction
        self.risk_controlled_training = self._trainer.risk_controlled_training
        
        # History tracking
        self.ece_history = self._trainer.ece_history
        self.mce_history = self._trainer.mce_history
        self.last_adjustment_step = self._trainer.last_adjustment_step
    
    # Delegate all methods to new trainer
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate with calibration metrics (original API)."""
        return self._trainer.evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def should_adjust_learning_rate(self):
        """Check if learning rate should be adjusted (original API)."""
        return self._trainer.should_adjust_learning_rate()
    
    def adjust_learning_rate(self, factor=None):
        """Adjust learning rate (original API)."""
        return self._trainer.adjust_learning_rate(factor)
    
    def compute_abstention_aware_loss(self, model, inputs, return_outputs=False):
        """Compute abstention-aware loss (original API)."""
        return self._trainer.compute_abstention_aware_loss(model, inputs, return_outputs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss (original API)."""
        return self._trainer.compute_loss(model, inputs, return_outputs)
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging (original API)."""
        return self._trainer.log(logs)
    
    # Private methods
    def _get_predictions_and_labels(self, dataloader):
        """Get predictions and labels (original API)."""
        return self._trainer._get_predictions_and_labels(dataloader)
    
    def _compute_calibration_metrics(self, predictions, labels, n_bins=10):
        """Compute calibration metrics (original API)."""
        return self._trainer._compute_calibration_metrics(predictions, labels, n_bins)
    
    def _calibrate_conformal_predictor(self, predictions, labels):
        """Calibrate conformal predictor (original API)."""
        return self._trainer._calibrate_conformal_predictor(predictions, labels)
    
    # Forward any other attributes
    def __getattr__(self, name):
        """Forward undefined attributes to new trainer."""
        return getattr(self._trainer, name)


class CalibrationMonitorCallback:
    """
    Backward compatibility facade for CalibrationMonitorCallback.
    
    This class maintains the original API while delegating to new components.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 adjustment_threshold: float = 0.05,
                 lr_reduction_factor: float = 0.8,
                 patience: int = 3):
        """Initialize with backward compatibility."""
        self._callback = NewCalibrationMonitorCallback(
            config=config,
            adjustment_threshold=adjustment_threshold,
            lr_reduction_factor=lr_reduction_factor,
            patience=patience
        )
        
        # Expose original attributes
        self.adjustment_threshold = self._callback.adjustment_threshold
        self.lr_reduction_factor = self._callback.lr_reduction_factor
        self.patience = self._callback.patience
        self.ece_history = self._callback.ece_history
        self.bad_calibration_count = self._callback.bad_calibration_count
        self.last_adjustment_step = self._callback.last_adjustment_step
    
    def on_evaluate(self, args, state, control, trainer=None, **kwargs):
        """Called during evaluation (original API)."""
        return self._callback.on_evaluate(args, state, control, trainer, **kwargs)
    
    def on_log(self, args, state, control, trainer=None, **kwargs):
        """Add calibration information to logs (original API)."""
        return self._callback.on_log(args, state, control, trainer, **kwargs)
    
    def _adjust_learning_rate(self, trainer, state):
        """Adjust learning rate (original API)."""
        return self._callback._adjust_learning_rate(trainer, state)
    
    # Forward any other attributes
    def __getattr__(self, name):
        """Forward undefined attributes to new callback."""
        return getattr(self._callback, name)


class EnhancedLoRASFTTrainer:
    """
    Backward compatibility facade for EnhancedLoRASFTTrainer.
    
    This class maintains the original API while delegating to new components.
    """
    
    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        """Initialize with backward compatibility."""
        self._trainer = NewEnhancedLoRASFTTrainer(config_path)
        
        # Expose original attributes for backward compatibility
        self.config_path = self._trainer.config_path
        self.config = self._trainer.config
        self.model_config = self._trainer.model_config
        self.model_id = self._trainer.model_id
        self.tokenizer_id = self._trainer.tokenizer_id
        self.lora_config = self._trainer.lora_config
        self.training_config = self._trainer.training_config
        self.calibration_config = self._trainer.calibration_config
        self.high_stakes_config = self._trainer.high_stakes_config
        self.conformal_config = self._trainer.conformal_config
        
        # Components
        self.model_manager = self._trainer.model_manager
        self.high_stakes_auditor = self._trainer.high_stakes_auditor
        self.metrics_aggregator = self._trainer.metrics_aggregator
        self.conformal_predictor = self._trainer.conformal_predictor
        
        # Training state
        self.model = self._trainer.model
        self.tokenizer = self._trainer.tokenizer
        self.train_dataset = self._trainer.train_dataset
        self.eval_dataset = self._trainer.eval_dataset
    
    def get_quantization_config(self):
        """Get quantization configuration (original API)."""
        return self._trainer.get_quantization_config()
    
    def get_peft_config(self):
        """Get PEFT configuration (original API)."""
        return self._trainer.get_peft_config()
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer (original API)."""
        result = self._trainer.setup_model_and_tokenizer()
        
        # Update exposed attributes
        self.model = self._trainer.model
        self.tokenizer = self._trainer.tokenizer
        
        return result
    
    def get_model_specific_prompt(self, text: str, metadata: Dict = None) -> str:
        """Get model-specific prompt (original API)."""
        return self._trainer.get_model_specific_prompt(text, metadata)
    
    def prepare_dataset(self, dataset):
        """Prepare dataset (original API)."""
        return self._trainer.prepare_dataset(dataset)
    
    def train(self):
        """Execute training (original API)."""
        return self._trainer.train()
    
    def set_datasets(self, train_dataset, eval_dataset=None):
        """Set datasets (original API)."""
        result = self._trainer.set_datasets(train_dataset, eval_dataset)
        
        # Update exposed attributes
        self.train_dataset = self._trainer.train_dataset
        self.eval_dataset = self._trainer.eval_dataset
        
        return result
    
    def load_model(self, model_path):
        """Load model (original API)."""
        result = self._trainer.load_model(model_path)
        
        # Update exposed attributes
        self.model = self._trainer.model
        self.tokenizer = self._trainer.tokenizer
        
        return result
    
    def get_training_info(self):
        """Get training info (original API)."""
        return self._trainer.get_training_info()
    
    def save_model(self, save_path):
        """Save model (original API)."""
        return self._trainer.save_model(save_path)
    
    # Private methods for backward compatibility
    def _load_config(self):
        """Load configuration (original API)."""
        return self._trainer._load_config()
    
    def _get_default_config(self):
        """Get default config (original API)."""
        return self._trainer._get_default_config()
    
    def _initialize_high_stakes_components(self):
        """Initialize high-stakes components (original API)."""
        return self._trainer._initialize_high_stakes_components()
    
    def _add_training_callbacks(self, trainer):
        """Add training callbacks (original API)."""
        return self._trainer._add_training_callbacks(trainer)
    
    # Forward any other attributes
    def __getattr__(self, name):
        """Forward undefined attributes to new trainer."""
        return getattr(self._trainer, name)


# Export original class names for full backward compatibility
__all__ = [
    'CalibratedTrainer',
    'CalibrationMonitorCallback', 
    'EnhancedLoRASFTTrainer'
]