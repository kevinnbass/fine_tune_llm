"""
Base trainer interface for model training.

This module provides the abstract base class for all trainers with
common training functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from ...core.interfaces import BaseComponent


class BaseTrainer(BaseComponent, ABC):
    """Abstract base class for model trainers."""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.training_config = self.config.get('training', {})
        
        # Training components
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.training_args = None
        
        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.current_step = 0
        
        # Callbacks and metrics
        self.callbacks = []
        self.metrics_aggregator = None
        
    @abstractmethod
    def setup_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Setup model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        pass
    
    @abstractmethod
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Processed dataset
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute training process.
        
        Returns:
            Training results and metrics
        """
        pass
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.training_config = self.config.get('training', {})
    
    def cleanup(self) -> None:
        """Clean up training resources."""
        if self.model:
            del self.model
            torch.cuda.empty_cache()
    
    @property
    def name(self) -> str:
        """Component name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def add_callback(self, callback):
        """Add training callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback):
        """Remove training callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def set_metrics_aggregator(self, aggregator):
        """Set metrics aggregator."""
        self.metrics_aggregator = aggregator
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get training arguments from configuration."""
        training_args = TrainingArguments(
            output_dir=self.training_config.get('output_dir', 'artifacts/models/training'),
            num_train_epochs=self.training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=self.training_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=self.training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 1),
            learning_rate=self.training_config.get('learning_rate', 2e-4),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            warmup_ratio=self.training_config.get('warmup_ratio', 0.03),
            lr_scheduler_type=self.training_config.get('lr_scheduler_type', 'cosine'),
            logging_steps=self.training_config.get('logging_steps', 10),
            evaluation_strategy=self.training_config.get('evaluation_strategy', 'steps'),
            eval_steps=self.training_config.get('eval_steps', 100),
            save_strategy=self.training_config.get('save_strategy', 'steps'),
            save_steps=self.training_config.get('save_steps', 500),
            load_best_model_at_end=self.training_config.get('load_best_model_at_end', True),
            metric_for_best_model=self.training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=self.training_config.get('greater_is_better', False),
            report_to=self.training_config.get('report_to', []),
            run_name=self.training_config.get('run_name', None),
            seed=self.training_config.get('seed', 42),
            data_seed=self.training_config.get('data_seed', 42),
            bf16=self.training_config.get('bf16', True),
            fp16=self.training_config.get('fp16', False),
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 0),
            remove_unused_columns=self.training_config.get('remove_unused_columns', False),
        )
        
        return training_args
    
    def validate_configuration(self) -> bool:
        """Validate training configuration."""
        required_fields = ['output_dir', 'num_train_epochs', 'learning_rate']
        
        for field in required_fields:
            if field not in self.training_config:
                raise ValueError(f"Missing required training config field: {field}")
        
        # Validate values
        if self.training_config.get('learning_rate', 0) <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training_config.get('num_train_epochs', 0) <= 0:
            raise ValueError("Number of epochs must be positive")
        
        return True
    
    def save_model(self, save_path: Union[str, Path]):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, model_path: Union[str, Path]):
        """Load trained model."""
        # Implementation depends on specific trainer type
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        info = {
            'model_type': type(self.model).__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(next(self.model.parameters()).device) if self.model.parameters() else 'unknown'
        }
        
        # Add memory usage if on GPU
        if torch.cuda.is_available():
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
        
        return info
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training metrics."""
        if self.metrics_aggregator:
            self.metrics_aggregator.log_metrics(metrics, step)
        
        # Also log to callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_log'):
                callback.on_log(metrics, step)
    
    def should_stop_training(self) -> bool:
        """Check if training should stop early."""
        # Check callbacks for early stopping signals
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop():
                return True
        
        return False