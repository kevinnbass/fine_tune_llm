"""
Training callbacks for monitoring and control.

This module provides various callbacks for training monitoring,
calibration adjustment, and advanced metrics tracking.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class BaseTrainingCallback(TrainerCallback, BaseComponent):
    """Base class for training callbacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base callback."""
        super().__init__()
        self.config = config or {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    def name(self) -> str:
        """Component name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"


class CalibrationMonitorCallback(BaseTrainingCallback):
    """
    Callback for monitoring calibration metrics and adjusting training.
    
    This callback monitors Expected Calibration Error (ECE) and Maximum
    Calibration Error (MCE) during training and can trigger learning rate
    adjustments when calibration degrades.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 adjustment_threshold: float = 0.05,
                 lr_reduction_factor: float = 0.8,
                 patience: int = 3):
        """
        Initialize calibration monitor callback.
        
        Args:
            config: Configuration dictionary
            adjustment_threshold: ECE threshold for LR adjustment
            lr_reduction_factor: Factor to reduce LR by
            patience: Number of evaluations to wait before adjusting
        """
        super().__init__(config)
        
        self.adjustment_threshold = adjustment_threshold
        self.lr_reduction_factor = lr_reduction_factor
        self.patience = patience
        
        # Monitoring state
        self.ece_history = []
        self.bad_calibration_count = 0
        self.last_adjustment_step = 0
        
        logger.info(f"Initialized CalibrationMonitorCallback:")
        logger.info(f"  - Adjustment threshold: {adjustment_threshold}")
        logger.info(f"  - LR reduction factor: {lr_reduction_factor}")
        logger.info(f"  - Patience: {patience}")
    
    def on_evaluate(self, 
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    trainer=None,
                    **kwargs):
        """
        Called during evaluation to monitor calibration.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            trainer: Trainer instance
        """
        if trainer is None:
            return
        
        # Get evaluation metrics
        eval_metrics = kwargs.get('logs', {})
        
        # Check for ECE metric
        ece = eval_metrics.get('eval_ece', None)
        
        if ece is not None:
            self.ece_history.append(ece)
            
            # Check if calibration is poor
            if ece > self.adjustment_threshold:
                self.bad_calibration_count += 1
            else:
                self.bad_calibration_count = 0
            
            # Trigger adjustment if needed
            if (self.bad_calibration_count >= self.patience and 
                state.global_step - self.last_adjustment_step > 200):
                
                self._adjust_learning_rate(trainer, state)
            
            # Log calibration status
            logger.info(f"Step {state.global_step}: ECE = {ece:.4f}, "
                       f"bad_count = {self.bad_calibration_count}")
    
    def _adjust_learning_rate(self, trainer, state):
        """Adjust learning rate for better calibration."""
        if not hasattr(trainer, 'optimizer'):
            return
        
        # Get current learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        new_lr = current_lr * self.lr_reduction_factor
        
        # Update learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.last_adjustment_step = state.global_step
        self.bad_calibration_count = 0
        
        logger.info(f"Calibration adjustment: LR {current_lr:.2e} -> {new_lr:.2e} "
                   f"at step {state.global_step}")
    
    def on_log(self, 
               args: TrainingArguments,
               state: TrainerState, 
               control: TrainerControl,
               trainer=None,
               **kwargs):
        """Add calibration information to logs."""
        logs = kwargs.get('logs', {})
        
        if self.ece_history:
            logs['calibration_ece_latest'] = self.ece_history[-1]
            logs['calibration_bad_count'] = self.bad_calibration_count
            logs['calibration_history_len'] = len(self.ece_history)


class MetricsAggregatorCallback(BaseTrainingCallback):
    """Callback for aggregating and storing training metrics."""
    
    def __init__(self, 
                 metrics_aggregator,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics aggregator callback.
        
        Args:
            metrics_aggregator: Metrics aggregation component
            config: Configuration dictionary
        """
        super().__init__(config)
        self.metrics_aggregator = metrics_aggregator
        
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               trainer=None,
               **kwargs):
        """Aggregate metrics on each log event."""
        if self.metrics_aggregator is None:
            return
        
        logs = kwargs.get('logs', {})
        
        if logs:
            try:
                # Add step information
                enriched_logs = logs.copy()
                enriched_logs.update({
                    'global_step': state.global_step,
                    'epoch': state.epoch,
                    'learning_rate': self._get_current_lr(trainer)
                })
                
                # Send to aggregator
                self.metrics_aggregator.add_metrics(enriched_logs)
                
            except Exception as e:
                logger.error(f"Error aggregating metrics: {e}")
    
    def _get_current_lr(self, trainer):
        """Get current learning rate."""
        if trainer and hasattr(trainer, 'optimizer'):
            return trainer.optimizer.param_groups[0]['lr']
        return 0.0


class EarlyStoppingCallback(BaseTrainingCallback):
    """Enhanced early stopping callback."""
    
    def __init__(self,
                 early_stopping_patience: int = 3,
                 early_stopping_threshold: float = 0.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize early stopping callback.
        
        Args:
            early_stopping_patience: Number of evaluations to wait
            early_stopping_threshold: Minimum improvement threshold
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        # State tracking
        self.best_metric = None
        self.wait_count = 0
        self.should_stop = False
        
    def on_evaluate(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    trainer=None,
                    **kwargs):
        """Check for early stopping conditions."""
        eval_metrics = kwargs.get('logs', {})
        
        # Get the metric to monitor
        metric_key = f"eval_{args.metric_for_best_model}"
        current_metric = eval_metrics.get(metric_key)
        
        if current_metric is None:
            return
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = current_metric
            return
        
        # Check improvement direction
        improved = (
            (args.greater_is_better and current_metric > self.best_metric + self.early_stopping_threshold) or
            (not args.greater_is_better and current_metric < self.best_metric - self.early_stopping_threshold)
        )
        
        if improved:
            self.best_metric = current_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Check for early stopping
        if self.wait_count >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.wait_count} evaluations "
                       f"without improvement (best: {self.best_metric:.4f})")
            control.should_training_stop = True
            self.should_stop = True


class ProgressCallback(BaseTrainingCallback):
    """Callback for detailed progress reporting."""
    
    def __init__(self, 
                 report_interval: int = 100,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize progress callback.
        
        Args:
            report_interval: Steps between detailed reports
            config: Configuration dictionary
        """
        super().__init__(config)
        self.report_interval = report_interval
        self.start_time = None
        
    def on_train_begin(self,
                      args: TrainingArguments,
                      state: TrainerState,
                      control: TrainerControl,
                      trainer=None,
                      **kwargs):
        """Record training start time."""
        import time
        self.start_time = time.time()
        
        total_steps = state.max_steps
        logger.info(f"Training started: {total_steps} total steps planned")
    
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               trainer=None,
               **kwargs):
        """Provide detailed progress reports."""
        if state.global_step % self.report_interval != 0:
            return
        
        import time
        
        # Calculate progress
        progress_pct = (state.global_step / state.max_steps) * 100
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Estimate remaining time
        if state.global_step > 0:
            time_per_step = elapsed_time / state.global_step
            remaining_steps = state.max_steps - state.global_step
            eta = remaining_steps * time_per_step
        else:
            eta = 0
        
        # Get current metrics
        logs = kwargs.get('logs', {})
        loss = logs.get('train_loss', 'N/A')
        lr = logs.get('learning_rate', 'N/A')
        
        logger.info(f"Progress: {progress_pct:.1f}% | "
                   f"Step: {state.global_step}/{state.max_steps} | "
                   f"Loss: {loss} | LR: {lr} | "
                   f"Elapsed: {elapsed_time/60:.1f}m | "
                   f"ETA: {eta/60:.1f}m")


class ResourceMonitorCallback(BaseTrainingCallback):
    """Callback for monitoring system resources during training."""
    
    def __init__(self,
                 monitor_interval: int = 50,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize resource monitor callback.
        
        Args:
            monitor_interval: Steps between resource checks
            config: Configuration dictionary
        """
        super().__init__(config)
        self.monitor_interval = monitor_interval
        
    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               trainer=None,
               **kwargs):
        """Monitor system resources."""
        if state.global_step % self.monitor_interval != 0:
            return
        
        try:
            import torch
            import psutil
            
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.memory_reserved() / 1024**3  # GB
            else:
                gpu_memory_used = gpu_memory_total = 0
            
            # System memory
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Log resource usage
            resource_info = {
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_total_gb': gpu_memory_total,
                'system_memory_percent': memory.percent,
                'cpu_percent': cpu_percent
            }
            
            logger.debug(f"Resources at step {state.global_step}: {resource_info}")
            
            # Add to logs for aggregation
            logs = kwargs.get('logs', {})
            logs.update(resource_info)
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")


# Callback factory for easy creation
def create_callback(callback_type: str, **kwargs) -> BaseTrainingCallback:
    """
    Factory function for creating training callbacks.
    
    Args:
        callback_type: Type of callback to create
        **kwargs: Callback-specific arguments
        
    Returns:
        Callback instance
    """
    callback_map = {
        'calibration_monitor': CalibrationMonitorCallback,
        'metrics_aggregator': MetricsAggregatorCallback,
        'early_stopping': EarlyStoppingCallback,
        'progress': ProgressCallback,
        'resource_monitor': ResourceMonitorCallback
    }
    
    if callback_type not in callback_map:
        raise ValueError(f"Unknown callback type: {callback_type}")
    
    return callback_map[callback_type](**kwargs)