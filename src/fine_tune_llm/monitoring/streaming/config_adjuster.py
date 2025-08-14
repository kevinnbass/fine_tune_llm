"""
Real-time Configuration Adjustment System.

This module provides dynamic configuration adjustment capabilities
during training with live updates and rollback support.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import json
from pathlib import Path
import copy

from ...config.manager import ConfigManager
from ...config.versioning import ConfigVersionManager, ChangeType
from ...core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class AdjustmentType(Enum):
    """Types of configuration adjustments."""
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    WARMUP_STEPS = "warmup_steps"
    MAX_STEPS = "max_steps"
    LOGGING_STEPS = "logging_steps"
    SAVE_STEPS = "save_steps"
    EVAL_STEPS = "eval_steps"
    EARLY_STOPPING = "early_stopping"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    CUSTOM = "custom"


class AdjustmentStrategy(Enum):
    """Configuration adjustment strategies."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    SCHEDULED = "scheduled"
    METRIC_BASED = "metric_based"
    ADAPTIVE = "adaptive"


@dataclass
class ConfigAdjustment:
    """Configuration adjustment request."""
    adjustment_type: AdjustmentType
    parameter: str
    old_value: Any
    new_value: Any
    strategy: AdjustmentStrategy = AdjustmentStrategy.IMMEDIATE
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    rollback_value: Optional[Any] = None


@dataclass
class AdjustmentRule:
    """Rule for automatic configuration adjustment."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    adjustment: Callable[[Dict[str, Any]], ConfigAdjustment]
    enabled: bool = True
    priority: int = 0
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None


class ConfigurationAdjuster:
    """
    Real-time configuration adjustment coordinator.
    
    Manages dynamic configuration changes during training
    with validation, rollback, and event notification.
    """
    
    def __init__(self, 
                 config_manager: Optional[ConfigManager] = None,
                 version_manager: Optional[ConfigVersionManager] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize configuration adjuster.
        
        Args:
            config_manager: Configuration manager
            version_manager: Version manager for rollback
            event_bus: Event bus for notifications
        """
        self.config_manager = config_manager or ConfigManager()
        self.version_manager = version_manager or ConfigVersionManager()
        self.event_bus = event_bus or EventBus()
        
        # Adjustment history
        self.adjustment_history: List[ConfigAdjustment] = []
        
        # Adjustment rules
        self.rules: Dict[str, AdjustmentRule] = {}
        
        # Validators
        self.validators: Dict[AdjustmentType, Callable] = {}
        
        # Listeners
        self.listeners: Set[Callable] = set()
        
        # Current training state
        self.training_state: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default validators
        self._initialize_validators()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Initialized ConfigurationAdjuster")
    
    def _initialize_validators(self):
        """Initialize default parameter validators."""
        self.validators[AdjustmentType.LEARNING_RATE] = self._validate_learning_rate
        self.validators[AdjustmentType.BATCH_SIZE] = self._validate_batch_size
        self.validators[AdjustmentType.GRADIENT_ACCUMULATION] = self._validate_gradient_accumulation
        self.validators[AdjustmentType.WARMUP_STEPS] = self._validate_warmup_steps
        self.validators[AdjustmentType.MAX_STEPS] = self._validate_max_steps
    
    def _initialize_default_rules(self):
        """Initialize default adjustment rules."""
        # Learning rate decay on plateau
        self.add_rule(AdjustmentRule(
            name="lr_plateau_decay",
            condition=self._check_loss_plateau,
            adjustment=self._adjust_lr_on_plateau,
            priority=1,
            cooldown_seconds=300
        ))
        
        # Gradient accumulation on OOM
        self.add_rule(AdjustmentRule(
            name="oom_gradient_accumulation",
            condition=self._check_oom_error,
            adjustment=self._increase_gradient_accumulation,
            priority=10,
            cooldown_seconds=60
        ))
        
        # Early stopping on overfitting
        self.add_rule(AdjustmentRule(
            name="overfitting_early_stop",
            condition=self._check_overfitting,
            adjustment=self._enable_early_stopping,
            priority=5,
            cooldown_seconds=600
        ))
    
    def adjust_configuration(self, 
                            adjustment: ConfigAdjustment,
                            validate: bool = True,
                            save_version: bool = True) -> bool:
        """
        Apply configuration adjustment.
        
        Args:
            adjustment: Configuration adjustment
            validate: Whether to validate adjustment
            save_version: Whether to save version for rollback
            
        Returns:
            True if adjustment applied successfully
        """
        try:
            with self._lock:
                # Validate adjustment
                if validate and not self._validate_adjustment(adjustment):
                    logger.error(f"Adjustment validation failed: {adjustment.parameter}")
                    return False
                
                # Save current version for rollback
                if save_version:
                    current_value = self.config_manager.get(adjustment.parameter)
                    adjustment.rollback_value = current_value
                    
                    self.version_manager.create_version(
                        config={adjustment.parameter: current_value},
                        change_type=ChangeType.UPDATE,
                        description=f"Before adjustment: {adjustment.parameter}"
                    )
                
                # Apply adjustment based on strategy
                success = self._apply_adjustment_strategy(adjustment)
                
                if success:
                    # Mark as applied
                    adjustment.applied = True
                    adjustment.timestamp = datetime.now()
                    
                    # Add to history
                    self.adjustment_history.append(adjustment)
                    
                    # Notify listeners
                    self._notify_listeners(adjustment)
                    
                    # Publish event
                    self._publish_adjustment_event(adjustment)
                    
                    logger.info(f"Applied configuration adjustment: {adjustment.parameter} = {adjustment.new_value}")
                    return True
                else:
                    logger.error(f"Failed to apply adjustment: {adjustment.parameter}")
                    return False
                
        except Exception as e:
            logger.error(f"Configuration adjustment failed: {e}")
            return False
    
    def rollback_adjustment(self, 
                           adjustment: ConfigAdjustment,
                           cascade: bool = True) -> bool:
        """
        Rollback configuration adjustment.
        
        Args:
            adjustment: Adjustment to rollback
            cascade: Whether to rollback dependent adjustments
            
        Returns:
            True if rollback successful
        """
        try:
            with self._lock:
                if not adjustment.applied or adjustment.rollback_value is None:
                    logger.warning("Cannot rollback: adjustment not applied or no rollback value")
                    return False
                
                # Apply rollback
                rollback_adjustment = ConfigAdjustment(
                    adjustment_type=adjustment.adjustment_type,
                    parameter=adjustment.parameter,
                    old_value=adjustment.new_value,
                    new_value=adjustment.rollback_value,
                    strategy=AdjustmentStrategy.IMMEDIATE,
                    metadata={'rollback': True, 'original_adjustment': adjustment.metadata}
                )
                
                success = self.adjust_configuration(
                    rollback_adjustment, 
                    validate=False,
                    save_version=False
                )
                
                if success:
                    # Mark original as rolled back
                    adjustment.applied = False
                    
                    # Cascade rollback if needed
                    if cascade:
                        self._cascade_rollback(adjustment)
                    
                    logger.info(f"Rolled back adjustment: {adjustment.parameter}")
                    return True
                else:
                    logger.error(f"Rollback failed: {adjustment.parameter}")
                    return False
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def add_rule(self, rule: AdjustmentRule) -> bool:
        """
        Add automatic adjustment rule.
        
        Args:
            rule: Adjustment rule
            
        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                self.rules[rule.name] = rule
            
            logger.info(f"Added adjustment rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add rule {rule.name}: {e}")
            return False
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove adjustment rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if removed successfully
        """
        try:
            with self._lock:
                if rule_name in self.rules:
                    del self.rules[rule_name]
                    logger.info(f"Removed adjustment rule: {rule_name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove rule {rule_name}: {e}")
            return False
    
    def update_training_state(self, state: Dict[str, Any]):
        """
        Update current training state.
        
        Args:
            state: Training state dictionary
        """
        with self._lock:
            self.training_state.update(state)
            
            # Check and apply rules
            self._check_and_apply_rules()
    
    def _check_and_apply_rules(self):
        """Check conditions and apply adjustment rules."""
        current_time = datetime.now()
        
        # Sort rules by priority
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                elapsed = (current_time - rule.last_triggered).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue
            
            # Check condition
            try:
                if rule.condition(self.training_state):
                    # Generate adjustment
                    adjustment = rule.adjustment(self.training_state)
                    
                    # Apply adjustment
                    if self.adjust_configuration(adjustment):
                        rule.last_triggered = current_time
                        logger.info(f"Applied rule {rule.name}: {adjustment.parameter}")
                        
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {e}")
    
    def _apply_adjustment_strategy(self, adjustment: ConfigAdjustment) -> bool:
        """Apply adjustment based on strategy."""
        if adjustment.strategy == AdjustmentStrategy.IMMEDIATE:
            return self._apply_immediate(adjustment)
        elif adjustment.strategy == AdjustmentStrategy.GRADUAL:
            return self._apply_gradual(adjustment)
        elif adjustment.strategy == AdjustmentStrategy.SCHEDULED:
            return self._apply_scheduled(adjustment)
        elif adjustment.strategy == AdjustmentStrategy.METRIC_BASED:
            return self._apply_metric_based(adjustment)
        elif adjustment.strategy == AdjustmentStrategy.ADAPTIVE:
            return self._apply_adaptive(adjustment)
        else:
            return self._apply_immediate(adjustment)
    
    def _apply_immediate(self, adjustment: ConfigAdjustment) -> bool:
        """Apply adjustment immediately."""
        try:
            self.config_manager.set(adjustment.parameter, adjustment.new_value)
            return True
        except Exception as e:
            logger.error(f"Immediate adjustment failed: {e}")
            return False
    
    def _apply_gradual(self, adjustment: ConfigAdjustment) -> bool:
        """Apply adjustment gradually over time."""
        # Create gradual adjustment schedule
        steps = adjustment.metadata.get('steps', 10)
        
        if isinstance(adjustment.old_value, (int, float)) and isinstance(adjustment.new_value, (int, float)):
            step_size = (adjustment.new_value - adjustment.old_value) / steps
            
            # Schedule gradual updates
            def gradual_update():
                current = adjustment.old_value
                for i in range(steps):
                    current += step_size
                    self.config_manager.set(adjustment.parameter, current)
                    import time
                    time.sleep(1)  # Adjust interval as needed
            
            # Run in background thread
            thread = threading.Thread(target=gradual_update)
            thread.daemon = True
            thread.start()
            
            return True
        else:
            # Fall back to immediate for non-numeric values
            return self._apply_immediate(adjustment)
    
    def _apply_scheduled(self, adjustment: ConfigAdjustment) -> bool:
        """Apply adjustment at scheduled time."""
        schedule_time = adjustment.metadata.get('schedule_time')
        
        if schedule_time:
            # Schedule adjustment for later
            def scheduled_apply():
                delay = (schedule_time - datetime.now()).total_seconds()
                if delay > 0:
                    import time
                    time.sleep(delay)
                self._apply_immediate(adjustment)
            
            thread = threading.Thread(target=scheduled_apply)
            thread.daemon = True
            thread.start()
            
            return True
        else:
            return self._apply_immediate(adjustment)
    
    def _apply_metric_based(self, adjustment: ConfigAdjustment) -> bool:
        """Apply adjustment based on metric conditions."""
        metric_name = adjustment.metadata.get('metric_name')
        threshold = adjustment.metadata.get('threshold')
        
        if metric_name and threshold:
            # Monitor metric and apply when condition met
            def metric_monitor():
                while True:
                    metric_value = self.training_state.get(metric_name)
                    if metric_value and metric_value >= threshold:
                        self._apply_immediate(adjustment)
                        break
                    import time
                    time.sleep(10)
            
            thread = threading.Thread(target=metric_monitor)
            thread.daemon = True
            thread.start()
            
            return True
        else:
            return self._apply_immediate(adjustment)
    
    def _apply_adaptive(self, adjustment: ConfigAdjustment) -> bool:
        """Apply adjustment adaptively based on feedback."""
        # Implement adaptive adjustment logic
        # This could involve monitoring the effect and adjusting further
        return self._apply_immediate(adjustment)
    
    def _validate_adjustment(self, adjustment: ConfigAdjustment) -> bool:
        """Validate configuration adjustment."""
        # Check if validator exists for adjustment type
        validator = self.validators.get(adjustment.adjustment_type)
        
        if validator:
            return validator(adjustment)
        
        # Default validation
        return True
    
    def _validate_learning_rate(self, adjustment: ConfigAdjustment) -> bool:
        """Validate learning rate adjustment."""
        new_value = adjustment.new_value
        
        if not isinstance(new_value, (int, float)):
            return False
        
        if new_value <= 0 or new_value > 1:
            return False
        
        # Check for too large changes
        if adjustment.old_value:
            ratio = new_value / adjustment.old_value
            if ratio > 10 or ratio < 0.1:
                logger.warning(f"Large learning rate change: {ratio}x")
        
        return True
    
    def _validate_batch_size(self, adjustment: ConfigAdjustment) -> bool:
        """Validate batch size adjustment."""
        new_value = adjustment.new_value
        
        if not isinstance(new_value, int):
            return False
        
        if new_value <= 0 or new_value > 1024:
            return False
        
        # Check if power of 2 (recommended)
        if new_value & (new_value - 1) != 0:
            logger.warning(f"Batch size {new_value} is not a power of 2")
        
        return True
    
    def _validate_gradient_accumulation(self, adjustment: ConfigAdjustment) -> bool:
        """Validate gradient accumulation adjustment."""
        new_value = adjustment.new_value
        
        if not isinstance(new_value, int):
            return False
        
        if new_value <= 0 or new_value > 128:
            return False
        
        return True
    
    def _validate_warmup_steps(self, adjustment: ConfigAdjustment) -> bool:
        """Validate warmup steps adjustment."""
        new_value = adjustment.new_value
        
        if not isinstance(new_value, int):
            return False
        
        if new_value < 0:
            return False
        
        # Check against max steps
        max_steps = self.training_state.get('max_steps')
        if max_steps and new_value > max_steps * 0.5:
            logger.warning("Warmup steps exceed 50% of max steps")
        
        return True
    
    def _validate_max_steps(self, adjustment: ConfigAdjustment) -> bool:
        """Validate max steps adjustment."""
        new_value = adjustment.new_value
        
        if not isinstance(new_value, int):
            return False
        
        if new_value <= 0:
            return False
        
        # Check against current step
        current_step = self.training_state.get('current_step', 0)
        if new_value <= current_step:
            return False
        
        return True
    
    def _check_loss_plateau(self, state: Dict[str, Any]) -> bool:
        """Check if loss has plateaued."""
        loss_history = state.get('loss_history', [])
        
        if len(loss_history) < 10:
            return False
        
        # Check if loss hasn't improved in last 5 steps
        recent_losses = loss_history[-5:]
        if recent_losses:
            improvement = min(recent_losses) - recent_losses[0]
            return improvement > -0.001  # Less than 0.1% improvement
        
        return False
    
    def _adjust_lr_on_plateau(self, state: Dict[str, Any]) -> ConfigAdjustment:
        """Generate learning rate adjustment for plateau."""
        current_lr = state.get('learning_rate', 1e-4)
        new_lr = current_lr * 0.5  # Reduce by 50%
        
        return ConfigAdjustment(
            adjustment_type=AdjustmentType.LEARNING_RATE,
            parameter='learning_rate',
            old_value=current_lr,
            new_value=new_lr,
            strategy=AdjustmentStrategy.IMMEDIATE,
            metadata={'reason': 'loss_plateau'}
        )
    
    def _check_oom_error(self, state: Dict[str, Any]) -> bool:
        """Check for out-of-memory errors."""
        return state.get('oom_error', False)
    
    def _increase_gradient_accumulation(self, state: Dict[str, Any]) -> ConfigAdjustment:
        """Increase gradient accumulation to handle OOM."""
        current_accumulation = state.get('gradient_accumulation_steps', 1)
        new_accumulation = current_accumulation * 2
        
        return ConfigAdjustment(
            adjustment_type=AdjustmentType.GRADIENT_ACCUMULATION,
            parameter='gradient_accumulation_steps',
            old_value=current_accumulation,
            new_value=new_accumulation,
            strategy=AdjustmentStrategy.IMMEDIATE,
            metadata={'reason': 'oom_error'}
        )
    
    def _check_overfitting(self, state: Dict[str, Any]) -> bool:
        """Check for overfitting."""
        train_loss = state.get('train_loss')
        val_loss = state.get('val_loss')
        
        if train_loss and val_loss:
            # Check if validation loss is significantly worse
            return val_loss > train_loss * 1.5
        
        return False
    
    def _enable_early_stopping(self, state: Dict[str, Any]) -> ConfigAdjustment:
        """Enable early stopping for overfitting."""
        return ConfigAdjustment(
            adjustment_type=AdjustmentType.EARLY_STOPPING,
            parameter='early_stopping',
            old_value=False,
            new_value=True,
            strategy=AdjustmentStrategy.IMMEDIATE,
            metadata={'reason': 'overfitting', 'patience': 3}
        )
    
    def _cascade_rollback(self, adjustment: ConfigAdjustment):
        """Cascade rollback to dependent adjustments."""
        # Find dependent adjustments
        dependent_adjustments = [
            adj for adj in self.adjustment_history
            if adj.applied and adj.metadata.get('depends_on') == adjustment.parameter
        ]
        
        for dep_adj in dependent_adjustments:
            self.rollback_adjustment(dep_adj, cascade=False)
    
    def _notify_listeners(self, adjustment: ConfigAdjustment):
        """Notify all listeners of adjustment."""
        for listener in self.listeners:
            try:
                listener(adjustment)
            except Exception as e:
                logger.error(f"Listener notification failed: {e}")
    
    def _publish_adjustment_event(self, adjustment: ConfigAdjustment):
        """Publish adjustment event to event bus."""
        event = Event(
            type=EventType.CONFIGURATION_CHANGED,
            data={
                'parameter': adjustment.parameter,
                'old_value': adjustment.old_value,
                'new_value': adjustment.new_value,
                'adjustment_type': adjustment.adjustment_type.value,
                'strategy': adjustment.strategy.value
            },
            source="ConfigurationAdjuster"
        )
        
        self.event_bus.publish(event)
    
    def add_listener(self, listener: Callable) -> bool:
        """Add adjustment listener."""
        self.listeners.add(listener)
        return True
    
    def remove_listener(self, listener: Callable) -> bool:
        """Remove adjustment listener."""
        self.listeners.discard(listener)
        return True
    
    def get_adjustment_history(self, 
                              limit: Optional[int] = None,
                              parameter: Optional[str] = None) -> List[ConfigAdjustment]:
        """
        Get adjustment history.
        
        Args:
            limit: Maximum number of adjustments to return
            parameter: Filter by parameter name
            
        Returns:
            List of adjustments
        """
        with self._lock:
            history = self.adjustment_history.copy()
            
            if parameter:
                history = [adj for adj in history if adj.parameter == parameter]
            
            if limit:
                history = history[-limit:]
            
            return history
    
    def export_adjustments(self, filepath: Path) -> bool:
        """
        Export adjustment history to file.
        
        Args:
            filepath: Export file path
            
        Returns:
            True if exported successfully
        """
        try:
            with self._lock:
                export_data = []
                
                for adj in self.adjustment_history:
                    export_data.append({
                        'timestamp': adj.timestamp.isoformat(),
                        'type': adj.adjustment_type.value,
                        'parameter': adj.parameter,
                        'old_value': str(adj.old_value),
                        'new_value': str(adj.new_value),
                        'strategy': adj.strategy.value,
                        'applied': adj.applied,
                        'metadata': adj.metadata
                    })
                
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Exported {len(export_data)} adjustments to {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export adjustments: {e}")
            return False


# Global adjuster instance
_config_adjuster = None

def get_config_adjuster() -> ConfigurationAdjuster:
    """Get global configuration adjuster instance."""
    global _config_adjuster
    if _config_adjuster is None:
        _config_adjuster = ConfigurationAdjuster()
    return _config_adjuster


# Convenience functions

def adjust_learning_rate(new_rate: float, strategy: AdjustmentStrategy = AdjustmentStrategy.IMMEDIATE) -> bool:
    """
    Adjust learning rate.
    
    Args:
        new_rate: New learning rate
        strategy: Adjustment strategy
        
    Returns:
        True if adjusted successfully
    """
    adjuster = get_config_adjuster()
    
    adjustment = ConfigAdjustment(
        adjustment_type=AdjustmentType.LEARNING_RATE,
        parameter='learning_rate',
        old_value=adjuster.training_state.get('learning_rate'),
        new_value=new_rate,
        strategy=strategy
    )
    
    return adjuster.adjust_configuration(adjustment)


def enable_adaptive_adjustments(rules: Optional[List[AdjustmentRule]] = None) -> bool:
    """
    Enable adaptive configuration adjustments.
    
    Args:
        rules: Additional adjustment rules
        
    Returns:
        True if enabled successfully
    """
    adjuster = get_config_adjuster()
    
    if rules:
        for rule in rules:
            adjuster.add_rule(rule)
    
    # Enable all rules
    for rule in adjuster.rules.values():
        rule.enabled = True
    
    return True