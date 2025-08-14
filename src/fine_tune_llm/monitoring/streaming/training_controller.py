"""
Training Process Control System.

This module provides comprehensive training process control capabilities
from dashboards with pause, resume, stop, and parameter adjustment features.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import asyncio
import json
from pathlib import Path
import signal
import os
import psutil

from ...core.events import EventBus, Event, EventType
from .metrics_streamer import MetricsStreamer, StreamMessage, StreamEvent
from .config_adjuster import ConfigurationAdjuster, ConfigAdjustment

logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """Training process states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    PAUSING = "pausing"
    RESUMING = "resuming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


class ControlCommand(Enum):
    """Training control commands."""
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    RESTART = "restart"
    ADJUST_CONFIG = "adjust_config"
    SAVE_CHECKPOINT = "save_checkpoint"
    RELOAD_DATA = "reload_data"
    VALIDATE = "validate"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class TrainingStatus:
    """Current training status information."""
    state: TrainingState
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None
    last_checkpoint: Optional[str] = None
    current_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ControlRequest:
    """Training control request."""
    command: ControlCommand
    parameters: Dict[str, Any] = field(default_factory=dict)
    requestor: str = "dashboard"
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0


class TrainingProcess:
    """Wrapper for training process management."""
    
    def __init__(self, process_id: Optional[int] = None):
        """Initialize training process wrapper."""
        self.process_id = process_id
        self.process: Optional[psutil.Process] = None
        
        if process_id:
            try:
                self.process = psutil.Process(process_id)
            except psutil.NoSuchProcess:
                logger.warning(f"Process {process_id} not found")
    
    def is_running(self) -> bool:
        """Check if process is running."""
        if not self.process:
            return False
        
        try:
            return self.process.is_running() and self.process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get process memory usage."""
        if not self.process or not self.is_running():
            return {}
        
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': memory_percent
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def get_cpu_usage(self) -> float:
        """Get process CPU usage."""
        if not self.process or not self.is_running():
            return 0.0
        
        try:
            return self.process.cpu_percent()
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return 0.0
    
    def send_signal(self, signal_type: int) -> bool:
        """Send signal to process."""
        if not self.process or not self.is_running():
            return False
        
        try:
            self.process.send_signal(signal_type)
            return True
        except Exception as e:
            logger.error(f"Failed to send signal {signal_type}: {e}")
            return False
    
    def terminate(self, timeout: int = 30) -> bool:
        """Terminate process gracefully."""
        if not self.process or not self.is_running():
            return True
        
        try:
            self.process.terminate()
            
            # Wait for graceful termination
            try:
                self.process.wait(timeout=timeout)
                return True
            except psutil.TimeoutExpired:
                # Force kill if necessary
                self.process.kill()
                return True
                
        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")
            return False


class TrainingController:
    """
    Comprehensive training process controller.
    
    Provides centralized control of training processes with state management,
    command processing, and dashboard integration.
    """
    
    def __init__(self,
                 metrics_streamer: Optional[MetricsStreamer] = None,
                 config_adjuster: Optional[ConfigurationAdjuster] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize training controller.
        
        Args:
            metrics_streamer: Metrics streaming system
            config_adjuster: Configuration adjustment system
            event_bus: Event bus for notifications
        """
        self.metrics_streamer = metrics_streamer
        self.config_adjuster = config_adjuster
        self.event_bus = event_bus or EventBus()
        
        # Training status
        self.status = TrainingStatus(state=TrainingState.IDLE)
        
        # Training process
        self.training_process: Optional[TrainingProcess] = None
        
        # Control queue
        self.control_queue: List[ControlRequest] = []
        
        # Command handlers
        self.command_handlers: Dict[ControlCommand, Callable] = {
            ControlCommand.START: self._handle_start,
            ControlCommand.PAUSE: self._handle_pause,
            ControlCommand.RESUME: self._handle_resume,
            ControlCommand.STOP: self._handle_stop,
            ControlCommand.RESTART: self._handle_restart,
            ControlCommand.ADJUST_CONFIG: self._handle_adjust_config,
            ControlCommand.SAVE_CHECKPOINT: self._handle_save_checkpoint,
            ControlCommand.RELOAD_DATA: self._handle_reload_data,
            ControlCommand.VALIDATE: self._handle_validate,
            ControlCommand.EMERGENCY_STOP: self._handle_emergency_stop
        }
        
        # State transition handlers
        self.state_handlers: Dict[TrainingState, Callable] = {
            TrainingState.STARTING: self._handle_starting_state,
            TrainingState.RUNNING: self._handle_running_state,
            TrainingState.PAUSING: self._handle_pausing_state,
            TrainingState.RESUMING: self._handle_resuming_state,
            TrainingState.STOPPING: self._handle_stopping_state
        }
        
        # Status history
        self.status_history: List[TrainingStatus] = []
        
        # Control workers
        self.control_worker_thread: Optional[threading.Thread] = None
        self.status_worker_thread: Optional[threading.Thread] = None
        self.worker_running = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.commands_processed = 0
        self.state_changes = 0
        
        logger.info("Initialized TrainingController")
    
    def start_controller(self) -> bool:
        """
        Start training controller.
        
        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                if self.worker_running:
                    return True
                
                # Start worker threads
                self.worker_running = True
                
                self.control_worker_thread = threading.Thread(target=self._control_worker_loop)
                self.control_worker_thread.daemon = True
                self.control_worker_thread.start()
                
                self.status_worker_thread = threading.Thread(target=self._status_worker_loop)
                self.status_worker_thread.daemon = True
                self.status_worker_thread.start()
                
                # Subscribe to metrics updates
                if self.metrics_streamer:
                    self.metrics_streamer.subscribe(self._on_metrics_update)
                
                logger.info("Training controller started")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start controller: {e}")
            return False
    
    def stop_controller(self):
        """Stop training controller."""
        try:
            with self._lock:
                self.worker_running = False
                
                # Wait for workers to finish
                if self.control_worker_thread:
                    self.control_worker_thread.join(timeout=5)
                
                if self.status_worker_thread:
                    self.status_worker_thread.join(timeout=5)
                
                # Unsubscribe from metrics
                if self.metrics_streamer:
                    self.metrics_streamer.unsubscribe(self._on_metrics_update)
                
                logger.info("Training controller stopped")
                
        except Exception as e:
            logger.error(f"Error stopping controller: {e}")
    
    def send_command(self, 
                    command: ControlCommand,
                    parameters: Optional[Dict[str, Any]] = None,
                    priority: int = 0) -> bool:
        """
        Send control command.
        
        Args:
            command: Control command
            parameters: Command parameters
            priority: Command priority (higher = more urgent)
            
        Returns:
            True if command queued successfully
        """
        try:
            request = ControlRequest(
                command=command,
                parameters=parameters or {},
                priority=priority
            )
            
            with self._lock:
                # Insert by priority
                inserted = False
                for i, existing in enumerate(self.control_queue):
                    if request.priority > existing.priority:
                        self.control_queue.insert(i, request)
                        inserted = True
                        break
                
                if not inserted:
                    self.control_queue.append(request)
            
            logger.info(f"Queued command: {command.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue command {command.value}: {e}")
            return False
    
    def get_status(self) -> TrainingStatus:
        """Get current training status."""
        with self._lock:
            return self.status
    
    def set_training_process(self, process_id: int):
        """Set training process ID."""
        with self._lock:
            self.training_process = TrainingProcess(process_id)
            logger.info(f"Set training process: {process_id}")
    
    def _control_worker_loop(self):
        """Control command processing loop."""
        while self.worker_running:
            try:
                # Process commands
                commands_to_process = []
                
                with self._lock:
                    if self.control_queue:
                        # Process up to 5 commands per iteration
                        commands_to_process = self.control_queue[:5]
                        self.control_queue = self.control_queue[5:]
                
                for request in commands_to_process:
                    self._process_command(request)
                    self.commands_processed += 1
                
                # Small delay
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Control worker error: {e}")
    
    def _status_worker_loop(self):
        """Status monitoring loop."""
        while self.worker_running:
            try:
                # Update status
                self._update_status()
                
                # Check state transitions
                self._check_state_transitions()
                
                # Stream status updates
                self._stream_status_update()
                
                # Small delay
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Status worker error: {e}")
    
    def _process_command(self, request: ControlRequest):
        """Process individual command."""
        try:
            handler = self.command_handlers.get(request.command)
            
            if handler:
                logger.info(f"Processing command: {request.command.value}")
                success = handler(request.parameters)
                
                # Publish command event
                self._publish_command_event(request, success)
                
                if success:
                    logger.info(f"Command completed: {request.command.value}")
                else:
                    logger.error(f"Command failed: {request.command.value}")
            else:
                logger.error(f"Unknown command: {request.command.value}")
                
        except Exception as e:
            logger.error(f"Command processing error: {e}")
    
    def _handle_start(self, parameters: Dict[str, Any]) -> bool:
        """Handle start command."""
        if self.status.state not in [TrainingState.IDLE, TrainingState.STOPPED]:
            logger.warning(f"Cannot start from state: {self.status.state}")
            return False
        
        try:
            self._change_state(TrainingState.STARTING)
            
            # Extract start parameters
            script_path = parameters.get('script_path', 'scripts/train_lora_sft.py')
            config_path = parameters.get('config_path', 'configs/llm_lora.yaml')
            
            # Start training process
            import subprocess
            process = subprocess.Popen([
                'python', script_path,
                '--config', config_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.set_training_process(process.pid)
            self._change_state(TrainingState.RUNNING)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self._change_state(TrainingState.ERROR)
            return False
    
    def _handle_pause(self, parameters: Dict[str, Any]) -> bool:
        """Handle pause command."""
        if self.status.state != TrainingState.RUNNING:
            logger.warning(f"Cannot pause from state: {self.status.state}")
            return False
        
        try:
            self._change_state(TrainingState.PAUSING)
            
            # Send SIGTERM to training process
            if self.training_process:
                success = self.training_process.send_signal(signal.SIGTERM)
                if success:
                    self._change_state(TrainingState.PAUSED)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to pause training: {e}")
            return False
    
    def _handle_resume(self, parameters: Dict[str, Any]) -> bool:
        """Handle resume command."""
        if self.status.state != TrainingState.PAUSED:
            logger.warning(f"Cannot resume from state: {self.status.state}")
            return False
        
        try:
            self._change_state(TrainingState.RESUMING)
            
            # Send SIGCONT to training process
            if self.training_process:
                success = self.training_process.send_signal(signal.SIGCONT)
                if success:
                    self._change_state(TrainingState.RUNNING)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resume training: {e}")
            return False
    
    def _handle_stop(self, parameters: Dict[str, Any]) -> bool:
        """Handle stop command."""
        if self.status.state in [TrainingState.IDLE, TrainingState.STOPPED]:
            return True
        
        try:
            self._change_state(TrainingState.STOPPING)
            
            # Gracefully terminate training process
            if self.training_process:
                timeout = parameters.get('timeout', 30)
                success = self.training_process.terminate(timeout)
                
                if success:
                    self._change_state(TrainingState.STOPPED)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return False
    
    def _handle_restart(self, parameters: Dict[str, Any]) -> bool:
        """Handle restart command."""
        # Stop first
        if not self._handle_stop(parameters):
            return False
        
        # Wait a moment
        import time
        time.sleep(2)
        
        # Start again
        return self._handle_start(parameters)
    
    def _handle_adjust_config(self, parameters: Dict[str, Any]) -> bool:
        """Handle configuration adjustment."""
        if not self.config_adjuster:
            logger.error("Configuration adjuster not available")
            return False
        
        try:
            # Extract adjustment parameters
            param_name = parameters.get('parameter')
            new_value = parameters.get('value')
            
            if not param_name or new_value is None:
                logger.error("Missing parameter name or value")
                return False
            
            # Create adjustment
            from .config_adjuster import ConfigAdjustment, AdjustmentType, AdjustmentStrategy
            
            adjustment = ConfigAdjustment(
                adjustment_type=AdjustmentType.CUSTOM,
                parameter=param_name,
                old_value=None,  # Will be filled by adjuster
                new_value=new_value,
                strategy=AdjustmentStrategy.IMMEDIATE,
                metadata=parameters.get('metadata', {})
            )
            
            return self.config_adjuster.adjust_configuration(adjustment)
            
        except Exception as e:
            logger.error(f"Configuration adjustment failed: {e}")
            return False
    
    def _handle_save_checkpoint(self, parameters: Dict[str, Any]) -> bool:
        """Handle save checkpoint command."""
        try:
            # Send save signal to training process
            if self.training_process:
                # Use custom signal for checkpoint saving
                return self.training_process.send_signal(signal.SIGUSR1)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _handle_reload_data(self, parameters: Dict[str, Any]) -> bool:
        """Handle reload data command."""
        try:
            # Send reload signal to training process
            if self.training_process:
                return self.training_process.send_signal(signal.SIGUSR2)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to reload data: {e}")
            return False
    
    def _handle_validate(self, parameters: Dict[str, Any]) -> bool:
        """Handle validation command."""
        try:
            # Trigger validation run
            # This would depend on training framework integration
            logger.info("Validation triggered")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _handle_emergency_stop(self, parameters: Dict[str, Any]) -> bool:
        """Handle emergency stop command."""
        try:
            if self.training_process:
                # Force kill immediately
                success = self.training_process.send_signal(signal.SIGKILL)
                if success:
                    self._change_state(TrainingState.STOPPED)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False
    
    def _change_state(self, new_state: TrainingState):
        """Change training state."""
        with self._lock:
            if new_state != self.status.state:
                old_state = self.status.state
                self.status.state = new_state
                self.status.timestamp = datetime.now()
                self.state_changes += 1
                
                # Add to history
                self.status_history.append(self.status)
                
                # Publish state change event
                self._publish_state_change_event(old_state, new_state)
                
                logger.info(f"State changed: {old_state.value} -> {new_state.value}")
    
    def _update_status(self):
        """Update training status information."""
        with self._lock:
            # Update process information
            if self.training_process:
                if not self.training_process.is_running():
                    if self.status.state == TrainingState.RUNNING:
                        self._change_state(TrainingState.STOPPED)
                
                # Update resource usage
                memory_usage = self.training_process.get_memory_usage()
                cpu_usage = self.training_process.get_cpu_usage()
                
                self.status.metadata.update({
                    'memory_usage': memory_usage,
                    'cpu_usage': cpu_usage
                })
    
    def _check_state_transitions(self):
        """Check and handle state transitions."""
        handler = self.state_handlers.get(self.status.state)
        if handler:
            handler()
    
    def _handle_starting_state(self):
        """Handle starting state."""
        # Check if process has started successfully
        if self.training_process and self.training_process.is_running():
            # Auto-transition to running after successful start
            import time
            time.sleep(1)  # Give it a moment
            if self.status.state == TrainingState.STARTING:
                self._change_state(TrainingState.RUNNING)
    
    def _handle_running_state(self):
        """Handle running state."""
        # Monitor for completion or errors
        if self.training_process and not self.training_process.is_running():
            # Process has ended
            self._change_state(TrainingState.COMPLETED)
    
    def _handle_pausing_state(self):
        """Handle pausing state."""
        # Check if process has actually paused
        if self.training_process:
            try:
                status = self.training_process.process.status()
                if status == psutil.STATUS_STOPPED:
                    self._change_state(TrainingState.PAUSED)
            except:
                pass
    
    def _handle_resuming_state(self):
        """Handle resuming state."""
        # Check if process has resumed
        if self.training_process:
            try:
                status = self.training_process.process.status()
                if status == psutil.STATUS_RUNNING:
                    self._change_state(TrainingState.RUNNING)
            except:
                pass
    
    def _handle_stopping_state(self):
        """Handle stopping state."""
        # Check if process has stopped
        if not self.training_process or not self.training_process.is_running():
            self._change_state(TrainingState.STOPPED)
    
    def _stream_status_update(self):
        """Stream status update to dashboard."""
        if self.metrics_streamer:
            message = StreamMessage(
                event=StreamEvent.INFO,
                data={
                    'type': 'training_status',
                    'status': {
                        'state': self.status.state.value,
                        'current_epoch': self.status.current_epoch,
                        'current_step': self.status.current_step,
                        'progress_percent': self.status.progress_percent,
                        'elapsed_time': self.status.elapsed_time,
                        'current_metrics': self.status.current_metrics,
                        'metadata': self.status.metadata
                    }
                },
                source="TrainingController"
            )
            
            self.metrics_streamer.connection.send(message)
    
    def _on_metrics_update(self, update):
        """Handle metrics update from streamer."""
        with self._lock:
            # Update current metrics
            self.status.current_metrics[update.metric_name] = update.value
            
            # Update progress if step information available
            if hasattr(update, 'step'):
                self.status.current_step = update.step
                
                if self.status.total_steps > 0:
                    self.status.progress_percent = (self.status.current_step / self.status.total_steps) * 100
    
    def _publish_command_event(self, request: ControlRequest, success: bool):
        """Publish command execution event."""
        event = Event(
            type=EventType.SYSTEM,
            data={
                'command': request.command.value,
                'parameters': request.parameters,
                'success': success,
                'timestamp': request.timestamp.isoformat()
            },
            source="TrainingController"
        )
        
        self.event_bus.publish(event)
    
    def _publish_state_change_event(self, old_state: TrainingState, new_state: TrainingState):
        """Publish state change event."""
        event = Event(
            type=EventType.TRAINING_STATE_CHANGED,
            data={
                'old_state': old_state.value,
                'new_state': new_state.value,
                'timestamp': datetime.now().isoformat()
            },
            source="TrainingController"
        )
        
        self.event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        with self._lock:
            return {
                'current_state': self.status.state.value,
                'commands_processed': self.commands_processed,
                'state_changes': self.state_changes,
                'queue_size': len(self.control_queue),
                'worker_running': self.worker_running,
                'process_running': self.training_process.is_running() if self.training_process else False
            }


# Global controller instance
_training_controller = None

def get_training_controller() -> TrainingController:
    """Get global training controller instance."""
    global _training_controller
    if _training_controller is None:
        _training_controller = TrainingController()
    return _training_controller


# Convenience functions

def start_training(script_path: str = 'scripts/train_lora_sft.py',
                  config_path: str = 'configs/llm_lora.yaml') -> bool:
    """
    Start training process.
    
    Args:
        script_path: Path to training script
        config_path: Path to configuration file
        
    Returns:
        True if started successfully
    """
    controller = get_training_controller()
    return controller.send_command(
        ControlCommand.START,
        {'script_path': script_path, 'config_path': config_path}
    )


def pause_training() -> bool:
    """Pause training process."""
    controller = get_training_controller()
    return controller.send_command(ControlCommand.PAUSE)


def resume_training() -> bool:
    """Resume training process."""
    controller = get_training_controller()
    return controller.send_command(ControlCommand.RESUME)


def stop_training(timeout: int = 30) -> bool:
    """
    Stop training process.
    
    Args:
        timeout: Termination timeout in seconds
        
    Returns:
        True if stopped successfully
    """
    controller = get_training_controller()
    return controller.send_command(
        ControlCommand.STOP,
        {'timeout': timeout}
    )


def emergency_stop() -> bool:
    """Emergency stop training process."""
    controller = get_training_controller()
    return controller.send_command(ControlCommand.EMERGENCY_STOP)