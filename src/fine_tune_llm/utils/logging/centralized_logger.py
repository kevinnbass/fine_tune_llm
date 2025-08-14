"""
Centralized Logging System.

This module provides a unified logging infrastructure across all platform
components with structured logging, performance monitoring, and advanced features.
"""

import logging
import logging.handlers
import json
import sys
import os
import threading
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import traceback
import inspect
from collections import deque, defaultdict
import gzip
import hashlib

from ...core.events import EventBus, Event, EventType

# Create module logger
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Enhanced log levels with custom levels."""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5
    PERFORMANCE = 15
    AUDIT = 25
    SECURITY = 35


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    STRUCTURED = "structured"
    STANDARD = "standard"
    COMPACT = "compact"
    CUSTOM = "custom"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: LogLevel = LogLevel.INFO
    logger_name: str = ""
    message: str = ""
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: int = 0
    process_id: int = 0
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    component: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'logger': self.logger_name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line': self.line_number,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'component': self.component,
            'tags': self.tags,
            'metadata': self.metadata,
            'exception': self.exception,
            'stack_trace': self.stack_trace,
            'performance': self.performance_data
        }
    
    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class LoggerConfig:
    """Configuration for centralized logger."""
    name: str
    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.STRUCTURED
    output_file: Optional[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    compression: bool = True
    console_output: bool = True
    structured_output: bool = True
    buffer_size: int = 1000
    flush_interval: int = 5
    include_performance: bool = True
    include_caller_info: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, format_type: LogFormat = LogFormat.JSON, include_extra: bool = True):
        """Initialize structured formatter."""
        super().__init__()
        self.format_type = format_type
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        try:
            # Create structured log entry
            entry = self._create_log_entry(record)
            
            if self.format_type == LogFormat.JSON:
                return entry.to_json()
            elif self.format_type == LogFormat.STRUCTURED:
                return self._format_structured(entry)
            elif self.format_type == LogFormat.COMPACT:
                return self._format_compact(entry)
            else:
                return self._format_standard(entry)
                
        except Exception as e:
            # Fallback formatting
            return f"LOGGING_ERROR: {e} | Original: {record.getMessage()}"
    
    def _create_log_entry(self, record: logging.LogRecord) -> LogEntry:
        """Create structured log entry from log record."""
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created, timezone.utc),
            level=LogLevel(record.levelno) if record.levelno in [l.value for l in LogLevel] else LogLevel.INFO,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module if hasattr(record, 'module') else '',
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process
        )
        
        # Add extra attributes
        if self.include_extra:
            extra_attrs = [
                'session_id', 'request_id', 'user_id', 'component',
                'tags', 'metadata', 'performance_data'
            ]
            
            for attr in extra_attrs:
                if hasattr(record, attr):
                    setattr(entry, attr, getattr(record, attr))
        
        # Handle exceptions
        if record.exc_info:
            entry.exception = str(record.exc_info[1])
            entry.stack_trace = self.formatException(record.exc_info)
        
        return entry
    
    def _format_structured(self, entry: LogEntry) -> str:
        """Format as structured text."""
        parts = [
            f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]",
            f"[{entry.level.name}]",
            f"[{entry.logger_name}]",
            f"[{entry.module}:{entry.function}:{entry.line_number}]",
            entry.message
        ]
        
        if entry.tags:
            parts.append(f"tags={','.join(entry.tags)}")
        
        if entry.metadata:
            metadata_str = ' '.join(f"{k}={v}" for k, v in entry.metadata.items())
            parts.append(f"metadata=({metadata_str})")
        
        if entry.exception:
            parts.append(f"exception={entry.exception}")
        
        return ' '.join(parts)
    
    def _format_compact(self, entry: LogEntry) -> str:
        """Format as compact text."""
        return f"{entry.timestamp.strftime('%H:%M:%S')} {entry.level.name[0]} {entry.module}:{entry.line_number} {entry.message}"
    
    def _format_standard(self, entry: LogEntry) -> str:
        """Format as standard text."""
        return f"{entry.timestamp} - {entry.logger_name} - {entry.level.name} - {entry.message}"


class PerformanceTracker:
    """Performance tracking for logging operations."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def track_operation(self, operation: str, duration: float):
        """Track operation performance."""
        with self._lock:
            self.operation_times[operation].append(duration)
            self.counters[f"{operation}_count"] += 1
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if operation:
                times = list(self.operation_times.get(operation, []))
                if times:
                    return {
                        'operation': operation,
                        'count': len(times),
                        'avg_duration': sum(times) / len(times),
                        'min_duration': min(times),
                        'max_duration': max(times),
                        'total_duration': sum(times)
                    }
                return {'operation': operation, 'count': 0}
            else:
                stats = {}
                for op, times in self.operation_times.items():
                    times_list = list(times)
                    if times_list:
                        stats[op] = {
                            'count': len(times_list),
                            'avg_duration': sum(times_list) / len(times_list),
                            'min_duration': min(times_list),
                            'max_duration': max(times_list),
                            'total_duration': sum(times_list)
                        }
                
                return {
                    'operations': stats,
                    'counters': dict(self.counters)
                }


class LogBuffer:
    """Thread-safe log buffer for batching."""
    
    def __init__(self, max_size: int = 1000, flush_interval: int = 5):
        """Initialize log buffer."""
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: deque = deque(maxlen=max_size)
        self.last_flush = time.time()
        self._lock = threading.RLock()
        self.flush_callbacks: List[Callable] = []
    
    def add_entry(self, entry: LogEntry) -> bool:
        """Add log entry to buffer."""
        with self._lock:
            self.buffer.append(entry)
            
            # Check if flush is needed
            current_time = time.time()
            if (len(self.buffer) >= self.max_size or 
                current_time - self.last_flush >= self.flush_interval):
                self._flush()
            
            return True
    
    def add_flush_callback(self, callback: Callable[[List[LogEntry]], None]):
        """Add callback for buffer flush."""
        self.flush_callbacks.append(callback)
    
    def _flush(self):
        """Flush buffer contents."""
        if not self.buffer:
            return
        
        entries = list(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()
        
        # Call flush callbacks
        for callback in self.flush_callbacks:
            try:
                callback(entries)
            except Exception as e:
                # Use standard logging to avoid recursion
                print(f"Log flush callback error: {e}")
    
    def force_flush(self):
        """Force immediate flush."""
        with self._lock:
            self._flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                'size': len(self.buffer),
                'max_size': self.max_size,
                'utilization': len(self.buffer) / self.max_size,
                'last_flush': self.last_flush,
                'flush_interval': self.flush_interval
            }


class CentralizedLogger:
    """
    Centralized logging system.
    
    Provides unified logging across all platform components with
    structured logging, performance monitoring, and advanced features.
    """
    
    def __init__(self, 
                 config: Optional[LoggerConfig] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize centralized logger.
        
        Args:
            config: Logger configuration
            event_bus: Event bus for notifications
        """
        self.config = config or LoggerConfig(name="fine_tune_llm")
        self.event_bus = event_bus or EventBus()
        
        # Component loggers
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Log buffer
        self.buffer = LogBuffer(
            max_size=self.config.buffer_size,
            flush_interval=self.config.flush_interval
        )
        
        # Performance tracker
        self.performance = PerformanceTracker()
        
        # Session tracking
        self.current_session_id = self._generate_session_id()
        self.context_stack: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'total_logs': 0,
            'logs_by_level': defaultdict(int),
            'logs_by_component': defaultdict(int),
            'errors_count': 0,
            'warnings_count': 0,
            'performance_logs': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup logging infrastructure
        self._setup_logging()
        
        # Register flush callback
        self.buffer.add_flush_callback(self._write_buffered_logs)
        
        logger.info("Initialized CentralizedLogger")
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        # Create root logger for platform
        root_logger = logging.getLogger(self.config.name)
        root_logger.setLevel(self.config.level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = StructuredFormatter(
            format_type=self.config.format_type,
            include_extra=True
        )
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.config.level.value)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config.output_file:
            file_path = Path(self.config.output_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.compression:
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.config.output_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
                # Setup compression for rotated files
                file_handler.rotator = self._compress_rotated_file
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.config.output_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
            
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.config.level.value)
            root_logger.addHandler(file_handler)
        
        # Add custom levels
        self._add_custom_levels()
        
        # Store root logger
        self.loggers['root'] = root_logger
    
    def _add_custom_levels(self):
        """Add custom logging levels."""
        custom_levels = [LogLevel.TRACE, LogLevel.PERFORMANCE, LogLevel.AUDIT, LogLevel.SECURITY]
        
        for level in custom_levels:
            logging.addLevelName(level.value, level.name)
    
    def _compress_rotated_file(self, source: str, dest: str):
        """Compress rotated log file."""
        try:
            with open(source, 'rb') as f_in:
                with gzip.open(dest + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(source)
        except Exception as e:
            print(f"Failed to compress log file {source}: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_logger(self, component: str) -> logging.Logger:
        """
        Get logger for component.
        
        Args:
            component: Component name
            
        Returns:
            Logger instance
        """
        if component not in self.loggers:
            logger_name = f"{self.config.name}.{component}"
            component_logger = logging.getLogger(logger_name)
            
            # Ensure it inherits from root logger
            component_logger.parent = self.loggers['root']
            
            self.loggers[component] = component_logger
        
        return self.loggers[component]
    
    def log(self, 
           level: Union[LogLevel, int], 
           message: str,
           component: str = "",
           tags: Optional[List[str]] = None,
           metadata: Optional[Dict[str, Any]] = None,
           exception: Optional[Exception] = None,
           performance_data: Optional[Dict[str, Any]] = None,
           **kwargs) -> bool:
        """
        Log message with enhanced context.
        
        Args:
            level: Log level
            message: Log message
            component: Component name
            tags: Log tags
            metadata: Additional metadata
            exception: Exception object
            performance_data: Performance metrics
            **kwargs: Additional context
            
        Returns:
            True if logged successfully
        """
        try:
            start_time = time.time()
            
            # Convert level if needed
            if isinstance(level, int):
                level = LogLevel(level)
            
            # Get caller information
            caller_info = self._get_caller_info() if self.config.include_caller_info else {}
            
            # Create log entry
            entry = LogEntry(
                level=level,
                message=message,
                component=component,
                tags=tags or [],
                metadata={**(metadata or {}), **kwargs},
                session_id=self.current_session_id,
                performance_data=performance_data,
                **caller_info
            )
            
            # Add context from stack
            if self.context_stack:
                context = {}
                for ctx in self.context_stack:
                    context.update(ctx)
                entry.metadata.update(context)
            
            # Handle exception
            if exception:
                entry.exception = str(exception)
                entry.stack_trace = traceback.format_exc()
            
            # Add to buffer
            if self.config.structured_output:
                self.buffer.add_entry(entry)
            
            # Standard logging
            component_logger = self.get_logger(component or 'general')
            
            # Create log record with extra data
            extra_data = {
                'component': component,
                'tags': tags or [],
                'metadata': entry.metadata,
                'session_id': self.current_session_id,
                'performance_data': performance_data
            }
            
            component_logger.log(level.value, message, extra=extra_data)
            
            # Update statistics
            self._update_stats(level, component)
            
            # Track performance
            duration = time.time() - start_time
            self.performance.track_operation('log_operation', duration)
            
            # Publish log event
            self._publish_log_event(entry)
            
            return True
            
        except Exception as e:
            # Fallback logging to avoid recursion
            print(f"Centralized logging error: {e}")
            return False
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get caller information."""
        try:
            frame = inspect.currentframe()
            # Skip log(), _get_caller_info(), and wrapper frames
            for _ in range(4):
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                return {
                    'module': frame.f_globals.get('__name__', ''),
                    'function': frame.f_code.co_name,
                    'line_number': frame.f_lineno
                }
        except Exception:
            pass
        
        return {'module': '', 'function': '', 'line_number': 0}
    
    def _update_stats(self, level: LogLevel, component: str):
        """Update logging statistics."""
        with self._lock:
            self.stats['total_logs'] += 1
            self.stats['logs_by_level'][level.name] += 1
            self.stats['logs_by_component'][component] += 1
            
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.stats['errors_count'] += 1
            elif level == LogLevel.WARNING:
                self.stats['warnings_count'] += 1
            elif level == LogLevel.PERFORMANCE:
                self.stats['performance_logs'] += 1
    
    def _write_buffered_logs(self, entries: List[LogEntry]):
        """Write buffered log entries."""
        if not entries or not self.config.output_file:
            return
        
        try:
            # Write to structured log file
            structured_file = self.config.output_file.replace('.log', '_structured.jsonl')
            
            with open(structured_file, 'a', encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry.to_json() + '\n')
                    
        except Exception as e:
            print(f"Failed to write buffered logs: {e}")
    
    def _publish_log_event(self, entry: LogEntry):
        """Publish log event to event bus."""
        if entry.level.value >= LogLevel.WARNING.value:  # Only publish warnings and above
            event = Event(
                type=EventType.SYSTEM,
                data={
                    'log_level': entry.level.name,
                    'message': entry.message,
                    'component': entry.component,
                    'timestamp': entry.timestamp.isoformat(),
                    'session_id': entry.session_id
                },
                source="CentralizedLogger"
            )
            
            self.event_bus.publish(event)
    
    # Convenience methods for different log levels
    
    def debug(self, message: str, **kwargs) -> bool:
        """Log debug message."""
        return self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> bool:
        """Log info message."""
        return self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> bool:
        """Log warning message."""
        return self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> bool:
        """Log error message."""
        return self.log(LogLevel.ERROR, message, exception=exception, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> bool:
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, message, exception=exception, **kwargs)
    
    def trace(self, message: str, **kwargs) -> bool:
        """Log trace message."""
        return self.log(LogLevel.TRACE, message, **kwargs)
    
    def performance(self, message: str, metrics: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """Log performance message."""
        return self.log(LogLevel.PERFORMANCE, message, performance_data=metrics, **kwargs)
    
    def audit(self, message: str, **kwargs) -> bool:
        """Log audit message."""
        return self.log(LogLevel.AUDIT, message, **kwargs)
    
    def security(self, message: str, **kwargs) -> bool:
        """Log security message."""
        return self.log(LogLevel.SECURITY, message, **kwargs)
    
    def push_context(self, context: Dict[str, Any]):
        """Push logging context."""
        self.context_stack.append(context)
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop logging context."""
        return self.context_stack.pop() if self.context_stack else None
    
    def set_session_id(self, session_id: str):
        """Set current session ID."""
        self.current_session_id = session_id
    
    def flush(self):
        """Force flush all buffers."""
        self.buffer.force_flush()
        
        # Flush all handlers
        for logger_instance in self.loggers.values():
            for handler in logger_instance.handlers:
                handler.flush()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            return {
                'general': dict(self.stats),
                'buffer': self.buffer.get_stats(),
                'performance': self.performance.get_stats(),
                'loggers': {
                    name: {
                        'level': logger_instance.level,
                        'handlers': len(logger_instance.handlers),
                        'enabled': logger_instance.disabled
                    }
                    for name, logger_instance in self.loggers.items()
                },
                'session_id': self.current_session_id,
                'context_depth': len(self.context_stack)
            }
    
    def configure_component(self, 
                           component: str, 
                           level: Optional[LogLevel] = None,
                           tags: Optional[List[str]] = None) -> bool:
        """
        Configure component-specific logging.
        
        Args:
            component: Component name
            level: Log level for component
            tags: Default tags for component
            
        Returns:
            True if configured successfully
        """
        try:
            component_logger = self.get_logger(component)
            
            if level:
                component_logger.setLevel(level.value)
            
            # Store component configuration
            if not hasattr(component_logger, 'component_config'):
                component_logger.component_config = {}
            
            component_logger.component_config.update({
                'tags': tags or [],
                'level': level
            })
            
            logger.info(f"Configured logging for component: {component}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure component {component}: {e}")
            return False


# Context manager for logging context
class LoggingContext:
    """Context manager for adding logging context."""
    
    def __init__(self, logger: CentralizedLogger, context: Dict[str, Any]):
        """Initialize logging context."""
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        """Enter context."""
        self.logger.push_context(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.pop_context()


# Global logger instance
_centralized_logger = None

def get_centralized_logger() -> CentralizedLogger:
    """Get global centralized logger instance."""
    global _centralized_logger
    if _centralized_logger is None:
        config = LoggerConfig(
            name="fine_tune_llm",
            level=LogLevel.INFO,
            output_file="logs/fine_tune_llm.log",
            format_type=LogFormat.JSON,
            console_output=True,
            structured_output=True
        )
        _centralized_logger = CentralizedLogger(config)
    return _centralized_logger


# Convenience functions

def get_component_logger(component: str) -> logging.Logger:
    """
    Get logger for component.
    
    Args:
        component: Component name
        
    Returns:
        Logger instance
    """
    return get_centralized_logger().get_logger(component)


def log_performance(operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
    """
    Log performance metrics.
    
    Args:
        operation: Operation name
        duration: Operation duration in seconds
        metadata: Additional metadata
    """
    logger = get_centralized_logger()
    performance_data = {
        'operation': operation,
        'duration': duration,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    if metadata:
        performance_data.update(metadata)
    
    logger.performance(
        f"Performance: {operation} completed in {duration:.3f}s",
        metrics=performance_data
    )


def create_logging_context(context: Dict[str, Any]) -> LoggingContext:
    """
    Create logging context manager.
    
    Args:
        context: Context dictionary
        
    Returns:
        Logging context manager
    """
    return LoggingContext(get_centralized_logger(), context)