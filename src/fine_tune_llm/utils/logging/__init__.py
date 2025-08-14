"""
Centralized Logging Module.

This module provides unified logging infrastructure across all platform
components with structured logging, performance monitoring, and advanced features.
"""

# New centralized logging system
from .centralized_logger import (
    LogLevel,
    LogFormat,
    LogEntry,
    LoggerConfig,
    CentralizedLogger,
    LoggingContext,
    get_centralized_logger,
    get_component_logger,
    log_performance,
    create_logging_context
)

# Legacy logging utilities (for backwards compatibility)
try:
    from .manager import LoggerManager
    from .setup import setup_logging, get_logger
    from .formatters import CustomFormatter, JSONFormatter
    from .handlers import RotatingFileHandler, StreamHandler
    
    _legacy_imports = [
        'LoggerManager', 'setup_logging', 'get_logger',
        'CustomFormatter', 'JSONFormatter', 'RotatingFileHandler', 'StreamHandler'
    ]
except ImportError:
    _legacy_imports = []

__all__ = [
    # New centralized logging system
    'LogLevel',
    'LogFormat',
    'LogEntry',
    'LoggerConfig',
    'CentralizedLogger',
    'LoggingContext',
    'get_centralized_logger',
    'get_component_logger',
    'log_performance',
    'create_logging_context'
] + _legacy_imports