"""
Utilities package providing common functionality across the platform.

This package contains utility modules for logging, I/O operations,
decorators, validation, and helper functions.
"""

from .logging import LoggerManager, get_logger, setup_logging
from .io import FileManager, safe_write, safe_read
from .decorators import retry, timeout, cache, validate_types
from .validators import validate_config, validate_data_schema
from .helpers import ensure_dir, get_timestamp, format_duration

__all__ = [
    # Logging
    'LoggerManager',
    'get_logger',
    'setup_logging',
    
    # I/O Operations
    'FileManager',
    'safe_write',
    'safe_read',
    
    # Decorators
    'retry',
    'timeout', 
    'cache',
    'validate_types',
    
    # Validators
    'validate_config',
    'validate_data_schema',
    
    # Helpers
    'ensure_dir',
    'get_timestamp',
    'format_duration',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Utility functions and classes for LLM fine-tuning platform'