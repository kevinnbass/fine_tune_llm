"""
Logging setup and configuration utilities.

This module provides functions for setting up logging configuration
across the entire platform with consistent formatting and handling.
"""

import logging
import logging.config
from typing import Dict, Any, Optional
from pathlib import Path
import sys

from .formatters import CustomFormatter, JSONFormatter
from .handlers import setup_file_handler, setup_console_handler


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_format: str = "standard",
    enable_json_logs: bool = False,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> None:
    """
    Set up logging configuration for the entire platform.
    
    Args:
        config: Optional logging configuration dictionary
        log_level: Default logging level
        log_dir: Directory for log files
        log_format: Format style ('standard', 'detailed', 'minimal')
        enable_json_logs: Enable JSON formatted logs
        enable_file_logging: Enable file logging
        enable_console_logging: Enable console logging
    """
    if config:
        # Use provided configuration
        logging.config.dictConfig(config)
        return
    
    # Create default configuration
    log_dir = log_dir or Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Set up formatters
    if enable_json_logs:
        formatter = JSONFormatter()
    else:
        formatter = CustomFormatter(format_style=log_format)
    
    # Set up console handler
    if enable_console_logging:
        console_handler = setup_console_handler(formatter)
        root_logger.addHandler(console_handler)
    
    # Set up file handlers
    if enable_file_logging:
        # Main log file
        main_file_handler = setup_file_handler(
            log_dir / "fine_tune_llm.log",
            formatter,
            level=log_level
        )
        root_logger.addHandler(main_file_handler)
        
        # Error log file
        error_file_handler = setup_file_handler(
            log_dir / "error.log", 
            formatter,
            level="ERROR"
        )
        root_logger.addHandler(error_file_handler)
    
    # Configure specific loggers
    _configure_component_loggers(log_level)
    
    logging.info("Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def _configure_component_loggers(default_level: str) -> None:
    """Configure loggers for specific components."""
    
    # Component-specific logging levels
    component_levels = {
        'fine_tune_llm.training': 'INFO',
        'fine_tune_llm.inference': 'INFO', 
        'fine_tune_llm.evaluation': 'INFO',
        'fine_tune_llm.data': 'WARNING',  # Reduce noise from data processing
        'fine_tune_llm.models': 'INFO',
        'fine_tune_llm.config': 'WARNING',
        'fine_tune_llm.services': 'INFO',
        'fine_tune_llm.monitoring': 'INFO',
        
        # External libraries
        'transformers': 'WARNING',
        'torch': 'WARNING',
        'accelerate': 'WARNING',
        'datasets': 'WARNING',
        'urllib3': 'WARNING',
        'requests': 'WARNING',
    }
    
    for component, level in component_levels.items():
        logger = logging.getLogger(component)
        logger.setLevel(getattr(logging, level.upper()))


def setup_development_logging() -> None:
    """Set up logging configuration optimized for development."""
    setup_logging(
        log_level="DEBUG",
        log_format="detailed",
        enable_json_logs=False,
        enable_file_logging=True,
        enable_console_logging=True
    )


def setup_production_logging(log_dir: Path) -> None:
    """
    Set up logging configuration optimized for production.
    
    Args:
        log_dir: Directory for production log files
    """
    setup_logging(
        log_level="INFO",
        log_dir=log_dir,
        log_format="standard", 
        enable_json_logs=True,
        enable_file_logging=True,
        enable_console_logging=False
    )


def setup_testing_logging() -> None:
    """Set up logging configuration for testing (minimal output)."""
    setup_logging(
        log_level="ERROR",
        log_format="minimal",
        enable_json_logs=False,
        enable_file_logging=False,
        enable_console_logging=True
    )


def get_logging_config() -> Dict[str, Any]:
    """
    Get current logging configuration.
    
    Returns:
        Dictionary representation of current logging config
    """
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {},
        'handlers': {},
        'loggers': {},
        'root': {}
    }
    
    # Get root logger info
    root_logger = logging.getLogger()
    config['root'] = {
        'level': logging.getLevelName(root_logger.level),
        'handlers': [h.__class__.__name__ for h in root_logger.handlers]
    }
    
    # Get handler info
    for handler in root_logger.handlers:
        handler_name = handler.__class__.__name__
        config['handlers'][handler_name] = {
            'level': logging.getLevelName(handler.level),
            'formatter': handler.formatter.__class__.__name__ if handler.formatter else None
        }
    
    return config


def silence_external_loggers() -> None:
    """Silence noisy external library loggers."""
    noisy_loggers = [
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_base',
        'transformers.modeling_utils',
        'torch.distributed',
        'torch.nn.parallel',
        'accelerate',
        'datasets.arrow_dataset',
        'datasets.builder',
        'PIL.PngImagePlugin',
        'matplotlib.font_manager',
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool'
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)


def enable_debug_logging(component: Optional[str] = None) -> None:
    """
    Enable debug logging for a specific component or all components.
    
    Args:
        component: Optional component name (None for all)
    """
    if component:
        logger = logging.getLogger(f'fine_tune_llm.{component}')
        logger.setLevel(logging.DEBUG)
        logging.info(f"Enabled debug logging for {component}")
    else:
        root_logger = logging.getLogger('fine_tune_llm')
        root_logger.setLevel(logging.DEBUG)
        logging.info("Enabled debug logging for all components")


def disable_debug_logging(component: Optional[str] = None) -> None:
    """
    Disable debug logging for a specific component or all components.
    
    Args:
        component: Optional component name (None for all)
    """
    if component:
        logger = logging.getLogger(f'fine_tune_llm.{component}')
        logger.setLevel(logging.INFO)
        logging.info(f"Disabled debug logging for {component}")
    else:
        root_logger = logging.getLogger('fine_tune_llm')
        root_logger.setLevel(logging.INFO)
        logging.info("Disabled debug logging for all components")


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import sys
    
    logger = get_logger(__name__)
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Architecture: {platform.architecture()}")
    logger.info(f"  Processor: {platform.processor()}")
    
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        logger.info("  PyTorch: Not installed")
    
    try:
        import transformers
        logger.info(f"  Transformers: {transformers.__version__}")
    except ImportError:
        logger.info("  Transformers: Not installed")


def configure_library_logging() -> None:
    """Configure logging for external libraries."""
    
    # Configure transformers logging
    try:
        import transformers
        transformers.logging.set_verbosity_warning()
    except ImportError:
        pass
    
    # Configure datasets logging  
    try:
        import datasets
        datasets.logging.set_verbosity_warning()
    except ImportError:
        pass
    
    # Configure torch distributed logging
    try:
        import torch.distributed
        if torch.distributed.is_available():
            logging.getLogger('torch.distributed').setLevel(logging.WARNING)
    except ImportError:
        pass