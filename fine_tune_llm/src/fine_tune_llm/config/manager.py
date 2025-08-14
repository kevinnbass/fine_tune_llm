"""
Configuration Manager

Unified configuration management with validation, hot-reloading,
and environment-specific configurations.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .validation import ConfigValidator, ValidationError
from .schemas import BaseConfig

logger = logging.getLogger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """Handle configuration file changes for hot-reloading."""
    
    def __init__(self, manager: 'ConfigManager'):
        self.manager = manager
        
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path == str(self.manager.config_path):
            logger.info(f"Configuration file changed: {event.src_path}")
            self.manager._reload_config()


@dataclass
class ConfigMetadata:
    """Configuration metadata tracking."""
    version: str
    last_modified: float
    source_path: str
    environment: str
    validation_errors: List[str]


class ConfigManager:
    """
    Unified configuration management system.
    
    Features:
    - Multiple format support (YAML, JSON)
    - Environment-specific configurations
    - Hot-reloading capability
    - Validation and schema enforcement
    - Configuration versioning and rollback
    """
    
    def __init__(self, 
                 config_path: Union[str, Path],
                 schema_class: Optional[type] = None,
                 enable_hot_reload: bool = False,
                 environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
            schema_class: Configuration schema class for validation
            enable_hot_reload: Enable automatic config reloading
            environment: Environment name (dev, staging, prod)
        """
        self.config_path = Path(config_path)
        self.schema_class = schema_class or BaseConfig
        self.enable_hot_reload = enable_hot_reload
        self.environment = environment or os.getenv('ENVIRONMENT', 'dev')
        
        # Internal state
        self._config: Dict[str, Any] = {}
        self._metadata: Optional[ConfigMetadata] = None
        self._validator = ConfigValidator()
        self._observer: Optional[Observer] = None
        self._change_callbacks: List[callable] = []
        
        # Load initial configuration
        self.load_config()
        
        # Setup hot-reloading if enabled
        if self.enable_hot_reload:
            self._setup_hot_reload()
            
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file with environment overrides.
        
        Returns:
            Loaded configuration dictionary
        """
        try:
            # Load base configuration
            base_config = self._load_file(self.config_path)
            
            # Apply environment-specific overrides
            env_config = self._load_environment_config()
            if env_config:
                base_config = self._merge_configs(base_config, env_config)
            
            # Validate configuration
            validation_errors = self._validator.validate(base_config, self.schema_class)
            
            # Update metadata
            self._metadata = ConfigMetadata(
                version=base_config.get('version', '1.0.0'),
                last_modified=self.config_path.stat().st_mtime,
                source_path=str(self.config_path),
                environment=self.environment,
                validation_errors=validation_errors
            )
            
            # Store validated config
            self._config = base_config
            
            if validation_errors:
                logger.warning(f"Configuration validation warnings: {validation_errors}")
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set value
        config[keys[-1]] = value
        
        # Trigger change callbacks
        self._notify_change_callbacks(key, value)
        
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
            
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save to (defaults to original path)
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            if save_path.suffix.lower() in ['.yml', '.yaml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration format: {save_path.suffix}")
                
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
            
    def add_change_callback(self, callback: callable) -> None:
        """
        Add callback for configuration changes.
        
        Args:
            callback: Function to call on config changes
        """
        self._change_callbacks.append(callback)
        
    def remove_change_callback(self, callback: callable) -> None:
        """
        Remove configuration change callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            
    def get_metadata(self) -> ConfigMetadata:
        """Get configuration metadata."""
        return self._metadata
        
    def validate_current_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors
        """
        return self._validator.validate(self._config, self.schema_class)
        
    def _load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {path.suffix}")
                    
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file {path}: {e}")
            
    def _load_environment_config(self) -> Optional[Dict[str, Any]]:
        """Load environment-specific configuration overrides."""
        env_path = self.config_path.parent / f"{self.config_path.stem}.{self.environment}.yaml"
        
        if env_path.exists():
            logger.info(f"Loading environment config: {env_path}")
            return self._load_file(env_path)
            
        return None
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _setup_hot_reload(self) -> None:
        """Setup file watching for hot-reloading."""
        try:
            self._observer = Observer()
            handler = ConfigChangeHandler(self)
            
            self._observer.schedule(
                handler,
                path=str(self.config_path.parent),
                recursive=False
            )
            
            self._observer.start()
            logger.info("Hot-reloading enabled for configuration")
            
        except Exception as e:
            logger.warning(f"Failed to setup hot-reloading: {e}")
            
    def _reload_config(self) -> None:
        """Reload configuration from file."""
        try:
            old_config = self._config.copy()
            self.load_config()
            
            # Notify callbacks of changes
            self._notify_change_callbacks('__reload__', self._config)
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            
    def _notify_change_callbacks(self, key: str, value: Any) -> None:
        """Notify all change callbacks."""
        for callback in self._change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")
                
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if self._observer:
            self._observer.stop()
            self._observer.join()


class ConfigurationError(Exception):
    """Configuration-related error."""
    pass