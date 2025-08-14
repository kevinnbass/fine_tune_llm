"""
Configuration manager with hot-reload and environment support.

This module provides a centralized configuration management system
with hot-reload capabilities, environment variable integration,
and schema validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from threading import Lock
from dataclasses import dataclass
import logging
from datetime import datetime, timezone

from .validation import ValidationEngine
from .schemas import SchemaRegistry
from ..core.events import EventBus
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Represents a configuration change."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str


class ConfigManager:
    """
    Centralized configuration manager with hot-reload and validation.
    
    Provides configuration loading, validation, hot-reload, and
    environment variable integration capabilities.
    """
    
    def __init__(self, 
                 config_dir: Optional[Path] = None,
                 event_bus: Optional[EventBus] = None,
                 enable_hot_reload: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            event_bus: Event bus for configuration change notifications
            enable_hot_reload: Enable file watching for hot reload
        """
        self.config_dir = config_dir or Path("configs")
        self.event_bus = event_bus or EventBus()
        self.enable_hot_reload = enable_hot_reload
        
        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._file_configs: Dict[str, Dict[str, Any]] = {}
        self._env_overrides: Dict[str, Any] = {}
        self._default_configs: Dict[str, Any] = {}
        
        # Validation and schema
        self.validation_engine = ValidationEngine()
        self.schema_registry = SchemaRegistry()
        
        # Hot reload support
        self._file_watchers: Dict[str, Any] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._config_lock = Lock()
        
        # Change tracking
        self._change_history: List[ConfigChange] = []
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # Load initial configuration
        self._load_initial_config()
        self._load_environment_overrides()
        
        if self.enable_hot_reload:
            self._setup_file_watching()
        
        logger.info(f"ConfigManager initialized with config_dir: {self.config_dir}")
    
    def _load_initial_config(self):
        """Load initial configuration from files."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return
        
        # Load configuration files
        config_files = [
            "config.yaml",
            "config.json", 
            "llm_lora.yaml",
            "labels.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    config_data = self._load_config_file(config_path)
                    self._file_configs[config_file] = config_data
                    logger.info(f"Loaded config file: {config_file}")
                except Exception as e:
                    logger.error(f"Failed to load config file {config_file}: {e}")
        
        # Merge all configurations
        self._merge_configurations()
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a single file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        env_prefix = "FINE_TUNE_LLM_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Keep as string if not valid JSON
                    parsed_value = value
                
                self._env_overrides[config_key] = parsed_value
                logger.info(f"Environment override: {config_key} = {parsed_value}")
    
    def _merge_configurations(self):
        """Merge all configuration sources in priority order."""
        with self._config_lock:
            # Start with defaults
            merged_config = self._default_configs.copy()
            
            # Add file configurations
            for file_name, file_config in self._file_configs.items():
                merged_config = self._deep_merge(merged_config, file_config)
            
            # Apply environment overrides
            for key, value in self._env_overrides.items():
                self._set_nested_value(merged_config, key, value)
            
            # Validate merged configuration
            if self.schema_registry.has_schemas():
                validation_result = self.validation_engine.validate_config(
                    merged_config, self.schema_registry
                )
                if not validation_result.is_valid:
                    logger.warning(f"Configuration validation warnings: {validation_result.errors}")
            
            # Update configuration atomically
            old_config = self._config.copy()
            self._config = merged_config
            
            # Track changes and notify
            self._track_config_changes(old_config, merged_config)
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _track_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Track and notify about configuration changes."""
        changes = self._find_changes(old_config, new_config)
        
        for change in changes:
            self._change_history.append(change)
            
            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(change.key, change.old_value, change.new_value)
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")
            
            # Publish event
            self.event_bus.publish('ConfigurationChanged', {
                'key': change.key,
                'old_value': change.old_value,
                'new_value': change.new_value,
                'timestamp': change.timestamp.isoformat(),
                'source': change.source
            })
    
    def _find_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], prefix: str = "") -> List[ConfigChange]:
        """Find all changes between two configurations."""
        changes = []
        
        # Find added/changed keys
        for key, new_value in new_config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                changes.append(ConfigChange(
                    key=full_key,
                    old_value=None,
                    new_value=new_value,
                    timestamp=datetime.now(timezone.utc),
                    source='merge'
                ))
            elif old_config[key] != new_value:
                if isinstance(old_config[key], dict) and isinstance(new_value, dict):
                    # Recursively check nested changes
                    nested_changes = self._find_changes(old_config[key], new_value, full_key)
                    changes.extend(nested_changes)
                else:
                    changes.append(ConfigChange(
                        key=full_key,
                        old_value=old_config[key],
                        new_value=new_value,
                        timestamp=datetime.now(timezone.utc),
                        source='merge'
                    ))
        
        # Find removed keys
        for key, old_value in old_config.items():
            if key not in new_config:
                full_key = f"{prefix}.{key}" if prefix else key
                changes.append(ConfigChange(
                    key=full_key,
                    old_value=old_value,
                    new_value=None,
                    timestamp=datetime.now(timezone.utc),
                    source='merge'
                ))
        
        return changes
    
    def _setup_file_watching(self):
        """Set up file watching for hot reload."""
        try:
            import watchdog
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager
                
                def on_modified(self, event):
                    if not event.is_directory:
                        file_path = Path(event.src_path)
                        if file_path.suffix.lower() in ['.yaml', '.yml', '.json']:
                            self.config_manager._reload_config_file(file_path)
            
            self._observer = Observer()
            self._observer.schedule(
                ConfigFileHandler(self),
                str(self.config_dir),
                recursive=False
            )
            self._observer.start()
            
            logger.info("File watching enabled for hot reload")
            
        except ImportError:
            logger.warning("Watchdog not available, hot reload disabled")
            self.enable_hot_reload = False
    
    def _reload_config_file(self, file_path: Path):
        """Reload a specific configuration file."""
        file_name = file_path.name
        
        try:
            config_data = self._load_config_file(file_path)
            
            with self._config_lock:
                self._file_configs[file_name] = config_data
            
            self._merge_configurations()
            logger.info(f"Reloaded config file: {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to reload config file {file_name}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any, source: str = 'runtime'):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            source: Source of the change
        """
        with self._config_lock:
            old_value = self.get(key)
            
            # Set the value
            keys = key.split('.')
            current = self._config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                elif not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            
            # Track change
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value,
                timestamp=datetime.now(timezone.utc),
                source=source
            )
            
            self._change_history.append(change)
            
            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(key, old_value, value)
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")
            
            # Publish event
            self.event_bus.publish('ConfigurationChanged', {
                'key': key,
                'old_value': old_value,
                'new_value': value,
                'timestamp': change.timestamp.isoformat(),
                'source': source
            })
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        with self._config_lock:
            return self._config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service."""
        service_config = self.get_section('services')
        return service_config.get(service_name, {})
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Add callback for configuration changes."""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Remove configuration change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def get_change_history(self, limit: Optional[int] = None) -> List[ConfigChange]:
        """Get configuration change history."""
        history = self._change_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def export_config(self, output_path: Path, format: str = 'yaml'):
        """Export current configuration to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.get_all()
        
        with open(output_path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_data, f, indent=2, default=str)
            else:
                raise ConfigurationError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported configuration to {output_path}")
    
    def reload_from_files(self):
        """Manually reload configuration from files."""
        self._load_initial_config()
        logger.info("Configuration reloaded from files")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration against schemas."""
        if not self.schema_registry.has_schemas():
            return {'valid': True, 'message': 'No schemas registered'}
        
        result = self.validation_engine.validate_config(
            self._config, self.schema_registry
        )
        
        return {
            'valid': result.is_valid,
            'errors': result.errors,
            'warnings': result.warnings
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_observer') and self._observer:
            self._observer.stop()
            self._observer.join()
        
        self._change_callbacks.clear()
        logger.info("ConfigManager cleaned up")
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction