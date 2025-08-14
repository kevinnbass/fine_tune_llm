"""
Pipeline Configuration System.

This module provides a unified configuration system for the entire pipeline
with dynamic updates, environment support, and comprehensive validation.
"""

import logging
import os
import json
import yaml
import toml
from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import threading
import copy
import hashlib
from collections import defaultdict

from ..core.events import EventBus, Event, EventType
from ..utils.logging import get_centralized_logger
from .manager import ConfigManager
from .validation import ValidationEngine
from .defaults import DefaultManager
from .versioning import ConfigVersionManager

logger = get_centralized_logger().get_logger("pipeline_config")


class ConfigScope(Enum):
    """Configuration scopes."""
    GLOBAL = "global"
    PIPELINE = "pipeline"
    COMPONENT = "component"
    RUNTIME = "runtime"
    USER = "user"
    ENVIRONMENT = "environment"


class ConfigFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class UpdateStrategy(Enum):
    """Configuration update strategies."""
    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    BATCH = "batch"
    CONDITIONAL = "conditional"


@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: Union[str, Path]
    format: ConfigFormat
    scope: ConfigScope
    priority: int = 0
    required: bool = False
    watch: bool = False
    environment_override: bool = True
    validation_schema: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.path = Path(self.path)


@dataclass
class ConfigUpdate:
    """Configuration update request."""
    scope: ConfigScope
    path: str
    value: Any
    source: str = "runtime"
    strategy: UpdateStrategy = UpdateStrategy.IMMEDIATE
    validate: bool = True
    persist: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigProfile:
    """Configuration profile for different environments."""
    name: str
    description: str
    sources: List[ConfigSource] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    active: bool = False


class ConfigWatcher:
    """Watches configuration files for changes."""
    
    def __init__(self, callback: Callable[[Path], None]):
        """Initialize configuration watcher."""
        self.callback = callback
        self.watched_files: Set[Path] = set()
        self.file_mtimes: Dict[Path, float] = {}
        self.watching = False
        self.watch_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    def add_file(self, path: Path):
        """Add file to watch list."""
        with self._lock:
            if path.exists():
                self.watched_files.add(path)
                self.file_mtimes[path] = path.stat().st_mtime
    
    def remove_file(self, path: Path):
        """Remove file from watch list."""
        with self._lock:
            self.watched_files.discard(path)
            self.file_mtimes.pop(path, None)
    
    def start_watching(self, interval: float = 1.0):
        """Start watching files."""
        if self.watching:
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            name="ConfigWatcher"
        )
        self.watch_thread.daemon = True
        self.watch_thread.start()
    
    def stop_watching(self):
        """Stop watching files."""
        self.watching = False
        if self.watch_thread and self.watch_thread.is_alive():
            self.watch_thread.join(timeout=5)
    
    def _watch_loop(self, interval: float):
        """File watching loop."""
        import time
        
        while self.watching:
            try:
                with self._lock:
                    for path in list(self.watched_files):
                        if not path.exists():
                            continue
                        
                        current_mtime = path.stat().st_mtime
                        last_mtime = self.file_mtimes.get(path, 0)
                        
                        if current_mtime > last_mtime:
                            self.file_mtimes[path] = current_mtime
                            try:
                                self.callback(path)
                            except Exception as e:
                                logger.error(f"Config file change callback failed: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Config watcher error: {e}")


class PipelineConfigManager:
    """
    Unified configuration system for the entire pipeline.
    
    Provides centralized configuration management with dynamic updates,
    environment support, and comprehensive validation.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize pipeline configuration manager.
        
        Args:
            event_bus: Event bus for notifications
        """
        self.event_bus = event_bus or EventBus()
        
        # Core managers
        self.config_manager = ConfigManager()
        self.validation_engine = ValidationEngine()
        self.default_manager = DefaultManager()
        self.version_manager = ConfigVersionManager()
        
        # Configuration state
        self.config_data: Dict[ConfigScope, Dict[str, Any]] = {
            scope: {} for scope in ConfigScope
        }
        
        # Configuration sources
        self.sources: Dict[str, ConfigSource] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self.active_profile: Optional[str] = None
        
        # File watcher
        self.watcher = ConfigWatcher(self._on_file_changed)
        
        # Update management
        self.pending_updates: List[ConfigUpdate] = []
        self.update_listeners: List[Callable] = []
        
        # Threading
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'failed_updates': 0,
            'sources_loaded': 0,
            'validation_errors': 0,
            'last_reload': None
        }
        
        # Initialize default configuration
        self._initialize_default_config()
        
        logger.info("Initialized PipelineConfigManager")
    
    def _initialize_default_config(self):
        """Initialize default configuration structure."""
        # Global defaults
        global_defaults = {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'data_dir': os.getenv('DATA_DIR', 'data'),
            'model_dir': os.getenv('MODEL_DIR', 'models'),
            'output_dir': os.getenv('OUTPUT_DIR', 'outputs'),
            'cache_dir': os.getenv('CACHE_DIR', 'cache')
        }
        
        self.config_data[ConfigScope.GLOBAL] = global_defaults
        
        # Pipeline defaults
        pipeline_defaults = {
            'name': 'fine_tune_llm_pipeline',
            'version': '1.0.0',
            'max_workers': int(os.getenv('MAX_WORKERS', '4')),
            'timeout': int(os.getenv('TIMEOUT', '3600')),
            'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3'))
        }
        
        self.config_data[ConfigScope.PIPELINE] = pipeline_defaults
        
        # Component defaults
        component_defaults = {
            'training': {
                'batch_size': 4,
                'learning_rate': 2e-4,
                'epochs': 3,
                'save_steps': 500,
                'eval_steps': 100,
                'logging_steps': 10,
                'warmup_ratio': 0.03,
                'max_grad_norm': 1.0
            },
            'inference': {
                'batch_size': 1,
                'max_length': 2048,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'batch_size': 8,
                'save_predictions': True,
                'compute_confidence': True
            },
            'monitoring': {
                'enabled': True,
                'collection_interval': 10,
                'alert_thresholds': {
                    'cpu_usage': 80,
                    'memory_usage': 85,
                    'error_rate': 5
                }
            }
        }
        
        self.config_data[ConfigScope.COMPONENT] = component_defaults
    
    def add_source(self, source: ConfigSource) -> bool:
        """
        Add configuration source.
        
        Args:
            source: Configuration source
            
        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                # Validate source
                if not source.path.exists() and source.required:
                    logger.error(f"Required configuration file not found: {source.path}")
                    return False
                
                self.sources[source.name] = source
                
                # Load configuration from source
                if source.path.exists():
                    self._load_source(source)
                
                # Add to file watcher if requested
                if source.watch and source.path.exists():
                    self.watcher.add_file(source.path)
                    
                    # Start watcher if not already running
                    if not self.watcher.watching:
                        self.watcher.start_watching()
                
                logger.info(f"Added configuration source: {source.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add configuration source {source.name}: {e}")
            return False
    
    def remove_source(self, source_name: str) -> bool:
        """
        Remove configuration source.
        
        Args:
            source_name: Name of source to remove
            
        Returns:
            True if removed successfully
        """
        try:
            with self._lock:
                if source_name not in self.sources:
                    return False
                
                source = self.sources[source_name]
                
                # Remove from watcher
                if source.watch:
                    self.watcher.remove_file(source.path)
                
                # Remove source
                del self.sources[source_name]
                
                logger.info(f"Removed configuration source: {source_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove configuration source {source_name}: {e}")
            return False
    
    def _load_source(self, source: ConfigSource) -> bool:
        """Load configuration from source."""
        try:
            if not source.path.exists():
                if source.required:
                    logger.error(f"Required configuration file not found: {source.path}")
                    return False
                else:
                    logger.warning(f"Optional configuration file not found: {source.path}")
                    return True
            
            # Load configuration data based on format
            config_data = self._load_config_file(source.path, source.format)
            
            if config_data is None:
                return False
            
            # Apply environment overrides
            if source.environment_override:
                config_data = self._apply_environment_overrides(config_data)
            
            # Validate configuration
            if source.validation_schema:
                validation_result = self.validation_engine.validate(
                    config_data, 
                    source.validation_schema
                )
                
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed for {source.name}")
                    for error in validation_result.errors:
                        logger.error(f"  {error}")
                    self.stats['validation_errors'] += 1
                    return False
            
            # Merge into configuration data
            self._merge_config_data(config_data, source.scope, source.priority)
            
            self.stats['sources_loaded'] += 1
            logger.info(f"Loaded configuration from {source.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration source {source.name}: {e}")
            return False
    
    def _load_config_file(self, path: Path, format: ConfigFormat) -> Optional[Dict[str, Any]]:
        """Load configuration file based on format."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if format == ConfigFormat.YAML:
                    return yaml.safe_load(f)
                elif format == ConfigFormat.JSON:
                    return json.load(f)
                elif format == ConfigFormat.TOML:
                    return toml.load(f)
                elif format == ConfigFormat.INI:
                    import configparser
                    config = configparser.ConfigParser()
                    config.read(path)
                    return {section: dict(config[section]) for section in config.sections()}
                else:
                    logger.error(f"Unsupported configuration format: {format}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to load configuration file {path}: {e}")
            return None
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        def override_recursive(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                env_key = f"{prefix}{key}".upper()
                
                if isinstance(value, dict):
                    override_recursive(value, f"{env_key}_")
                else:
                    env_value = os.getenv(env_key)
                    if env_value is not None:
                        # Try to convert to appropriate type
                        try:
                            if isinstance(value, bool):
                                data[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                            elif isinstance(value, int):
                                data[key] = int(env_value)
                            elif isinstance(value, float):
                                data[key] = float(env_value)
                            else:
                                data[key] = env_value
                        except (ValueError, TypeError):
                            data[key] = env_value
        
        config_copy = copy.deepcopy(config_data)
        override_recursive(config_copy)
        return config_copy
    
    def _merge_config_data(self, new_data: Dict[str, Any], scope: ConfigScope, priority: int):
        """Merge new configuration data into existing configuration."""
        with self._lock:
            # Get existing data for scope
            existing_data = self.config_data[scope]
            
            # Merge based on priority (higher priority overwrites)
            merged_data = self._deep_merge(existing_data, new_data)
            
            # Update configuration data
            self.config_data[scope] = merged_data
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def get(self, path: str, default: Any = None, scope: Optional[ConfigScope] = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Configuration path (dot notation)
            default: Default value if not found
            scope: Specific scope to search (all scopes if None)
            
        Returns:
            Configuration value
        """
        with self._lock:
            if scope:
                # Search specific scope
                return self._get_from_scope(path, self.config_data[scope], default)
            else:
                # Search all scopes in priority order
                search_order = [
                    ConfigScope.RUNTIME,
                    ConfigScope.USER,
                    ConfigScope.ENVIRONMENT,
                    ConfigScope.COMPONENT,
                    ConfigScope.PIPELINE,
                    ConfigScope.GLOBAL
                ]
                
                for search_scope in search_order:
                    value = self._get_from_scope(path, self.config_data[search_scope], None)
                    if value is not None:
                        return value
                
                # Try default manager
                return self.default_manager.get_default(path, fallback=default)
    
    def _get_from_scope(self, path: str, data: Dict[str, Any], default: Any) -> Any:
        """Get value from specific scope data."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, path: str, value: Any, scope: ConfigScope = ConfigScope.RUNTIME, 
            persist: bool = False, validate: bool = True) -> bool:
        """
        Set configuration value.
        
        Args:
            path: Configuration path (dot notation)
            value: Value to set
            scope: Configuration scope
            persist: Whether to persist to file
            validate: Whether to validate the change
            
        Returns:
            True if set successfully
        """
        try:
            update = ConfigUpdate(
                scope=scope,
                path=path,
                value=value,
                persist=persist,
                validate=validate
            )
            
            return self.apply_update(update)
            
        except Exception as e:
            logger.error(f"Failed to set configuration {path}: {e}")
            return False
    
    def apply_update(self, update: ConfigUpdate) -> bool:
        """
        Apply configuration update.
        
        Args:
            update: Configuration update
            
        Returns:
            True if applied successfully
        """
        try:
            with self._lock:
                # Validate update if requested
                if update.validate:
                    if not self._validate_update(update):
                        self.stats['failed_updates'] += 1
                        return False
                
                # Create version before update
                if update.persist:
                    current_config = copy.deepcopy(self.config_data[update.scope])
                    self.version_manager.create_version(
                        config=current_config,
                        description=f"Before update: {update.path}"
                    )
                
                # Apply update based on strategy
                success = self._apply_update_strategy(update)
                
                if success:
                    self.stats['total_updates'] += 1
                    
                    # Persist if requested
                    if update.persist:
                        self._persist_update(update)
                    
                    # Notify listeners
                    self._notify_update_listeners(update)
                    
                    # Publish event
                    self._publish_update_event(update)
                else:
                    self.stats['failed_updates'] += 1
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to apply configuration update: {e}")
            self.stats['failed_updates'] += 1
            return False
    
    def _validate_update(self, update: ConfigUpdate) -> bool:
        """Validate configuration update."""
        try:
            # Basic type validation
            current_value = self.get(update.path, scope=update.scope)
            
            if current_value is not None:
                # Check type compatibility
                if type(current_value) != type(update.value):
                    logger.warning(f"Type mismatch for {update.path}: {type(current_value)} vs {type(update.value)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Update validation failed: {e}")
            return False
    
    def _apply_update_strategy(self, update: ConfigUpdate) -> bool:
        """Apply update based on strategy."""
        if update.strategy == UpdateStrategy.IMMEDIATE:
            return self._apply_immediate_update(update)
        elif update.strategy == UpdateStrategy.DEFERRED:
            self.pending_updates.append(update)
            return True
        elif update.strategy == UpdateStrategy.BATCH:
            self.pending_updates.append(update)
            return True
        else:
            return self._apply_immediate_update(update)
    
    def _apply_immediate_update(self, update: ConfigUpdate) -> bool:
        """Apply immediate configuration update."""
        try:
            # Get scope data
            scope_data = self.config_data[update.scope]
            
            # Navigate to parent of target key
            keys = update.path.split('.')
            current = scope_data
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set value
            current[keys[-1]] = update.value
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply immediate update: {e}")
            return False
    
    def _persist_update(self, update: ConfigUpdate):
        """Persist update to configuration file."""
        # Find source file for the scope
        for source in self.sources.values():
            if source.scope == update.scope and source.path.exists():
                try:
                    # Save current configuration to file
                    config_data = self.config_data[update.scope]
                    self._save_config_file(source.path, config_data, source.format)
                    logger.info(f"Persisted configuration update to {source.path}")
                    break
                except Exception as e:
                    logger.error(f"Failed to persist update to {source.path}: {e}")
    
    def _save_config_file(self, path: Path, data: Dict[str, Any], format: ConfigFormat):
        """Save configuration to file."""
        with open(path, 'w', encoding='utf-8') as f:
            if format == ConfigFormat.YAML:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format == ConfigFormat.JSON:
                json.dump(data, f, indent=2, default=str)
            elif format == ConfigFormat.TOML:
                toml.dump(data, f)
            else:
                raise ValueError(f"Unsupported format for saving: {format}")
    
    def _on_file_changed(self, path: Path):
        """Handle configuration file change."""
        logger.info(f"Configuration file changed: {path}")
        
        # Find source for this path
        source = None
        for src in self.sources.values():
            if src.path == path:
                source = src
                break
        
        if source:
            self._load_source(source)
            self._publish_reload_event(source.name)
    
    def reload_all(self) -> bool:
        """Reload all configuration sources."""
        try:
            with self._lock:
                success_count = 0
                
                for source in self.sources.values():
                    if self._load_source(source):
                        success_count += 1
                
                self.stats['last_reload'] = datetime.now(timezone.utc)
                
                logger.info(f"Reloaded {success_count}/{len(self.sources)} configuration sources")
                return success_count == len(self.sources)
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def add_update_listener(self, listener: Callable[[ConfigUpdate], None]):
        """Add configuration update listener."""
        self.update_listeners.append(listener)
    
    def remove_update_listener(self, listener: Callable[[ConfigUpdate], None]):
        """Remove configuration update listener."""
        if listener in self.update_listeners:
            self.update_listeners.remove(listener)
    
    def _notify_update_listeners(self, update: ConfigUpdate):
        """Notify all update listeners."""
        for listener in self.update_listeners:
            try:
                listener(update)
            except Exception as e:
                logger.error(f"Update listener failed: {e}")
    
    def _publish_update_event(self, update: ConfigUpdate):
        """Publish configuration update event."""
        event = Event(
            type=EventType.CONFIGURATION_CHANGED,
            data={
                'scope': update.scope.value,
                'path': update.path,
                'value': str(update.value),
                'source': update.source,
                'timestamp': update.timestamp.isoformat()
            },
            source="PipelineConfigManager"
        )
        
        self.event_bus.publish(event)
    
    def _publish_reload_event(self, source_name: str):
        """Publish configuration reload event."""
        event = Event(
            type=EventType.CONFIGURATION_RELOADED,
            data={
                'source': source_name,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            source="PipelineConfigManager"
        )
        
        self.event_bus.publish(event)
    
    def get_all_config(self, flatten: bool = False) -> Dict[str, Any]:
        """
        Get all configuration data.
        
        Args:
            flatten: Whether to flatten the structure
            
        Returns:
            Complete configuration data
        """
        with self._lock:
            if flatten:
                result = {}
                for scope, data in self.config_data.items():
                    self._flatten_dict(data, result, f"{scope.value}.")
                return result
            else:
                return {
                    scope.value: copy.deepcopy(data) 
                    for scope, data in self.config_data.items()
                }
    
    def _flatten_dict(self, data: Dict[str, Any], result: Dict[str, Any], prefix: str = ""):
        """Flatten nested dictionary."""
        for key, value in data.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict):
                self._flatten_dict(value, result, f"{full_key}.")
            else:
                result[full_key] = value
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        with self._lock:
            return {
                'general': dict(self.stats),
                'sources': {
                    name: {
                        'path': str(source.path),
                        'format': source.format.value,
                        'scope': source.scope.value,
                        'priority': source.priority,
                        'required': source.required,
                        'watch': source.watch,
                        'exists': source.path.exists()
                    }
                    for name, source in self.sources.items()
                },
                'scopes': {
                    scope.value: len(data) for scope, data in self.config_data.items()
                },
                'pending_updates': len(self.pending_updates),
                'update_listeners': len(self.update_listeners),
                'watcher_active': self.watcher.watching,
                'watched_files': len(self.watcher.watched_files)
            }


# Global pipeline config manager
_pipeline_config = None

def get_pipeline_config() -> PipelineConfigManager:
    """Get global pipeline configuration manager."""
    global _pipeline_config
    if _pipeline_config is None:
        _pipeline_config = PipelineConfigManager()
    return _pipeline_config


# Convenience functions

def get_config(path: str, default: Any = None, scope: Optional[ConfigScope] = None) -> Any:
    """
    Get configuration value.
    
    Args:
        path: Configuration path
        default: Default value
        scope: Specific scope
        
    Returns:
        Configuration value
    """
    config = get_pipeline_config()
    return config.get(path, default, scope)


def set_config(path: str, value: Any, scope: ConfigScope = ConfigScope.RUNTIME, 
               persist: bool = False) -> bool:
    """
    Set configuration value.
    
    Args:
        path: Configuration path
        value: Value to set
        scope: Configuration scope
        persist: Whether to persist
        
    Returns:
        True if set successfully
    """
    config = get_pipeline_config()
    return config.set(path, value, scope, persist)


def add_config_source(name: str, path: Union[str, Path], format: ConfigFormat, 
                     scope: ConfigScope = ConfigScope.COMPONENT, **kwargs) -> bool:
    """
    Add configuration source.
    
    Args:
        name: Source name
        path: File path
        format: Configuration format
        scope: Configuration scope
        **kwargs: Additional source options
        
    Returns:
        True if added successfully
    """
    config = get_pipeline_config()
    source = ConfigSource(name=name, path=path, format=format, scope=scope, **kwargs)
    return config.add_source(source)


def reload_config() -> bool:
    """Reload all configuration sources."""
    config = get_pipeline_config()
    return config.reload_all()