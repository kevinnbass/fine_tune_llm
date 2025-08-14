"""
Unified Configuration System for all components.

This module provides a centralized configuration system that integrates
all component configurations with consistent validation and management.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type, TypeVar
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .parser import ConfigurationParser, ParseResult, ParserConfig
from .manager import ConfigManager
from .validation import ValidationEngine
from .versioning import ConfigVersionManager
from .secrets import SecretManager
from ..core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ComponentConfig:
    """Component-specific configuration."""
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedConfigData:
    """Complete unified configuration data."""
    # Core platform config
    platform: Dict[str, Any] = field(default_factory=dict)
    
    # Component configurations
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    
    # Global settings
    global_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Environment-specific settings
    environments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedConfigManager:
    """
    Unified configuration manager for all platform components.
    
    Provides centralized configuration management with component registration,
    validation, versioning, and hot-reload capabilities.
    """
    
    def __init__(self, 
                 config_dir: Optional[Path] = None,
                 environment: str = "default",
                 enable_hot_reload: bool = True):
        """
        Initialize unified configuration manager.
        
        Args:
            config_dir: Configuration directory
            environment: Current environment name
            enable_hot_reload: Enable hot-reload of configurations
        """
        self.config_dir = config_dir or Path("configs")
        self.environment = environment
        self.enable_hot_reload = enable_hot_reload
        
        # Core managers
        self.parser = ConfigurationParser()
        self.config_manager = ConfigManager()
        self.validation_engine = ValidationEngine()
        self.version_manager = ConfigVersionManager()
        self.secret_manager = SecretManager()
        
        # Unified configuration
        self.unified_config = UnifiedConfigData()
        
        # Component registry
        self.registered_components: Dict[str, Type] = {}
        self.component_schemas: Dict[str, Dict[str, Any]] = {}
        self.component_defaults: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load configurations
        self._initialize()
        
        logger.info(f"Initialized UnifiedConfigManager for environment: {environment}")
    
    def _initialize(self):
        """Initialize configuration system."""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Load platform configuration
            self._load_platform_config()
            
            # Load component configurations
            self._load_component_configs()
            
            # Apply environment-specific overrides
            self._apply_environment_config()
            
            # Validate all configurations
            self._validate_all_configs()
            
            # Set up hot-reload if enabled
            if self.enable_hot_reload:
                self._setup_hot_reload()
                
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration system: {e}")
    
    def register_component(self, 
                          name: str,
                          component_class: Type,
                          schema: Optional[Dict[str, Any]] = None,
                          defaults: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register component with configuration system.
        
        Args:
            name: Component name
            component_class: Component class
            schema: Configuration schema
            defaults: Default configuration values
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.registered_components[name] = component_class
                
                if schema:
                    self.component_schemas[name] = schema
                
                if defaults:
                    self.component_defaults[name] = defaults
                
                # Load component configuration if not already loaded
                if name not in self.unified_config.components:
                    self._load_component_config(name)
            
            logger.info(f"Registered component: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {name}: {e}")
            return False
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister component.
        
        Args:
            name: Component name
            
        Returns:
            True if unregistered successfully
        """
        try:
            with self._lock:
                self.registered_components.pop(name, None)
                self.component_schemas.pop(name, None)
                self.component_defaults.pop(name, None)
                self.unified_config.components.pop(name, None)
            
            logger.info(f"Unregistered component: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister component {name}: {e}")
            return False
    
    def get_component_config(self, name: str) -> Optional[ComponentConfig]:
        """
        Get component configuration.
        
        Args:
            name: Component name
            
        Returns:
            Component configuration or None
        """
        with self._lock:
            return self.unified_config.components.get(name)
    
    def set_component_config(self, 
                           name: str, 
                           config: Union[ComponentConfig, Dict[str, Any]]) -> bool:
        """
        Set component configuration.
        
        Args:
            name: Component name
            config: Component configuration
            
        Returns:
            True if set successfully
        """
        try:
            with self._lock:
                if isinstance(config, dict):
                    # Convert dict to ComponentConfig
                    existing = self.unified_config.components.get(name)
                    if existing:
                        existing.config.update(config)
                        component_config = existing
                    else:
                        component_config = ComponentConfig(
                            name=name,
                            config=config.copy()
                        )
                else:
                    component_config = config
                
                # Validate configuration
                errors = self._validate_component_config(name, component_config.config)
                if errors:
                    raise ValidationError(f"Configuration validation failed for {name}: {errors}")
                
                # Update configuration
                self.unified_config.components[name] = component_config
                
                # Create version snapshot
                self.version_manager.create_version(
                    component_config.config,
                    change_type="update",
                    description=f"Updated configuration for {name}"
                )
            
            logger.info(f"Updated configuration for component: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set component config for {name}: {e}")
            return False
    
    def get_config_value(self, 
                        path: str, 
                        default: Any = None,
                        component: Optional[str] = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Configuration path (dot-separated)
            default: Default value if not found
            component: Specific component name
            
        Returns:
            Configuration value
        """
        with self._lock:
            try:
                if component:
                    # Get from specific component
                    comp_config = self.unified_config.components.get(component)
                    if comp_config:
                        return self._get_nested_value(comp_config.config, path, default)
                else:
                    # Search in order: platform, global_settings, components
                    sources = [
                        self.unified_config.platform,
                        self.unified_config.global_settings
                    ]
                    
                    for source in sources:
                        value = self._get_nested_value(source, path, None)
                        if value is not None:
                            return value
                
                return default
                
            except Exception as e:
                logger.error(f"Error getting config value {path}: {e}")
                return default
    
    def set_config_value(self, 
                        path: str, 
                        value: Any,
                        component: Optional[str] = None) -> bool:
        """
        Set configuration value by path.
        
        Args:
            path: Configuration path (dot-separated)
            value: Value to set
            component: Specific component name
            
        Returns:
            True if set successfully
        """
        try:
            with self._lock:
                if component:
                    # Set in specific component
                    if component not in self.unified_config.components:
                        self.unified_config.components[component] = ComponentConfig(name=component)
                    
                    comp_config = self.unified_config.components[component]
                    self._set_nested_value(comp_config.config, path, value)
                else:
                    # Set in global settings
                    self._set_nested_value(self.unified_config.global_settings, path, value)
            
            logger.debug(f"Set config value {path} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config value {path}: {e}")
            return False
    
    def reload_config(self, component: Optional[str] = None) -> bool:
        """
        Reload configuration from files.
        
        Args:
            component: Specific component to reload (all if None)
            
        Returns:
            True if reloaded successfully
        """
        try:
            with self._lock:
                if component:
                    # Reload specific component
                    self._load_component_config(component)
                else:
                    # Reload all configurations
                    self._load_platform_config()
                    self._load_component_configs()
                    self._apply_environment_config()
                
                # Validate after reload
                self._validate_all_configs()
            
            logger.info(f"Reloaded configuration for: {component or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def export_config(self, export_path: Path, format: str = "yaml") -> bool:
        """
        Export unified configuration to file.
        
        Args:
            export_path: Export file path
            format: Export format (yaml, json, toml)
            
        Returns:
            True if exported successfully
        """
        try:
            # Convert to exportable format
            export_data = {
                'platform': self.unified_config.platform,
                'global_settings': self.unified_config.global_settings,
                'components': {
                    name: {
                        'name': comp.name,
                        'version': comp.version,
                        'enabled': comp.enabled,
                        'config': comp.config,
                        'dependencies': comp.dependencies,
                        'metadata': comp.metadata
                    }
                    for name, comp in self.unified_config.components.items()
                },
                'environments': self.unified_config.environments,
                'metadata': {
                    **self.unified_config.metadata,
                    'exported_at': datetime.now().isoformat(),
                    'environment': self.environment
                }
            }
            
            # Write to file
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                import json
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'yaml':
                import yaml
                with open(export_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            elif format.lower() == 'toml':
                import toml
                with open(export_path, 'w') as f:
                    toml.dump(export_data, f)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported configuration to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """
        Import configuration from file.
        
        Args:
            import_path: Import file path
            
        Returns:
            True if imported successfully
        """
        try:
            # Parse imported configuration
            result = self.parser.parse_file(import_path)
            imported_data = result.data
            
            with self._lock:
                # Update configurations
                if 'platform' in imported_data:
                    self.unified_config.platform.update(imported_data['platform'])
                
                if 'global_settings' in imported_data:
                    self.unified_config.global_settings.update(imported_data['global_settings'])
                
                if 'components' in imported_data:
                    for comp_name, comp_data in imported_data['components'].items():
                        component_config = ComponentConfig(
                            name=comp_data.get('name', comp_name),
                            version=comp_data.get('version', '1.0.0'),
                            enabled=comp_data.get('enabled', True),
                            config=comp_data.get('config', {}),
                            dependencies=comp_data.get('dependencies', []),
                            metadata=comp_data.get('metadata', {})
                        )
                        self.unified_config.components[comp_name] = component_config
                
                if 'environments' in imported_data:
                    self.unified_config.environments.update(imported_data['environments'])
                
                # Validate imported configuration
                self._validate_all_configs()
            
            logger.info(f"Imported configuration from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import configuration from {import_path}: {e}")
            return False
    
    def _load_platform_config(self):
        """Load platform configuration."""
        platform_config_file = self.config_dir / "platform.yaml"
        
        if platform_config_file.exists():
            result = self.parser.parse_file(platform_config_file)
            self.unified_config.platform = result.data
        else:
            # Create default platform config
            default_platform = {
                'name': 'fine-tune-llm',
                'version': '2.0.0',
                'description': 'LLM Fine-tuning Platform',
                'author': 'Fine-Tune LLM Team'
            }
            self.unified_config.platform = default_platform
    
    def _load_component_configs(self):
        """Load all component configurations."""
        components_dir = self.config_dir / "components"
        
        if components_dir.exists():
            for config_file in components_dir.rglob("*.{yaml,yml,json,toml}"):
                component_name = config_file.stem
                self._load_component_config(component_name, config_file)
    
    def _load_component_config(self, 
                              component_name: str,
                              config_file: Optional[Path] = None):
        """Load specific component configuration."""
        if config_file is None:
            # Find config file
            components_dir = self.config_dir / "components"
            for ext in ['yaml', 'yml', 'json', 'toml']:
                config_file = components_dir / f"{component_name}.{ext}"
                if config_file.exists():
                    break
            else:
                # No config file found, use defaults
                defaults = self.component_defaults.get(component_name, {})
                component_config = ComponentConfig(
                    name=component_name,
                    config=defaults.copy()
                )
                self.unified_config.components[component_name] = component_config
                return
        
        try:
            result = self.parser.parse_file(config_file)
            config_data = result.data
            
            # Create component config
            component_config = ComponentConfig(
                name=config_data.get('name', component_name),
                version=config_data.get('version', '1.0.0'),
                enabled=config_data.get('enabled', True),
                config=config_data.get('config', config_data),
                dependencies=config_data.get('dependencies', []),
                metadata=config_data.get('metadata', {})
            )
            
            # Apply defaults
            defaults = self.component_defaults.get(component_name, {})
            merged_config = self._merge_with_defaults(component_config.config, defaults)
            component_config.config = merged_config
            
            self.unified_config.components[component_name] = component_config
            
        except Exception as e:
            logger.error(f"Failed to load config for component {component_name}: {e}")
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides."""
        env_config_file = self.config_dir / "environments" / f"{self.environment}.yaml"
        
        if env_config_file.exists():
            result = self.parser.parse_file(env_config_file)
            env_data = result.data
            
            # Store environment config
            self.unified_config.environments[self.environment] = env_data
            
            # Apply overrides
            if 'platform' in env_data:
                self.unified_config.platform.update(env_data['platform'])
            
            if 'global_settings' in env_data:
                self.unified_config.global_settings.update(env_data['global_settings'])
            
            if 'components' in env_data:
                for comp_name, comp_overrides in env_data['components'].items():
                    if comp_name in self.unified_config.components:
                        existing = self.unified_config.components[comp_name]
                        if 'config' in comp_overrides:
                            existing.config.update(comp_overrides['config'])
                        if 'enabled' in comp_overrides:
                            existing.enabled = comp_overrides['enabled']
    
    def _validate_all_configs(self):
        """Validate all configurations."""
        # Validate each component config
        for name, component_config in self.unified_config.components.items():
            errors = self._validate_component_config(name, component_config.config)
            if errors:
                logger.warning(f"Validation errors for component {name}: {errors}")
    
    def _validate_component_config(self, name: str, config: Dict[str, Any]) -> List[str]:
        """Validate component configuration."""
        schema = self.component_schemas.get(name)
        if schema:
            return self.validation_engine.validate_config(config, schema)
        return []
    
    def _get_nested_value(self, data: Dict[str, Any], path: str, default: Any) -> Any:
        """Get nested dictionary value by dot-separated path."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value by dot-separated path."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_with_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with defaults."""
        result = defaults.copy()
        result.update(config)
        return result
    
    def _setup_hot_reload(self):
        """Set up hot-reload of configuration files."""
        # This would implement file watching for hot-reload
        # For now, it's a placeholder
        logger.debug("Hot-reload setup (placeholder)")
    
    def get_component_names(self) -> List[str]:
        """Get list of registered component names."""
        with self._lock:
            return list(self.unified_config.components.keys())
    
    def is_component_enabled(self, name: str) -> bool:
        """Check if component is enabled."""
        with self._lock:
            component = self.unified_config.components.get(name)
            return component.enabled if component else False
    
    def enable_component(self, name: str) -> bool:
        """Enable component."""
        return self._set_component_enabled(name, True)
    
    def disable_component(self, name: str) -> bool:
        """Disable component."""
        return self._set_component_enabled(name, False)
    
    def _set_component_enabled(self, name: str, enabled: bool) -> bool:
        """Set component enabled state."""
        try:
            with self._lock:
                if name in self.unified_config.components:
                    self.unified_config.components[name].enabled = enabled
                    logger.info(f"{'Enabled' if enabled else 'Disabled'} component: {name}")
                    return True
                else:
                    logger.warning(f"Component not found: {name}")
                    return False
        except Exception as e:
            logger.error(f"Failed to set enabled state for {name}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration system statistics."""
        with self._lock:
            return {
                'total_components': len(self.unified_config.components),
                'enabled_components': sum(1 for c in self.unified_config.components.values() if c.enabled),
                'registered_components': len(self.registered_components),
                'schemas_registered': len(self.component_schemas),
                'defaults_registered': len(self.component_defaults),
                'environment': self.environment,
                'config_dir': str(self.config_dir),
                'hot_reload_enabled': self.enable_hot_reload
            }
    
    def close(self):
        """Close configuration manager and clean up resources."""
        try:
            self.secret_manager.close()
            logger.info("Unified configuration manager closed")
        except Exception as e:
            logger.error(f"Error closing unified configuration manager: {e}")


# Global instance
_unified_config_manager = None

def get_unified_config() -> UnifiedConfigManager:
    """Get global unified configuration manager instance."""
    global _unified_config_manager
    if _unified_config_manager is None:
        _unified_config_manager = UnifiedConfigManager()
    return _unified_config_manager