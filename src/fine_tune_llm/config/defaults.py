"""
Centralized Default Value Management System.

This module provides a comprehensive default value management system
that can be shared across all components with hierarchical defaults,
environment-specific overrides, and dynamic default computation.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import copy

logger = logging.getLogger(__name__)


class DefaultScope(Enum):
    """Scope for default values."""
    GLOBAL = "global"
    COMPONENT = "component"
    ENVIRONMENT = "environment"
    USER = "user"
    RUNTIME = "runtime"


class DefaultPriority(Enum):
    """Priority levels for default values."""
    LOWEST = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4


@dataclass
class DefaultValue:
    """Default value configuration."""
    key: str
    value: Any
    scope: DefaultScope = DefaultScope.GLOBAL
    priority: DefaultPriority = DefaultPriority.MEDIUM
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None
    dependencies: List[str] = field(default_factory=list)
    environment_override: bool = True
    computed: bool = False
    compute_fn: Optional[Callable[[], Any]] = None


@dataclass
class DefaultProfile:
    """Collection of default values for a specific context."""
    name: str
    description: str
    defaults: Dict[str, DefaultValue] = field(default_factory=dict)
    parent: Optional[str] = None
    active: bool = True


class DefaultManager:
    """
    Centralized default value management system.
    
    Manages hierarchical default values with scope-based priorities,
    environment overrides, and dynamic computation capabilities.
    """
    
    def __init__(self):
        """Initialize default manager."""
        # Default profiles
        self.profiles: Dict[str, DefaultProfile] = {}
        
        # Component defaults
        self.component_defaults: Dict[str, Dict[str, DefaultValue]] = {}
        
        # Runtime context
        self.runtime_context: Dict[str, Any] = {}
        
        # Environment mappings
        self.env_mappings: Dict[str, str] = {}
        
        # Computed defaults cache
        self._computed_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize built-in defaults
        self._initialize_builtin_defaults()
        
        logger.info("Initialized DefaultManager")
    
    def _initialize_builtin_defaults(self):
        """Initialize built-in default values."""
        # Create global defaults profile
        global_profile = DefaultProfile(
            name="global",
            description="Global default values for the platform",
        )
        
        # Platform defaults
        platform_defaults = {
            # Core platform settings
            "platform.name": DefaultValue(
                key="platform.name",
                value="fine-tune-llm",
                description="Platform name identifier",
                priority=DefaultPriority.HIGH
            ),
            "platform.version": DefaultValue(
                key="platform.version", 
                value="2.0.0",
                description="Platform version",
                priority=DefaultPriority.HIGH
            ),
            "platform.debug": DefaultValue(
                key="platform.debug",
                value=False,
                description="Enable debug mode",
                environment_override=True
            ),
            
            # Logging defaults
            "logging.level": DefaultValue(
                key="logging.level",
                value="INFO",
                description="Default logging level",
                environment_override=True,
                validator=lambda v: v in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            "logging.format": DefaultValue(
                key="logging.format",
                value="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                description="Default logging format"
            ),
            "logging.file": DefaultValue(
                key="logging.file",
                value=None,
                description="Log file path (None for console only)",
                environment_override=True
            ),
            
            # Directory structure defaults
            "dirs.config": DefaultValue(
                key="dirs.config",
                value="configs",
                description="Configuration directory",
                computed=True,
                compute_fn=lambda: os.environ.get("FTL_CONFIG_DIR", "configs")
            ),
            "dirs.data": DefaultValue(
                key="dirs.data",
                value="data",
                description="Data directory",
                computed=True,
                compute_fn=lambda: os.environ.get("FTL_DATA_DIR", "data")
            ),
            "dirs.models": DefaultValue(
                key="dirs.models",
                value="artifacts/models",
                description="Models directory",
                computed=True,
                compute_fn=lambda: os.environ.get("FTL_MODELS_DIR", "artifacts/models")
            ),
            "dirs.logs": DefaultValue(
                key="dirs.logs",
                value="logs",
                description="Logs directory",
                computed=True,
                compute_fn=lambda: os.environ.get("FTL_LOGS_DIR", "logs")
            ),
            
            # Network defaults
            "network.timeout": DefaultValue(
                key="network.timeout",
                value=30,
                description="Default network timeout in seconds",
                validator=lambda v: isinstance(v, int) and 0 < v <= 300
            ),
            "network.retries": DefaultValue(
                key="network.retries",
                value=3,
                description="Default number of network retries",
                validator=lambda v: isinstance(v, int) and 0 <= v <= 10
            ),
            
            # Security defaults
            "security.secret_key": DefaultValue(
                key="security.secret_key",
                value=None,
                description="Secret key for encryption",
                environment_override=True,
                computed=True,
                compute_fn=self._generate_secret_key
            ),
            "security.token_expiry": DefaultValue(
                key="security.token_expiry",
                value=3600,
                description="Token expiry time in seconds",
                validator=lambda v: isinstance(v, int) and v > 0
            ),
        }
        
        global_profile.defaults.update(platform_defaults)
        self.profiles["global"] = global_profile
        
        # Component-specific defaults
        self._initialize_component_defaults()
        
        # Environment mappings
        self._initialize_env_mappings()
        
        logger.info(f"Initialized {len(platform_defaults)} global defaults")
    
    def _initialize_component_defaults(self):
        """Initialize component-specific defaults."""
        # Training component defaults
        training_defaults = {
            "model.base_model": DefaultValue(
                key="model.base_model",
                value="ZHIPU-AI/glm-4-9b-chat",
                description="Base model for training",
                scope=DefaultScope.COMPONENT
            ),
            "model.tokenizer": DefaultValue(
                key="model.tokenizer",
                value=None,
                description="Tokenizer (defaults to base model)",
                scope=DefaultScope.COMPONENT,
                computed=True,
                compute_fn=lambda: self.get_default("model.base_model")
            ),
            "training.learning_rate": DefaultValue(
                key="training.learning_rate",
                value=2e-4,
                description="Learning rate for training",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, (int, float)) and 0 < v <= 1
            ),
            "training.batch_size": DefaultValue(
                key="training.batch_size",
                value=4,
                description="Batch size for training",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and v > 0
            ),
            "training.epochs": DefaultValue(
                key="training.epochs",
                value=3,
                description="Number of training epochs",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and v > 0
            ),
            "training.max_length": DefaultValue(
                key="training.max_length",
                value=2048,
                description="Maximum sequence length",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and 0 < v <= 10000
            ),
            "lora.rank": DefaultValue(
                key="lora.rank",
                value=16,
                description="LoRA rank",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and v in [8, 16, 32, 64, 128]
            ),
            "lora.alpha": DefaultValue(
                key="lora.alpha",
                value=32,
                description="LoRA alpha (typically 2x rank)",
                scope=DefaultScope.COMPONENT,
                computed=True,
                compute_fn=lambda: self.get_default("lora.rank") * 2
            ),
            "lora.dropout": DefaultValue(
                key="lora.dropout",
                value=0.1,
                description="LoRA dropout rate",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, (int, float)) and 0 <= v <= 1
            )
        }
        
        self.component_defaults["training"] = training_defaults
        
        # UI component defaults
        ui_defaults = {
            "ui.theme": DefaultValue(
                key="ui.theme",
                value="default",
                description="Default UI theme",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: v in ["default", "dark", "light", "blue", "green", "purple", "professional"]
            ),
            "ui.port": DefaultValue(
                key="ui.port",
                value=8501,
                description="Default UI port",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and 1024 <= v <= 65535
            ),
            "ui.auto_refresh": DefaultValue(
                key="ui.auto_refresh",
                value=True,
                description="Enable auto-refresh in UI",
                scope=DefaultScope.COMPONENT
            ),
            "ui.refresh_interval": DefaultValue(
                key="ui.refresh_interval",
                value=5,
                description="UI refresh interval in seconds",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and 1 <= v <= 60
            )
        }
        
        self.component_defaults["ui"] = ui_defaults
        
        # Monitoring defaults
        monitoring_defaults = {
            "monitoring.enabled": DefaultValue(
                key="monitoring.enabled",
                value=True,
                description="Enable monitoring",
                scope=DefaultScope.COMPONENT
            ),
            "monitoring.interval": DefaultValue(
                key="monitoring.interval",
                value=10,
                description="Monitoring interval in seconds",
                scope=DefaultScope.COMPONENT,
                validator=lambda v: isinstance(v, int) and v > 0
            ),
            "monitoring.alerts": DefaultValue(
                key="monitoring.alerts",
                value=True,
                description="Enable monitoring alerts",
                scope=DefaultScope.COMPONENT
            )
        }
        
        self.component_defaults["monitoring"] = monitoring_defaults
        
        logger.info(f"Initialized defaults for {len(self.component_defaults)} components")
    
    def _initialize_env_mappings(self):
        """Initialize environment variable mappings."""
        self.env_mappings = {
            "FTL_DEBUG": "platform.debug",
            "FTL_LOG_LEVEL": "logging.level",
            "FTL_LOG_FILE": "logging.file",
            "FTL_CONFIG_DIR": "dirs.config",
            "FTL_DATA_DIR": "dirs.data",
            "FTL_MODELS_DIR": "dirs.models",
            "FTL_LOGS_DIR": "dirs.logs",
            "FTL_SECRET_KEY": "security.secret_key",
            "FTL_NETWORK_TIMEOUT": "network.timeout",
            "FTL_NETWORK_RETRIES": "network.retries",
            "FTL_UI_THEME": "ui.theme",
            "FTL_UI_PORT": "ui.port",
            "FTL_LEARNING_RATE": "training.learning_rate",
            "FTL_BATCH_SIZE": "training.batch_size",
            "FTL_EPOCHS": "training.epochs",
            "FTL_LORA_RANK": "lora.rank",
            "FTL_LORA_ALPHA": "lora.alpha"
        }
    
    def register_profile(self, profile: DefaultProfile) -> bool:
        """
        Register default profile.
        
        Args:
            profile: Default profile to register
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.profiles[profile.name] = profile
            
            logger.info(f"Registered default profile: {profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register profile {profile.name}: {e}")
            return False
    
    def register_component_defaults(self, component: str, defaults: Dict[str, DefaultValue]) -> bool:
        """
        Register component defaults.
        
        Args:
            component: Component name
            defaults: Component default values
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                if component not in self.component_defaults:
                    self.component_defaults[component] = {}
                self.component_defaults[component].update(defaults)
            
            logger.info(f"Registered {len(defaults)} defaults for component: {component}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register defaults for {component}: {e}")
            return False
    
    def get_default(self, 
                   key: str,
                   component: Optional[str] = None,
                   profile: Optional[str] = None,
                   fallback: Any = None) -> Any:
        """
        Get default value with hierarchical resolution.
        
        Args:
            key: Configuration key
            component: Component name for component-specific defaults
            profile: Profile name
            fallback: Fallback value if not found
            
        Returns:
            Default value or fallback
        """
        with self._lock:
            try:
                # Resolution order:
                # 1. Environment variables
                # 2. Runtime context
                # 3. Component defaults
                # 4. Profile defaults
                # 5. Global defaults
                # 6. Fallback
                
                # Check environment variables
                env_value = self._get_env_value(key)
                if env_value is not None:
                    return env_value
                
                # Check runtime context
                if key in self.runtime_context:
                    return self.runtime_context[key]
                
                # Check component defaults
                if component and component in self.component_defaults:
                    comp_defaults = self.component_defaults[component]
                    if key in comp_defaults:
                        default_value = comp_defaults[key]
                        return self._resolve_default_value(default_value)
                
                # Check profile defaults
                profile_name = profile or "global"
                if profile_name in self.profiles:
                    profile_defaults = self.profiles[profile_name]
                    if key in profile_defaults.defaults:
                        default_value = profile_defaults.defaults[key]
                        return self._resolve_default_value(default_value)
                    
                    # Check parent profile
                    if profile_defaults.parent:
                        return self.get_default(key, component, profile_defaults.parent, fallback)
                
                # Check global defaults
                if "global" in self.profiles and key in self.profiles["global"].defaults:
                    default_value = self.profiles["global"].defaults[key]
                    return self._resolve_default_value(default_value)
                
                # Return fallback
                return fallback
                
            except Exception as e:
                logger.error(f"Error getting default for {key}: {e}")
                return fallback
    
    def set_runtime_default(self, key: str, value: Any):
        """
        Set runtime default value.
        
        Args:
            key: Configuration key
            value: Default value
        """
        with self._lock:
            self.runtime_context[key] = value
        
        logger.debug(f"Set runtime default: {key} = {value}")
    
    def get_all_defaults(self, 
                        component: Optional[str] = None,
                        profile: Optional[str] = None,
                        include_computed: bool = True) -> Dict[str, Any]:
        """
        Get all default values for component/profile.
        
        Args:
            component: Component name
            profile: Profile name
            include_computed: Whether to include computed defaults
            
        Returns:
            Dictionary of all default values
        """
        defaults = {}
        
        try:
            with self._lock:
                # Collect from global profile
                if "global" in self.profiles:
                    for key, default_value in self.profiles["global"].defaults.items():
                        if not default_value.computed or include_computed:
                            defaults[key] = self._resolve_default_value(default_value)
                
                # Collect from specific profile
                if profile and profile in self.profiles:
                    for key, default_value in self.profiles[profile].defaults.items():
                        if not default_value.computed or include_computed:
                            defaults[key] = self._resolve_default_value(default_value)
                
                # Collect from component defaults
                if component and component in self.component_defaults:
                    for key, default_value in self.component_defaults[component].items():
                        if not default_value.computed or include_computed:
                            defaults[key] = self._resolve_default_value(default_value)
                
                # Apply environment overrides
                for env_var, config_key in self.env_mappings.items():
                    env_value = self._get_env_value(config_key)
                    if env_value is not None:
                        defaults[config_key] = env_value
                
                # Apply runtime context
                defaults.update(self.runtime_context)
            
        except Exception as e:
            logger.error(f"Error getting all defaults: {e}")
        
        return defaults
    
    def create_config_with_defaults(self, 
                                  config: Dict[str, Any],
                                  component: Optional[str] = None,
                                  profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Create configuration with defaults applied.
        
        Args:
            config: Base configuration
            component: Component name
            profile: Profile name
            
        Returns:
            Configuration with defaults applied
        """
        # Start with defaults
        result = self.get_all_defaults(component, profile)
        
        # Apply provided configuration (overrides defaults)
        result.update(config)
        
        return result
    
    def validate_defaults(self, 
                         component: Optional[str] = None,
                         profile: Optional[str] = None) -> List[str]:
        """
        Validate default values.
        
        Args:
            component: Component name
            profile: Profile name
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            with self._lock:
                # Get all defaults
                defaults = self.get_all_defaults(component, profile)
                
                # Validate each default
                sources = []
                
                # Add global defaults
                if "global" in self.profiles:
                    sources.extend(self.profiles["global"].defaults.values())
                
                # Add profile defaults
                if profile and profile in self.profiles:
                    sources.extend(self.profiles[profile].defaults.values())
                
                # Add component defaults
                if component and component in self.component_defaults:
                    sources.extend(self.component_defaults[component].values())
                
                # Validate each default
                for default_value in sources:
                    if default_value.validator:
                        try:
                            resolved_value = self._resolve_default_value(default_value)
                            if not default_value.validator(resolved_value):
                                errors.append(f"Default value validation failed for {default_value.key}: {resolved_value}")
                        except Exception as e:
                            errors.append(f"Error validating default {default_value.key}: {e}")
        
        except Exception as e:
            errors.append(f"Default validation failed: {e}")
        
        return errors
    
    def _resolve_default_value(self, default_value: DefaultValue) -> Any:
        """Resolve default value (handle computed values)."""
        if not default_value.computed:
            return default_value.value
        
        # Check cache
        cache_key = f"{default_value.key}:{id(default_value)}"
        current_time = datetime.now().timestamp()
        
        if (cache_key in self._computed_cache and 
            cache_key in self._cache_timestamps and
            current_time - self._cache_timestamps[cache_key] < 300):  # 5 minute cache
            return self._computed_cache[cache_key]
        
        # Compute value
        try:
            if default_value.compute_fn:
                computed_value = default_value.compute_fn()
                
                # Cache result
                self._computed_cache[cache_key] = computed_value
                self._cache_timestamps[cache_key] = current_time
                
                return computed_value
            else:
                return default_value.value
                
        except Exception as e:
            logger.error(f"Error computing default value for {default_value.key}: {e}")
            return default_value.value
    
    def _get_env_value(self, key: str) -> Optional[Any]:
        """Get value from environment variables."""
        # Find corresponding environment variable
        env_var = None
        for env_name, config_key in self.env_mappings.items():
            if config_key == key:
                env_var = env_name
                break
        
        if not env_var:
            return None
        
        env_value = os.environ.get(env_var)
        if env_value is None:
            return None
        
        # Convert to appropriate type
        return self._convert_env_value(env_value)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def _generate_secret_key(self) -> str:
        """Generate secret key if not provided."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def clear_computed_cache(self):
        """Clear computed defaults cache."""
        with self._lock:
            self._computed_cache.clear()
            self._cache_timestamps.clear()
        
        logger.debug("Cleared computed defaults cache")
    
    def export_defaults(self, 
                       component: Optional[str] = None,
                       profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Export defaults for external use.
        
        Args:
            component: Component name
            profile: Profile name
            
        Returns:
            Exported defaults configuration
        """
        defaults = self.get_all_defaults(component, profile)
        
        return {
            'defaults': defaults,
            'metadata': {
                'component': component,
                'profile': profile,
                'exported_at': datetime.now().isoformat(),
                'total_defaults': len(defaults)
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get default manager statistics."""
        with self._lock:
            return {
                'total_profiles': len(self.profiles),
                'total_components': len(self.component_defaults),
                'total_env_mappings': len(self.env_mappings),
                'runtime_context_size': len(self.runtime_context),
                'computed_cache_size': len(self._computed_cache),
                'active_profiles': [name for name, profile in self.profiles.items() if profile.active]
            }


# Global default manager instance
_default_manager = None

def get_default_manager() -> DefaultManager:
    """Get global default manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DefaultManager()
    return _default_manager


# Convenience functions

def get_default(key: str, 
               component: Optional[str] = None,
               fallback: Any = None) -> Any:
    """
    Get default value using global default manager.
    
    Args:
        key: Configuration key
        component: Component name
        fallback: Fallback value
        
    Returns:
        Default value or fallback
    """
    manager = get_default_manager()
    return manager.get_default(key, component=component, fallback=fallback)


def create_config_with_defaults(config: Dict[str, Any],
                               component: Optional[str] = None) -> Dict[str, Any]:
    """
    Create configuration with defaults applied.
    
    Args:
        config: Base configuration
        component: Component name
        
    Returns:
        Configuration with defaults
    """
    manager = get_default_manager()
    return manager.create_config_with_defaults(config, component=component)


def register_component_defaults(component: str, defaults: Dict[str, DefaultValue]) -> bool:
    """
    Register component defaults globally.
    
    Args:
        component: Component name
        defaults: Component defaults
        
    Returns:
        True if registered successfully
    """
    manager = get_default_manager()
    return manager.register_component_defaults(component, defaults)