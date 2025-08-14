"""
Base UI component for shared functionality.

This module provides the base class for all UI components with
common functionality like theming, configuration, and lifecycle management.
"""

import abc
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..utils import ComponentTheme, UIConfig, LayoutManager
from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class BaseUIComponent(BaseComponent):
    """
    Abstract base class for UI components.
    
    Provides common functionality for all UI components including
    theme management, configuration handling, and lifecycle methods.
    """
    
    def __init__(self, 
                 config: Optional[UIConfig] = None,
                 theme: Optional[ComponentTheme] = None,
                 **kwargs):
        """
        Initialize base UI component.
        
        Args:
            config: UI configuration
            theme: Component theme
            **kwargs: Additional configuration options
        """
        self.config = config or UIConfig()
        self.theme = theme or ComponentTheme.from_config(self.config.theme_config)
        self.layout_manager = LayoutManager(self.config.layout)
        
        # Component state
        self._initialized = False
        self._running = False
        self._error_state = False
        self._last_error = None
        
        # Custom configuration from kwargs
        self.custom_config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Component name identifier."""
        pass
    
    @property 
    @abc.abstractmethod
    def version(self) -> str:
        """Component version."""
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Component description."""
        pass
    
    @property
    @abc.abstractmethod
    def dependencies(self) -> List[str]:
        """Required dependencies."""
        pass
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize component with configuration.
        
        Args:
            config: Configuration dictionary
        """
        try:
            # Update configuration
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self.custom_config[key] = value
            
            # Check dependencies
            if not self._check_dependencies():
                raise RuntimeError(f"Missing dependencies for {self.name}")
            
            # Component-specific initialization
            self._initialize_component()
            
            self._initialized = True
            logger.info(f"Initialized component {self.name}")
            
        except Exception as e:
            self._error_state = True
            self._last_error = str(e)
            logger.error(f"Failed to initialize {self.name}: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up component resources."""
        try:
            # Component-specific cleanup
            self._cleanup_component()
            
            self._running = False
            logger.info(f"Cleaned up component {self.name}")
            
        except Exception as e:
            logger.error(f"Error during cleanup of {self.name}: {e}")
    
    @abc.abstractmethod
    def _initialize_component(self) -> None:
        """Component-specific initialization logic."""
        pass
    
    @abc.abstractmethod
    def _cleanup_component(self) -> None:
        """Component-specific cleanup logic."""
        pass
    
    @abc.abstractmethod
    def render(self) -> None:
        """Render the UI component."""
        pass
    
    def start(self) -> bool:
        """
        Start the UI component.
        
        Returns:
            True if started successfully
        """
        try:
            if not self._initialized:
                raise RuntimeError(f"Component {self.name} not initialized")
            
            if self._running:
                logger.warning(f"Component {self.name} is already running")
                return True
            
            # Component-specific startup
            self._start_component()
            
            self._running = True
            self._error_state = False
            self._last_error = None
            
            logger.info(f"Started component {self.name}")
            return True
            
        except Exception as e:
            self._error_state = True
            self._last_error = str(e)
            logger.error(f"Failed to start {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the UI component.
        
        Returns:
            True if stopped successfully
        """
        try:
            if not self._running:
                logger.warning(f"Component {self.name} is not running")
                return True
            
            # Component-specific shutdown
            self._stop_component()
            
            self._running = False
            logger.info(f"Stopped component {self.name}")
            return True
            
        except Exception as e:
            self._error_state = True
            self._last_error = str(e)
            logger.error(f"Failed to stop {self.name}: {e}")
            return False
    
    @abc.abstractmethod
    def _start_component(self) -> None:
        """Component-specific start logic."""
        pass
    
    @abc.abstractmethod
    def _stop_component(self) -> None:
        """Component-specific stop logic."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Status information dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'initialized': self._initialized,
            'running': self._running,
            'error_state': self._error_state,
            'last_error': self._last_error,
            'dependencies': self.dependencies,
            'theme': self.theme.name
        }
    
    def apply_theme(self, theme: ComponentTheme) -> None:
        """
        Apply new theme to component.
        
        Args:
            theme: Theme to apply
        """
        self.theme = theme
        
        # Component-specific theme application
        self._apply_theme()
        
        logger.info(f"Applied theme {theme.name} to {self.name}")
    
    @abc.abstractmethod
    def _apply_theme(self) -> None:
        """Component-specific theme application."""
        pass
    
    def get_css(self) -> str:
        """
        Get component CSS styles.
        
        Returns:
            CSS stylesheet content
        """
        base_css = self.layout_manager.get_component_css(self.theme)
        component_css = self._get_component_css()
        
        return f"{base_css}\n\n{component_css}"
    
    @abc.abstractmethod
    def _get_component_css(self) -> str:
        """Get component-specific CSS."""
        pass
    
    def handle_error(self, error: Exception) -> None:
        """
        Handle component error.
        
        Args:
            error: Exception that occurred
        """
        self._error_state = True
        self._last_error = str(error)
        
        logger.error(f"Error in component {self.name}: {error}")
        
        # Component-specific error handling
        self._handle_error(error)
    
    def _handle_error(self, error: Exception) -> None:
        """Component-specific error handling."""
        pass
    
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are available."""
        try:
            import importlib
            
            for dep in self.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    logger.error(f"Missing dependency: {dep}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for the component.
        
        Returns:
            JSON schema for configuration
        """
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "theme": {"type": "string"},
                "auto_refresh": {"type": "boolean"},
                "refresh_interval": {"type": "integer", "minimum": 1}
            },
            "additionalProperties": True
        }
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            import jsonschema
            schema = self.get_config_schema()
            jsonschema.validate(config, schema)
            
        except ImportError:
            # jsonschema not available, basic validation
            required_keys = ['title']
            for key in required_keys:
                if key not in config:
                    errors.append(f"Missing required configuration: {key}")
                    
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export current configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'theme': self.theme.name,
            'config': self.custom_config.copy()
        }
    
    def import_config(self, config: Dict[str, Any]) -> bool:
        """
        Import configuration.
        
        Args:
            config: Configuration to import
            
        Returns:
            True if imported successfully
        """
        try:
            errors = self.validate_config(config)
            if errors:
                logger.error(f"Configuration validation failed: {errors}")
                return False
            
            # Apply configuration
            if 'theme' in config:
                theme_config = {'preset': config['theme']}
                new_theme = ComponentTheme.from_config(theme_config)
                self.apply_theme(new_theme)
            
            if 'config' in config:
                self.custom_config.update(config['config'])
            
            logger.info(f"Imported configuration for {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False