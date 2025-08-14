"""
Component factory system for fine-tune LLM library.

Provides centralized component creation with plugin support,
dependency injection, and dynamic registration.
"""

from typing import Dict, Any, Type, Optional, List, Callable
from abc import ABC, abstractmethod
import inspect
import importlib
import logging
from pathlib import Path

from .interfaces import BaseComponent, BaseFactory
from .exceptions import ConfigurationError, SystemError
from .protocols import ConfigProtocol

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for component types and their factories."""
    
    def __init__(self):
        self._components: Dict[str, Type[BaseComponent]] = {}
        self._factories: Dict[str, Type[BaseFactory]] = {}
        self._instances: Dict[str, BaseComponent] = {}
        self._singletons: Dict[str, BaseComponent] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register_component(self, name: str, component_class: Type[BaseComponent],
                          singleton: bool = False, dependencies: Optional[List[str]] = None) -> None:
        """Register a component type."""
        self._components[name] = component_class
        
        if singleton:
            self._singletons[name] = None
        
        if dependencies:
            self._dependencies[name] = dependencies
            
        logger.info(f"Registered component: {name} ({component_class.__name__})")
    
    def register_factory(self, name: str, factory_class: Type[BaseFactory]) -> None:
        """Register a factory type."""
        self._factories[name] = factory_class
        logger.info(f"Registered factory: {name} ({factory_class.__name__})")
    
    def get_component_class(self, name: str) -> Optional[Type[BaseComponent]]:
        """Get component class by name."""
        return self._components.get(name)
    
    def get_factory_class(self, name: str) -> Optional[Type[BaseFactory]]:
        """Get factory class by name."""
        return self._factories.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())
    
    def list_factories(self) -> List[str]:
        """List all registered factory names."""
        return list(self._factories.keys())
    
    def is_singleton(self, name: str) -> bool:
        """Check if component is configured as singleton."""
        return name in self._singletons
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get component dependencies."""
        return self._dependencies.get(name, [])

class DependencyInjector:
    """Dependency injection container."""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self._resolving: set = set()
    
    def create_component(self, name: str, config: Dict[str, Any]) -> BaseComponent:
        """Create component with dependency injection."""
        if name in self._resolving:
            raise ConfigurationError(f"Circular dependency detected for component: {name}")
        
        # Check if singleton exists
        if self.registry.is_singleton(name) and name in self.registry._singletons:
            instance = self.registry._singletons[name]
            if instance is not None:
                return instance
        
        self._resolving.add(name)
        
        try:
            # Get component class
            component_class = self.registry.get_component_class(name)
            if not component_class:
                raise ConfigurationError(f"Unknown component: {name}")
            
            # Resolve dependencies
            dependencies = self.registry.get_dependencies(name)
            resolved_deps = {}
            
            for dep_name in dependencies:
                dep_instance = self.create_component(dep_name, config)
                resolved_deps[dep_name] = dep_instance
            
            # Inspect constructor parameters
            sig = inspect.signature(component_class.__init__)
            init_params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Check if dependency injection is needed
                if param_name in resolved_deps:
                    init_params[param_name] = resolved_deps[param_name]
                elif param_name in config:
                    init_params[param_name] = config[param_name]
                elif param.default is not inspect.Parameter.empty:
                    # Use default value
                    pass
                else:
                    logger.warning(f"Missing parameter {param_name} for {name}")
            
            # Create instance
            instance = component_class(**init_params)
            
            # Initialize with configuration
            if hasattr(instance, 'initialize'):
                instance.initialize(config)
            
            # Store singleton if needed
            if self.registry.is_singleton(name):
                self.registry._singletons[name] = instance
            
            logger.info(f"Created component: {name}")
            return instance
            
        finally:
            self._resolving.remove(name)

class PluginManager:
    """Plugin discovery and loading system."""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.plugin_paths: List[Path] = []
        self.loaded_plugins: Dict[str, Any] = {}
    
    def add_plugin_path(self, path: Path) -> None:
        """Add path to search for plugins."""
        if path.exists() and path.is_dir():
            self.plugin_paths.append(path)
            logger.info(f"Added plugin path: {path}")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins."""
        plugins = []
        
        for plugin_path in self.plugin_paths:
            for py_file in plugin_path.glob("**/*.py"):
                if py_file.name.startswith("plugin_"):
                    plugin_name = py_file.stem
                    plugins.append(plugin_name)
        
        return plugins
    
    def load_plugin(self, plugin_name: str) -> None:
        """Load a plugin module."""
        try:
            # Find plugin file
            plugin_file = None
            for plugin_path in self.plugin_paths:
                candidate = plugin_path / f"{plugin_name}.py"
                if candidate.exists():
                    plugin_file = candidate
                    break
            
            if not plugin_file:
                raise SystemError(f"Plugin file not found: {plugin_name}")
            
            # Import plugin module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # Look for plugin registration function
            if hasattr(plugin_module, 'register_plugin'):
                plugin_module.register_plugin(self.registry)
                self.loaded_plugins[plugin_name] = plugin_module
                logger.info(f"Loaded plugin: {plugin_name}")
            else:
                logger.warning(f"Plugin {plugin_name} has no register_plugin function")
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise
    
    def load_all_plugins(self) -> None:
        """Load all discovered plugins."""
        plugins = self.discover_plugins()
        for plugin_name in plugins:
            try:
                self.load_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")

class ComponentFactory:
    """Main factory for creating all components."""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self.injector = DependencyInjector(self.registry)
        self.plugin_manager = PluginManager(self.registry)
        self._factories: Dict[str, BaseFactory] = {}
        
        # Register built-in components
        self._register_builtin_components()
    
    def register_component(self, name: str, component_class: Type[BaseComponent],
                          singleton: bool = False, dependencies: Optional[List[str]] = None) -> None:
        """Register a component."""
        self.registry.register_component(name, component_class, singleton, dependencies)
    
    def register_factory(self, name: str, factory_class: Type[BaseFactory]) -> None:
        """Register a factory."""
        self.registry.register_factory(name, factory_class)
    
    def create(self, component_name: str, config: Dict[str, Any]) -> BaseComponent:
        """Create component instance."""
        return self.injector.create_component(component_name, config)
    
    def create_with_factory(self, factory_name: str, component_type: str, 
                           config: Dict[str, Any]) -> BaseComponent:
        """Create component using specific factory."""
        if factory_name not in self._factories:
            factory_class = self.registry.get_factory_class(factory_name)
            if not factory_class:
                raise ConfigurationError(f"Unknown factory: {factory_name}")
            self._factories[factory_name] = factory_class()
        
        factory = self._factories[factory_name]
        return factory.create(component_type, config)
    
    def list_components(self) -> List[str]:
        """List available components."""
        return self.registry.list_components()
    
    def list_factories(self) -> List[str]:
        """List available factories."""
        return self.registry.list_factories()
    
    def add_plugin_path(self, path: Path) -> None:
        """Add plugin search path."""
        self.plugin_manager.add_plugin_path(path)
    
    def load_plugins(self) -> None:
        """Load all plugins."""
        self.plugin_manager.load_all_plugins()
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get component information."""
        component_class = self.registry.get_component_class(name)
        if not component_class:
            return None
        
        return {
            "name": name,
            "class": component_class.__name__,
            "module": component_class.__module__,
            "singleton": self.registry.is_singleton(name),
            "dependencies": self.registry.get_dependencies(name),
            "docstring": component_class.__doc__
        }
    
    def _register_builtin_components(self) -> None:
        """Register built-in components."""
        # This will be populated as components are created
        # For now, we'll register placeholders that will be replaced
        # by actual implementations
        pass

# Global factory instance
_global_factory: Optional[ComponentFactory] = None

def get_component_factory() -> ComponentFactory:
    """Get global component factory."""
    global _global_factory
    if _global_factory is None:
        _global_factory = ComponentFactory()
    return _global_factory

def create_component(name: str, config: Dict[str, Any]) -> BaseComponent:
    """Convenience function to create component."""
    factory = get_component_factory()
    return factory.create(name, config)

def register_component(name: str, component_class: Type[BaseComponent],
                      singleton: bool = False, dependencies: Optional[List[str]] = None) -> None:
    """Convenience function to register component."""
    factory = get_component_factory()
    factory.register_component(name, component_class, singleton, dependencies)

# Decorators for component registration
def component(name: str, singleton: bool = False, dependencies: Optional[List[str]] = None):
    """Decorator to register a component."""
    def decorator(cls):
        register_component(name, cls, singleton, dependencies)
        return cls
    return decorator

def singleton_component(name: str, dependencies: Optional[List[str]] = None):
    """Decorator to register a singleton component."""
    return component(name, singleton=True, dependencies=dependencies)

def factory(name: str):
    """Decorator to register a factory."""
    def decorator(cls):
        factory_instance = get_component_factory()
        factory_instance.register_factory(name, cls)
        return cls
    return decorator