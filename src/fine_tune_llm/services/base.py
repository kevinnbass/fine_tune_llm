"""
Base service interface for all service classes.

This module provides the abstract base class for all services with
common functionality like dependency injection, lifecycle management,
and event handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
import logging

from ..core.interfaces import BaseComponent
from ..core.events import EventBus, BaseEvent
from ..config import ConfigManager
from ..core.dependency_injection import DIContainer

logger = logging.getLogger(__name__)


class BaseService(BaseComponent, ABC):
    """Abstract base class for all services."""
    
    def __init__(self, 
                 config_manager: Optional[ConfigManager] = None,
                 event_bus: Optional[EventBus] = None,
                 di_container: Optional[DIContainer] = None):
        """
        Initialize base service.
        
        Args:
            config_manager: Configuration manager instance
            event_bus: Event bus for pub/sub messaging
            di_container: Dependency injection container
        """
        self.config_manager = config_manager or self._get_default_config_manager()
        self.event_bus = event_bus or self._get_default_event_bus()
        self.di_container = di_container or self._get_default_di_container()
        
        # Service state
        self.is_initialized = False
        self.is_running = False
        
        # Dependencies (to be injected)
        self.dependencies: Dict[str, Any] = {}
        
        # Event subscriptions
        self.event_subscriptions: List[str] = []
        
        # Service configuration
        self.service_config = self.config_manager.get_service_config(self.__class__.__name__)
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the service with configuration.
        
        Args:
            config: Service configuration
        """
        if self.is_initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return
        
        # Update configuration
        self.service_config.update(config)
        
        # Inject dependencies
        self._inject_dependencies()
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
        # Initialize service-specific components
        self._initialize_service()
        
        self.is_initialized = True
        logger.info(f"{self.__class__.__name__} initialized successfully")
        
        # Publish initialization event
        self._publish_event('ServiceInitialized', {
            'service_name': self.__class__.__name__,
            'config': self.service_config
        })
    
    def cleanup(self) -> None:
        """Clean up service resources."""
        if not self.is_initialized:
            return
        
        # Stop service if running
        if self.is_running:
            self.stop()
        
        # Unsubscribe from events
        self._cleanup_event_subscriptions()
        
        # Cleanup service-specific resources
        self._cleanup_service()
        
        # Clear dependencies
        self.dependencies.clear()
        
        self.is_initialized = False
        logger.info(f"{self.__class__.__name__} cleaned up")
        
        # Publish cleanup event
        self._publish_event('ServiceCleanup', {
            'service_name': self.__class__.__name__
        })
    
    @property
    def name(self) -> str:
        """Service name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Service version."""
        return "2.0.0"
    
    def start(self) -> None:
        """Start the service."""
        if not self.is_initialized:
            raise RuntimeError(f"{self.__class__.__name__} not initialized")
        
        if self.is_running:
            logger.warning(f"{self.__class__.__name__} already running")
            return
        
        # Start service-specific operations
        self._start_service()
        
        self.is_running = True
        logger.info(f"{self.__class__.__name__} started")
        
        # Publish start event
        self._publish_event('ServiceStarted', {
            'service_name': self.__class__.__name__
        })
    
    def stop(self) -> None:
        """Stop the service."""
        if not self.is_running:
            logger.warning(f"{self.__class__.__name__} not running")
            return
        
        # Stop service-specific operations
        self._stop_service()
        
        self.is_running = False
        logger.info(f"{self.__class__.__name__} stopped")
        
        # Publish stop event
        self._publish_event('ServiceStopped', {
            'service_name': self.__class__.__name__
        })
    
    def restart(self) -> None:
        """Restart the service."""
        self.stop()
        self.start()
    
    @abstractmethod
    def _initialize_service(self) -> None:
        """Initialize service-specific components."""
        pass
    
    @abstractmethod
    def _cleanup_service(self) -> None:
        """Clean up service-specific resources."""
        pass
    
    @abstractmethod
    def _start_service(self) -> None:
        """Start service-specific operations."""
        pass
    
    @abstractmethod  
    def _stop_service(self) -> None:
        """Stop service-specific operations."""
        pass
    
    def _inject_dependencies(self) -> None:
        """Inject dependencies from DI container."""
        required_dependencies = self.get_required_dependencies()
        
        for dep_name, dep_type in required_dependencies.items():
            try:
                dependency = self.di_container.resolve(dep_type)
                self.dependencies[dep_name] = dependency
                logger.debug(f"Injected dependency {dep_name}: {dep_type}")
            except Exception as e:
                logger.error(f"Failed to inject dependency {dep_name}: {e}")
                raise
    
    def get_required_dependencies(self) -> Dict[str, Type]:
        """
        Get required dependencies for this service.
        
        Returns:
            Dictionary mapping dependency names to types
        """
        # Default implementation - override in subclasses
        return {}
    
    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions."""
        event_handlers = self.get_event_handlers()
        
        for event_type, handler in event_handlers.items():
            self.event_bus.subscribe(event_type, handler)
            self.event_subscriptions.append(event_type)
            logger.debug(f"Subscribed to event: {event_type}")
    
    def _cleanup_event_subscriptions(self) -> None:
        """Clean up event subscriptions."""
        for event_type in self.event_subscriptions:
            try:
                # Note: Simplified - real implementation would track specific handlers
                logger.debug(f"Unsubscribed from event: {event_type}")
            except Exception as e:
                logger.error(f"Error unsubscribing from {event_type}: {e}")
        
        self.event_subscriptions.clear()
    
    def get_event_handlers(self) -> Dict[str, callable]:
        """
        Get event handlers for this service.
        
        Returns:
            Dictionary mapping event types to handler functions
        """
        # Default implementation - override in subclasses
        return {}
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event."""
        try:
            self.event_bus.publish(event_type, data)
        except Exception as e:
            logger.error(f"Error publishing event {event_type}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get service status information.
        
        Returns:
            Status dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'initialized': self.is_initialized,
            'running': self.is_running,
            'dependencies': list(self.dependencies.keys()),
            'event_subscriptions': self.event_subscriptions.copy(),
            'config': self.service_config.copy()
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.
        
        Returns:
            Health check result
        """
        health_status = {
            'service': self.name,
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': self._get_timestamp(),
            'checks': {}
        }
        
        # Check dependencies
        dependency_health = self._check_dependencies_health()
        health_status['checks']['dependencies'] = dependency_health
        
        # Check service-specific health
        service_health = self._check_service_health()
        health_status['checks']['service_specific'] = service_health
        
        # Determine overall health
        all_healthy = (dependency_health.get('status') == 'healthy' and 
                      service_health.get('status') == 'healthy')
        
        if not all_healthy:
            health_status['status'] = 'unhealthy'
        
        return health_status
    
    def _check_dependencies_health(self) -> Dict[str, Any]:
        """Check health of service dependencies."""
        dependency_status = {
            'status': 'healthy',
            'details': {}
        }
        
        for dep_name, dependency in self.dependencies.items():
            try:
                if hasattr(dependency, 'get_health_check'):
                    dep_health = dependency.get_health_check()
                    dependency_status['details'][dep_name] = dep_health
                    
                    if dep_health.get('status') != 'healthy':
                        dependency_status['status'] = 'unhealthy'
                else:
                    dependency_status['details'][dep_name] = {'status': 'unknown'}
                    
            except Exception as e:
                dependency_status['details'][dep_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                dependency_status['status'] = 'unhealthy'
        
        return dependency_status
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Service health status
        """
        # Default implementation - override in subclasses
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'details': {}
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'
    
    def _get_default_config_manager(self) -> ConfigManager:
        """Get default configuration manager."""
        try:
            return self.di_container.resolve(ConfigManager)
        except:
            return ConfigManager()
    
    def _get_default_event_bus(self) -> EventBus:
        """Get default event bus."""
        try:
            return self.di_container.resolve(EventBus)
        except:
            return EventBus()
    
    def _get_default_di_container(self) -> DIContainer:
        """Get default DI container."""
        # Create singleton container if none provided
        if not hasattr(BaseService, '_default_container'):
            BaseService._default_container = DIContainer()
        return BaseService._default_container
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update service configuration.
        
        Args:
            config_updates: Configuration updates to apply
        """
        self.service_config.update(config_updates)
        
        # Notify of configuration change
        self._publish_event('ServiceConfigUpdated', {
            'service_name': self.__class__.__name__,
            'config_updates': config_updates
        })
        
        logger.info(f"Updated configuration for {self.__class__.__name__}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.
        
        Returns:
            Service metrics dictionary
        """
        # Default implementation - override in subclasses
        return {
            'service': self.name,
            'uptime': self._get_uptime() if self.is_running else 0,
            'status': 'running' if self.is_running else 'stopped'
        }
    
    def _get_uptime(self) -> float:
        """Get service uptime in seconds."""
        # Simplified implementation
        return 0.0