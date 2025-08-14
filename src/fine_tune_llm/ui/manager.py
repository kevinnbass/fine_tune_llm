"""
UI Manager for centralized user interface component management.

This module provides a unified interface for managing all UI components,
including dashboards, prediction interfaces, and monitoring tools.
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
import time

from .utils import UIConfig, ComponentTheme
from .components import get_available_components
from ..core.exceptions import SystemError, ConfigurationError

logger = logging.getLogger(__name__)


class UIType(Enum):
    """Available UI component types."""
    DASHBOARD = "dashboard"
    RISK_PREDICTION = "risk_prediction" 
    MONITORING = "monitoring"
    CONFIGURATION = "configuration"
    ANALYSIS = "analysis"


@dataclass
class UIComponent:
    """UI component configuration."""
    name: str
    type: UIType
    description: str
    entry_point: str
    port: int
    config: Dict[str, Any]
    dependencies: List[str]
    auto_start: bool = False
    process: Optional[subprocess.Popen] = None


class UIManager:
    """
    Centralized UI component manager.
    
    Manages all user interface components with unified configuration,
    theming, and lifecycle management.
    """
    
    def __init__(self, config: Optional[UIConfig] = None):
        """
        Initialize UI manager.
        
        Args:
            config: UI configuration object
        """
        self.config = config or UIConfig()
        self.theme = ComponentTheme.from_config(self.config.theme_config)
        
        # Component registry
        self._components: Dict[str, UIComponent] = {}
        self._running_components: Dict[str, subprocess.Popen] = {}
        
        # Status tracking
        self._status = {
            'initialized': True,
            'components_loaded': False,
            'theme_applied': False
        }
        
        # Initialize components
        self._load_components()
        self._apply_theme()
        
        logger.info(f"Initialized UIManager with theme: {self.theme.name}")
    
    def _load_components(self):
        """Load available UI components."""
        try:
            # Training dashboard
            self._components['training_dashboard'] = UIComponent(
                name="Training Dashboard",
                type=UIType.DASHBOARD,
                description="Real-time training metrics and visualization",
                entry_point="streamlit run scripts/run_dashboard.py",
                port=8501,
                config={
                    'auto_refresh': True,
                    'refresh_interval': 5,
                    'show_advanced_metrics': True
                },
                dependencies=['streamlit', 'plotly'],
                auto_start=self.config.auto_start_dashboard
            )
            
            # Risk prediction interface
            self._components['risk_prediction'] = UIComponent(
                name="Risk Prediction Interface",
                type=UIType.RISK_PREDICTION,
                description="Interactive risk-controlled predictions",
                entry_point="streamlit run scripts/risk_prediction_ui.py",
                port=8502,
                config={
                    'confidence_levels': [0.8, 0.9, 0.95],
                    'cost_matrix_editable': True,
                    'show_uncertainty': True
                },
                dependencies=['streamlit', 'plotly', 'numpy'],
                auto_start=self.config.auto_start_prediction
            )
            
            # Configuration interface
            self._components['configuration'] = UIComponent(
                name="Configuration Manager",
                type=UIType.CONFIGURATION,
                description="Dynamic configuration management",
                entry_point="python -m src.fine_tune_llm.ui.components.config_ui",
                port=8503,
                config={
                    'allow_hot_reload': True,
                    'show_validation_errors': True,
                    'backup_on_change': True
                },
                dependencies=['streamlit'],
                auto_start=False
            )
            
            # Monitoring interface
            self._components['monitoring'] = UIComponent(
                name="System Monitor",
                type=UIType.MONITORING,
                description="System health and resource monitoring",
                entry_point="python -m src.fine_tune_llm.ui.components.monitor_ui",
                port=8504,
                config={
                    'show_gpu_metrics': True,
                    'alert_thresholds': {
                        'cpu_usage': 80,
                        'memory_usage': 85,
                        'disk_usage': 90
                    }
                },
                dependencies=['streamlit', 'psutil'],
                auto_start=False
            )
            
            # Analysis interface  
            self._components['analysis'] = UIComponent(
                name="Model Analysis",
                type=UIType.ANALYSIS,
                description="Model performance analysis and debugging",
                entry_point="python -m src.fine_tune_llm.ui.components.analysis_ui",
                port=8505,
                config={
                    'show_bias_analysis': True,
                    'show_calibration_plots': True,
                    'interactive_exploration': True
                },
                dependencies=['streamlit', 'plotly', 'scikit-learn'],
                auto_start=False
            )
            
            self._status['components_loaded'] = True
            logger.info(f"Loaded {len(self._components)} UI components")
            
        except Exception as e:
            raise SystemError(f"Failed to load UI components: {e}")
    
    def _apply_theme(self):
        """Apply theme configuration to components."""
        try:
            # Set Streamlit theme environment variables
            theme_env = {
                'STREAMLIT_THEME_PRIMARY_COLOR': self.theme.primary_color,
                'STREAMLIT_THEME_BACKGROUND_COLOR': self.theme.background_color,
                'STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR': self.theme.secondary_color,
                'STREAMLIT_THEME_TEXT_COLOR': self.theme.text_color,
                'STREAMLIT_THEME_FONT': self.theme.font_family
            }
            
            os.environ.update(theme_env)
            self._status['theme_applied'] = True
            
            logger.info(f"Applied theme: {self.theme.name}")
            
        except Exception as e:
            logger.warning(f"Failed to apply theme: {e}")
    
    def list_components(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available UI components.
        
        Returns:
            Dictionary of component information
        """
        components_info = {}
        
        for comp_id, component in self._components.items():
            components_info[comp_id] = {
                'name': component.name,
                'type': component.type.value,
                'description': component.description,
                'port': component.port,
                'running': comp_id in self._running_components,
                'auto_start': component.auto_start,
                'dependencies': component.dependencies
            }
        
        return components_info
    
    def start_component(self, 
                       component_id: str, 
                       args: Optional[Dict[str, Any]] = None,
                       wait_for_startup: bool = True) -> bool:
        """
        Start a UI component.
        
        Args:
            component_id: Component identifier
            args: Additional arguments for component
            wait_for_startup: Wait for component to start
            
        Returns:
            True if started successfully
        """
        if component_id not in self._components:
            raise ConfigurationError(f"Unknown component: {component_id}")
        
        component = self._components[component_id]
        
        # Check if already running
        if component_id in self._running_components:
            logger.warning(f"Component {component_id} is already running")
            return True
        
        try:
            # Check dependencies
            if not self._check_dependencies(component.dependencies):
                raise SystemError(f"Missing dependencies for {component_id}")
            
            # Build command
            cmd = component.entry_point.split()
            
            # Add port argument if supported
            if 'streamlit' in component.entry_point:
                cmd.extend(['--server.port', str(component.port)])
                cmd.extend(['--server.headless', 'true'])
                
                # Apply theme
                if self.theme.name != 'default':
                    cmd.extend(['--theme.base', self.theme.base_theme])
            
            # Add custom arguments
            if args:
                for key, value in args.items():
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(value)])
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path.cwd())
            )
            
            # Wait for startup if requested
            if wait_for_startup:
                if not self._wait_for_startup(component.port, timeout=30):
                    process.terminate()
                    raise SystemError(f"Component {component_id} failed to start")
            
            self._running_components[component_id] = process
            component.process = process
            
            logger.info(f"Started component {component_id} on port {component.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start component {component_id}: {e}")
            return False
    
    def stop_component(self, component_id: str, force: bool = False) -> bool:
        """
        Stop a UI component.
        
        Args:
            component_id: Component identifier
            force: Force kill if graceful shutdown fails
            
        Returns:
            True if stopped successfully
        """
        if component_id not in self._running_components:
            logger.warning(f"Component {component_id} is not running")
            return True
        
        try:
            process = self._running_components[component_id]
            
            # Graceful shutdown
            process.terminate()
            
            # Wait for shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if force:
                    process.kill()
                    process.wait(timeout=5)
                else:
                    logger.warning(f"Component {component_id} did not shutdown gracefully")
                    return False
            
            # Clean up
            del self._running_components[component_id]
            self._components[component_id].process = None
            
            logger.info(f"Stopped component {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop component {component_id}: {e}")
            return False
    
    def restart_component(self, component_id: str, **kwargs) -> bool:
        """
        Restart a UI component.
        
        Args:
            component_id: Component identifier
            **kwargs: Arguments for start_component
            
        Returns:
            True if restarted successfully
        """
        if component_id in self._running_components:
            if not self.stop_component(component_id):
                return False
        
        return self.start_component(component_id, **kwargs)
    
    def start_all(self, auto_start_only: bool = True) -> Dict[str, bool]:
        """
        Start all UI components.
        
        Args:
            auto_start_only: Only start components with auto_start=True
            
        Returns:
            Dictionary of start results per component
        """
        results = {}
        
        for comp_id, component in self._components.items():
            if auto_start_only and not component.auto_start:
                continue
            
            try:
                results[comp_id] = self.start_component(
                    comp_id, 
                    wait_for_startup=False  # Don't wait for each component
                )
            except Exception as e:
                logger.error(f"Failed to start {comp_id}: {e}")
                results[comp_id] = False
        
        return results
    
    def stop_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Stop all running UI components.
        
        Args:
            force: Force kill components
            
        Returns:
            Dictionary of stop results per component
        """
        results = {}
        
        for comp_id in list(self._running_components.keys()):
            results[comp_id] = self.stop_component(comp_id, force=force)
        
        return results
    
    def get_component_status(self, component_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Component status information
        """
        if component_id not in self._components:
            raise ConfigurationError(f"Unknown component: {component_id}")
        
        component = self._components[component_id]
        is_running = component_id in self._running_components
        
        status = {
            'name': component.name,
            'type': component.type.value,
            'running': is_running,
            'port': component.port,
            'auto_start': component.auto_start,
            'config': component.config.copy()
        }
        
        if is_running:
            process = self._running_components[component_id]
            status.update({
                'pid': process.pid,
                'poll': process.poll(),
                'url': f"http://localhost:{component.port}"
            })
        
        return status
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall UI system status.
        
        Returns:
            System status information
        """
        running_count = len(self._running_components)
        total_count = len(self._components)
        
        return {
            'ui_manager': self._status,
            'components': {
                'total': total_count,
                'running': running_count,
                'stopped': total_count - running_count
            },
            'theme': {
                'name': self.theme.name,
                'applied': self._status['theme_applied']
            },
            'ports_in_use': [
                self._components[comp_id].port 
                for comp_id in self._running_components
            ]
        }
    
    def open_component(self, component_id: str) -> bool:
        """
        Open component in browser.
        
        Args:
            component_id: Component identifier
            
        Returns:
            True if opened successfully
        """
        if component_id not in self._running_components:
            logger.error(f"Component {component_id} is not running")
            return False
        
        try:
            import webbrowser
            port = self._components[component_id].port
            url = f"http://localhost:{port}"
            webbrowser.open(url)
            logger.info(f"Opened {component_id} in browser: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open {component_id} in browser: {e}")
            return False
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if component dependencies are available."""
        try:
            import importlib
            
            for dep in dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    logger.error(f"Missing dependency: {dep}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    def _wait_for_startup(self, port: int, timeout: int = 30) -> bool:
        """Wait for component to start on specified port."""
        import socket
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    return True
                    
            except Exception:
                pass
            
            time.sleep(1)
        
        return False
    
    def close(self):
        """Close UI manager and stop all components."""
        try:
            self.stop_all(force=True)
            logger.info("UI manager closed")
            
        except Exception as e:
            logger.error(f"Error closing UI manager: {e}")


# Convenience functions

def start_dashboard(port: int = 8501, **kwargs) -> UIManager:
    """
    Start training dashboard quickly.
    
    Args:
        port: Dashboard port
        **kwargs: Additional arguments
        
    Returns:
        UIManager instance
    """
    config = UIConfig(auto_start_dashboard=True)
    manager = UIManager(config)
    
    # Override port if specified
    if port != 8501:
        manager._components['training_dashboard'].port = port
    
    manager.start_component('training_dashboard', kwargs)
    return manager


def start_prediction_ui(port: int = 8502, **kwargs) -> UIManager:
    """
    Start risk prediction UI quickly.
    
    Args:
        port: UI port
        **kwargs: Additional arguments
        
    Returns:
        UIManager instance
    """
    config = UIConfig(auto_start_prediction=True)
    manager = UIManager(config)
    
    # Override port if specified
    if port != 8502:
        manager._components['risk_prediction'].port = port
    
    manager.start_component('risk_prediction', kwargs)
    return manager


def start_all_ui(**kwargs) -> UIManager:
    """
    Start all UI components.
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        UIManager instance
    """
    config = UIConfig(
        auto_start_dashboard=True,
        auto_start_prediction=True,
        **kwargs
    )
    manager = UIManager(config)
    manager.start_all()
    return manager