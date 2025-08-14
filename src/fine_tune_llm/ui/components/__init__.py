"""
UI Components for the fine-tune LLM platform.

This package provides individual UI components that can be used
independently or orchestrated through the UIManager.
"""

from .base import BaseUIComponent
from .dashboard import TrainingDashboard
from .prediction import RiskPredictionInterface
from .config import ConfigurationInterface
from .monitor import SystemMonitor
from .analysis import ModelAnalysis

# Component registry
AVAILABLE_COMPONENTS = {
    'dashboard': TrainingDashboard,
    'prediction': RiskPredictionInterface,
    'config': ConfigurationInterface,
    'monitor': SystemMonitor,
    'analysis': ModelAnalysis
}

def get_available_components():
    """Get dictionary of available UI components."""
    return AVAILABLE_COMPONENTS.copy()

def create_component(component_type: str, **kwargs):
    """Create UI component instance by type."""
    if component_type not in AVAILABLE_COMPONENTS:
        raise ValueError(f"Unknown component type: {component_type}")
    
    component_class = AVAILABLE_COMPONENTS[component_type]
    return component_class(**kwargs)

__all__ = [
    'BaseUIComponent',
    'TrainingDashboard',
    'RiskPredictionInterface',
    'ConfigurationInterface',
    'SystemMonitor',
    'ModelAnalysis',
    'AVAILABLE_COMPONENTS',
    'get_available_components',
    'create_component'
]