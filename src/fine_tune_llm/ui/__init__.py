"""
Unified UI components for the fine-tune LLM platform.

This package provides a single entry point for all user interface components,
including training dashboards, risk prediction interfaces, and monitoring tools.
"""

from .manager import UIManager, start_dashboard, start_prediction_ui, start_all_ui
from .components import *
from .utils import UIConfig, ComponentTheme, LayoutManager, ThemePreset
from .theme_manager import ThemeManager, get_theme_manager, apply_theme_to_component
from .behavior_manager import BehaviorManager, get_behavior_manager, KeyboardShortcut, InteractionRule
from .shared import SharedWidgets, UIHelpers, SharedStyles

__all__ = [
    # Core managers
    'UIManager',
    'ThemeManager',
    'BehaviorManager',
    
    # Configuration and utilities
    'UIConfig',
    'ComponentTheme',
    'LayoutManager',
    'ThemePreset',
    
    # Shared components
    'SharedWidgets',
    'UIHelpers', 
    'SharedStyles',
    
    # Convenience functions
    'start_dashboard',
    'start_prediction_ui',
    'start_all_ui',
    'get_theme_manager',
    'get_behavior_manager',
    'apply_theme_to_component',
    
    # Behavior management
    'KeyboardShortcut',
    'InteractionRule',
    
    # Components will be exported dynamically
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Unified user interface components for LLM fine-tuning platform'

# Global managers
_ui_manager = None
_theme_manager = None
_behavior_manager = None

def initialize_ui_system(config: UIConfig = None) -> UIManager:
    """
    Initialize the complete UI system.
    
    Args:
        config: UI configuration
        
    Returns:
        Initialized UI manager
    """
    global _ui_manager, _theme_manager, _behavior_manager
    
    # Initialize managers
    _theme_manager = ThemeManager(config)
    _behavior_manager = BehaviorManager()
    _ui_manager = UIManager(config)
    
    # Connect managers
    _theme_manager.subscribe_to_theme_changes(
        lambda theme: _ui_manager._apply_theme()
    )
    
    return _ui_manager

def get_ui_system() -> tuple[UIManager, ThemeManager, BehaviorManager]:
    """
    Get all UI system managers.
    
    Returns:
        Tuple of (UIManager, ThemeManager, BehaviorManager)
    """
    global _ui_manager, _theme_manager, _behavior_manager
    
    if not all([_ui_manager, _theme_manager, _behavior_manager]):
        initialize_ui_system()
    
    return _ui_manager, _theme_manager, _behavior_manager

def shutdown_ui_system():
    """Shutdown the UI system and clean up resources."""
    global _ui_manager, _theme_manager, _behavior_manager
    
    if _ui_manager:
        _ui_manager.close()
        _ui_manager = None
    
    if _theme_manager:
        _theme_manager.close()
        _theme_manager = None
    
    if _behavior_manager:
        _behavior_manager.close()
        _behavior_manager = None