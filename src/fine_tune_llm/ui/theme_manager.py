"""
Theme Manager for consistent styling across all UI components.

This module provides centralized theme management with dynamic theme switching,
CSS generation, and consistent styling enforcement across all UI components.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import time

from .utils import ComponentTheme, ThemePreset, UIConfig
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ThemeRegistry:
    """Registry of available themes."""
    themes: Dict[str, ComponentTheme]
    active_theme: str
    custom_themes: Dict[str, ComponentTheme]
    
    def __post_init__(self):
        if not self.themes:
            self.themes = {}
        if not self.custom_themes:
            self.custom_themes = {}


class ThemeManager:
    """
    Centralized theme management system.
    
    Manages theme application, CSS generation, and consistent styling
    across all UI components with hot-reloading capabilities.
    """
    
    def __init__(self, config: Optional[UIConfig] = None):
        """
        Initialize theme manager.
        
        Args:
            config: UI configuration
        """
        self.config = config or UIConfig()
        
        # Theme registry
        self.registry = ThemeRegistry(
            themes={},
            active_theme='default',
            custom_themes={}
        )
        
        # Component subscribers for theme changes
        self._subscribers: List[Callable[[ComponentTheme], None]] = []
        self._lock = threading.RLock()
        
        # CSS cache for performance
        self._css_cache: Dict[str, str] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Initialize built-in themes
        self._load_builtin_themes()
        
        # Apply initial theme
        initial_theme = self.config.theme_config.get('preset', 'default')
        self.set_active_theme(initial_theme)
        
        logger.info(f"Initialized ThemeManager with {len(self.registry.themes)} themes")
    
    def _load_builtin_themes(self):
        """Load built-in theme presets."""
        try:
            for preset in ThemePreset:
                theme = ComponentTheme.from_preset(preset)
                self.registry.themes[preset.value] = theme
            
            logger.info(f"Loaded {len(ThemePreset)} built-in themes")
            
        except Exception as e:
            logger.error(f"Failed to load built-in themes: {e}")
    
    def register_theme(self, name: str, theme: ComponentTheme) -> bool:
        """
        Register a custom theme.
        
        Args:
            name: Theme name
            theme: Theme configuration
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.registry.custom_themes[name] = theme
                self.registry.themes[name] = theme
                
                # Clear CSS cache for this theme
                self._invalidate_css_cache(name)
            
            logger.info(f"Registered custom theme: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register theme {name}: {e}")
            return False
    
    def unregister_theme(self, name: str) -> bool:
        """
        Unregister a custom theme.
        
        Args:
            name: Theme name
            
        Returns:
            True if unregistered successfully
        """
        try:
            # Don't allow unregistering built-in themes
            builtin_themes = {preset.value for preset in ThemePreset}
            if name in builtin_themes:
                logger.warning(f"Cannot unregister built-in theme: {name}")
                return False
            
            with self._lock:
                if name in self.registry.custom_themes:
                    del self.registry.custom_themes[name]
                
                if name in self.registry.themes:
                    del self.registry.themes[name]
                
                # Clear CSS cache
                self._invalidate_css_cache(name)
                
                # Switch to default if this was active
                if self.registry.active_theme == name:
                    self.set_active_theme('default')
            
            logger.info(f"Unregistered custom theme: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister theme {name}: {e}")
            return False
    
    def get_available_themes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available themes.
        
        Returns:
            Dictionary of theme information
        """
        themes_info = {}
        
        with self._lock:
            for name, theme in self.registry.themes.items():
                is_custom = name in self.registry.custom_themes
                themes_info[name] = {
                    'name': theme.name,
                    'base_theme': theme.base_theme,
                    'primary_color': theme.primary_color,
                    'is_custom': is_custom,
                    'is_active': name == self.registry.active_theme
                }
        
        return themes_info
    
    def get_theme(self, name: str) -> Optional[ComponentTheme]:
        """
        Get theme by name.
        
        Args:
            name: Theme name
            
        Returns:
            Theme object or None if not found
        """
        with self._lock:
            return self.registry.themes.get(name)
    
    def get_active_theme(self) -> ComponentTheme:
        """
        Get currently active theme.
        
        Returns:
            Active theme object
        """
        with self._lock:
            return self.registry.themes[self.registry.active_theme]
    
    def set_active_theme(self, name: str) -> bool:
        """
        Set active theme and notify subscribers.
        
        Args:
            name: Theme name
            
        Returns:
            True if set successfully
        """
        try:
            with self._lock:
                if name not in self.registry.themes:
                    raise ConfigurationError(f"Theme not found: {name}")
                
                old_theme = self.registry.active_theme
                self.registry.active_theme = name
                
                # Get new theme
                new_theme = self.registry.themes[name]
                
                # Notify subscribers
                self._notify_theme_change(new_theme)
            
            logger.info(f"Changed active theme from {old_theme} to {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active theme {name}: {e}")
            return False
    
    def subscribe_to_theme_changes(self, callback: Callable[[ComponentTheme], None]):
        """
        Subscribe to theme change notifications.
        
        Args:
            callback: Function to call when theme changes
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
                logger.debug(f"Added theme change subscriber: {callback.__name__}")
    
    def unsubscribe_from_theme_changes(self, callback: Callable[[ComponentTheme], None]):
        """
        Unsubscribe from theme change notifications.
        
        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
                logger.debug(f"Removed theme change subscriber: {callback.__name__}")
    
    def _notify_theme_change(self, theme: ComponentTheme):
        """Notify all subscribers of theme change."""
        for callback in self._subscribers:
            try:
                callback(theme)
            except Exception as e:
                logger.error(f"Error notifying theme change subscriber {callback.__name__}: {e}")
    
    def generate_global_css(self, theme_name: Optional[str] = None) -> str:
        """
        Generate global CSS for theme.
        
        Args:
            theme_name: Theme name (uses active theme if None)
            
        Returns:
            Complete CSS stylesheet
        """
        if theme_name is None:
            theme_name = self.registry.active_theme
        
        # Check cache first
        cache_key = f"{theme_name}_global"
        if self._is_css_cached(cache_key):
            return self._css_cache[cache_key]
        
        try:
            theme = self.get_theme(theme_name)
            if not theme:
                raise ConfigurationError(f"Theme not found: {theme_name}")
            
            # Generate CSS
            css_parts = [
                self._generate_css_variables(theme),
                self._generate_base_styles(theme),
                self._generate_component_styles(theme),
                self._generate_responsive_styles(theme),
                self._generate_animation_styles(),
                self._generate_utility_classes(theme)
            ]
            
            complete_css = '\n\n'.join(css_parts)
            
            # Cache result
            self._cache_css(cache_key, complete_css)
            
            return complete_css
            
        except Exception as e:
            logger.error(f"Failed to generate CSS for theme {theme_name}: {e}")
            return ""
    
    def generate_component_css(self, 
                             component_name: str,
                             theme_name: Optional[str] = None,
                             additional_styles: str = "") -> str:
        """
        Generate CSS for specific component.
        
        Args:
            component_name: Component name
            theme_name: Theme name (uses active theme if None)
            additional_styles: Additional CSS to include
            
        Returns:
            Component-specific CSS
        """
        if theme_name is None:
            theme_name = self.registry.active_theme
        
        cache_key = f"{theme_name}_{component_name}"
        if self._is_css_cached(cache_key) and not additional_styles:
            return self._css_cache[cache_key]
        
        try:
            theme = self.get_theme(theme_name)
            if not theme:
                raise ConfigurationError(f"Theme not found: {theme_name}")
            
            # Generate component-specific CSS
            css_parts = [
                f"/* CSS for {component_name} component with {theme_name} theme */",
                self._generate_css_variables(theme),
                self._generate_component_base_styles(component_name, theme),
                additional_styles
            ]
            
            complete_css = '\n\n'.join(filter(None, css_parts))
            
            # Cache if no additional styles
            if not additional_styles:
                self._cache_css(cache_key, complete_css)
            
            return complete_css
            
        except Exception as e:
            logger.error(f"Failed to generate component CSS for {component_name}: {e}")
            return ""
    
    def _generate_css_variables(self, theme: ComponentTheme) -> str:
        """Generate CSS custom properties from theme."""
        vars_css = theme.to_css_vars()
        
        # Add computed colors
        accent_colors = theme.get_accent_colors(5)
        for i, color in enumerate(accent_colors):
            vars_css[f'--accent-color-{i+1}'] = color
        
        # Generate CSS
        var_declarations = []
        for var_name, var_value in vars_css.items():
            var_declarations.append(f"  {var_name}: {var_value};")
        
        return f"""
:root {{
{chr(10).join(var_declarations)}
}}
"""
    
    def _generate_base_styles(self, theme: ComponentTheme) -> str:
        """Generate base styles for theme."""
        return f"""
/* Base Styles */
* {{
    box-sizing: border-box;
}}

body {{
    font-family: var(--font-family);
    background-color: var(--background-color);
    color: var(--text-color);
    margin: 0;
    padding: 0;
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--spacing);
}}

.card {{
    background: var(--secondary-color);
    border-radius: var(--border-radius);
    padding: var(--spacing);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: var(--spacing);
}}

.btn {{
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: calc(var(--spacing) * 0.5) var(--spacing);
    font-family: var(--font-family);
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-block;
    text-decoration: none;
    text-align: center;
}}

.btn:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
}}

.btn:active {{
    transform: translateY(0);
}}

.btn-secondary {{
    background: var(--secondary-color);
    color: var(--text-color);
    border: 1px solid var(--primary-color);
}}

.btn-outline {{
    background: transparent;
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
}}

.btn-danger {{
    background: #dc3545;
}}

.btn-success {{
    background: #28a745;
}}

.btn-warning {{
    background: #ffc107;
    color: #212529;
}}
"""
    
    def _generate_component_styles(self, theme: ComponentTheme) -> str:
        """Generate common component styles."""
        return """
/* Component Styles */
.header {{
    background: var(--secondary-color);
    padding: var(--spacing);
    border-bottom: 1px solid var(--primary-color);
    margin-bottom: var(--spacing);
}}

.sidebar {{
    background: var(--secondary-color);
    padding: var(--spacing);
    border-radius: var(--border-radius);
    min-height: 100vh;
}}

.main-content {{
    padding: var(--spacing);
}}

.form-group {{
    margin-bottom: var(--spacing);
}}

.form-label {{
    display: block;
    margin-bottom: calc(var(--spacing) * 0.25);
    font-weight: bold;
}}

.form-input {{
    width: 100%;
    padding: calc(var(--spacing) * 0.5);
    border: 1px solid var(--secondary-color);
    border-radius: var(--border-radius);
    font-family: var(--font-family);
    background: var(--background-color);
    color: var(--text-color);
}}

.form-input:focus {{
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
}}

.grid {{
    display: grid;
    gap: var(--spacing);
}}

.grid-2 {{ grid-template-columns: 1fr 1fr; }}
.grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.grid-4 {{ grid-template-columns: 1fr 1fr 1fr 1fr; }}

.flex {{
    display: flex;
    gap: var(--spacing);
}}

.flex-center {{
    align-items: center;
    justify-content: center;
}}

.flex-between {{
    justify-content: space-between;
}}

.flex-column {{
    flex-direction: column;
}}
"""
    
    def _generate_responsive_styles(self, theme: ComponentTheme) -> str:
        """Generate responsive design styles."""
        return """
/* Responsive Styles */
@media (max-width: 768px) {
    .container {
        padding: calc(var(--spacing) * 0.5);
    }
    
    .grid-2, .grid-3, .grid-4 {
        grid-template-columns: 1fr;
    }
    
    .sidebar {
        min-height: auto;
    }
    
    .flex {
        flex-direction: column;
    }
    
    .btn {
        width: 100%;
        margin-bottom: calc(var(--spacing) * 0.5);
    }
}

@media (min-width: 769px) and (max-width: 1024px) {
    .grid-4 {
        grid-template-columns: 1fr 1fr;
    }
}
"""
    
    def _generate_animation_styles(self) -> str:
        """Generate animation and transition styles."""
        return """
/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideIn {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.fade-in {
    animation: fadeIn 0.3s ease-in-out;
}

.slide-in {
    animation: slideIn 0.3s ease-in-out;
}

.pulse {
    animation: pulse 2s infinite;
}

.spin {
    animation: spin 1s linear infinite;
}

.transition-all {
    transition: all 0.2s ease;
}
"""
    
    def _generate_utility_classes(self, theme: ComponentTheme) -> str:
        """Generate utility classes."""
        accent_colors = theme.get_accent_colors(5)
        
        # Generate color utilities
        color_utilities = []
        for i, color in enumerate(accent_colors):
            color_utilities.append(f".text-accent-{i+1} {{ color: {color}; }}")
            color_utilities.append(f".bg-accent-{i+1} {{ background-color: {color}; }}")
        
        return f"""
/* Utility Classes */
.text-primary {{ color: var(--primary-color); }}
.text-secondary {{ color: var(--secondary-color); }}
.text-muted {{ opacity: 0.7; }}

.bg-primary {{ background-color: var(--primary-color); }}
.bg-secondary {{ background-color: var(--secondary-color); }}

{chr(10).join(color_utilities)}

.p-0 {{ padding: 0; }}
.p-1 {{ padding: calc(var(--spacing) * 0.25); }}
.p-2 {{ padding: calc(var(--spacing) * 0.5); }}
.p-3 {{ padding: var(--spacing); }}
.p-4 {{ padding: calc(var(--spacing) * 1.5); }}

.m-0 {{ margin: 0; }}
.m-1 {{ margin: calc(var(--spacing) * 0.25); }}
.m-2 {{ margin: calc(var(--spacing) * 0.5); }}
.m-3 {{ margin: var(--spacing); }}
.m-4 {{ margin: calc(var(--spacing) * 1.5); }}

.text-left {{ text-align: left; }}
.text-center {{ text-align: center; }}
.text-right {{ text-align: right; }}

.font-bold {{ font-weight: bold; }}
.font-normal {{ font-weight: normal; }}

.hidden {{ display: none; }}
.visible {{ display: block; }}

.rounded {{ border-radius: var(--border-radius); }}
.rounded-full {{ border-radius: 50%; }}

.shadow {{ box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }}
.shadow-lg {{ box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); }}
"""
    
    def _generate_component_base_styles(self, component_name: str, theme: ComponentTheme) -> str:
        """Generate base styles for specific component."""
        component_styles = {
            'dashboard': """
.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--spacing);
}

.dashboard-metric {
    background: var(--secondary-color);
    padding: var(--spacing);
    border-radius: var(--border-radius);
    border-left: 4px solid var(--primary-color);
}
""",
            'prediction': """
.prediction-interface {
    display: grid;
    grid-template-rows: auto 1fr auto;
    gap: var(--spacing);
    height: 100vh;
}

.confidence-bar {
    height: 6px;
    background: var(--secondary-color);
    border-radius: 3px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: var(--primary-color);
    transition: width 0.3s ease;
}
""",
            'monitor': """
.monitor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing);
}

.resource-meter {
    position: relative;
    height: 8px;
    background: var(--secondary-color);
    border-radius: 4px;
    overflow: hidden;
}

.resource-fill {
    height: 100%;
    transition: width 0.3s ease, background-color 0.3s ease;
}
"""
        }
        
        return component_styles.get(component_name, "/* No specific styles for this component */")
    
    def _is_css_cached(self, cache_key: str) -> bool:
        """Check if CSS is cached and valid."""
        if cache_key not in self._css_cache:
            return False
        
        # Check if cache is still valid (1 hour)
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return time.time() - timestamp < 3600
    
    def _cache_css(self, cache_key: str, css: str):
        """Cache CSS result."""
        self._css_cache[cache_key] = css
        self._cache_timestamps[cache_key] = time.time()
    
    def _invalidate_css_cache(self, theme_name: str):
        """Invalidate CSS cache for theme."""
        keys_to_remove = [key for key in self._css_cache.keys() if theme_name in key]
        for key in keys_to_remove:
            del self._css_cache[key]
            del self._cache_timestamps[key]
    
    def export_theme(self, theme_name: str, export_path: Path) -> bool:
        """
        Export theme to file.
        
        Args:
            theme_name: Theme to export
            export_path: Export file path
            
        Returns:
            True if exported successfully
        """
        try:
            theme = self.get_theme(theme_name)
            if not theme:
                raise ConfigurationError(f"Theme not found: {theme_name}")
            
            # Convert to exportable format
            theme_data = asdict(theme)
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(theme_data, f, indent=2)
            
            logger.info(f"Exported theme {theme_name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export theme {theme_name}: {e}")
            return False
    
    def import_theme(self, import_path: Path, theme_name: Optional[str] = None) -> bool:
        """
        Import theme from file.
        
        Args:
            import_path: Import file path
            theme_name: Name for imported theme (uses file name if None)
            
        Returns:
            True if imported successfully
        """
        try:
            with open(import_path, 'r') as f:
                theme_data = json.load(f)
            
            # Create theme object
            theme = ComponentTheme(**theme_data)
            
            # Use filename as theme name if not specified
            if theme_name is None:
                theme_name = import_path.stem
            
            # Register theme
            return self.register_theme(theme_name, theme)
            
        except Exception as e:
            logger.error(f"Failed to import theme from {import_path}: {e}")
            return False
    
    def create_theme_variant(self, 
                           base_theme: str,
                           variant_name: str,
                           modifications: Dict[str, Any]) -> bool:
        """
        Create theme variant based on existing theme.
        
        Args:
            base_theme: Base theme name
            variant_name: New variant name
            modifications: Theme properties to modify
            
        Returns:
            True if created successfully
        """
        try:
            base = self.get_theme(base_theme)
            if not base:
                raise ConfigurationError(f"Base theme not found: {base_theme}")
            
            # Create variant
            variant_data = asdict(base)
            variant_data.update(modifications)
            variant_data['name'] = variant_name
            
            variant_theme = ComponentTheme(**variant_data)
            
            return self.register_theme(variant_name, variant_theme)
            
        except Exception as e:
            logger.error(f"Failed to create theme variant {variant_name}: {e}")
            return False
    
    def get_theme_statistics(self) -> Dict[str, Any]:
        """
        Get theme manager statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                'total_themes': len(self.registry.themes),
                'builtin_themes': len(ThemePreset),
                'custom_themes': len(self.registry.custom_themes),
                'active_theme': self.registry.active_theme,
                'subscribers': len(self._subscribers),
                'css_cache_size': len(self._css_cache),
                'available_themes': list(self.registry.themes.keys())
            }
    
    def clear_css_cache(self):
        """Clear CSS cache."""
        with self._lock:
            self._css_cache.clear()
            self._cache_timestamps.clear()
        logger.info("Cleared CSS cache")
    
    def close(self):
        """Clean up theme manager."""
        try:
            with self._lock:
                self._subscribers.clear()
                self.clear_css_cache()
            
            logger.info("Theme manager closed")
            
        except Exception as e:
            logger.error(f"Error closing theme manager: {e}")


# Convenience functions

def get_theme_manager() -> ThemeManager:
    """Get global theme manager instance."""
    if not hasattr(get_theme_manager, '_instance'):
        get_theme_manager._instance = ThemeManager()
    return get_theme_manager._instance


def apply_theme_to_component(component_name: str, theme_name: str = None) -> str:
    """
    Apply theme to component and return CSS.
    
    Args:
        component_name: Component name
        theme_name: Theme name (uses active if None)
        
    Returns:
        CSS for component
    """
    manager = get_theme_manager()
    return manager.generate_component_css(component_name, theme_name)