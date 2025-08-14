"""
UI utilities for configuration, theming, and layout management.

This module provides utility classes and functions for managing UI components
with consistent styling, configuration, and behavior.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import colorsys

logger = logging.getLogger(__name__)


class ThemePreset(Enum):
    """Predefined theme presets."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    BLUE = "blue"
    GREEN = "green"
    PURPLE = "purple"
    PROFESSIONAL = "professional"


@dataclass
class ComponentTheme:
    """UI theme configuration."""
    name: str
    base_theme: str  # 'light' or 'dark'
    primary_color: str
    background_color: str
    secondary_color: str
    text_color: str
    font_family: str
    border_radius: str
    spacing: str
    
    @classmethod
    def from_preset(cls, preset: ThemePreset) -> 'ComponentTheme':
        """Create theme from preset."""
        presets = {
            ThemePreset.DEFAULT: cls(
                name="default",
                base_theme="light",
                primary_color="#FF4B4B",
                background_color="#FFFFFF",
                secondary_color="#F0F2F6",
                text_color="#262730",
                font_family="sans serif",
                border_radius="0.25rem",
                spacing="1rem"
            ),
            ThemePreset.DARK: cls(
                name="dark",
                base_theme="dark",
                primary_color="#FF6B6B",
                background_color="#0E1117",
                secondary_color="#262730",
                text_color="#FAFAFA",
                font_family="sans serif",
                border_radius="0.25rem",
                spacing="1rem"
            ),
            ThemePreset.LIGHT: cls(
                name="light",
                base_theme="light", 
                primary_color="#1F77B4",
                background_color="#FFFFFF",
                secondary_color="#F8F9FA",
                text_color="#212529",
                font_family="sans serif",
                border_radius="0.25rem",
                spacing="1rem"
            ),
            ThemePreset.BLUE: cls(
                name="blue",
                base_theme="light",
                primary_color="#0066CC",
                background_color="#F8F9FF",
                secondary_color="#E6F2FF",
                text_color="#1A1A1A",
                font_family="sans serif",
                border_radius="0.5rem",
                spacing="1rem"
            ),
            ThemePreset.GREEN: cls(
                name="green",
                base_theme="light",
                primary_color="#28A745",
                background_color="#F8FFF8",
                secondary_color="#E6F7E6",
                text_color="#1A1A1A",
                font_family="sans serif",
                border_radius="0.5rem",
                spacing="1rem"
            ),
            ThemePreset.PURPLE: cls(
                name="purple",
                base_theme="dark",
                primary_color="#8B5CF6",
                background_color="#1A1625",
                secondary_color="#2D2438",
                text_color="#F3F4F6",
                font_family="sans serif",
                border_radius="0.75rem",
                spacing="1.2rem"
            ),
            ThemePreset.PROFESSIONAL: cls(
                name="professional",
                base_theme="light",
                primary_color="#2563EB",
                background_color="#FEFEFE",
                secondary_color="#F1F5F9",
                text_color="#334155",
                font_family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                border_radius="0.375rem",
                spacing="1rem"
            )
        }
        
        return presets[preset]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ComponentTheme':
        """Create theme from configuration dictionary."""
        if 'preset' in config:
            preset = ThemePreset(config['preset'])
            theme = cls.from_preset(preset)
            
            # Override with custom values
            for key, value in config.items():
                if key != 'preset' and hasattr(theme, key):
                    setattr(theme, key, value)
            
            return theme
        
        # Create custom theme
        return cls(
            name=config.get('name', 'custom'),
            base_theme=config.get('base_theme', 'light'),
            primary_color=config.get('primary_color', '#FF4B4B'),
            background_color=config.get('background_color', '#FFFFFF'),
            secondary_color=config.get('secondary_color', '#F0F2F6'),
            text_color=config.get('text_color', '#262730'),
            font_family=config.get('font_family', 'sans serif'),
            border_radius=config.get('border_radius', '0.25rem'),
            spacing=config.get('spacing', '1rem')
        )
    
    def to_css_vars(self) -> Dict[str, str]:
        """Convert theme to CSS custom properties."""
        return {
            '--primary-color': self.primary_color,
            '--background-color': self.background_color,
            '--secondary-color': self.secondary_color,
            '--text-color': self.text_color,
            '--font-family': self.font_family,
            '--border-radius': self.border_radius,
            '--spacing': self.spacing
        }
    
    def to_streamlit_config(self) -> Dict[str, str]:
        """Convert theme to Streamlit theme configuration."""
        return {
            'primaryColor': self.primary_color,
            'backgroundColor': self.background_color,
            'secondaryBackgroundColor': self.secondary_color,
            'textColor': self.text_color,
            'font': self.font_family
        }
    
    def get_accent_colors(self, count: int = 5) -> List[str]:
        """Generate accent colors based on primary color."""
        # Convert hex to HSV
        primary_rgb = tuple(int(self.primary_color[i:i+2], 16) for i in (1, 3, 5))
        primary_hsv = colorsys.rgb_to_hsv(*[c/255.0 for c in primary_rgb])
        
        colors = []
        for i in range(count):
            # Vary hue slightly
            hue_offset = (i - count//2) * 0.1
            new_hue = (primary_hsv[0] + hue_offset) % 1.0
            
            # Vary saturation and value slightly
            sat_factor = 0.8 + (i / count) * 0.4
            val_factor = 0.7 + (i / count) * 0.3
            
            new_rgb = colorsys.hsv_to_rgb(
                new_hue,
                min(1.0, primary_hsv[1] * sat_factor),
                min(1.0, primary_hsv[2] * val_factor)
            )
            
            # Convert back to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(new_rgb[0] * 255),
                int(new_rgb[1] * 255),
                int(new_rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors


@dataclass
class LayoutConfig:
    """Layout configuration for UI components."""
    sidebar_width: str = "300px"
    main_width: str = "auto"
    header_height: str = "60px"
    footer_height: str = "40px"
    padding: str = "1rem"
    gap: str = "1rem"
    columns: List[str] = None
    responsive_breakpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = ["1fr"]
        
        if self.responsive_breakpoints is None:
            self.responsive_breakpoints = {
                'mobile': '768px',
                'tablet': '1024px', 
                'desktop': '1440px'
            }


@dataclass
class UIConfig:
    """Configuration for UI components."""
    # General settings
    title: str = "Fine-Tune LLM Platform"
    favicon: Optional[str] = None
    logo: Optional[str] = None
    
    # Theme settings
    theme_config: Dict[str, Any] = None
    
    # Layout settings
    layout: LayoutConfig = None
    
    # Component settings
    auto_start_dashboard: bool = False
    auto_start_prediction: bool = False
    auto_refresh: bool = True
    refresh_interval: int = 5
    
    # Advanced settings
    enable_authentication: bool = False
    enable_analytics: bool = True
    debug_mode: bool = False
    
    # Network settings
    host: str = "localhost"
    base_port: int = 8500
    max_components: int = 10
    
    def __post_init__(self):
        if self.theme_config is None:
            self.theme_config = {'preset': 'default'}
        
        if self.layout is None:
            self.layout = LayoutConfig()
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'UIConfig':
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Handle nested layout config
            if 'layout' in data:
                data['layout'] = LayoutConfig(**data['layout'])
            
            return cls(**data)
            
        except Exception as e:
            logger.error(f"Failed to load UI config from {config_path}: {e}")
            return cls()  # Return default config
    
    def to_file(self, config_path: Path):
        """Save configuration to file."""
        try:
            # Convert to dict and handle nested objects
            data = asdict(self)
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved UI config to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save UI config to {config_path}: {e}")


class LayoutManager:
    """Manages UI layouts and responsive design."""
    
    def __init__(self, config: LayoutConfig):
        self.config = config
    
    def get_grid_template(self, columns: Optional[List[str]] = None) -> str:
        """Get CSS grid template for layout."""
        cols = columns or self.config.columns
        return f"grid-template-columns: {' '.join(cols)};"
    
    def get_responsive_css(self) -> str:
        """Generate responsive CSS based on breakpoints."""
        css_rules = []
        
        for device, width in self.config.responsive_breakpoints.items():
            if device == 'mobile':
                # Mobile styles
                css_rules.append(f"""
                @media (max-width: {width}) {{
                    .ui-container {{
                        padding: 0.5rem;
                        grid-template-columns: 1fr;
                    }}
                    .ui-sidebar {{
                        display: none;
                    }}
                }}
                """)
            elif device == 'tablet':
                # Tablet styles
                css_rules.append(f"""
                @media (min-width: {self.config.responsive_breakpoints['mobile']}) and (max-width: {width}) {{
                    .ui-container {{
                        padding: 0.75rem;
                        grid-template-columns: 250px 1fr;
                    }}
                }}
                """)
        
        return '\n'.join(css_rules)
    
    def get_component_css(self, theme: ComponentTheme) -> str:
        """Generate component CSS with theme variables."""
        css_vars = theme.to_css_vars()
        var_declarations = '\n'.join([f"    {k}: {v};" for k, v in css_vars.items()])
        
        return f"""
        :root {{
{var_declarations}
        }}
        
        .ui-container {{
            font-family: var(--font-family);
            color: var(--text-color);
            background-color: var(--background-color);
            padding: var(--spacing);
            gap: var(--spacing);
            border-radius: var(--border-radius);
        }}
        
        .ui-card {{
            background-color: var(--secondary-color);
            border-radius: var(--border-radius);
            padding: var(--spacing);
            margin-bottom: var(--spacing);
        }}
        
        .ui-button {{
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: calc(var(--spacing) * 0.5) var(--spacing);
            font-family: var(--font-family);
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        
        .ui-button:hover {{
            opacity: 0.9;
        }}
        
        {self.get_responsive_css()}
        """


def generate_color_palette(base_color: str, count: int = 8) -> List[str]:
    """
    Generate a color palette from a base color.
    
    Args:
        base_color: Base color in hex format
        count: Number of colors to generate
        
    Returns:
        List of hex color codes
    """
    # Convert hex to RGB
    rgb = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
    hsv = colorsys.rgb_to_hsv(*[c/255.0 for c in rgb])
    
    colors = [base_color]  # Include original color
    
    for i in range(1, count):
        # Create variations by adjusting saturation and value
        sat_factor = 0.3 + (i / count) * 0.7
        val_factor = 0.4 + (i / count) * 0.6
        
        new_rgb = colorsys.hsv_to_rgb(
            hsv[0],
            min(1.0, hsv[1] * sat_factor),
            min(1.0, hsv[2] * val_factor)
        )
        
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(new_rgb[0] * 255),
            int(new_rgb[1] * 255),
            int(new_rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors


def validate_css_color(color: str) -> bool:
    """
    Validate CSS color format.
    
    Args:
        color: Color string to validate
        
    Returns:
        True if valid CSS color
    """
    import re
    
    # Hex colors
    if re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color):
        return True
    
    # RGB/RGBA colors
    if re.match(r'^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*(0|1|0?\.\d+))?\s*\)$', color):
        return True
    
    # Named colors (basic set)
    named_colors = {
        'black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
        'silver', 'gray', 'maroon', 'olive', 'lime', 'aqua', 'teal', 'navy',
        'fuchsia', 'purple', 'transparent'
    }
    
    if color.lower() in named_colors:
        return True
    
    return False


def create_gradient(start_color: str, end_color: str, steps: int = 10) -> List[str]:
    """
    Create gradient colors between two colors.
    
    Args:
        start_color: Starting color in hex format
        end_color: Ending color in hex format  
        steps: Number of gradient steps
        
    Returns:
        List of gradient colors
    """
    # Convert hex to RGB
    start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    
    colors = []
    
    for i in range(steps):
        ratio = i / (steps - 1)
        
        # Interpolate RGB values
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        colors.append(hex_color)
    
    return colors