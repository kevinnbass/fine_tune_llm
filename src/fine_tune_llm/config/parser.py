"""
Unified Configuration Parser for all components.

This module provides a single, centralized configuration parser that can handle
all component configurations with consistent parsing, validation, and transformation.
"""

import os
import json
import yaml
import toml
import logging
from typing import Dict, Any, List, Optional, Union, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from ..core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


@dataclass
class ParseResult:
    """Configuration parsing result."""
    data: Dict[str, Any]
    format: ConfigFormat
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ParserConfig:
    """Parser configuration."""
    allow_environment_override: bool = True
    environment_prefix: str = "FTL_"
    merge_strategy: str = "deep"  # "shallow", "deep", "replace"
    include_support: bool = True
    template_support: bool = True
    validation_enabled: bool = True
    auto_type_conversion: bool = True
    preserve_comments: bool = False


class ConfigurationParser:
    """
    Unified configuration parser for all components.
    
    Supports multiple configuration formats, environment variable overrides,
    configuration includes, templating, and comprehensive validation.
    """
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize configuration parser.
        
        Args:
            config: Parser configuration
        """
        self.config = config or ParserConfig()
        
        # Format handlers
        self._format_handlers: Dict[ConfigFormat, Callable] = {
            ConfigFormat.JSON: self._parse_json,
            ConfigFormat.YAML: self._parse_yaml,
            ConfigFormat.TOML: self._parse_toml,
            ConfigFormat.INI: self._parse_ini,
            ConfigFormat.ENV: self._parse_env
        }
        
        # Type converters
        self._type_converters: Dict[str, Callable] = {
            'int': int,
            'float': float,
            'bool': self._parse_bool,
            'list': self._parse_list,
            'dict': self._parse_dict,
            'path': Path,
            'datetime': self._parse_datetime
        }
        
        # Template variables
        self._template_vars: Dict[str, Any] = {}
        
        # Include cache
        self._include_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized ConfigurationParser")
    
    def parse_file(self, file_path: Union[str, Path]) -> ParseResult:
        """
        Parse configuration from file.
        
        Args:
            file_path: Configuration file path
            
        Returns:
            Parse result with configuration data
            
        Raises:
            ConfigurationError: If parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        # Detect format from extension
        format_type = self._detect_format(file_path)
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Parse content
            result = self.parse_content(content, format_type, str(file_path))
            
            # Process includes if enabled
            if self.config.include_support:
                result.data = self._process_includes(result.data, file_path.parent)
            
            # Apply environment overrides
            if self.config.allow_environment_override:
                result.data = self._apply_environment_overrides(result.data)
            
            # Process templates if enabled
            if self.config.template_support:
                result.data = self._process_templates(result.data)
            
            # Add metadata
            result.metadata.update({
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                'parsed_at': datetime.now()
            })
            
            logger.info(f"Successfully parsed configuration from {file_path}")
            return result
            
        except Exception as e:
            raise ConfigurationError(f"Failed to parse {file_path}: {e}")
    
    def parse_content(self, 
                     content: str, 
                     format_type: ConfigFormat,
                     source: str = "<content>") -> ParseResult:
        """
        Parse configuration from content string.
        
        Args:
            content: Configuration content
            format_type: Content format
            source: Source identifier
            
        Returns:
            Parse result
        """
        try:
            # Get format handler
            handler = self._format_handlers.get(format_type)
            if not handler:
                raise ConfigurationError(f"Unsupported format: {format_type}")
            
            # Parse content
            data = handler(content)
            
            # Type conversion if enabled
            if self.config.auto_type_conversion:
                data = self._convert_types(data)
            
            # Create result
            result = ParseResult(
                data=data,
                format=format_type,
                source=source
            )
            
            return result
            
        except Exception as e:
            raise ConfigurationError(f"Failed to parse content from {source}: {e}")
    
    def parse_directory(self, 
                       directory: Union[str, Path],
                       pattern: str = "**/*.{yaml,yml,json,toml}",
                       merge: bool = True) -> Union[ParseResult, List[ParseResult]]:
        """
        Parse configuration from directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            merge: Whether to merge all configs into one result
            
        Returns:
            Single merged result or list of results
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise ConfigurationError(f"Directory not found: {directory}")
        
        # Find configuration files
        config_files = []
        for ext in ['yaml', 'yml', 'json', 'toml']:
            config_files.extend(directory.rglob(f"*.{ext}"))
        
        if not config_files:
            logger.warning(f"No configuration files found in {directory}")
            return ParseResult(data={}, format=ConfigFormat.JSON, source=str(directory))
        
        # Parse all files
        results = []
        for config_file in sorted(config_files):
            try:
                result = self.parse_file(config_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse {config_file}: {e}")
        
        if not merge:
            return results
        
        # Merge all configurations
        merged_data = {}
        merged_errors = []
        merged_warnings = []
        
        for result in results:
            merged_data = self._merge_configs(merged_data, result.data)
            merged_errors.extend(result.errors)
            merged_warnings.extend(result.warnings)
        
        return ParseResult(
            data=merged_data,
            format=ConfigFormat.JSON,  # Merged format is JSON-like
            source=str(directory),
            errors=merged_errors,
            warnings=merged_warnings,
            metadata={'files_parsed': len(results)}
        )
    
    def parse_environment(self, prefix: Optional[str] = None) -> ParseResult:
        """
        Parse configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Parse result with environment configuration
        """
        prefix = prefix or self.config.environment_prefix
        
        env_data = {}
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            # Remove prefix and convert to nested structure
            config_key = key[len(prefix):] if prefix else key
            config_key = config_key.lower()
            
            # Convert underscores to nested structure
            # e.g., MODEL_TRAINING_EPOCHS -> model.training.epochs
            keys = config_key.split('_')
            
            # Set nested value
            current = env_data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Convert value type
            current[keys[-1]] = self._convert_env_value(value)
        
        return ParseResult(
            data=env_data,
            format=ConfigFormat.ENV,
            source="environment"
        )
    
    def validate_config(self, 
                       config_data: Dict[str, Any],
                       schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate configuration data.
        
        Args:
            config_data: Configuration data to validate
            schema: Optional JSON schema for validation
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            if schema:
                # JSON Schema validation if available
                try:
                    import jsonschema
                    jsonschema.validate(config_data, schema)
                except ImportError:
                    logger.warning("jsonschema not available for validation")
                except jsonschema.ValidationError as e:
                    errors.append(f"Schema validation failed: {e.message}")
            
            # Custom validation rules
            errors.extend(self._validate_custom_rules(config_data))
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        extension = file_path.suffix.lower()
        
        format_map = {
            '.json': ConfigFormat.JSON,
            '.yaml': ConfigFormat.YAML,
            '.yml': ConfigFormat.YAML,
            '.toml': ConfigFormat.TOML,
            '.ini': ConfigFormat.INI,
            '.cfg': ConfigFormat.INI,
            '.conf': ConfigFormat.INI,
            '.env': ConfigFormat.ENV
        }
        
        return format_map.get(extension, ConfigFormat.JSON)
    
    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON content."""
        return json.loads(content)
    
    def _parse_yaml(self, content: str) -> Dict[str, Any]:
        """Parse YAML content."""
        try:
            return yaml.safe_load(content) or {}
        except ImportError:
            raise ConfigurationError("PyYAML is required for YAML parsing")
    
    def _parse_toml(self, content: str) -> Dict[str, Any]:
        """Parse TOML content."""
        try:
            return toml.loads(content)
        except ImportError:
            raise ConfigurationError("toml is required for TOML parsing")
    
    def _parse_ini(self, content: str) -> Dict[str, Any]:
        """Parse INI content."""
        import configparser
        
        parser = configparser.ConfigParser()
        parser.read_string(content)
        
        # Convert to nested dict
        config = {}
        for section_name in parser.sections():
            section = parser[section_name]
            config[section_name] = dict(section)
        
        return config
    
    def _parse_env(self, content: str) -> Dict[str, Any]:
        """Parse .env file content."""
        config = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse key=value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                config[key] = self._convert_env_value(value)
        
        return config
    
    def _convert_types(self, data: Any) -> Any:
        """Convert string values to appropriate types."""
        if isinstance(data, dict):
            return {key: self._convert_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_types(item) for item in data]
        elif isinstance(data, str):
            return self._convert_string_value(data)
        else:
            return data
    
    def _convert_string_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Check for type hints in value
        if ':' in value and value.count(':') == 1:
            type_hint, actual_value = value.split(':', 1)
            type_hint = type_hint.strip()
            actual_value = actual_value.strip()
            
            converter = self._type_converters.get(type_hint)
            if converter:
                try:
                    return converter(actual_value)
                except Exception:
                    pass
        
        # Auto-detect type
        value = value.strip()
        
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value."""
        return self._convert_string_value(value)
    
    def _parse_bool(self, value: str) -> bool:
        """Parse boolean value."""
        return str(value).lower() in ('true', '1', 'yes', 'on')
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse list value."""
        if value.startswith('[') and value.endswith(']'):
            # JSON-style list
            return json.loads(value)
        else:
            # Comma-separated
            return [item.strip() for item in value.split(',')]
    
    def _parse_dict(self, value: str) -> Dict[str, Any]:
        """Parse dictionary value."""
        return json.loads(value)
    
    def _parse_datetime(self, value: str) -> datetime:
        """Parse datetime value."""
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Cannot parse datetime: {value}")
    
    def _process_includes(self, data: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """Process configuration includes."""
        if not isinstance(data, dict):
            return data
        
        # Check for include directives
        includes = data.get('include', [])
        if isinstance(includes, str):
            includes = [includes]
        
        # Process includes
        for include_path in includes:
            include_file = base_path / include_path
            
            # Check cache first
            cache_key = str(include_file)
            if cache_key in self._include_cache:
                included_data = self._include_cache[cache_key]
            else:
                try:
                    result = self.parse_file(include_file)
                    included_data = result.data
                    self._include_cache[cache_key] = included_data
                except Exception as e:
                    logger.error(f"Failed to include {include_file}: {e}")
                    continue
            
            # Merge included data
            data = self._merge_configs(included_data, data)
        
        # Remove include directive
        data.pop('include', None)
        
        # Process nested includes
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = self._process_includes(value, base_path)
        
        return data
    
    def _apply_environment_overrides(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_result = self.parse_environment()
        
        if env_result.data:
            data = self._merge_configs(data, env_result.data)
        
        return data
    
    def _process_templates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process template variables in configuration."""
        if not isinstance(data, dict):
            return data
        
        # Update template variables
        self._template_vars.update({
            'env': dict(os.environ),
            'cwd': os.getcwd(),
            'home': str(Path.home())
        })
        
        # Process templates recursively
        return self._process_templates_recursive(data)
    
    def _process_templates_recursive(self, obj: Any) -> Any:
        """Recursively process templates in configuration."""
        if isinstance(obj, dict):
            return {key: self._process_templates_recursive(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._process_templates_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return self._expand_template_string(obj)
        else:
            return obj
    
    def _expand_template_string(self, template: str) -> str:
        """Expand template variables in string."""
        # Simple variable substitution: ${var} or $var
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            
            # Look in template variables
            if var_name in self._template_vars:
                value = self._template_vars[var_name]
                return str(value) if not isinstance(value, dict) else json.dumps(value)
            
            # Look in environment
            if var_name in os.environ:
                return os.environ[var_name]
            
            # Return unchanged if not found
            return match.group(0)
        
        return re.sub(pattern, replace_var, template)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        if self.config.merge_strategy == "replace":
            return override.copy()
        elif self.config.merge_strategy == "shallow":
            result = base.copy()
            result.update(override)
            return result
        else:  # deep merge
            return self._deep_merge(base, override)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
    
    def _validate_custom_rules(self, config_data: Dict[str, Any]) -> List[str]:
        """Apply custom validation rules."""
        errors = []
        
        # Check for required fields
        required_fields = ['name', 'version']
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Required field missing: {field}")
        
        # Validate port numbers
        for key, value in config_data.items():
            if key.endswith('_port') and isinstance(value, int):
                if not (1 <= value <= 65535):
                    errors.append(f"Invalid port number {key}: {value}")
        
        # Validate paths
        for key, value in config_data.items():
            if key.endswith('_path') and isinstance(value, str):
                path = Path(value)
                if not path.exists() and not key.startswith('output_'):
                    errors.append(f"Path does not exist {key}: {value}")
        
        return errors
    
    def add_template_variable(self, name: str, value: Any):
        """Add template variable."""
        self._template_vars[name] = value
    
    def set_merge_strategy(self, strategy: str):
        """Set configuration merge strategy."""
        if strategy not in ('shallow', 'deep', 'replace'):
            raise ValueError(f"Invalid merge strategy: {strategy}")
        self.config.merge_strategy = strategy
    
    def clear_include_cache(self):
        """Clear include file cache."""
        self._include_cache.clear()
    
    def get_supported_formats(self) -> List[ConfigFormat]:
        """Get list of supported configuration formats."""
        return list(self._format_handlers.keys())


# Convenience functions

def parse_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse configuration file with default settings.
    
    Args:
        file_path: Configuration file path
        
    Returns:
        Parsed configuration data
    """
    parser = ConfigurationParser()
    result = parser.parse_file(file_path)
    return result.data


def parse_config_directory(directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse all configuration files in directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Merged configuration data
    """
    parser = ConfigurationParser()
    result = parser.parse_directory(directory, merge=True)
    return result.data


def create_parser(allow_env: bool = True, 
                 template_support: bool = True,
                 include_support: bool = True) -> ConfigurationParser:
    """
    Create configuration parser with common settings.
    
    Args:
        allow_env: Allow environment variable overrides
        template_support: Enable template processing
        include_support: Enable include processing
        
    Returns:
        Configured parser instance
    """
    config = ParserConfig(
        allow_environment_override=allow_env,
        template_support=template_support,
        include_support=include_support
    )
    return ConfigurationParser(config)