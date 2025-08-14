"""
Shared Configuration Validation System.

This module provides a comprehensive validation framework that can be shared
across all components with standardized validation rules and error reporting.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import ipaddress

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationType(Enum):
    """Standard validation types."""
    REQUIRED = "required"
    TYPE = "type"
    FORMAT = "format"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"
    SCHEMA = "schema"


@dataclass
class ValidationError:
    """Validation error details."""
    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    type: ValidationType = ValidationType.CUSTOM
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationRule:
    """Configuration validation rule."""
    name: str
    validator: Callable[[Any, Dict[str, Any]], List[ValidationError]]
    description: str
    type: ValidationType = ValidationType.CUSTOM
    severity: ValidationSeverity = ValidationSeverity.ERROR
    applies_to: Optional[Set[str]] = None  # Component names this rule applies to


class SharedValidationEngine:
    """
    Shared validation engine for all configuration validation.
    
    Provides a comprehensive set of validation rules and utilities
    that can be used consistently across all platform components.
    """
    
    def __init__(self):
        """Initialize shared validation engine."""
        # Built-in validation rules
        self.rules: Dict[str, ValidationRule] = {}
        
        # Component-specific rules
        self.component_rules: Dict[str, List[str]] = {}
        
        # Global validation context
        self.context: Dict[str, Any] = {}
        
        # Initialize built-in rules
        self._initialize_builtin_rules()
        
        logger.info("Initialized SharedValidationEngine")
    
    def _initialize_builtin_rules(self):
        """Initialize built-in validation rules."""
        # Required field validation
        self.register_rule(ValidationRule(
            name="required_fields",
            validator=self._validate_required_fields,
            description="Validate required configuration fields",
            type=ValidationType.REQUIRED,
            severity=ValidationSeverity.ERROR
        ))
        
        # Type validation
        self.register_rule(ValidationRule(
            name="field_types",
            validator=self._validate_field_types,
            description="Validate field types",
            type=ValidationType.TYPE,
            severity=ValidationSeverity.ERROR
        ))
        
        # Range validation
        self.register_rule(ValidationRule(
            name="numeric_ranges",
            validator=self._validate_numeric_ranges,
            description="Validate numeric value ranges",
            type=ValidationType.RANGE,
            severity=ValidationSeverity.ERROR
        ))
        
        # Path validation
        self.register_rule(ValidationRule(
            name="file_paths",
            validator=self._validate_file_paths,
            description="Validate file and directory paths",
            type=ValidationType.FORMAT,
            severity=ValidationSeverity.WARNING
        ))
        
        # URL validation
        self.register_rule(ValidationRule(
            name="urls",
            validator=self._validate_urls,
            description="Validate URL formats",
            type=ValidationType.FORMAT,
            severity=ValidationSeverity.ERROR
        ))
        
        # Port validation
        self.register_rule(ValidationRule(
            name="ports",
            validator=self._validate_ports,
            description="Validate port numbers",
            type=ValidationType.RANGE,
            severity=ValidationSeverity.ERROR
        ))
        
        # Email validation
        self.register_rule(ValidationRule(
            name="emails",
            validator=self._validate_emails,
            description="Validate email addresses",
            type=ValidationType.FORMAT,
            severity=ValidationSeverity.ERROR
        ))
        
        # Model ID validation (LLM-specific)
        self.register_rule(ValidationRule(
            name="model_ids",
            validator=self._validate_model_ids,
            description="Validate model identifiers",
            type=ValidationType.FORMAT,
            severity=ValidationSeverity.ERROR,
            applies_to={"training", "inference", "evaluation"}
        ))
        
        # Learning rate validation (ML-specific)
        self.register_rule(ValidationRule(
            name="learning_rates",
            validator=self._validate_learning_rates,
            description="Validate learning rate values",
            type=ValidationType.RANGE,
            severity=ValidationSeverity.WARNING,
            applies_to={"training"}
        ))
        
        # Batch size validation (ML-specific)
        self.register_rule(ValidationRule(
            name="batch_sizes",
            validator=self._validate_batch_sizes,
            description="Validate batch size values",
            type=ValidationType.RANGE,
            severity=ValidationSeverity.WARNING,
            applies_to={"training"}
        ))
        
        # Dependencies validation
        self.register_rule(ValidationRule(
            name="dependencies",
            validator=self._validate_dependencies,
            description="Validate component dependencies",
            type=ValidationType.DEPENDENCY,
            severity=ValidationSeverity.ERROR
        ))
        
        # Schema validation
        self.register_rule(ValidationRule(
            name="json_schema",
            validator=self._validate_json_schema,
            description="Validate against JSON schema",
            type=ValidationType.SCHEMA,
            severity=ValidationSeverity.ERROR
        ))
        
        logger.info(f"Initialized {len(self.rules)} built-in validation rules")
    
    def register_rule(self, rule: ValidationRule) -> bool:
        """
        Register validation rule.
        
        Args:
            rule: Validation rule to register
            
        Returns:
            True if registered successfully
        """
        try:
            self.rules[rule.name] = rule
            logger.debug(f"Registered validation rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register validation rule {rule.name}: {e}")
            return False
    
    def unregister_rule(self, rule_name: str) -> bool:
        """
        Unregister validation rule.
        
        Args:
            rule_name: Rule name to unregister
            
        Returns:
            True if unregistered successfully
        """
        try:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.debug(f"Unregistered validation rule: {rule_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister validation rule {rule_name}: {e}")
            return False
    
    def validate(self, 
                config: Dict[str, Any],
                component: Optional[str] = None,
                rules: Optional[List[str]] = None,
                schema: Optional[Dict[str, Any]] = None) -> List[ValidationError]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            component: Component name for component-specific rules
            rules: Specific rules to apply (all if None)
            schema: JSON schema for validation
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Determine which rules to apply
            rules_to_apply = rules or list(self.rules.keys())
            
            # Filter by component if specified
            if component:
                component_rules = self.component_rules.get(component, [])
                rules_to_apply = [
                    rule for rule in rules_to_apply
                    if (rule in component_rules or 
                        self.rules.get(rule, {}).applies_to is None or
                        component in self.rules.get(rule, {}).applies_to)
                ]
            
            # Apply each rule
            for rule_name in rules_to_apply:
                rule = self.rules.get(rule_name)
                if not rule:
                    continue
                
                try:
                    # Prepare validation context
                    context = {
                        'component': component,
                        'schema': schema,
                        'global_context': self.context,
                        'full_config': config
                    }
                    
                    # Run validation
                    rule_errors = rule.validator(config, context)
                    errors.extend(rule_errors)
                    
                except Exception as e:
                    logger.error(f"Error running validation rule {rule_name}: {e}")
                    errors.append(ValidationError(
                        path="<validation>",
                        message=f"Validation rule {rule_name} failed: {e}",
                        severity=ValidationSeverity.ERROR,
                        type=ValidationType.CUSTOM
                    ))
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            errors.append(ValidationError(
                path="<root>",
                message=f"Validation process failed: {e}",
                severity=ValidationSeverity.ERROR
            ))
        
        return errors
    
    def validate_field(self, 
                      config: Dict[str, Any],
                      field_path: str,
                      expected_type: Optional[Type] = None,
                      required: bool = True,
                      validator: Optional[Callable] = None) -> List[ValidationError]:
        """
        Validate specific field.
        
        Args:
            config: Configuration dictionary
            field_path: Dot-separated field path
            expected_type: Expected field type
            required: Whether field is required
            validator: Custom validator function
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Get field value
            value = self._get_nested_value(config, field_path)
            
            # Check if required
            if required and value is None:
                errors.append(ValidationError(
                    path=field_path,
                    message=f"Required field missing: {field_path}",
                    type=ValidationType.REQUIRED,
                    severity=ValidationSeverity.ERROR
                ))
                return errors
            
            # Skip validation if field is optional and missing
            if value is None:
                return errors
            
            # Type validation
            if expected_type and not isinstance(value, expected_type):
                errors.append(ValidationError(
                    path=field_path,
                    message=f"Invalid type for {field_path}",
                    type=ValidationType.TYPE,
                    expected=expected_type.__name__,
                    actual=type(value).__name__,
                    severity=ValidationSeverity.ERROR
                ))
            
            # Custom validation
            if validator:
                try:
                    is_valid, error_message = validator(value)
                    if not is_valid:
                        errors.append(ValidationError(
                            path=field_path,
                            message=error_message,
                            type=ValidationType.CUSTOM,
                            actual=value,
                            severity=ValidationSeverity.ERROR
                        ))
                except Exception as e:
                    errors.append(ValidationError(
                        path=field_path,
                        message=f"Custom validation failed: {e}",
                        type=ValidationType.CUSTOM,
                        severity=ValidationSeverity.ERROR
                    ))
        
        except Exception as e:
            errors.append(ValidationError(
                path=field_path,
                message=f"Field validation failed: {e}",
                severity=ValidationSeverity.ERROR
            ))
        
        return errors
    
    def _validate_required_fields(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate required fields based on schema."""
        errors = []
        
        schema = context.get('schema', {})
        required_fields = schema.get('required', [])
        
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(ValidationError(
                    path=field,
                    message=f"Required field missing: {field}",
                    type=ValidationType.REQUIRED,
                    severity=ValidationSeverity.ERROR,
                    suggestion=f"Add {field} to configuration"
                ))
        
        return errors
    
    def _validate_field_types(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate field types based on schema."""
        errors = []
        
        schema = context.get('schema', {})
        properties = schema.get('properties', {})
        
        for field, field_schema in properties.items():
            if field not in config:
                continue
            
            value = config[field]
            expected_type = field_schema.get('type')
            
            if expected_type and not self._check_json_type(value, expected_type):
                errors.append(ValidationError(
                    path=field,
                    message=f"Invalid type for field {field}",
                    type=ValidationType.TYPE,
                    expected=expected_type,
                    actual=type(value).__name__,
                    severity=ValidationSeverity.ERROR
                ))
        
        return errors
    
    def _validate_numeric_ranges(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate numeric value ranges."""
        errors = []
        
        # Check common numeric fields
        numeric_checks = [
            ('port', 1, 65535),
            ('timeout', 0, 3600),
            ('max_length', 1, 100000),
            ('batch_size', 1, 1000),
            ('epochs', 1, 1000),
            ('learning_rate', 0.0001, 1.0)
        ]
        
        for field, min_val, max_val in numeric_checks:
            value = self._get_nested_value(config, field)
            if value is not None and isinstance(value, (int, float)):
                if not (min_val <= value <= max_val):
                    errors.append(ValidationError(
                        path=field,
                        message=f"Value {value} out of range [{min_val}, {max_val}] for {field}",
                        type=ValidationType.RANGE,
                        expected=f"[{min_val}, {max_val}]",
                        actual=str(value),
                        severity=ValidationSeverity.WARNING
                    ))
        
        return errors
    
    def _validate_file_paths(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate file and directory paths."""
        errors = []
        
        # Find path fields
        path_fields = []
        for key, value in config.items():
            if ('path' in key.lower() or 'dir' in key.lower() or 'file' in key.lower()) and isinstance(value, str):
                path_fields.append((key, value))
        
        for field, path_str in path_fields:
            try:
                path = Path(path_str)
                
                # Check if path exists (for input paths)
                if not field.startswith('output_') and not path.exists():
                    errors.append(ValidationError(
                        path=field,
                        message=f"Path does not exist: {path_str}",
                        type=ValidationType.FORMAT,
                        actual=path_str,
                        severity=ValidationSeverity.WARNING,
                        suggestion="Ensure the path exists or will be created"
                    ))
                
                # Check permissions for output paths
                if field.startswith('output_'):
                    parent_dir = path.parent
                    if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
                        errors.append(ValidationError(
                            path=field,
                            message=f"No write permission for output path: {path_str}",
                            type=ValidationType.FORMAT,
                            severity=ValidationSeverity.ERROR
                        ))
                        
            except Exception as e:
                errors.append(ValidationError(
                    path=field,
                    message=f"Invalid path format: {path_str} ({e})",
                    type=ValidationType.FORMAT,
                    severity=ValidationSeverity.ERROR
                ))
        
        return errors
    
    def _validate_urls(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate URL formats."""
        errors = []
        
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for key, value in config.items():
            if ('url' in key.lower() or 'endpoint' in key.lower()) and isinstance(value, str):
                if not url_pattern.match(value):
                    errors.append(ValidationError(
                        path=key,
                        message=f"Invalid URL format: {value}",
                        type=ValidationType.FORMAT,
                        actual=value,
                        severity=ValidationSeverity.ERROR,
                        suggestion="Use format: http://domain.com or https://domain.com"
                    ))
        
        return errors
    
    def _validate_ports(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate port numbers."""
        errors = []
        
        for key, value in config.items():
            if 'port' in key.lower() and isinstance(value, int):
                if not (1 <= value <= 65535):
                    errors.append(ValidationError(
                        path=key,
                        message=f"Invalid port number: {value}",
                        type=ValidationType.RANGE,
                        expected="1-65535",
                        actual=str(value),
                        severity=ValidationSeverity.ERROR
                    ))
        
        return errors
    
    def _validate_emails(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate email addresses."""
        errors = []
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for key, value in config.items():
            if 'email' in key.lower() and isinstance(value, str):
                if not email_pattern.match(value):
                    errors.append(ValidationError(
                        path=key,
                        message=f"Invalid email format: {value}",
                        type=ValidationType.FORMAT,
                        actual=value,
                        severity=ValidationSeverity.ERROR,
                        suggestion="Use format: user@domain.com"
                    ))
        
        return errors
    
    def _validate_model_ids(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate model identifiers."""
        errors = []
        
        model_fields = ['model_id', 'model_name', 'base_model', 'tokenizer_id']
        
        for field in model_fields:
            value = self._get_nested_value(config, field)
            if value and isinstance(value, str):
                # Check for valid model ID format
                if not re.match(r'^[a-zA-Z0-9._/-]+$', value):
                    errors.append(ValidationError(
                        path=field,
                        message=f"Invalid model ID format: {value}",
                        type=ValidationType.FORMAT,
                        actual=value,
                        severity=ValidationSeverity.ERROR,
                        suggestion="Use alphanumeric characters, dots, underscores, hyphens, and slashes"
                    ))
        
        return errors
    
    def _validate_learning_rates(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate learning rate values."""
        errors = []
        
        lr_fields = ['learning_rate', 'lr', 'base_lr', 'max_lr']
        
        for field in lr_fields:
            value = self._get_nested_value(config, field)
            if value and isinstance(value, (int, float)):
                if not (0.00001 <= value <= 1.0):
                    errors.append(ValidationError(
                        path=field,
                        message=f"Learning rate {value} may be too high or too low",
                        type=ValidationType.RANGE,
                        expected="0.00001-1.0",
                        actual=str(value),
                        severity=ValidationSeverity.WARNING,
                        suggestion="Typical range: 1e-5 to 1e-3"
                    ))
        
        return errors
    
    def _validate_batch_sizes(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate batch size values."""
        errors = []
        
        batch_fields = ['batch_size', 'train_batch_size', 'eval_batch_size', 'per_device_batch_size']
        
        for field in batch_fields:
            value = self._get_nested_value(config, field)
            if value and isinstance(value, int):
                if not (1 <= value <= 1024):
                    errors.append(ValidationError(
                        path=field,
                        message=f"Batch size {value} may be too large or too small",
                        type=ValidationType.RANGE,
                        expected="1-1024",
                        actual=str(value),
                        severity=ValidationSeverity.WARNING,
                        suggestion="Consider memory constraints and training efficiency"
                    ))
        
        return errors
    
    def _validate_dependencies(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate component dependencies."""
        errors = []
        
        dependencies = config.get('dependencies', [])
        if not isinstance(dependencies, list):
            return errors
        
        # Check for circular dependencies (simplified)
        component = context.get('component')
        if component and component in dependencies:
            errors.append(ValidationError(
                path="dependencies",
                message=f"Circular dependency detected: {component} depends on itself",
                type=ValidationType.DEPENDENCY,
                severity=ValidationSeverity.ERROR
            ))
        
        return errors
    
    def _validate_json_schema(self, config: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationError]:
        """Validate against JSON schema."""
        errors = []
        
        schema = context.get('schema')
        if not schema:
            return errors
        
        try:
            import jsonschema
            
            # Validate against schema
            validator = jsonschema.Draft7Validator(schema)
            
            for error in validator.iter_errors(config):
                errors.append(ValidationError(
                    path='.'.join(str(p) for p in error.path),
                    message=error.message,
                    type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.ERROR
                ))
                
        except ImportError:
            logger.warning("jsonschema package not available for schema validation")
        except Exception as e:
            errors.append(ValidationError(
                path="<schema>",
                message=f"Schema validation failed: {e}",
                type=ValidationType.SCHEMA,
                severity=ValidationSeverity.ERROR
            ))
        
        return errors
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested dictionary value by dot-separated path."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _check_json_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches JSON schema type."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def add_component_rules(self, component: str, rules: List[str]):
        """Add rules specific to a component."""
        self.component_rules[component] = rules
    
    def set_validation_context(self, context: Dict[str, Any]):
        """Set global validation context."""
        self.context.update(context)
    
    def get_validation_summary(self, errors: List[ValidationError]) -> Dict[str, Any]:
        """Get validation summary statistics."""
        summary = {
            'total_errors': len(errors),
            'error_count': len([e for e in errors if e.severity == ValidationSeverity.ERROR]),
            'warning_count': len([e for e in errors if e.severity == ValidationSeverity.WARNING]),
            'info_count': len([e for e in errors if e.severity == ValidationSeverity.INFO]),
            'errors_by_type': {},
            'errors_by_path': {}
        }
        
        # Group by type
        for error in errors:
            error_type = error.type.value
            if error_type not in summary['errors_by_type']:
                summary['errors_by_type'][error_type] = 0
            summary['errors_by_type'][error_type] += 1
        
        # Group by path
        for error in errors:
            path = error.path
            if path not in summary['errors_by_path']:
                summary['errors_by_path'][path] = 0
            summary['errors_by_path'][path] += 1
        
        return summary
    
    def format_errors(self, errors: List[ValidationError], format_type: str = "text") -> str:
        """Format validation errors for display."""
        if format_type == "json":
            import json
            return json.dumps([
                {
                    'path': e.path,
                    'message': e.message,
                    'severity': e.severity.value,
                    'type': e.type.value,
                    'expected': e.expected,
                    'actual': e.actual,
                    'suggestion': e.suggestion
                }
                for e in errors
            ], indent=2)
        
        elif format_type == "html":
            html = ["<ul>"]
            for error in errors:
                severity_class = error.severity.value
                html.append(f'<li class="validation-{severity_class}">')
                html.append(f'<strong>{error.path}</strong>: {error.message}')
                if error.suggestion:
                    html.append(f' <em>(Suggestion: {error.suggestion})</em>')
                html.append('</li>')
            html.append("</ul>")
            return '\n'.join(html)
        
        else:  # text format
            lines = []
            for error in errors:
                severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(error.severity.value, '•')
                lines.append(f"{severity_icon} {error.path}: {error.message}")
                if error.expected and error.actual:
                    lines.append(f"   Expected: {error.expected}, Got: {error.actual}")
                if error.suggestion:
                    lines.append(f"   💡 {error.suggestion}")
            
            return '\n'.join(lines)


# Global shared validation engine instance
_shared_validator = None

def get_shared_validator() -> SharedValidationEngine:
    """Get global shared validation engine instance."""
    global _shared_validator
    if _shared_validator is None:
        _shared_validator = SharedValidationEngine()
    return _shared_validator


# Convenience validation functions

def validate_config(config: Dict[str, Any], 
                   component: Optional[str] = None,
                   schema: Optional[Dict[str, Any]] = None) -> List[ValidationError]:
    """
    Validate configuration using shared validation engine.
    
    Args:
        config: Configuration to validate
        component: Component name
        schema: Optional JSON schema
        
    Returns:
        List of validation errors
    """
    validator = get_shared_validator()
    return validator.validate(config, component=component, schema=schema)


def validate_field(config: Dict[str, Any],
                  field_path: str,
                  expected_type: Optional[Type] = None,
                  required: bool = True) -> List[ValidationError]:
    """
    Validate specific configuration field.
    
    Args:
        config: Configuration dictionary
        field_path: Field path to validate
        expected_type: Expected field type
        required: Whether field is required
        
    Returns:
        List of validation errors
    """
    validator = get_shared_validator()
    return validator.validate_field(config, field_path, expected_type, required)


def register_validation_rule(rule: ValidationRule) -> bool:
    """
    Register custom validation rule globally.
    
    Args:
        rule: Validation rule to register
        
    Returns:
        True if registered successfully
    """
    validator = get_shared_validator()
    return validator.register_rule(rule)