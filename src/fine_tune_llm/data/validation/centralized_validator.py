"""
Centralized Data Validation System.

This module provides comprehensive data validation across all components
with pluggable validators, schema enforcement, and detailed reporting.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationType(Enum):
    """Types of validation checks."""
    SCHEMA = "schema"
    FORMAT = "format"
    RANGE = "range"
    CONTENT = "content"
    CONSISTENCY = "consistency"
    BUSINESS_RULE = "business_rule"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CUSTOM = "custom"


@dataclass
class ValidationError:
    """Validation error details."""
    field: str
    message: str
    severity: ValidationSeverity
    validation_type: ValidationType
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: ValidationError):
        """Add validation error."""
        if error.severity == ValidationSeverity.CRITICAL or error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        else:
            self.warnings.append(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'critical_errors': len([e for e in self.errors if e.severity == ValidationSeverity.CRITICAL]),
            'error_types': list(set(e.validation_type.value for e in self.errors)),
            'warning_types': list(set(e.validation_type.value for e in self.warnings))
        }


class BaseValidator:
    """Base class for data validators."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize validator."""
        self.name = name
        self.config = config or {}
        self.enabled = True
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data."""
        raise NotImplementedError
    
    def configure(self, config: Dict[str, Any]):
        """Update validator configuration."""
        self.config.update(config)


class SchemaValidator(BaseValidator):
    """JSON schema validator."""
    
    def __init__(self, schema: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Initialize schema validator."""
        super().__init__("schema", config)
        self.schema = schema
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against JSON schema."""
        result = ValidationResult(is_valid=True)
        
        try:
            import jsonschema
            
            # Validate against schema
            validator = jsonschema.Draft7Validator(self.schema)
            
            for error in validator.iter_errors(data):
                validation_error = ValidationError(
                    field='.'.join(str(p) for p in error.path),
                    message=error.message,
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.SCHEMA,
                    expected=self.schema.get('type'),
                    actual=type(data).__name__,
                    context={'schema_path': list(error.schema_path)}
                )
                result.add_error(validation_error)
        
        except ImportError:
            result.add_error(ValidationError(
                field="schema",
                message="jsonschema package not available",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.SCHEMA
            ))
        except Exception as e:
            result.add_error(ValidationError(
                field="schema",
                message=f"Schema validation failed: {e}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.SCHEMA
            ))
        
        return result


class FormatValidator(BaseValidator):
    """Data format validator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize format validator."""
        super().__init__("format", config)
        
        # Built-in format patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
            'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'ip_address': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            'datetime_iso': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$'
        }
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data format."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(data, dict):
            return result
        
        format_rules = self.config.get('format_rules', {})
        
        for field, format_spec in format_rules.items():
            value = self._get_nested_value(data, field)
            
            if value is None:
                continue
            
            if not isinstance(value, str):
                result.add_error(ValidationError(
                    field=field,
                    message=f"Expected string for format validation, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.FORMAT,
                    actual=type(value).__name__
                ))
                continue
            
            # Get pattern
            if isinstance(format_spec, str):
                pattern = self.patterns.get(format_spec, format_spec)
            elif isinstance(format_spec, dict):
                pattern = format_spec.get('pattern', format_spec.get('format', ''))
            else:
                continue
            
            # Validate pattern
            if pattern and not re.match(pattern, value, re.IGNORECASE):
                result.add_error(ValidationError(
                    field=field,
                    message=f"Invalid format for {field}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.FORMAT,
                    expected=format_spec,
                    actual=value,
                    suggestion=f"Expected format: {format_spec}"
                ))
        
        return result
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value by dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class RangeValidator(BaseValidator):
    """Numeric range validator."""
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate numeric ranges."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(data, dict):
            return result
        
        range_rules = self.config.get('range_rules', {})
        
        for field, range_spec in range_rules.items():
            value = self._get_nested_value(data, field)
            
            if value is None:
                continue
            
            if not isinstance(value, (int, float, np.number)):
                result.add_error(ValidationError(
                    field=field,
                    message=f"Expected numeric value for range validation, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    actual=type(value).__name__
                ))
                continue
            
            # Extract range bounds
            min_val = range_spec.get('min')
            max_val = range_spec.get('max')
            
            # Validate bounds
            if min_val is not None and value < min_val:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Value {value} below minimum {min_val}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    expected=f">= {min_val}",
                    actual=value
                ))
            
            if max_val is not None and value > max_val:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Value {value} above maximum {max_val}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    expected=f"<= {max_val}",
                    actual=value
                ))
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value by dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class ContentValidator(BaseValidator):
    """Content quality validator."""
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate content quality."""
        result = ValidationResult(is_valid=True)
        
        content_rules = self.config.get('content_rules', {})
        
        for field, rules in content_rules.items():
            value = self._get_nested_value(data, field) if isinstance(data, dict) else data
            
            if value is None:
                continue
            
            # Text content validation
            if isinstance(value, str):
                self._validate_text_content(field, value, rules, result)
            
            # List content validation
            elif isinstance(value, list):
                self._validate_list_content(field, value, rules, result)
            
            # DataFrame validation
            elif isinstance(value, pd.DataFrame):
                self._validate_dataframe_content(field, value, rules, result)
        
        return result
    
    def _validate_text_content(self, field: str, text: str, rules: Dict[str, Any], result: ValidationResult):
        """Validate text content."""
        # Length validation
        min_length = rules.get('min_length')
        max_length = rules.get('max_length')
        
        if min_length and len(text) < min_length:
            result.add_error(ValidationError(
                field=field,
                message=f"Text too short: {len(text)} < {min_length}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=f">= {min_length} characters",
                actual=f"{len(text)} characters"
            ))
        
        if max_length and len(text) > max_length:
            result.add_error(ValidationError(
                field=field,
                message=f"Text too long: {len(text)} > {max_length}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=f"<= {max_length} characters",
                actual=f"{len(text)} characters"
            ))
        
        # Content checks
        if rules.get('no_empty') and not text.strip():
            result.add_error(ValidationError(
                field=field,
                message="Empty text not allowed",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT
            ))
        
        # Forbidden words
        forbidden_words = rules.get('forbidden_words', [])
        for word in forbidden_words:
            if word.lower() in text.lower():
                result.add_error(ValidationError(
                    field=field,
                    message=f"Forbidden word found: {word}",
                    severity=ValidationSeverity.WARNING,
                    validation_type=ValidationType.CONTENT,
                    actual=word
                ))
        
        # Required words
        required_words = rules.get('required_words', [])
        for word in required_words:
            if word.lower() not in text.lower():
                result.add_error(ValidationError(
                    field=field,
                    message=f"Required word missing: {word}",
                    severity=ValidationSeverity.WARNING,
                    validation_type=ValidationType.CONTENT,
                    expected=word
                ))
    
    def _validate_list_content(self, field: str, items: List[Any], rules: Dict[str, Any], result: ValidationResult):
        """Validate list content."""
        # Size validation
        min_items = rules.get('min_items')
        max_items = rules.get('max_items')
        
        if min_items and len(items) < min_items:
            result.add_error(ValidationError(
                field=field,
                message=f"Too few items: {len(items)} < {min_items}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=f">= {min_items} items",
                actual=f"{len(items)} items"
            ))
        
        if max_items and len(items) > max_items:
            result.add_error(ValidationError(
                field=field,
                message=f"Too many items: {len(items)} > {max_items}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=f"<= {max_items} items",
                actual=f"{len(items)} items"
            ))
        
        # Uniqueness check
        if rules.get('unique') and len(items) != len(set(str(item) for item in items)):
            result.add_error(ValidationError(
                field=field,
                message="Duplicate items found",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.CONTENT
            ))
    
    def _validate_dataframe_content(self, field: str, df: pd.DataFrame, rules: Dict[str, Any], result: ValidationResult):
        """Validate DataFrame content."""
        # Shape validation
        min_rows = rules.get('min_rows')
        max_rows = rules.get('max_rows')
        
        if min_rows and len(df) < min_rows:
            result.add_error(ValidationError(
                field=field,
                message=f"Too few rows: {len(df)} < {min_rows}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=f">= {min_rows} rows",
                actual=f"{len(df)} rows"
            ))
        
        if max_rows and len(df) > max_rows:
            result.add_error(ValidationError(
                field=field,
                message=f"Too many rows: {len(df)} > {max_rows}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=f"<= {max_rows} rows",
                actual=f"{len(df)} rows"
            ))
        
        # Required columns
        required_columns = rules.get('required_columns', [])
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            result.add_error(ValidationError(
                field=field,
                message=f"Missing required columns: {missing_columns}",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT,
                expected=required_columns,
                actual=list(df.columns)
            ))
        
        # Null value checks
        if rules.get('no_nulls'):
            null_columns = df.columns[df.isnull().any()].tolist()
            if null_columns:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Null values found in columns: {null_columns}",
                    severity=ValidationSeverity.WARNING,
                    validation_type=ValidationType.CONTENT,
                    actual=null_columns
                ))
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value by dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class SecurityValidator(BaseValidator):
    """Security-focused validator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security validator."""
        super().__init__("security", config)
        
        # Security patterns to detect
        self.security_patterns = {
            'sql_injection': r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b|\'|\"|;|--)',
            'xss': r'<script[^>]*>.*?</script>|javascript:|vbscript:|onload=|onerror=',
            'path_traversal': r'\.\./|\.\.\\'',
            'command_injection': r'[\|;&`$()]',
            'secrets': r'(password|secret|key|token|api_key)[\s]*=[\s]*[\'"]?([^\s\'"]+)',
        }
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate for security issues."""
        result = ValidationResult(is_valid=True)
        
        security_rules = self.config.get('security_rules', {})
        
        # Check for security patterns in string data
        self._check_security_patterns(data, result)
        
        # Check for sensitive data exposure
        self._check_sensitive_data(data, result)
        
        # Check data size limits
        self._check_size_limits(data, result)
        
        return result
    
    def _check_security_patterns(self, data: Any, result: ValidationResult, path: str = ""):
        """Recursively check for security patterns."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self._check_security_patterns(value, result, current_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._check_security_patterns(item, result, current_path)
        
        elif isinstance(data, str):
            for pattern_name, pattern in self.security_patterns.items():
                if re.search(pattern, data, re.IGNORECASE):
                    severity = ValidationSeverity.CRITICAL if pattern_name in ['sql_injection', 'xss', 'command_injection'] else ValidationSeverity.WARNING
                    
                    result.add_error(ValidationError(
                        field=path or "data",
                        message=f"Potential {pattern_name.replace('_', ' ')} detected",
                        severity=severity,
                        validation_type=ValidationType.SECURITY,
                        actual=data[:100] + "..." if len(data) > 100 else data,
                        suggestion=f"Review and sanitize input for {pattern_name}"
                    ))
    
    def _check_sensitive_data(self, data: Any, result: ValidationResult):
        """Check for sensitive data exposure."""
        sensitive_fields = ['password', 'secret', 'token', 'api_key', 'private_key', 'credit_card']
        
        if isinstance(data, dict):
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    if isinstance(value, str) and len(value) > 0:
                        result.add_error(ValidationError(
                            field=key,
                            message=f"Sensitive data detected in field: {key}",
                            severity=ValidationSeverity.CRITICAL,
                            validation_type=ValidationType.SECURITY,
                            suggestion="Mask or encrypt sensitive data"
                        ))
    
    def _check_size_limits(self, data: Any, result: ValidationResult):
        """Check data size limits."""
        max_size = self.config.get('max_size_mb', 100)
        
        try:
            data_size = len(json.dumps(data, default=str)) / (1024 * 1024)  # MB
            
            if data_size > max_size:
                result.add_error(ValidationError(
                    field="data_size",
                    message=f"Data size {data_size:.2f}MB exceeds limit {max_size}MB",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.SECURITY,
                    expected=f"<= {max_size}MB",
                    actual=f"{data_size:.2f}MB"
                ))
        except Exception:
            # Skip size check if serialization fails
            pass


class CentralizedValidator:
    """
    Centralized data validation system.
    
    Coordinates multiple validators and provides comprehensive
    data validation across all platform components.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize centralized validator.
        
        Args:
            event_bus: Event bus for notifications
        """
        self.event_bus = event_bus or EventBus()
        
        # Registered validators
        self.validators: Dict[str, BaseValidator] = {}
        
        # Validation profiles
        self.profiles: Dict[str, List[str]] = {}
        
        # Validation history
        self.validation_history: List[Tuple[str, ValidationResult]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize built-in validators
        self._initialize_builtin_validators()
        
        # Initialize default profiles
        self._initialize_default_profiles()
        
        logger.info("Initialized CentralizedValidator")
    
    def _initialize_builtin_validators(self):
        """Initialize built-in validators."""
        # Format validator
        format_config = {
            'format_rules': {
                'email': 'email',
                'url': 'url',
                'model_id': r'^[a-zA-Z0-9._/-]+$',
                'file_path': r'^[a-zA-Z0-9._/\\-]+$'
            }
        }
        self.register_validator(FormatValidator(format_config))
        
        # Range validator
        range_config = {
            'range_rules': {
                'learning_rate': {'min': 0.00001, 'max': 1.0},
                'batch_size': {'min': 1, 'max': 1024},
                'epochs': {'min': 1, 'max': 1000},
                'max_length': {'min': 1, 'max': 100000}
            }
        }
        self.register_validator(RangeValidator("range", range_config))
        
        # Content validator
        content_config = {
            'content_rules': {
                'text': {
                    'min_length': 1,
                    'max_length': 100000,
                    'no_empty': True
                },
                'training_data': {
                    'min_items': 1,
                    'max_items': 1000000
                }
            }
        }
        self.register_validator(ContentValidator("content", content_config))
        
        # Security validator
        security_config = {
            'max_size_mb': 100,
            'security_rules': {
                'check_injection': True,
                'check_xss': True,
                'check_secrets': True
            }
        }
        self.register_validator(SecurityValidator(security_config))
    
    def _initialize_default_profiles(self):
        """Initialize default validation profiles."""
        self.profiles = {
            'training_data': ['format', 'content', 'security'],
            'model_config': ['format', 'range', 'security'],
            'inference_data': ['format', 'content', 'security'],
            'evaluation_data': ['format', 'content'],
            'user_input': ['format', 'content', 'security'],
            'api_request': ['format', 'range', 'security'],
            'file_upload': ['content', 'security'],
            'configuration': ['format', 'range'],
            'minimal': ['security'],
            'comprehensive': ['format', 'range', 'content', 'security']
        }
    
    def register_validator(self, validator: BaseValidator) -> bool:
        """
        Register validator.
        
        Args:
            validator: Validator instance
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.validators[validator.name] = validator
            
            logger.info(f"Registered validator: {validator.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register validator {validator.name}: {e}")
            return False
    
    def unregister_validator(self, name: str) -> bool:
        """
        Unregister validator.
        
        Args:
            name: Validator name
            
        Returns:
            True if unregistered successfully
        """
        try:
            with self._lock:
                if name in self.validators:
                    del self.validators[name]
                    logger.info(f"Unregistered validator: {name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister validator {name}: {e}")
            return False
    
    def add_profile(self, name: str, validators: List[str]) -> bool:
        """
        Add validation profile.
        
        Args:
            name: Profile name
            validators: List of validator names
            
        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                self.profiles[name] = validators
            
            logger.info(f"Added validation profile: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add profile {name}: {e}")
            return False
    
    def validate(self, 
                data: Any,
                profile: str = "comprehensive",
                context: Optional[Dict[str, Any]] = None,
                schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data using specified profile.
        
        Args:
            data: Data to validate
            profile: Validation profile name
            context: Validation context
            schema: Optional JSON schema
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        try:
            with self._lock:
                # Get validators for profile
                validator_names = self.profiles.get(profile, [])
                
                # Add schema validator if schema provided
                if schema:
                    schema_validator = SchemaValidator(schema)
                    schema_result = schema_validator.validate(data, context)
                    self._merge_results(result, schema_result)
                
                # Run each validator
                for validator_name in validator_names:
                    validator = self.validators.get(validator_name)
                    
                    if validator and validator.enabled:
                        try:
                            validator_result = validator.validate(data, context)
                            self._merge_results(result, validator_result)
                            
                        except Exception as e:
                            result.add_error(ValidationError(
                                field="validator",
                                message=f"Validator {validator_name} failed: {e}",
                                severity=ValidationSeverity.ERROR,
                                validation_type=ValidationType.CUSTOM
                            ))
                
                # Store in history
                self.validation_history.append((profile, result))
                
                # Limit history size
                if len(self.validation_history) > 1000:
                    self.validation_history = self.validation_history[-1000:]
                
                # Publish validation event
                self._publish_validation_event(profile, result)
                
                return result
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            result.add_error(ValidationError(
                field="validation",
                message=f"Validation process failed: {e}",
                severity=ValidationSeverity.CRITICAL,
                validation_type=ValidationType.CUSTOM
            ))
            return result
    
    def _merge_results(self, target: ValidationResult, source: ValidationResult):
        """Merge validation results."""
        target.errors.extend(source.errors)
        target.warnings.extend(source.warnings)
        target.metadata.update(source.metadata)
        
        if not source.is_valid:
            target.is_valid = False
    
    def _publish_validation_event(self, profile: str, result: ValidationResult):
        """Publish validation event."""
        event = Event(
            type=EventType.DATA_VALIDATION,
            data={
                'profile': profile,
                'is_valid': result.is_valid,
                'error_count': len(result.errors),
                'warning_count': len(result.warnings),
                'summary': result.get_summary()
            },
            source="CentralizedValidator"
        )
        
        self.event_bus.publish(event)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        with self._lock:
            total_validations = len(self.validation_history)
            
            if total_validations == 0:
                return {'total_validations': 0}
            
            # Calculate statistics
            successful_validations = sum(1 for _, result in self.validation_history if result.is_valid)
            failed_validations = total_validations - successful_validations
            
            # Profile usage
            profile_usage = {}
            for profile, _ in self.validation_history:
                profile_usage[profile] = profile_usage.get(profile, 0) + 1
            
            # Error types
            error_types = {}
            for _, result in self.validation_history:
                for error in result.errors:
                    error_type = error.validation_type.value
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            return {
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'success_rate': successful_validations / total_validations,
                'profile_usage': profile_usage,
                'error_types': error_types,
                'active_validators': len([v for v in self.validators.values() if v.enabled]),
                'total_validators': len(self.validators),
                'available_profiles': list(self.profiles.keys())
            }
    
    def configure_validator(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Configure validator.
        
        Args:
            name: Validator name
            config: New configuration
            
        Returns:
            True if configured successfully
        """
        try:
            with self._lock:
                validator = self.validators.get(name)
                if validator:
                    validator.configure(config)
                    logger.info(f"Configured validator: {name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to configure validator {name}: {e}")
            return False
    
    def enable_validator(self, name: str) -> bool:
        """Enable validator."""
        return self._set_validator_state(name, True)
    
    def disable_validator(self, name: str) -> bool:
        """Disable validator."""
        return self._set_validator_state(name, False)
    
    def _set_validator_state(self, name: str, enabled: bool) -> bool:
        """Set validator enabled state."""
        try:
            with self._lock:
                validator = self.validators.get(name)
                if validator:
                    validator.enabled = enabled
                    logger.info(f"{'Enabled' if enabled else 'Disabled'} validator: {name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to set validator state {name}: {e}")
            return False


# Global validator instance
_centralized_validator = None

def get_centralized_validator() -> CentralizedValidator:
    """Get global centralized validator instance."""
    global _centralized_validator
    if _centralized_validator is None:
        _centralized_validator = CentralizedValidator()
    return _centralized_validator


# Convenience functions

def validate_data(data: Any, 
                 profile: str = "comprehensive",
                 context: Optional[Dict[str, Any]] = None,
                 schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """
    Validate data using centralized validator.
    
    Args:
        data: Data to validate
        profile: Validation profile
        context: Validation context
        schema: Optional JSON schema
        
    Returns:
        Validation result
    """
    validator = get_centralized_validator()
    return validator.validate(data, profile, context, schema)


def validate_training_data(data: Any) -> ValidationResult:
    """Validate training data."""
    return validate_data(data, profile="training_data")


def validate_user_input(data: Any) -> ValidationResult:
    """Validate user input."""
    return validate_data(data, profile="user_input")


def validate_api_request(data: Any, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Validate API request."""
    return validate_data(data, profile="api_request", schema=schema)