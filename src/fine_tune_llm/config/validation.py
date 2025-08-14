"""
Configuration validation system for fine-tune LLM library.

Provides comprehensive validation with schema definitions, 
custom validators, and detailed error reporting.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import re
import os
from pathlib import Path
import logging

from ..core.exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    CHOICES = "choices"
    PATTERN = "pattern"
    LENGTH = "length"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"
    CONDITIONAL = "conditional"

@dataclass
class ValidationRule:
    """Single validation rule definition."""
    field_path: str
    rule_type: ValidationType
    constraint: Any
    message: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None

class BaseValidator(ABC):
    """Abstract base validator."""
    
    @abstractmethod
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        """Validate value against constraint."""
        pass
    
    @abstractmethod
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        """Get error message for validation failure."""
        pass

class RequiredValidator(BaseValidator):
    """Validator for required fields."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        if constraint and (value is None or value == ""):
            return ["Field is required"]
        return []
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        return f"Field '{field}' is required but not provided"

class TypeValidator(BaseValidator):
    """Validator for data types."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        if value is None:
            return []
        
        expected_type = constraint
        if isinstance(expected_type, tuple):
            # Multiple types allowed
            if not isinstance(value, expected_type):
                return [f"Expected type {expected_type}, got {type(value).__name__}"]
        else:
            if not isinstance(value, expected_type):
                return [f"Expected type {expected_type.__name__}, got {type(value).__name__}"]
        return []
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        return f"Field '{field}' has incorrect type. Expected {constraint}, got {type(value).__name__}"

class RangeValidator(BaseValidator):
    """Validator for numeric ranges."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        if value is None or not isinstance(value, (int, float)):
            return []
        
        min_val, max_val = constraint
        errors = []
        
        if min_val is not None and value < min_val:
            errors.append(f"Value {value} is below minimum {min_val}")
        
        if max_val is not None and value > max_val:
            errors.append(f"Value {value} is above maximum {max_val}")
        
        return errors
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        min_val, max_val = constraint
        return f"Field '{field}' value {value} is outside allowed range [{min_val}, {max_val}]"

class ChoicesValidator(BaseValidator):
    """Validator for allowed choices."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        if value is None:
            return []
        
        allowed_choices = constraint
        if value not in allowed_choices:
            return [f"Value '{value}' not in allowed choices: {allowed_choices}"]
        return []
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        return f"Field '{field}' value '{value}' not in allowed choices: {constraint}"

class PatternValidator(BaseValidator):
    """Validator for regex patterns."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        if value is None or not isinstance(value, str):
            return []
        
        pattern = constraint
        if not re.match(pattern, value):
            return [f"Value '{value}' does not match required pattern: {pattern}"]
        return []
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        return f"Field '{field}' value '{value}' does not match pattern: {constraint}"

class LengthValidator(BaseValidator):
    """Validator for string/list length."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        if value is None:
            return []
        
        if not hasattr(value, '__len__'):
            return ["Value does not have length"]
        
        min_len, max_len = constraint
        length = len(value)
        errors = []
        
        if min_len is not None and length < min_len:
            errors.append(f"Length {length} is below minimum {min_len}")
        
        if max_len is not None and length > max_len:
            errors.append(f"Length {length} is above maximum {max_len}")
        
        return errors
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        min_len, max_len = constraint
        return f"Field '{field}' length {len(value)} is outside allowed range [{min_len}, {max_len}]"

class CustomValidator(BaseValidator):
    """Validator for custom validation functions."""
    
    def validate(self, value: Any, constraint: Any, context: Dict[str, Any]) -> List[str]:
        validator_func = constraint
        try:
            result = validator_func(value, context)
            if result is True:
                return []
            elif result is False:
                return ["Custom validation failed"]
            elif isinstance(result, str):
                return [result]
            elif isinstance(result, list):
                return result
            else:
                return ["Custom validation returned invalid result"]
        except Exception as e:
            return [f"Custom validation error: {str(e)}"]
    
    def get_error_message(self, field: str, value: Any, constraint: Any) -> str:
        return f"Custom validation failed for field '{field}'"

class ConfigValidator:
    """Main configuration validator."""
    
    def __init__(self):
        self.validators = {
            ValidationType.REQUIRED: RequiredValidator(),
            ValidationType.TYPE: TypeValidator(),
            ValidationType.RANGE: RangeValidator(),
            ValidationType.CHOICES: ChoicesValidator(),
            ValidationType.PATTERN: PatternValidator(),
            ValidationType.LENGTH: LengthValidator(),
            ValidationType.CUSTOM: CustomValidator()
        }
        self.rules: List[ValidationRule] = []
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self.rules.append(rule)
    
    def add_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Add validation schema."""
        self.schemas[name] = schema
        self._convert_schema_to_rules(name, schema)
    
    def validate(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> List[str]:
        """Validate configuration against rules or schema."""
        errors = []
        
        # Use schema if provided
        if schema_name and schema_name in self.schemas:
            schema_errors = self._validate_against_schema(config, self.schemas[schema_name])
            errors.extend(schema_errors)
        
        # Apply custom rules
        for rule in self.rules:
            if rule.condition and not rule.condition(config):
                continue  # Skip rule if condition not met
                
            field_value = self._get_nested_value(config, rule.field_path)
            validator = self.validators.get(rule.rule_type)
            
            if validator:
                rule_errors = validator.validate(field_value, rule.constraint, config)
                for error in rule_errors:
                    full_error = f"{rule.field_path}: {error}"
                    if rule.message:
                        full_error = f"{rule.field_path}: {rule.message}"
                    errors.append(full_error)
        
        return errors
    
    def is_valid(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> bool:
        """Check if configuration is valid."""
        return len(self.validate(config, schema_name)) == 0
    
    def validate_and_raise(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> None:
        """Validate and raise exception if invalid."""
        errors = self.validate(config, schema_name)
        if errors:
            raise ValidationError("Configuration validation failed", errors)
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema definition."""
        errors = []
        
        # Validate required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(f"Required field '{field}' is missing")
        
        # Validate field properties
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in config:
                field_errors = self._validate_field(field, config[field], field_schema, config)
                errors.extend([f"{field}: {error}" for error in field_errors])
        
        return errors
    
    def _validate_field(self, field_name: str, value: Any, field_schema: Dict[str, Any], 
                       context: Dict[str, Any]) -> List[str]:
        """Validate single field against its schema."""
        errors = []
        
        # Type validation
        if 'type' in field_schema:
            expected_type = field_schema['type']
            type_map = {
                'string': str,
                'integer': int,
                'number': (int, float),
                'boolean': bool,
                'array': list,
                'object': dict
            }
            
            if expected_type in type_map:
                python_type = type_map[expected_type]
                if not isinstance(value, python_type):
                    errors.append(f"Expected {expected_type}, got {type(value).__name__}")
        
        # Range validation
        if isinstance(value, (int, float)):
            if 'minimum' in field_schema and value < field_schema['minimum']:
                errors.append(f"Value {value} below minimum {field_schema['minimum']}")
            if 'maximum' in field_schema and value > field_schema['maximum']:
                errors.append(f"Value {value} above maximum {field_schema['maximum']}")
        
        # String validation
        if isinstance(value, str):
            if 'minLength' in field_schema and len(value) < field_schema['minLength']:
                errors.append(f"Length {len(value)} below minimum {field_schema['minLength']}")
            if 'maxLength' in field_schema and len(value) > field_schema['maxLength']:
                errors.append(f"Length {len(value)} above maximum {field_schema['maxLength']}")
            if 'pattern' in field_schema and not re.match(field_schema['pattern'], value):
                errors.append(f"Value does not match pattern: {field_schema['pattern']}")
        
        # Enum validation
        if 'enum' in field_schema and value not in field_schema['enum']:
            errors.append(f"Value '{value}' not in allowed choices: {field_schema['enum']}")
        
        # Array validation
        if isinstance(value, list):
            if 'minItems' in field_schema and len(value) < field_schema['minItems']:
                errors.append(f"Array length {len(value)} below minimum {field_schema['minItems']}")
            if 'maxItems' in field_schema and len(value) > field_schema['maxItems']:
                errors.append(f"Array length {len(value)} above maximum {field_schema['maxItems']}")
        
        return errors
    
    def _convert_schema_to_rules(self, schema_name: str, schema: Dict[str, Any]) -> None:
        """Convert JSON schema to validation rules."""
        # This is a simplified conversion - in a full implementation,
        # you'd want to handle nested schemas, references, etc.
        pass
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current

# Predefined validators for common use cases
def validate_file_path(path: str, context: Dict[str, Any]) -> Union[bool, str]:
    """Validate that path exists and is a file."""
    if not path:
        return "File path cannot be empty"
    
    path_obj = Path(path)
    if not path_obj.exists():
        return f"File does not exist: {path}"
    
    if not path_obj.is_file():
        return f"Path is not a file: {path}"
    
    return True

def validate_directory_path(path: str, context: Dict[str, Any]) -> Union[bool, str]:
    """Validate that path exists and is a directory."""
    if not path:
        return "Directory path cannot be empty"
    
    path_obj = Path(path)
    if not path_obj.exists():
        return f"Directory does not exist: {path}"
    
    if not path_obj.is_dir():
        return f"Path is not a directory: {path}"
    
    return True

def validate_model_id(model_id: str, context: Dict[str, Any]) -> Union[bool, str]:
    """Validate model ID format."""
    if not model_id:
        return "Model ID cannot be empty"
    
    # Basic format validation
    if not re.match(r'^[a-zA-Z0-9\-_./]+$', model_id):
        return "Model ID contains invalid characters"
    
    # Check for common patterns
    if '/' in model_id:
        parts = model_id.split('/')
        if len(parts) != 2:
            return "Model ID should be in format 'organization/model'"
        
        org, model = parts
        if not org or not model:
            return "Model ID organization and model parts cannot be empty"
    
    return True

def validate_learning_rate(lr: float, context: Dict[str, Any]) -> Union[bool, str]:
    """Validate learning rate value."""
    if lr <= 0:
        return "Learning rate must be positive"
    
    if lr > 1.0:
        return "Learning rate is unusually high (> 1.0)"
    
    if lr < 1e-8:
        return "Learning rate is unusually low (< 1e-8)"
    
    return True

def validate_batch_size(batch_size: int, context: Dict[str, Any]) -> Union[bool, str]:
    """Validate batch size value."""
    if batch_size <= 0:
        return "Batch size must be positive"
    
    if batch_size > 1024:
        return "Batch size is unusually large (> 1024)"
    
    # Check if batch size is power of 2 (recommended)
    if batch_size & (batch_size - 1) != 0:
        return "Batch size should ideally be a power of 2 for optimal performance"
    
    return True

# Factory function for creating common validators
def create_llm_config_validator() -> ConfigValidator:
    """Create validator for LLM configuration."""
    validator = ConfigValidator()
    
    # Add common validation rules
    validator.add_rule(ValidationRule(
        field_path="model.model_id",
        rule_type=ValidationType.CUSTOM,
        constraint=validate_model_id,
        message="Invalid model ID format"
    ))
    
    validator.add_rule(ValidationRule(
        field_path="training.learning_rate",
        rule_type=ValidationType.CUSTOM,
        constraint=validate_learning_rate,
        message="Invalid learning rate"
    ))
    
    validator.add_rule(ValidationRule(
        field_path="training.batch_size",
        rule_type=ValidationType.CUSTOM,
        constraint=validate_batch_size,
        message="Invalid batch size"
    ))
    
    # Add range validations
    validator.add_rule(ValidationRule(
        field_path="training.num_epochs",
        rule_type=ValidationType.RANGE,
        constraint=(1, 100),
        message="Number of epochs must be between 1 and 100"
    ))
    
    validator.add_rule(ValidationRule(
        field_path="lora.r",
        rule_type=ValidationType.RANGE,
        constraint=(1, 256),
        message="LoRA rank must be between 1 and 256"
    ))
    
    return validator