"""
Configuration management system for fine-tune LLM library.

Provides centralized configuration management with validation,
hot-reloading, environment support, and comprehensive schemas.
"""

from .manager import ConfigManager, ConfigChangeHandler, ConfigMetadata, ConfigurationError
from .validation import (
    ConfigValidator, ValidationRule, ValidationType, ValidationError,
    create_llm_config_validator,
    validate_file_path, validate_directory_path, validate_model_id,
    validate_learning_rate, validate_batch_size
)
from .schemas import (
    BaseConfig, ModelConfig, LoRAConfig, TrainingConfig,
    AbstentionLossConfig, CalibrationConfig, DataConfig,
    InferenceConfig, EvaluationConfig, MonitoringConfig,
    LLMLoRAConfig, ModelType, LoRAMethod, SchedulerType,
    LLM_CONFIG_SCHEMA
)
from .secrets import (
    SecretManager, SecretConfigMixin,
    get_global_secret_manager, set_secret, get_secret,
    delete_secret, has_secret
)
from .versioning import (
    ConfigVersionManager, ConfigVersion, ChangeType
)
# Unified configuration system
from .parser import (
    ConfigurationParser, ParseResult, ParserConfig, ConfigFormat,
    parse_config_file, parse_config_directory, create_parser
)
from .unified_config import (
    UnifiedConfigManager, ComponentConfig, UnifiedConfigData,
    get_unified_config
)
from .shared_validation import (
    SharedValidationEngine, ValidationError, ValidationRule, 
    ValidationSeverity, ValidationType, get_shared_validator,
    validate_config, validate_field, register_validation_rule
)
from .defaults import (
    DefaultManager, DefaultValue, DefaultScope, DefaultPriority, DefaultProfile,
    get_default_manager, get_default, create_config_with_defaults,
    register_component_defaults
)

__all__ = [
    # Configuration management
    "ConfigManager",
    "ConfigChangeHandler", 
    "ConfigMetadata",
    "ConfigurationError",
    
    # Validation system
    "ConfigValidator",
    "ValidationRule",
    "ValidationType", 
    "ValidationError",
    "create_llm_config_validator",
    
    # Validation functions
    "validate_file_path",
    "validate_directory_path",
    "validate_model_id",
    "validate_learning_rate",
    "validate_batch_size",
    
    # Configuration schemas
    "BaseConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "AbstentionLossConfig",
    "CalibrationConfig", 
    "DataConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "MonitoringConfig",
    "LLMLoRAConfig",
    
    # Enums
    "ModelType",
    "LoRAMethod",
    "SchedulerType",
    
    # JSON Schema
    "LLM_CONFIG_SCHEMA",
    
    # Secret management
    "SecretManager",
    "SecretConfigMixin",
    "get_global_secret_manager",
    "set_secret",
    "get_secret", 
    "delete_secret",
    "has_secret",
    
    # Configuration versioning
    "ConfigVersionManager",
    "ConfigVersion",
    "ChangeType",
    
    # Unified configuration system
    "ConfigurationParser",
    "ParseResult", 
    "ParserConfig",
    "ConfigFormat",
    "parse_config_file",
    "parse_config_directory",
    "create_parser",
    "UnifiedConfigManager",
    "ComponentConfig",
    "UnifiedConfigData",
    "get_unified_config"
]