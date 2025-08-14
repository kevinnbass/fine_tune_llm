"""
Exception hierarchy for fine-tune LLM library.

Provides a comprehensive, organized exception system with proper inheritance
and context information for debugging and error handling.
"""

from typing import Dict, Any, Optional, List
import traceback
import time

class FineTuneLLMError(Exception):
    """Root exception for all fine-tune LLM errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = time.time()
        self.traceback_str = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "traceback": self.traceback_str,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        base = f"{self.__class__.__name__}: {self.message}"
        if self.context:
            base += f" (Context: {self.context})"
        if self.cause:
            base += f" (Caused by: {self.cause})"
        return base

# Configuration-related errors
class ConfigurationError(FineTuneLLMError):
    """Configuration-related issues."""
    pass

class ValidationError(ConfigurationError):
    """Configuration validation errors."""
    
    def __init__(self, message: str, validation_errors: List[str], 
                 context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message, context, cause)
        self.validation_errors = validation_errors
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["validation_errors"] = self.validation_errors
        return result

class SchemaError(ConfigurationError):
    """Configuration schema errors."""
    pass

class EnvironmentError(ConfigurationError):
    """Environment-related configuration errors."""
    pass

# Model-related errors
class ModelError(FineTuneLLMError):
    """Model-related issues."""
    pass

class ModelLoadError(ModelError):
    """Model loading errors."""
    pass

class CheckpointError(ModelError):
    """Model checkpoint errors."""
    pass

class AdapterError(ModelError):
    """LoRA/adapter-related errors."""
    pass

# Training-related errors
class TrainingError(FineTuneLLMError):
    """Training-related issues."""
    pass

class ConvergenceError(TrainingError):
    """Training convergence issues."""
    pass

class ResourceError(TrainingError):
    """Training resource errors (memory, compute)."""
    pass

class CallbackError(TrainingError):
    """Training callback errors."""
    pass

# Inference-related errors
class InferenceError(FineTuneLLMError):
    """Inference-related issues."""
    pass

class PredictionError(InferenceError):
    """Prediction generation errors."""
    pass

class CalibrationError(InferenceError):
    """Model calibration errors."""
    pass

class UncertaintyError(InferenceError):
    """Uncertainty quantification errors."""
    pass

# Data-related errors
class DataError(FineTuneLLMError):
    """Data-related issues."""
    pass

class DataValidationError(DataError):
    """Data validation errors."""
    
    def __init__(self, message: str, invalid_samples: List[Dict[str, Any]], 
                 context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message, context, cause)
        self.invalid_samples = invalid_samples
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["invalid_samples"] = self.invalid_samples
        return result

class ProcessingError(DataError):
    """Data processing errors."""
    pass

class LoadingError(DataError):
    """Data loading errors."""
    pass

# Integration-related errors
class IntegrationError(FineTuneLLMError):
    """Integration-related issues."""
    pass

class ServiceError(IntegrationError):
    """Service integration errors."""
    pass

class APIError(IntegrationError):
    """API integration errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_data: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message, context, cause)
        self.status_code = status_code
        self.response_data = response_data
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["status_code"] = self.status_code
        result["response_data"] = self.response_data
        return result

class NetworkError(IntegrationError):
    """Network-related errors."""
    pass

# System-level errors
class SystemError(FineTuneLLMError):
    """System-level issues."""
    pass

class ResourceExhaustionError(SystemError):
    """System resource exhaustion."""
    
    def __init__(self, message: str, resource_type: str, current_usage: Optional[float] = None,
                 limit: Optional[float] = None, context: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        super().__init__(message, context, cause)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "resource_type": self.resource_type,
            "current_usage": self.current_usage,
            "limit": self.limit
        })
        return result

class PermissionError(SystemError):
    """System permission errors."""
    pass

class SystemEnvironmentError(SystemError):
    """System environment errors."""
    pass

# Security-related errors
class SecurityError(FineTuneLLMError):
    """Security-related issues."""
    pass

class CryptographyError(SecurityError):
    """Cryptography and encryption errors."""
    pass

class AuthenticationError(SecurityError):
    """Authentication-related errors."""
    pass

class AuthorizationError(SecurityError):
    """Authorization and permission errors."""
    pass

# Utility functions for exception handling
def create_error_context(
    component: str,
    operation: str,
    parameters: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error context."""
    context = {
        "component": component,
        "operation": operation,
        "timestamp": time.time()
    }
    
    if parameters:
        context["parameters"] = parameters
    
    if additional_info:
        context.update(additional_info)
    
    return context

def wrap_exception(original_exception: Exception, new_exception_class: type, 
                  message: str, context: Optional[Dict[str, Any]] = None) -> FineTuneLLMError:
    """Wrap an existing exception with a fine-tune LLM exception."""
    return new_exception_class(
        message=message,
        context=context,
        cause=original_exception
    )

# Export all exceptions
__all__ = [
    # Root exception
    "FineTuneLLMError",
    
    # Configuration exceptions
    "ConfigurationError",
    "ValidationError", 
    "SchemaError",
    "EnvironmentError",
    
    # Model exceptions
    "ModelError",
    "ModelLoadError",
    "CheckpointError",
    "AdapterError",
    
    # Training exceptions
    "TrainingError",
    "ConvergenceError",
    "ResourceError", 
    "CallbackError",
    
    # Inference exceptions
    "InferenceError",
    "PredictionError",
    "CalibrationError",
    "UncertaintyError",
    
    # Data exceptions
    "DataError",
    "DataValidationError",
    "ProcessingError",
    "LoadingError",
    
    # Integration exceptions
    "IntegrationError",
    "ServiceError",
    "APIError",
    "NetworkError",
    
    # System exceptions
    "SystemError",
    "ResourceExhaustionError",
    "PermissionError",
    "SystemEnvironmentError",
    
    # Security exceptions
    "SecurityError",
    "CryptographyError",
    "AuthenticationError",
    "AuthorizationError",
    
    # Utility functions
    "create_error_context",
    "wrap_exception"
]