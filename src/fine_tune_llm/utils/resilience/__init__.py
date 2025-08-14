"""
Resilience utilities for external service reliability.

This package provides reliability patterns including circuit breakers,
retry mechanisms, and bulkhead isolation for external service interactions.
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerError,
    CircuitState,
    get_circuit_breaker,
    circuit_breaker,
    get_all_circuit_breaker_stats,
    create_api_circuit_breaker_config,
    create_database_circuit_breaker_config
)

from .retry import (
    Retrier,
    AsyncRetrier,
    RetryConfig,
    RetryError,
    BackoffStrategy,
    retry,
    async_retry,
    create_api_retry_config,
    create_database_retry_config,
    create_file_operation_retry_config
)

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig", 
    "CircuitBreakerRegistry",
    "CircuitBreakerError",
    "CircuitState",
    "get_circuit_breaker",
    "circuit_breaker",
    "get_all_circuit_breaker_stats",
    "create_api_circuit_breaker_config",
    "create_database_circuit_breaker_config",
    
    # Retry mechanisms
    "Retrier",
    "AsyncRetrier", 
    "RetryConfig",
    "RetryError",
    "BackoffStrategy",
    "retry",
    "async_retry",
    "create_api_retry_config",
    "create_database_retry_config",
    "create_file_operation_retry_config"
]