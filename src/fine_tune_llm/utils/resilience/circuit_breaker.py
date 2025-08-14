"""
Circuit breaker pattern implementation for external service reliability.

This module provides circuit breaker functionality to prevent cascading failures
when external services become unavailable or unresponsive.
"""

import time
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
import logging

from ...core.exceptions import ServiceError, IntegrationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, blocking requests  
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0               # Request timeout in seconds
    expected_exception_types: tuple = (ServiceError, IntegrationError)
    monitor_requests: bool = True       # Whether to monitor all requests
    monitor_window_size: int = 100      # Size of sliding window for monitoring


@dataclass  
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_failure_streak: int = 0
    current_success_streak: int = 0


class CircuitBreakerError(ServiceError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str, state: CircuitState):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.state = state


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation for protecting external service calls.
    
    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is open, requests fail fast
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name identifier for this circuit breaker
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CircuitBreakerStats()
        
        # State transition tracking
        self._last_state_change = time.time()
        self._half_open_requests = 0
        
        logger.info(f"Initialized circuit breaker '{name}' with config: {self.config}")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect a function with circuit breaker."""
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        
        wrapper.__name__ = f"circuit_breaker_{func.__name__}"
        wrapper.__doc__ = f"Circuit breaker protected: {func.__doc__}"
        return wrapper
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails in closed/half-open state
        """
        with self._lock:
            self._check_state_transition()
            
            # Fail fast if circuit is open
            if self._state == CircuitState.OPEN:
                self.stats.total_requests += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN - failing fast",
                    self.name,
                    CircuitState.OPEN
                )
            
            # Limit concurrent requests in half-open state
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= 1:  # Only allow one request at a time
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN - too many concurrent requests",
                        self.name,
                        CircuitState.HALF_OPEN
                    )
                self._half_open_requests += 1
        
        # Execute the function
        start_time = time.time()
        try:
            result = self._execute_with_timeout(func, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record successful execution
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
        
        finally:
            # Clean up half-open request counter
            with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_requests = max(0, self._half_open_requests - 1)
    
    def _execute_with_timeout(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with timeout protection."""
        import signal
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Function execution timed out")
        
        # Set up timeout (Unix only - for Windows, we'll use a different approach)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except (AttributeError, OSError):
            # Windows or signal not available - use threading timeout
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def target():
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(result)
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.config.timeout)
            
            if thread.is_alive():
                # Thread is still running - timeout occurred
                raise TimeoutError("Function execution timed out")
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            if not result_queue.empty():
                return result_queue.get()
            
            raise TimeoutError("Function execution failed unexpectedly")
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()
            self.stats.current_failure_streak = 0
            self.stats.current_success_streak += 1
            
            # Transition from half-open to closed if enough successes
            if (self._state == CircuitState.HALF_OPEN and 
                self.stats.current_success_streak >= self.config.success_threshold):
                self._transition_to_closed()
        
        logger.debug(f"Circuit breaker '{self.name}': Success recorded "
                    f"(execution_time={execution_time:.2f}s)")
    
    def _record_failure(self, exception: Exception, execution_time: float):
        """Record failed execution."""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.time()
            self.stats.current_success_streak = 0
            
            # Only count expected exceptions as circuit breaker failures
            if isinstance(exception, self.config.expected_exception_types):
                self.stats.current_failure_streak += 1
                
                # Check if we should open the circuit
                if (self._state == CircuitState.CLOSED and
                    self.stats.current_failure_streak >= self.config.failure_threshold):
                    self._transition_to_open()
                
                elif (self._state == CircuitState.HALF_OPEN):
                    # Any failure in half-open state returns to open
                    self._transition_to_open()
            
            # Track timeout failures separately
            if "timeout" in str(exception).lower():
                self.stats.timeouts += 1
        
        logger.warning(f"Circuit breaker '{self.name}': Failure recorded "
                      f"(exception={type(exception).__name__}, "
                      f"execution_time={execution_time:.2f}s)")
    
    def _check_state_transition(self):
        """Check if circuit should transition states based on time."""
        current_time = time.time()
        
        if (self._state == CircuitState.OPEN and
            current_time - self._last_state_change >= self.config.recovery_timeout):
            self._transition_to_half_open()
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._last_state_change = time.time()
        self._half_open_requests = 0
        
        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> CLOSED")
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._last_state_change = time.time()
        self._half_open_requests = 0
        self.stats.circuit_opened_count += 1
        
        logger.warning(f"Circuit breaker '{self.name}': {old_state.value} -> OPEN "
                      f"(failures={self.stats.current_failure_streak})")
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._last_state_change = time.time()
        self._half_open_requests = 0
        
        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> HALF_OPEN")
    
    def force_open(self):
        """Manually force circuit to open state."""
        with self._lock:
            self._transition_to_open()
        
        logger.warning(f"Circuit breaker '{self.name}': Manually forced to OPEN")
    
    def force_closed(self):
        """Manually force circuit to closed state."""
        with self._lock:
            self._transition_to_closed()
            # Reset failure streak
            self.stats.current_failure_streak = 0
        
        logger.info(f"Circuit breaker '{self.name}': Manually forced to CLOSED")
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self.stats = CircuitBreakerStats()
        
        logger.info(f"Circuit breaker '{self.name}': Statistics reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics as dictionary."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': (
                    self.stats.successful_requests / max(1, self.stats.total_requests)
                ),
                'failure_rate': (
                    self.stats.failed_requests / max(1, self.stats.total_requests)
                ),
                'timeouts': self.stats.timeouts,
                'circuit_opened_count': self.stats.circuit_opened_count,
                'current_failure_streak': self.stats.current_failure_streak,
                'current_success_streak': self.stats.current_success_streak,
                'last_failure_time': self.stats.last_failure_time,
                'last_success_time': self.stats.last_success_time,
                'time_in_current_state': time.time() - self._last_state_change
            }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides centralized management and monitoring of circuit breakers
    across the application.
    """
    
    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
        logger.info("Initialized CircuitBreakerRegistry")
    
    def create_breaker(self, 
                      name: str, 
                      config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Create or get existing circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration (only used if creating new breaker)
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name in self._breakers:
                logger.debug(f"Returning existing circuit breaker '{name}'")
                return self._breakers[name]
            
            breaker = CircuitBreaker(name, config)
            self._breakers[name] = breaker
            
            logger.info(f"Created new circuit breaker '{name}'")
            return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def remove_breaker(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False
    
    def list_breakers(self) -> list:
        """List all circuit breaker names."""
        with self._lock:
            return list(self._breakers.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_stats() 
                   for name, breaker in self._breakers.items()}
    
    def force_open_all(self):
        """Force all circuit breakers to open state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_open()
        
        logger.warning("Forced all circuit breakers to OPEN")
    
    def force_closed_all(self):
        """Force all circuit breakers to closed state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_closed()
        
        logger.info("Forced all circuit breakers to CLOSED")
    
    def reset_all_stats(self):
        """Reset statistics for all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset_stats()
        
        logger.info("Reset statistics for all circuit breakers")


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, 
                       config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker from global registry."""
    return _global_registry.create_breaker(name, config)


def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to protect function with circuit breaker.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        
    Example:
        @circuit_breaker("api_service")
        def call_external_api():
            # API call code here
            pass
    """
    breaker = get_circuit_breaker(name, config)
    return breaker


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers in global registry."""
    return _global_registry.get_all_stats()


# Utility functions for common configurations
def create_api_circuit_breaker_config(timeout: float = 30.0,
                                     failure_threshold: int = 5,
                                     recovery_timeout: float = 60.0) -> CircuitBreakerConfig:
    """Create circuit breaker config optimized for API calls."""
    return CircuitBreakerConfig(
        timeout=timeout,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=3,
        expected_exception_types=(ServiceError, IntegrationError, ConnectionError, TimeoutError)
    )


def create_database_circuit_breaker_config(timeout: float = 10.0,
                                          failure_threshold: int = 3,
                                          recovery_timeout: float = 30.0) -> CircuitBreakerConfig:
    """Create circuit breaker config optimized for database calls."""
    return CircuitBreakerConfig(
        timeout=timeout,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=2,
        expected_exception_types=(ServiceError, IntegrationError, ConnectionError)
    )