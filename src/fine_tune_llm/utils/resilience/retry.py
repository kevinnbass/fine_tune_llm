"""
Retry mechanisms with exponential backoff and jitter.

This module provides sophisticated retry logic with various backoff strategies
to handle transient failures in external service calls.
"""

import time
import random
import functools
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Type, Tuple, List, Union
from dataclasses import dataclass
import logging

from ...core.exceptions import ServiceError, IntegrationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BackoffStrategy(Enum):
    """Backoff strategies for retry attempts."""
    FIXED = "fixed"                    # Fixed delay between attempts
    EXPONENTIAL = "exponential"        # Exponentially increasing delay
    LINEAR = "linear"                 # Linearly increasing delay  
    FIBONACCI = "fibonacci"           # Fibonacci sequence delays
    RANDOM = "random"                 # Random delay within bounds


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3                     # Maximum number of attempts
    base_delay: float = 1.0                  # Base delay in seconds
    max_delay: float = 300.0                 # Maximum delay in seconds
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    exponential_base: float = 2.0            # Base for exponential backoff
    jitter: bool = True                      # Add random jitter to prevent thundering herd
    jitter_range: float = 0.1               # Jitter as fraction of delay (±10%)
    
    # Exception handling
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ServiceError, IntegrationError, ConnectionError, TimeoutError
    )
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (
        ValueError, TypeError, AttributeError
    )
    
    # Conditions
    retry_on_result: Optional[Callable[[Any], bool]] = None  # Retry based on result
    stop_on_result: Optional[Callable[[Any], bool]] = None   # Stop based on result


@dataclass
class RetryState:
    """State tracking for retry attempts."""
    attempt: int = 0
    total_elapsed_time: float = 0.0
    last_exception: Optional[Exception] = None
    last_result: Optional[Any] = None
    start_time: float = 0.0
    delays: List[float] = None
    
    def __post_init__(self):
        if self.delays is None:
            self.delays = []


class BackoffCalculator(ABC):
    """Abstract base class for backoff calculators."""
    
    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        """Calculate delay for the given attempt number."""
        pass


class FixedBackoff(BackoffCalculator):
    """Fixed delay backoff strategy."""
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        return min(base_delay, max_delay)


class ExponentialBackoff(BackoffCalculator):
    """Exponential backoff strategy."""
    
    def __init__(self, base: float = 2.0):
        self.base = base
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay * (self.base ** (attempt - 1))
        return min(delay, max_delay)


class LinearBackoff(BackoffCalculator):
    """Linear backoff strategy."""
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay * attempt
        return min(delay, max_delay)


class FibonacciBackoff(BackoffCalculator):
    """Fibonacci sequence backoff strategy."""
    
    def __init__(self):
        self._cache = {0: 0, 1: 1}
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number with caching."""
        if n in self._cache:
            return self._cache[n]
        
        self._cache[n] = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        return self._cache[n]
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        fib_multiplier = self._fibonacci(max(1, attempt))
        delay = base_delay * fib_multiplier
        return min(delay, max_delay)


class RandomBackoff(BackoffCalculator):
    """Random backoff strategy."""
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        return random.uniform(base_delay, max_delay)


class RetryError(ServiceError):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, attempts: int, total_time: float, 
                 last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.attempts = attempts
        self.total_time = total_time
        self.last_exception = last_exception


class Retrier:
    """
    Advanced retry mechanism with configurable backoff strategies.
    
    Provides sophisticated retry logic with exponential backoff, jitter,
    and flexible exception handling for resilient service calls.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retrier with configuration.
        
        Args:
            config: Retry configuration settings
        """
        self.config = config or RetryConfig()
        self._backoff_calculator = self._create_backoff_calculator()
        
        logger.debug(f"Initialized Retrier with strategy: {self.config.backoff_strategy.value}")
    
    def _create_backoff_calculator(self) -> BackoffCalculator:
        """Create appropriate backoff calculator based on strategy."""
        strategy_map = {
            BackoffStrategy.FIXED: FixedBackoff(),
            BackoffStrategy.EXPONENTIAL: ExponentialBackoff(self.config.exponential_base),
            BackoffStrategy.LINEAR: LinearBackoff(),
            BackoffStrategy.FIBONACCI: FibonacciBackoff(),
            BackoffStrategy.RANDOM: RandomBackoff()
        }
        
        return strategy_map[self.config.backoff_strategy]
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to add retry behavior to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryError: If all retry attempts are exhausted
        """
        state = RetryState(start_time=time.time())
        
        for attempt in range(1, self.config.max_attempts + 1):
            state.attempt = attempt
            
            try:
                logger.debug(f"Retry attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Check if we should stop based on result
                if self.config.stop_on_result and self.config.stop_on_result(result):
                    logger.info(f"Stopping retries based on result after attempt {attempt}")
                    return result
                
                # Check if we should retry based on result
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if attempt < self.config.max_attempts:
                        state.last_result = result
                        self._wait_before_retry(state)
                        continue
                
                # Success - return result
                logger.debug(f"Function {func.__name__} succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                state.last_exception = e
                state.total_elapsed_time = time.time() - state.start_time
                
                # Check if this exception should not be retried
                if isinstance(e, self.config.non_retryable_exceptions):
                    logger.warning(f"Non-retryable exception {type(e).__name__}: {e}")
                    raise
                
                # Check if this exception should be retried
                if not isinstance(e, self.config.retryable_exceptions):
                    logger.warning(f"Exception {type(e).__name__} not in retryable list: {e}")
                    raise
                
                logger.warning(f"Attempt {attempt} failed with {type(e).__name__}: {e}")
                
                # If this was the last attempt, raise RetryError
                if attempt >= self.config.max_attempts:
                    raise RetryError(
                        f"Function {func.__name__} failed after {attempt} attempts "
                        f"(total time: {state.total_elapsed_time:.2f}s)",
                        attempt,
                        state.total_elapsed_time,
                        e
                    )
                
                # Wait before next retry
                self._wait_before_retry(state)
        
        # This should never be reached, but just in case
        raise RetryError(
            f"Function {func.__name__} failed after exhausting all retry attempts",
            state.attempt,
            state.total_elapsed_time,
            state.last_exception
        )
    
    def _wait_before_retry(self, state: RetryState):
        """Wait before the next retry attempt."""
        # Calculate base delay
        delay = self._backoff_calculator.calculate_delay(
            state.attempt,
            self.config.base_delay,
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        state.delays.append(delay)
        
        logger.debug(f"Waiting {delay:.2f}s before retry attempt {state.attempt + 1}")
        time.sleep(delay)


class AsyncRetrier:
    """
    Async version of Retrier for async functions.
    
    Note: This is a basic implementation. For production use with async,
    consider using libraries like tenacity or backoff.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize async retrier."""
        self.config = config or RetryConfig()
        self._backoff_calculator = self._create_backoff_calculator()
    
    def _create_backoff_calculator(self) -> BackoffCalculator:
        """Create appropriate backoff calculator based on strategy."""
        strategy_map = {
            BackoffStrategy.FIXED: FixedBackoff(),
            BackoffStrategy.EXPONENTIAL: ExponentialBackoff(self.config.exponential_base),
            BackoffStrategy.LINEAR: LinearBackoff(),
            BackoffStrategy.FIBONACCI: FibonacciBackoff(),
            BackoffStrategy.RANDOM: RandomBackoff()
        }
        
        return strategy_map[self.config.backoff_strategy]
    
    def __call__(self, func):
        """Decorator for async functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        
        return wrapper
    
    async def execute(self, func, *args, **kwargs):
        """Execute async function with retry logic."""
        import asyncio
        
        state = RetryState(start_time=time.time())
        
        for attempt in range(1, self.config.max_attempts + 1):
            state.attempt = attempt
            
            try:
                logger.debug(f"Async retry attempt {attempt}/{self.config.max_attempts}")
                
                # Execute the async function
                result = await func(*args, **kwargs)
                
                # Check conditions (same logic as sync version)
                if self.config.stop_on_result and self.config.stop_on_result(result):
                    return result
                
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if attempt < self.config.max_attempts:
                        state.last_result = result
                        await self._async_wait_before_retry(state)
                        continue
                
                return result
                
            except Exception as e:
                state.last_exception = e
                state.total_elapsed_time = time.time() - state.start_time
                
                if isinstance(e, self.config.non_retryable_exceptions):
                    raise
                
                if not isinstance(e, self.config.retryable_exceptions):
                    raise
                
                logger.warning(f"Async attempt {attempt} failed: {e}")
                
                if attempt >= self.config.max_attempts:
                    raise RetryError(
                        f"Async function failed after {attempt} attempts",
                        attempt,
                        state.total_elapsed_time,
                        e
                    )
                
                await self._async_wait_before_retry(state)
        
        raise RetryError(
            f"Async function failed after exhausting all retry attempts",
            state.attempt,
            state.total_elapsed_time,
            state.last_exception
        )
    
    async def _async_wait_before_retry(self, state: RetryState):
        """Async wait before next retry attempt."""
        import asyncio
        
        delay = self._backoff_calculator.calculate_delay(
            state.attempt,
            self.config.base_delay,
            self.config.max_delay
        )
        
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        state.delays.append(delay)
        
        logger.debug(f"Async waiting {delay:.2f}s before retry")
        await asyncio.sleep(delay)


# Convenience functions and decorators
def retry(max_attempts: int = 3,
          base_delay: float = 1.0,
          backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
          retryable_exceptions: Tuple[Type[Exception], ...] = None,
          **kwargs):
    """
    Decorator for adding retry behavior to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between attempts
        backoff_strategy: Strategy for calculating delays
        retryable_exceptions: Exception types to retry on
        **kwargs: Additional RetryConfig parameters
    
    Example:
        @retry(max_attempts=3, base_delay=2.0)
        def call_external_service():
            # Service call code
            pass
    """
    if retryable_exceptions is None:
        retryable_exceptions = (ServiceError, IntegrationError, ConnectionError, TimeoutError)
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy,
        retryable_exceptions=retryable_exceptions,
        **kwargs
    )
    
    return Retrier(config)


def async_retry(max_attempts: int = 3,
                base_delay: float = 1.0,
                backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
                **kwargs):
    """Decorator for adding retry behavior to async functions."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy,
        **kwargs
    )
    
    return AsyncRetrier(config)


# Predefined retry configurations for common use cases
def create_api_retry_config(max_attempts: int = 3,
                           base_delay: float = 1.0,
                           max_delay: float = 60.0) -> RetryConfig:
    """Create retry config optimized for API calls."""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=(
            ServiceError, IntegrationError, ConnectionError, 
            TimeoutError, IOError
        )
    )


def create_database_retry_config(max_attempts: int = 2,
                                base_delay: float = 0.5,
                                max_delay: float = 10.0) -> RetryConfig:
    """Create retry config optimized for database calls."""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        exponential_base=1.5,
        jitter=True,
        retryable_exceptions=(
            ServiceError, IntegrationError, ConnectionError
        )
    )


def create_file_operation_retry_config(max_attempts: int = 5,
                                      base_delay: float = 0.1,
                                      max_delay: float = 5.0) -> RetryConfig:
    """Create retry config optimized for file operations."""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=BackoffStrategy.LINEAR,
        jitter=False,  # File operations don't need jitter
        retryable_exceptions=(IOError, OSError, PermissionError)
    )