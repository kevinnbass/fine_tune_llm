"""
Retry decorators for handling transient failures.

This module provides decorators for implementing retry logic with
various backoff strategies and error handling.
"""

import functools
import time
import random
import logging
from typing import Callable, Any, Type, Union, Tuple, Optional, List

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    jitter: bool = True
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for each retry
        max_delay: Maximum delay between retries (seconds)
        exceptions: Exception types to retry on
        on_retry: Optional callback function called on each retry
        jitter: Add random jitter to delay
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise the exception
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with jitter
                    actual_delay = current_delay
                    if jitter:
                        actual_delay = current_delay * (0.5 + random.random() * 0.5)
                    
                    actual_delay = min(actual_delay, max_delay)
                    
                    # Log retry attempt
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {actual_delay:.2f} seconds..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt + 1, e)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    # Wait before retry
                    time.sleep(actual_delay)
                    
                    # Update delay for next attempt
                    current_delay = min(current_delay * backoff, max_delay)
            
            # This should never be reached due to the raise above
            raise last_exception
        
        return wrapper
    return decorator


def exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    max_attempts: int = 5,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    jitter: bool = True
) -> Callable:
    """
    Decorator for exponential backoff retry strategy.
    
    Args:
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        max_attempts: Maximum retry attempts
        exceptions: Exception types to retry on
        jitter: Add random jitter to delay
        
    Returns:
        Decorated function with exponential backoff retry
    """
    return retry(
        max_attempts=max_attempts,
        delay=base_delay,
        backoff=2.0,
        max_delay=max_delay,
        exceptions=exceptions,
        jitter=jitter
    )


def retry_on_condition(
    condition: Callable[[Exception], bool],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Callable:
    """
    Decorator for retrying based on a condition function.
    
    Args:
        condition: Function that takes an exception and returns True to retry
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        
    Returns:
        Decorated function with conditional retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry based on condition
                    if not condition(e) or attempt == max_attempts - 1:
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_with_circuit_breaker(
    max_attempts: int = 3,
    delay: float = 1.0,
    circuit_break_threshold: int = 5,
    circuit_break_timeout: float = 60.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
) -> Callable:
    """
    Decorator combining retry logic with circuit breaker pattern.
    
    Args:
        max_attempts: Maximum retry attempts per call
        delay: Delay between retries
        circuit_break_threshold: Number of consecutive failures before circuit opens
        circuit_break_timeout: Time to wait before attempting to close circuit
        exceptions: Exception types to handle
        
    Returns:
        Decorated function with retry and circuit breaker logic
    """
    # Circuit breaker state (shared across all decorated functions)
    circuit_state = {
        'failures': 0,
        'last_failure_time': 0,
        'is_open': False
    }
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check circuit breaker state
            current_time = time.time()
            
            if circuit_state['is_open']:
                if current_time - circuit_state['last_failure_time'] < circuit_break_timeout:
                    raise Exception("Circuit breaker is open")
                else:
                    # Try to close circuit
                    circuit_state['is_open'] = False
                    circuit_state['failures'] = 0
                    logger.info(f"Circuit breaker closed for {func.__name__}")
            
            # Attempt function with retry
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - reset circuit breaker
                    if circuit_state['failures'] > 0:
                        circuit_state['failures'] = 0
                        logger.info(f"Circuit breaker reset for {func.__name__}")
                    
                    return result
                    
                except exceptions as e:
                    circuit_state['failures'] += 1
                    circuit_state['last_failure_time'] = current_time
                    
                    # Check if circuit should open
                    if circuit_state['failures'] >= circuit_break_threshold:
                        circuit_state['is_open'] = True
                        logger.error(f"Circuit breaker opened for {func.__name__}")
                        raise Exception("Circuit breaker opened due to consecutive failures")
                    
                    if attempt == max_attempts - 1:
                        raise
                    
                    logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                    time.sleep(delay)
        
        return wrapper
    return decorator


class RetryableError(Exception):
    """Exception class for errors that should trigger retry."""
    pass


class NonRetryableError(Exception):
    """Exception class for errors that should not trigger retry."""
    pass


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception should trigger a retry.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if the exception is retryable
    """
    # Define retryable error conditions
    retryable_conditions = [
        isinstance(exception, RetryableError),
        isinstance(exception, ConnectionError),
        isinstance(exception, TimeoutError),
        "timeout" in str(exception).lower(),
        "connection" in str(exception).lower(),
        "temporary" in str(exception).lower(),
        "transient" in str(exception).lower(),
    ]
    
    # Define non-retryable error conditions
    non_retryable_conditions = [
        isinstance(exception, NonRetryableError),
        isinstance(exception, KeyboardInterrupt),
        isinstance(exception, SystemExit),
        isinstance(exception, ValueError),  # Usually indicates bad input
        isinstance(exception, TypeError),   # Usually indicates programming error
    ]
    
    # Check non-retryable conditions first
    if any(non_retryable_conditions):
        return False
    
    # Check retryable conditions
    return any(retryable_conditions)


def smart_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> Callable:
    """
    Smart retry decorator that automatically determines if errors are retryable.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        jitter: Add random jitter to delays
        
    Returns:
        Decorated function with smart retry logic
    """
    return retry_on_condition(
        condition=is_retryable_error,
        max_attempts=max_attempts,
        delay=base_delay,
        backoff=2.0
    )