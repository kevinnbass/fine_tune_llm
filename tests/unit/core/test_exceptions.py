"""
Unit tests for core exceptions system.

This test module provides comprehensive coverage for the exception hierarchy,
error handling, circuit breaker, and retry mechanisms with 100% line coverage.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
import logging
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.fine_tune_llm.core.exceptions import (
    FineTuneLLMError,
    ConfigurationError,
    ModelError,
    TrainingError,
    InferenceError,
    DataError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    StorageError,
    ExternalServiceError,
    CircuitBreakerError,
    RetryError,
    SerializationError,
    SubscriptionError,
    EventBusError,
    PluginError,
    ServiceError,
    ComponentError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorReporter,
    CircuitBreaker,
    CircuitState,
    RetryStrategy,
    ExponentialBackoff,
    LinearBackoff,
    ConstantBackoff,
    RetryPolicy,
    ErrorAnalytics,
    get_error_reporter,
    handle_error,
    with_error_handling,
    retry_on_failure,
    circuit_breaker
)


class TestExceptionHierarchy:
    """Test exception hierarchy and base functionality."""
    
    def test_base_exception_creation(self):
        """Test FineTuneLLMError base exception."""
        error = FineTuneLLMError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.severity == ErrorSeverity.ERROR
        assert error.category == ErrorCategory.GENERAL
        assert isinstance(error.timestamp, datetime)
        assert error.timestamp.tzinfo == timezone.utc
        assert error.context == {}
        assert error.retry_count == 0
        assert error.correlation_id is None
    
    def test_base_exception_with_context(self):
        """Test base exception with context and metadata."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            user_id="user123",
            session_id="session456"
        )
        
        error = FineTuneLLMError(
            "Test error",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            correlation_id="corr-789"
        )
        
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.context.component == "test_component"
        assert error.correlation_id == "corr-789"
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError(
            "Invalid configuration",
            config_path="model.batch_size",
            config_value=-1
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.config_path == "model.batch_size"
        assert error.config_value == -1
    
    def test_model_error(self):
        """Test ModelError exception."""
        error = ModelError(
            "Model loading failed",
            model_name="test-model",
            model_path="/path/to/model"
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.MODEL
        assert error.model_name == "test-model"
        assert error.model_path == "/path/to/model"
    
    def test_training_error(self):
        """Test TrainingError exception."""
        error = TrainingError(
            "Training failed",
            epoch=5,
            step=100,
            loss=float('inf')
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.TRAINING
        assert error.epoch == 5
        assert error.step == 100
        assert error.loss == float('inf')
    
    def test_inference_error(self):
        """Test InferenceError exception."""
        error = InferenceError(
            "Inference failed",
            input_text="test input",
            batch_size=4
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.INFERENCE
        assert error.input_text == "test input"
        assert error.batch_size == 4
    
    def test_data_error(self):
        """Test DataError exception."""
        error = DataError(
            "Data validation failed",
            data_path="/path/to/data",
            expected_format="json",
            actual_format="csv"
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.DATA
        assert error.data_path == "/path/to/data"
        assert error.expected_format == "json"
        assert error.actual_format == "csv"
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError(
            "Validation failed",
            field="temperature",
            value=2.0,
            constraint="between 0 and 1"
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.VALIDATION
        assert error.field == "temperature"
        assert error.value == 2.0
        assert error.constraint == "between 0 and 1"
    
    def test_external_service_error(self):
        """Test ExternalServiceError exception."""
        error = ExternalServiceError(
            "API call failed",
            service_name="openai",
            endpoint="/v1/chat/completions",
            status_code=429
        )
        
        assert isinstance(error, FineTuneLLMError)
        assert error.category == ErrorCategory.EXTERNAL_SERVICE
        assert error.service_name == "openai"
        assert error.endpoint == "/v1/chat/completions"
        assert error.status_code == 429
    
    def test_error_serialization(self):
        """Test error serialization to dict."""
        context = ErrorContext(component="test", operation="test_op")
        error = FineTuneLLMError(
            "Test error",
            severity=ErrorSeverity.WARNING,
            context=context,
            correlation_id="test-corr"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Test error"
        assert error_dict["severity"] == "WARNING"
        assert error_dict["category"] == "GENERAL"
        assert error_dict["correlation_id"] == "test-corr"
        assert "timestamp" in error_dict
        assert "context" in error_dict


class TestErrorContext:
    """Test ErrorContext class."""
    
    def test_context_creation(self):
        """Test error context creation."""
        context = ErrorContext(
            component="trainer",
            operation="forward_pass",
            user_id="user123",
            session_id="session456",
            additional_data={"batch_id": 42}
        )
        
        assert context.component == "trainer"
        assert context.operation == "forward_pass"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.additional_data["batch_id"] == 42
    
    def test_context_to_dict(self):
        """Test context serialization."""
        context = ErrorContext(
            component="model",
            operation="load",
            additional_data={"model_size": "7B"}
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["component"] == "model"
        assert context_dict["operation"] == "load"
        assert context_dict["additional_data"]["model_size"] == "7B"
    
    def test_context_from_dict(self):
        """Test context deserialization."""
        context_dict = {
            "component": "inference",
            "operation": "predict",
            "user_id": "user789",
            "additional_data": {"temperature": 0.7}
        }
        
        context = ErrorContext.from_dict(context_dict)
        
        assert context.component == "inference"
        assert context.operation == "predict"
        assert context.user_id == "user789"
        assert context.additional_data["temperature"] == 0.7


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.handler = ErrorHandler()
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        error = FineTuneLLMError("Test error")
        
        with patch.object(self.handler, '_log_error') as mock_log:
            with patch.object(self.handler, '_report_error') as mock_report:
                result = self.handler.handle_error(error)
                
                assert result == error
                mock_log.assert_called_once_with(error)
                mock_report.assert_called_once_with(error)
    
    def test_handle_error_with_context(self):
        """Test error handling with context."""
        context = ErrorContext(component="test", operation="test_op")
        error = FineTuneLLMError("Test error", context=context)
        
        with patch.object(self.handler, '_log_error') as mock_log:
            result = self.handler.handle_error(error)
            
            assert result == error
            mock_log.assert_called_once_with(error)
    
    def test_handle_error_with_callback(self):
        """Test error handling with callback."""
        error = FineTuneLLMError("Test error")
        callback = Mock()
        
        result = self.handler.handle_error(error, callback=callback)
        
        assert result == error
        callback.assert_called_once_with(error)
    
    def test_handle_error_re_raise(self):
        """Test error handling with re-raise."""
        error = FineTuneLLMError("Test error")
        
        with pytest.raises(FineTuneLLMError):
            self.handler.handle_error(error, re_raise=True)
    
    def test_handle_error_suppress(self):
        """Test error handling with suppression."""
        error = FineTuneLLMError("Test error")
        
        result = self.handler.handle_error(error, suppress=True)
        
        assert result == error
    
    def test_add_error_listener(self):
        """Test adding error listener."""
        listener = Mock()
        self.handler.add_error_listener(listener)
        
        error = FineTuneLLMError("Test error")
        self.handler.handle_error(error)
        
        listener.assert_called_once_with(error)
    
    def test_remove_error_listener(self):
        """Test removing error listener."""
        listener = Mock()
        self.handler.add_error_listener(listener)
        self.handler.remove_error_listener(listener)
        
        error = FineTuneLLMError("Test error")
        self.handler.handle_error(error)
        
        listener.assert_not_called()


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        assert breaker._failure_threshold == 3
        assert breaker._recovery_timeout == 60
        assert breaker._expected_exception == Exception
        assert breaker._state == CircuitState.CLOSED
        assert breaker._failure_count == 0
    
    def test_circuit_breaker_closed_success(self):
        """Test circuit breaker in closed state with success."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def successful_function():
            return "success"
        
        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker._state == CircuitState.CLOSED
        assert breaker._failure_count == 0
    
    def test_circuit_breaker_closed_failure(self):
        """Test circuit breaker in closed state with failure."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception, match="Test failure"):
            breaker.call(failing_function)
        
        assert breaker._state == CircuitState.CLOSED
        assert breaker._failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception, match="Test failure"):
            breaker.call(failing_function)
        
        assert breaker._state == CircuitState.OPEN
        assert breaker._failure_count == 2
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        def failing_function():
            raise Exception("Test failure")
        
        # Trigger circuit to open
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        assert breaker._state == CircuitState.OPEN
        
        # Should raise CircuitBreakerError when open
        with pytest.raises(CircuitBreakerError):
            breaker.call(failing_function)
    
    def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker half-open to closed transition."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def initially_failing_then_working():
            if breaker._state == CircuitState.HALF_OPEN:
                return "success"
            raise Exception("Still failing")
        
        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(initially_failing_then_working)
        
        assert breaker._state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open and then closed on success
        result = breaker.call(initially_failing_then_working)
        assert result == "success"
        assert breaker._state == CircuitState.CLOSED
        assert breaker._failure_count == 0
    
    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker half-open back to open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def always_failing():
            raise Exception("Always failing")
        
        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(always_failing)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open and back to open on failure
        with pytest.raises(Exception):
            breaker.call(always_failing)
        
        assert breaker._state == CircuitState.OPEN
    
    def test_circuit_breaker_statistics(self):
        """Test circuit breaker statistics."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def sometimes_failing():
            if breaker._success_count < 2:
                return "success"
            raise Exception("Failure")
        
        # Successful calls
        breaker.call(sometimes_failing)
        breaker.call(sometimes_failing)
        
        # Failed call
        with pytest.raises(Exception):
            breaker.call(sometimes_failing)
        
        stats = breaker.get_statistics()
        
        assert stats["state"] == "CLOSED"
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 2
        assert stats["total_calls"] == 3
        assert "last_failure_time" in stats
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        breaker = CircuitBreaker(failure_threshold=1)
        
        def failing_function():
            raise Exception("Failure")
        
        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        assert breaker._state == CircuitState.OPEN
        
        # Reset circuit
        breaker.reset()
        
        assert breaker._state == CircuitState.CLOSED
        assert breaker._failure_count == 0


class TestBackoffStrategies:
    """Test backoff strategies."""
    
    def test_constant_backoff(self):
        """Test constant backoff strategy."""
        backoff = ConstantBackoff(delay=1.0)
        
        assert backoff.calculate_delay(1) == 1.0
        assert backoff.calculate_delay(5) == 1.0
        assert backoff.calculate_delay(10) == 1.0
    
    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        backoff = LinearBackoff(base_delay=1.0, increment=0.5)
        
        assert backoff.calculate_delay(1) == 1.0
        assert backoff.calculate_delay(2) == 1.5
        assert backoff.calculate_delay(3) == 2.0
    
    def test_linear_backoff_with_max(self):
        """Test linear backoff with maximum delay."""
        backoff = LinearBackoff(base_delay=1.0, increment=2.0, max_delay=5.0)
        
        assert backoff.calculate_delay(1) == 1.0
        assert backoff.calculate_delay(2) == 3.0
        assert backoff.calculate_delay(3) == 5.0  # Capped at max_delay
        assert backoff.calculate_delay(4) == 5.0  # Still capped
    
    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        backoff = ExponentialBackoff(base_delay=1.0, multiplier=2.0)
        
        assert backoff.calculate_delay(1) == 1.0
        assert backoff.calculate_delay(2) == 2.0
        assert backoff.calculate_delay(3) == 4.0
        assert backoff.calculate_delay(4) == 8.0
    
    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter=True,
            max_delay=10.0
        )
        
        # With jitter, delays should be random but within expected range
        delay1 = backoff.calculate_delay(1)
        delay2 = backoff.calculate_delay(2)
        delay3 = backoff.calculate_delay(3)
        
        # Base delay should be between 0.5 and 1.5 (50% jitter)
        assert 0.5 <= delay1 <= 1.5
        assert 1.0 <= delay2 <= 3.0
        assert 2.0 <= delay3 <= 6.0
    
    def test_exponential_backoff_max_delay(self):
        """Test exponential backoff with maximum delay."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            max_delay=5.0
        )
        
        assert backoff.calculate_delay(1) == 1.0
        assert backoff.calculate_delay(2) == 2.0
        assert backoff.calculate_delay(3) == 4.0
        assert backoff.calculate_delay(4) == 5.0  # Capped at max_delay
        assert backoff.calculate_delay(10) == 5.0  # Still capped


class TestRetryPolicy:
    """Test RetryPolicy class."""
    
    def test_retry_policy_creation(self):
        """Test retry policy creation."""
        backoff = ExponentialBackoff(base_delay=1.0)
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=backoff,
            retry_exceptions=(ValueError, TypeError)
        )
        
        assert policy.max_attempts == 3
        assert policy.backoff_strategy == backoff
        assert policy.retry_exceptions == (ValueError, TypeError)
    
    def test_should_retry_exception_type(self):
        """Test retry decision based on exception type."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_exceptions=(ValueError,)
        )
        
        assert policy.should_retry(ValueError("test"), 1)
        assert not policy.should_retry(TypeError("test"), 1)
        assert not policy.should_retry(Exception("test"), 1)
    
    def test_should_retry_max_attempts(self):
        """Test retry decision based on max attempts."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_exceptions=(ValueError,)
        )
        
        error = ValueError("test")
        assert policy.should_retry(error, 1)
        assert policy.should_retry(error, 2)
        assert not policy.should_retry(error, 3)  # Reached max attempts
    
    def test_should_retry_custom_condition(self):
        """Test retry decision with custom condition."""
        def custom_condition(exception, attempt):
            return attempt < 2 and "retryable" in str(exception)
        
        policy = RetryPolicy(
            max_attempts=5,
            retry_condition=custom_condition
        )
        
        retryable_error = Exception("retryable error")
        non_retryable_error = Exception("fatal error")
        
        assert policy.should_retry(retryable_error, 1)
        assert not policy.should_retry(retryable_error, 2)  # Custom condition fails
        assert not policy.should_retry(non_retryable_error, 1)
    
    def test_execute_successful(self):
        """Test successful execution without retries."""
        policy = RetryPolicy(max_attempts=3)
        
        def successful_function():
            return "success"
        
        result = policy.execute(successful_function)
        assert result == "success"
    
    def test_execute_retry_and_succeed(self):
        """Test execution with retry that eventually succeeds."""
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=ConstantBackoff(0.01),  # Fast retry for testing
            retry_exceptions=(ValueError,)
        )
        
        call_count = 0
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return f"success on attempt {call_count}"
        
        result = policy.execute(flaky_function)
        assert result == "success on attempt 3"
        assert call_count == 3
    
    def test_execute_retry_exhausted(self):
        """Test execution with retry exhaustion."""
        policy = RetryPolicy(
            max_attempts=2,
            backoff_strategy=ConstantBackoff(0.01),
            retry_exceptions=(ValueError,)
        )
        
        def always_failing():
            raise ValueError("Always fails")
        
        with pytest.raises(RetryError) as exc_info:
            policy.execute(always_failing)
        
        retry_error = exc_info.value
        assert retry_error.attempts == 2
        assert isinstance(retry_error.last_exception, ValueError)
    
    def test_execute_non_retryable_exception(self):
        """Test execution with non-retryable exception."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_exceptions=(ValueError,)
        )
        
        def failing_function():
            raise TypeError("Non-retryable error")
        
        with pytest.raises(TypeError, match="Non-retryable error"):
            policy.execute(failing_function)


class TestErrorAnalytics:
    """Test ErrorAnalytics class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analytics = ErrorAnalytics()
    
    def test_record_error(self):
        """Test recording an error."""
        error = FineTuneLLMError("Test error")
        
        self.analytics.record_error(error)
        
        stats = self.analytics.get_statistics()
        assert stats["total_errors"] == 1
        assert "GENERAL" in stats["errors_by_category"]
        assert stats["errors_by_category"]["GENERAL"] == 1
    
    def test_record_multiple_errors(self):
        """Test recording multiple errors."""
        errors = [
            FineTuneLLMError("Error 1", category=ErrorCategory.MODEL),
            FineTuneLLMError("Error 2", category=ErrorCategory.TRAINING),
            FineTuneLLMError("Error 3", category=ErrorCategory.MODEL)
        ]
        
        for error in errors:
            self.analytics.record_error(error)
        
        stats = self.analytics.get_statistics()
        assert stats["total_errors"] == 3
        assert stats["errors_by_category"]["MODEL"] == 2
        assert stats["errors_by_category"]["TRAINING"] == 1
    
    def test_get_error_trends(self):
        """Test getting error trends."""
        # Record errors over time
        errors = [
            FineTuneLLMError("Error 1"),
            FineTuneLLMError("Error 2"),
            FineTuneLLMError("Error 3")
        ]
        
        for error in errors:
            self.analytics.record_error(error)
        
        trends = self.analytics.get_error_trends(hours=1)
        assert len(trends) > 0
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        context = ErrorContext(component="trainer")
        error = FineTuneLLMError(
            "Training error",
            severity=ErrorSeverity.CRITICAL,
            context=context
        )
        
        self.analytics.record_error(error)
        
        summary = self.analytics.get_error_summary()
        
        assert summary["total_errors"] == 1
        assert "CRITICAL" in summary["errors_by_severity"]
        assert "trainer" in summary["errors_by_component"]
    
    def test_clear_statistics(self):
        """Test clearing error statistics."""
        error = FineTuneLLMError("Test error")
        self.analytics.record_error(error)
        
        assert self.analytics.get_statistics()["total_errors"] == 1
        
        self.analytics.clear_statistics()
        
        assert self.analytics.get_statistics()["total_errors"] == 0


class TestErrorReporter:
    """Test ErrorReporter class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reporter = ErrorReporter()
    
    def test_report_error_basic(self):
        """Test basic error reporting."""
        error = FineTuneLLMError("Test error")
        
        with patch.object(self.reporter, '_send_to_analytics') as mock_analytics:
            with patch.object(self.reporter, '_send_to_logging') as mock_logging:
                self.reporter.report_error(error)
                
                mock_analytics.assert_called_once_with(error)
                mock_logging.assert_called_once_with(error)
    
    def test_report_error_with_context(self):
        """Test error reporting with context."""
        context = ErrorContext(component="test", user_id="user123")
        error = FineTuneLLMError("Test error", context=context)
        
        with patch.object(self.reporter, '_send_to_analytics') as mock_analytics:
            self.reporter.report_error(error)
            mock_analytics.assert_called_once_with(error)
    
    def test_add_error_sink(self):
        """Test adding error sink."""
        sink = Mock()
        self.reporter.add_error_sink(sink)
        
        error = FineTuneLLMError("Test error")
        self.reporter.report_error(error)
        
        sink.assert_called_once_with(error)
    
    def test_remove_error_sink(self):
        """Test removing error sink."""
        sink = Mock()
        self.reporter.add_error_sink(sink)
        self.reporter.remove_error_sink(sink)
        
        error = FineTuneLLMError("Test error")
        self.reporter.report_error(error)
        
        sink.assert_not_called()


class TestDecorators:
    """Test error handling decorators."""
    
    def test_with_error_handling_decorator(self):
        """Test with_error_handling decorator."""
        @with_error_handling(FineTuneLLMError, "Operation failed")
        def failing_function():
            raise ValueError("Original error")
        
        with pytest.raises(FineTuneLLMError, match="Operation failed"):
            failing_function()
    
    def test_with_error_handling_success(self):
        """Test with_error_handling decorator with success."""
        @with_error_handling(FineTuneLLMError, "Operation failed")
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_on_failure_decorator(self):
        """Test retry_on_failure decorator."""
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_failure_exhausted(self):
        """Test retry_on_failure decorator with exhausted retries."""
        @retry_on_failure(max_attempts=2, delay=0.01)
        def always_failing():
            raise ValueError("Always fails")
        
        with pytest.raises(RetryError):
            always_failing()
    
    def test_circuit_breaker_decorator(self):
        """Test circuit_breaker decorator."""
        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def sometimes_failing():
            if hasattr(sometimes_failing, 'call_count'):
                sometimes_failing.call_count += 1
            else:
                sometimes_failing.call_count = 1
            
            if sometimes_failing.call_count <= 2:
                raise Exception(f"Failure {sometimes_failing.call_count}")
            return "success"
        
        # First two calls should fail and open circuit
        with pytest.raises(Exception):
            sometimes_failing()
        
        with pytest.raises(Exception):
            sometimes_failing()
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            sometimes_failing()


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_get_error_reporter(self):
        """Test getting global error reporter."""
        reporter1 = get_error_reporter()
        reporter2 = get_error_reporter()
        
        # Should return same instance
        assert reporter1 is reporter2
        assert isinstance(reporter1, ErrorReporter)
    
    def test_handle_error_function(self):
        """Test global handle_error function."""
        error = FineTuneLLMError("Test error")
        
        with patch('src.fine_tune_llm.core.exceptions.get_error_reporter') as mock_get:
            mock_reporter = Mock()
            mock_get.return_value = mock_reporter
            
            result = handle_error(error)
            
            assert result == error
            mock_reporter.report_error.assert_called_once_with(error)


class TestIntegration:
    """Integration tests for error handling system."""
    
    def test_full_error_workflow(self):
        """Test complete error handling workflow."""
        # Setup components
        handler = ErrorHandler()
        analytics = ErrorAnalytics()
        reporter = ErrorReporter()
        
        # Connect components
        reporter.add_error_sink(analytics.record_error)
        handler.add_error_listener(reporter.report_error)
        
        # Create and handle error
        context = ErrorContext(component="test", operation="test_op")
        error = FineTuneLLMError(
            "Test error",
            severity=ErrorSeverity.WARNING,
            context=context
        )
        
        handler.handle_error(error)
        
        # Verify error was processed
        stats = analytics.get_statistics()
        assert stats["total_errors"] == 1
        assert stats["errors_by_severity"]["WARNING"] == 1
    
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry policy."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=ConstantBackoff(0.01)
        )
        
        call_count = 0
        def unreliable_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception(f"Service failure {call_count}")
            return "success"
        
        # First few calls should fail and eventually open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: policy.execute(unreliable_service))
        
        # After threshold, circuit should be open
        assert breaker._state == CircuitState.OPEN
        
        with pytest.raises(CircuitBreakerError):
            breaker.call(lambda: policy.execute(unreliable_service))
    
    def test_error_correlation(self):
        """Test error correlation across operations."""
        correlation_id = "test-correlation-123"
        analytics = ErrorAnalytics()
        
        # Create related errors with same correlation ID
        errors = [
            FineTuneLLMError(
                "Database connection failed",
                category=ErrorCategory.EXTERNAL_SERVICE,
                correlation_id=correlation_id
            ),
            FineTuneLLMError(
                "Training interrupted",
                category=ErrorCategory.TRAINING,
                correlation_id=correlation_id
            )
        ]
        
        for error in errors:
            analytics.record_error(error)
        
        # Verify errors can be correlated
        summary = analytics.get_error_summary()
        assert summary["total_errors"] == 2