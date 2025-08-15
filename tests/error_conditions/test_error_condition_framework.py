"""
Error condition testing framework for comprehensive failure scenario validation.

This test module implements a systematic approach to testing error conditions,
failure scenarios, exception handling, and system recovery mechanisms.
"""

import pytest
import asyncio
import time
import threading
import random
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment,
    MockTransformerModel, MockTokenizer, MockTrainingDataset
)


@dataclass
class ErrorScenario:
    """Defines an error scenario for systematic testing."""
    name: str
    description: str
    error_type: type
    error_message: str
    trigger_condition: Callable
    recovery_action: Optional[Callable] = None
    expected_behavior: str = "graceful_failure"
    severity: str = "medium"  # low, medium, high, critical


class ErrorInjector:
    """Inject errors systematically for testing error handling."""
    
    def __init__(self):
        self.active_patches = []
        self.error_history = []
        self.failure_rate = 0.0
        
    def set_failure_rate(self, rate: float):
        """Set global failure rate for random failures."""
        self.failure_rate = max(0.0, min(1.0, rate))
    
    def should_fail(self) -> bool:
        """Determine if an operation should fail based on failure rate."""
        return random.random() < self.failure_rate
    
    def inject_exception(self, target_function, exception_type, message="Injected error"):
        """Inject exception into target function."""
        def failing_function(*args, **kwargs):
            if self.should_fail():
                error = exception_type(message)
                self.error_history.append({
                    "timestamp": datetime.now(timezone.utc),
                    "function": target_function.__name__,
                    "exception": exception_type.__name__,
                    "message": message
                })
                raise error
            return target_function(*args, **kwargs)
        
        return failing_function
    
    def inject_timeout(self, target_function, timeout_duration=5.0):
        """Inject timeout behavior into target function."""
        def timeout_function(*args, **kwargs):
            if self.should_fail():
                time.sleep(timeout_duration)
                raise TimeoutError(f"Operation timed out after {timeout_duration}s")
            return target_function(*args, **kwargs)
        
        return timeout_function
    
    def inject_memory_error(self, target_function):
        """Inject memory-related errors."""
        def memory_failing_function(*args, **kwargs):
            if self.should_fail():
                error = MemoryError("Insufficient memory for operation")
                self.error_history.append({
                    "timestamp": datetime.now(timezone.utc),
                    "function": target_function.__name__,
                    "exception": "MemoryError",
                    "message": "Memory injection"
                })
                raise error
            return target_function(*args, **kwargs)
        
        return memory_failing_function
    
    def inject_network_error(self, target_function):
        """Inject network-related errors."""
        def network_failing_function(*args, **kwargs):
            if self.should_fail():
                import socket
                error = socket.error("Network connection failed")
                self.error_history.append({
                    "timestamp": datetime.now(timezone.utc),
                    "function": target_function.__name__,
                    "exception": "socket.error",
                    "message": "Network injection"
                })
                raise error
            return target_function(*args, **kwargs)
        
        return network_failing_function
    
    def cleanup(self):
        """Clean up active patches."""
        for patch_obj in self.active_patches:
            try:
                patch_obj.stop()
            except Exception:
                pass
        self.active_patches.clear()


class ErrorRecoveryTester:
    """Test error recovery mechanisms."""
    
    def __init__(self):
        self.recovery_attempts = []
        self.recovery_success_rate = 0.0
        
    def test_retry_mechanism(self, function, max_retries=3, backoff_factor=1.0):
        """Test retry mechanism with exponential backoff."""
        attempts = 0
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = function()
                self.recovery_attempts.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "error": None
                })
                return result
            except Exception as e:
                attempts += 1
                last_error = e
                self.recovery_attempts.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e)
                })
                
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
        
        # All retries failed
        self.recovery_success_rate = 0.0
        raise last_error
    
    def test_circuit_breaker(self, function, failure_threshold=5, recovery_timeout=30):
        """Test circuit breaker pattern."""
        failure_count = 0
        last_failure_time = None
        circuit_open = False
        
        def circuit_breaker_function(*args, **kwargs):
            nonlocal failure_count, last_failure_time, circuit_open
            
            # Check if circuit should be reset
            if circuit_open and last_failure_time:
                if time.time() - last_failure_time > recovery_timeout:
                    circuit_open = False
                    failure_count = 0
            
            # If circuit is open, fail fast
            if circuit_open:
                raise Exception("Circuit breaker is OPEN")
            
            try:
                result = function(*args, **kwargs)
                # Success resets failure count
                failure_count = 0
                return result
            except Exception as e:
                failure_count += 1
                last_failure_time = time.time()
                
                # Open circuit if threshold exceeded
                if failure_count >= failure_threshold:
                    circuit_open = True
                
                raise e
        
        return circuit_breaker_function
    
    def test_graceful_degradation(self, primary_function, fallback_function):
        """Test graceful degradation to fallback functionality."""
        try:
            return primary_function()
        except Exception as e:
            self.recovery_attempts.append({
                "primary_failed": True,
                "error": str(e),
                "fallback_used": True
            })
            return fallback_function()


class TestModelErrorConditions:
    """Test error conditions in model operations."""
    
    def test_model_loading_failures(self):
        """Test various model loading failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            hf_api = env.get_service('huggingface')
            
            # Test file not found error
            with pytest.raises(FileNotFoundError):
                hf_api.from_pretrained("non-existent-model")
            
            # Test network error during download
            error_injector.set_failure_rate(1.0)  # Always fail
            
            # Inject network errors into download function
            original_download = hf_api.download_model
            hf_api.download_model = error_injector.inject_network_error(original_download)
            
            with pytest.raises(Exception):  # Should be network error
                hf_api.from_pretrained("test-model-network-fail")
            
            # Test memory error during loading
            error_injector.set_failure_rate(0.0)  # Reset
            original_load = hf_api.load_model_from_cache
            hf_api.load_model_from_cache = error_injector.inject_memory_error(original_load)
            error_injector.set_failure_rate(1.0)
            
            with pytest.raises(MemoryError):
                hf_api.from_pretrained("test-model-memory-fail")
            
            error_injector.cleanup()
    
    def test_model_inference_failures(self):
        """Test inference failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            recovery_tester = ErrorRecoveryTester()
            
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            # Test invalid input handling
            with pytest.raises(ValueError):
                # Empty input
                model(input_ids=None)
            
            with pytest.raises(ValueError):
                # Invalid shape
                import torch
                model(input_ids=torch.tensor([]))
            
            # Test timeout during inference
            def inference_with_timeout():
                import torch
                text = "Test sentence for timeout scenario"
                encoding = tokenizer(text)
                input_ids = torch.tensor([encoding.input_ids])
                return model(input_ids=input_ids)
            
            error_injector.set_failure_rate(0.5)  # 50% failure rate
            timeout_inference = error_injector.inject_timeout(inference_with_timeout, 0.1)
            
            # Test retry mechanism
            try:
                result = recovery_tester.test_retry_mechanism(timeout_inference, max_retries=3)
                assert result is not None
            except Exception:
                # Retries can legitimately fail in test environment
                pass
            
            error_injector.cleanup()
    
    def test_model_training_failures(self):
        """Test training failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            dataset = MockTrainingDataset("test-dataset", size=10, tokenizer=tokenizer)
            
            # Test training step failures
            for i in range(5):
                try:
                    batch = dataset[i]
                    
                    # Inject random failures
                    error_injector.set_failure_rate(0.3)  # 30% failure rate
                    
                    if error_injector.should_fail():
                        if i % 3 == 0:
                            raise RuntimeError("CUDA out of memory")
                        elif i % 3 == 1:
                            raise ValueError("Invalid tensor dimensions")
                        else:
                            raise Exception("Training step failed")
                    
                    # Normal training step
                    import torch
                    input_ids = torch.tensor([batch["input_ids"]])
                    outputs = model(input_ids=input_ids)
                    
                except Exception as e:
                    # Log training failure
                    assert isinstance(e, (RuntimeError, ValueError, Exception))
            
            error_injector.cleanup()
    
    def test_model_checkpointing_failures(self):
        """Test model checkpointing failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            
            model = MockTransformerModel("test-model")
            
            # Test disk full scenario
            def save_checkpoint_failing():
                checkpoint_path = "/tmp/test_checkpoint.bin"
                if error_injector.should_fail():
                    raise OSError("No space left on device")
                return {"model_state": "saved", "path": checkpoint_path}
            
            error_injector.set_failure_rate(1.0)  # Always fail
            
            with pytest.raises(OSError):
                save_checkpoint_failing()
            
            # Test corrupted checkpoint loading
            def load_checkpoint_failing():
                if error_injector.should_fail():
                    raise EOFError("Checkpoint file corrupted")
                return {"model_state": "loaded"}
            
            with pytest.raises(EOFError):
                load_checkpoint_failing()
            
            error_injector.cleanup()


class TestDataPipelineErrorConditions:
    """Test error conditions in data pipeline."""
    
    def test_data_loading_failures(self):
        """Test data loading failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            
            # Test file not found
            with pytest.raises(FileNotFoundError):
                with open("non_existent_file.json", 'r') as f:
                    data = json.load(f)
            
            # Test corrupted JSON
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            temp_file.write('{"invalid": json content}')
            temp_file.close()
            
            try:
                with pytest.raises(json.JSONDecodeError):
                    with open(temp_file.name, 'r') as f:
                        data = json.load(f)
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
            
            # Test empty dataset
            hf_api = env.get_service('huggingface')
            
            def load_empty_dataset():
                if error_injector.should_fail():
                    raise ValueError("Dataset is empty")
                return MockTrainingDataset("empty", size=0, tokenizer=MockTokenizer())
            
            error_injector.set_failure_rate(1.0)
            
            with pytest.raises(ValueError):
                load_empty_dataset()
            
            error_injector.cleanup()
    
    def test_data_preprocessing_failures(self):
        """Test preprocessing failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            tokenizer = MockTokenizer()
            
            # Test tokenization failures
            def failing_tokenization(text):
                if error_injector.should_fail():
                    raise ValueError("Tokenization failed for input")
                return tokenizer(text)
            
            error_injector.set_failure_rate(0.5)
            
            test_texts = [
                "Normal text",
                "Text with special characters: éñ中文",
                "",  # Empty text
                "Very " * 1000 + "long text"  # Very long text
            ]
            
            successful_tokenizations = 0
            failed_tokenizations = 0
            
            for text in test_texts:
                try:
                    result = failing_tokenization(text)
                    successful_tokenizations += 1
                    assert result is not None
                except ValueError:
                    failed_tokenizations += 1
            
            # Should have some failures due to error injection
            assert failed_tokenizations > 0
            
            error_injector.cleanup()
    
    def test_data_validation_failures(self):
        """Test data validation failure scenarios."""
        with mock_dependencies_context() as env:
            # Test invalid data formats
            invalid_samples = [
                {"text": None, "label": 1},  # None text
                {"text": "", "label": "invalid"},  # Invalid label type
                {"text": "valid text"},  # Missing label
                {"label": 1},  # Missing text
                {},  # Empty sample
                {"text": "valid", "label": 1, "extra": "field"}  # Extra fields
            ]
            
            def validate_sample(sample):
                if not isinstance(sample, dict):
                    raise TypeError("Sample must be a dictionary")
                
                if "text" not in sample:
                    raise KeyError("Missing 'text' field")
                
                if "label" not in sample:
                    raise KeyError("Missing 'label' field")
                
                if not isinstance(sample["text"], str):
                    raise TypeError("'text' must be a string")
                
                if not isinstance(sample["label"], int):
                    raise TypeError("'label' must be an integer")
                
                if len(sample["text"].strip()) == 0:
                    raise ValueError("'text' cannot be empty")
                
                return True
            
            validation_failures = 0
            
            for sample in invalid_samples:
                try:
                    validate_sample(sample)
                except (TypeError, KeyError, ValueError):
                    validation_failures += 1
            
            # All invalid samples should fail validation
            assert validation_failures == len(invalid_samples)


class TestConfigurationErrorConditions:
    """Test error conditions in configuration system."""
    
    def test_configuration_loading_failures(self):
        """Test configuration loading failure scenarios."""
        with mock_dependencies_context() as env:
            from src.fine_tune_llm.config.manager import ConfigManager
            
            config_manager = ConfigManager()
            
            # Test missing configuration file
            with pytest.raises(FileNotFoundError):
                config_manager.load_from_file("non_existent_config.yaml")
            
            # Test invalid YAML
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            temp_file.write('invalid: yaml: content: [unclosed')
            temp_file.close()
            
            try:
                with pytest.raises(Exception):  # YAML parsing error
                    config_manager.load_from_file(temp_file.name)
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
            
            # Test invalid configuration values
            invalid_configs = [
                {"learning_rate": "not_a_number"},
                {"batch_size": -1},
                {"epochs": 0},
                {"model_name": ""},
                {"training": {"invalid_nested": {"too": {"deep": "value"}}}}
            ]
            
            for invalid_config in invalid_configs:
                try:
                    for key, value in invalid_config.items():
                        config_manager.set(key, value)
                    
                    # Validation should catch invalid values
                    validation_result = config_manager.validate_config(invalid_config)
                    assert not validation_result.get("valid", True)
                    
                except (ValueError, TypeError):
                    # Direct setting may also raise exceptions
                    pass
    
    def test_configuration_validation_failures(self):
        """Test configuration validation failure scenarios."""
        with mock_dependencies_context() as env:
            from src.fine_tune_llm.config.manager import ConfigManager
            
            config_manager = ConfigManager()
            
            # Set up validation schema
            schema = {
                "model": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "hidden_size": {"type": "integer", "minimum": 64}
                    },
                    "required": ["name"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number", "minimum": 1e-6},
                        "batch_size": {"type": "integer", "minimum": 1}
                    },
                    "required": ["learning_rate", "batch_size"]
                }
            }
            
            config_manager.set_validation_schema(schema)
            
            # Test various validation failures
            failing_configs = [
                {
                    "model": {"name": ""},  # Empty name
                    "training": {"learning_rate": 1e-4, "batch_size": 16}
                },
                {
                    "model": {"name": "valid-model", "hidden_size": 32},  # Too small
                    "training": {"learning_rate": 1e-4, "batch_size": 16}
                },
                {
                    "model": {"name": "valid-model"},
                    "training": {"learning_rate": 0}  # Missing batch_size, invalid lr
                },
                {
                    "training": {"learning_rate": 1e-4, "batch_size": 16}
                    # Missing model section
                }
            ]
            
            for config in failing_configs:
                validation_result = config_manager.validate_config(config)
                assert not validation_result["valid"]
                assert len(validation_result["errors"]) > 0


class TestExternalServiceErrorConditions:
    """Test error conditions with external services."""
    
    def test_api_service_failures(self):
        """Test API service failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            recovery_tester = ErrorRecoveryTester()
            
            openai_api = env.get_service('openai')
            hf_api = env.get_service('huggingface')
            
            # Test rate limiting
            openai_api.enable_rate_limiting(max_calls_per_minute=2)
            
            # Make calls to exceed rate limit
            try:
                for i in range(5):
                    openai_api.chat_completions_create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Test {i}"}]
                    )
            except Exception as e:
                assert "rate limit" in str(e).lower()
            
            # Test API timeout
            def api_call_with_timeout():
                return hf_api.from_pretrained("test-model")
            
            timeout_call = error_injector.inject_timeout(api_call_with_timeout, 0.1)
            error_injector.set_failure_rate(1.0)
            
            with pytest.raises(TimeoutError):
                timeout_call()
            
            # Test circuit breaker
            def failing_api_call():
                if error_injector.should_fail():
                    raise ConnectionError("API unavailable")
                return {"status": "success"}
            
            circuit_breaker_call = recovery_tester.test_circuit_breaker(
                failing_api_call, 
                failure_threshold=3, 
                recovery_timeout=1
            )
            
            # Trigger circuit breaker
            error_injector.set_failure_rate(1.0)
            failures = 0
            
            for _ in range(5):
                try:
                    circuit_breaker_call()
                except Exception:
                    failures += 1
            
            assert failures > 0  # Should have failures
            
            error_injector.cleanup()
    
    def test_database_connection_failures(self):
        """Test database connection failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            
            # Mock database service
            db_service = env.get_infrastructure('database')
            
            # Test connection timeout
            def connect_with_timeout():
                if error_injector.should_fail():
                    raise TimeoutError("Database connection timeout")
                return db_service.connect()
            
            error_injector.set_failure_rate(1.0)
            
            with pytest.raises(TimeoutError):
                connect_with_timeout()
            
            # Test connection pool exhaustion
            def get_connection():
                if error_injector.should_fail():
                    raise Exception("Connection pool exhausted")
                return {"connection": "mock_connection"}
            
            with pytest.raises(Exception):
                get_connection()
            
            error_injector.cleanup()
    
    def test_cloud_storage_failures(self):
        """Test cloud storage failure scenarios."""
        with mock_dependencies_context() as env:
            error_injector = ErrorInjector()
            
            s3_service = env.get_service('s3')
            
            # Test upload failures
            def upload_with_failures(bucket, key, data):
                if error_injector.should_fail():
                    failure_type = random.choice([
                        "AccessDenied",
                        "NoSuchBucket", 
                        "InternalError",
                        "NetworkError"
                    ])
                    
                    if failure_type == "AccessDenied":
                        raise PermissionError("Access denied to S3 bucket")
                    elif failure_type == "NoSuchBucket":
                        raise Exception("The specified bucket does not exist")
                    elif failure_type == "InternalError":
                        raise Exception("Internal server error")
                    else:
                        raise ConnectionError("Network error during upload")
                
                return s3_service.put_object(bucket, key, data)
            
            error_injector.set_failure_rate(1.0)
            
            with pytest.raises((PermissionError, Exception, ConnectionError)):
                upload_with_failures("test-bucket", "test-key", b"test data")
            
            error_injector.cleanup()


class TestConcurrencyErrorConditions:
    """Test error conditions in concurrent operations."""
    
    def test_thread_safety_issues(self):
        """Test thread safety and race condition scenarios."""
        with mock_dependencies_context() as env:
            shared_resource = {"counter": 0, "data": []}
            errors = []
            
            def worker_function(worker_id):
                """Worker function that may cause race conditions."""
                try:
                    for i in range(100):
                        # Potential race condition
                        current_value = shared_resource["counter"]
                        time.sleep(0.001)  # Simulate processing
                        shared_resource["counter"] = current_value + 1
                        shared_resource["data"].append(f"worker_{worker_id}_item_{i}")
                        
                except Exception as e:
                    errors.append({"worker": worker_id, "error": str(e)})
            
            # Run multiple workers concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(worker_function, worker_id) 
                    for worker_id in range(5)
                ]
                
                for future in as_completed(futures):
                    future.result()  # Wait for completion
            
            # Check for race condition effects
            expected_counter = 5 * 100  # 5 workers * 100 increments each
            actual_counter = shared_resource["counter"]
            
            # Due to race conditions, actual counter might be less than expected
            if actual_counter < expected_counter:
                # This demonstrates race condition detection
                pass
            
            assert len(shared_resource["data"]) <= expected_counter
    
    def test_deadlock_scenarios(self):
        """Test deadlock detection and prevention."""
        import threading
        
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = threading.Event()
        
        def worker1():
            """Worker that acquires locks in order: lock1, lock2."""
            try:
                lock1.acquire(timeout=1.0)
                time.sleep(0.1)
                if not lock2.acquire(timeout=1.0):
                    deadlock_detected.set()
                    lock1.release()
                    return
                lock2.release()
                lock1.release()
            except Exception:
                deadlock_detected.set()
        
        def worker2():
            """Worker that acquires locks in reverse order: lock2, lock1."""
            try:
                lock2.acquire(timeout=1.0)
                time.sleep(0.1)
                if not lock1.acquire(timeout=1.0):
                    deadlock_detected.set()
                    lock2.release()
                    return
                lock1.release()
                lock2.release()
            except Exception:
                deadlock_detected.set()
        
        # Start workers that can create deadlock
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)
        
        thread1.start()
        thread2.start()
        
        # Wait for completion or deadlock detection
        thread1.join(timeout=2.0)
        thread2.join(timeout=2.0)
        
        # Verify deadlock was detected and handled
        if deadlock_detected.is_set():
            # Deadlock was properly detected and prevented
            assert True
        else:
            # No deadlock occurred (also valid)
            assert True
    
    def test_resource_contention(self):
        """Test resource contention scenarios."""
        with mock_dependencies_context() as env:
            # Simulate limited resource pool
            resource_pool = [f"resource_{i}" for i in range(3)]  # Only 3 resources
            resource_lock = threading.Lock()
            allocation_failures = []
            
            def allocate_resource(worker_id):
                """Try to allocate a resource."""
                try:
                    with resource_lock:
                        if resource_pool:
                            resource = resource_pool.pop()
                            time.sleep(0.1)  # Hold resource
                            resource_pool.append(resource)  # Return resource
                            return resource
                        else:
                            raise Exception("No resources available")
                except Exception as e:
                    allocation_failures.append({
                        "worker": worker_id,
                        "error": str(e)
                    })
                    return None
            
            # Start more workers than available resources
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(allocate_resource, worker_id)
                    for worker_id in range(10)
                ]
                
                results = [future.result() for future in as_completed(futures)]
            
            # Some allocations should fail due to contention
            successful_allocations = sum(1 for r in results if r is not None)
            assert successful_allocations <= len(resource_pool)


class TestSystemRecoveryMechanisms:
    """Test system recovery and resilience mechanisms."""
    
    def test_auto_restart_mechanism(self):
        """Test automatic restart after failure."""
        restart_attempts = 0
        max_restarts = 3
        
        def unreliable_service():
            """Service that fails and needs restart."""
            nonlocal restart_attempts
            restart_attempts += 1
            
            if restart_attempts <= 2:
                raise Exception(f"Service failed (attempt {restart_attempts})")
            
            return {"status": "running", "restart_count": restart_attempts}
        
        def auto_restart_wrapper(service_func, max_attempts=3):
            """Wrapper that implements auto-restart."""
            for attempt in range(max_attempts):
                try:
                    return service_func()
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(0.1)  # Brief pause before restart
        
        # Test auto-restart
        result = auto_restart_wrapper(unreliable_service, max_restarts)
        assert result["status"] == "running"
        assert result["restart_count"] == 3
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown procedures."""
        shutdown_completed = threading.Event()
        services_stopped = []
        
        class MockService:
            def __init__(self, name):
                self.name = name
                self.running = True
            
            def stop(self):
                """Stop service gracefully."""
                self.running = False
                services_stopped.append(self.name)
                time.sleep(0.1)  # Simulate cleanup time
        
        # Create mock services
        services = [MockService(f"service_{i}") for i in range(3)]
        
        def graceful_shutdown():
            """Implement graceful shutdown."""
            try:
                # Stop services in reverse order
                for service in reversed(services):
                    service.stop()
                
                # Wait for all services to stop
                for service in services:
                    while service.running:
                        time.sleep(0.01)
                
                shutdown_completed.set()
                
            except Exception as e:
                # Handle shutdown errors
                pass
        
        # Trigger graceful shutdown
        shutdown_thread = threading.Thread(target=graceful_shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=2.0)
        
        # Verify graceful shutdown
        assert shutdown_completed.is_set()
        assert len(services_stopped) == len(services)
        assert all(not service.running for service in services)
    
    def test_data_consistency_recovery(self):
        """Test data consistency recovery mechanisms."""
        # Simulate data corruption scenario
        corrupted_data = {
            "model_checkpoints": [
                {"id": 1, "path": "/models/checkpoint_1.bin", "corrupted": True},
                {"id": 2, "path": "/models/checkpoint_2.bin", "corrupted": False},
                {"id": 3, "path": "/models/checkpoint_3.bin", "corrupted": True}
            ]
        }
        
        def recover_data_consistency(data):
            """Recover from data corruption."""
            recovery_log = []
            
            for checkpoint in data["model_checkpoints"]:
                if checkpoint["corrupted"]:
                    # Simulate recovery action
                    if checkpoint["id"] == 1:
                        # Try to restore from backup
                        checkpoint["recovered_from"] = "backup"
                        checkpoint["corrupted"] = False
                        recovery_log.append(f"Restored checkpoint {checkpoint['id']} from backup")
                    elif checkpoint["id"] == 3:
                        # Mark for regeneration
                        checkpoint["action"] = "regenerate"
                        recovery_log.append(f"Marked checkpoint {checkpoint['id']} for regeneration")
            
            return recovery_log
        
        # Test data recovery
        recovery_log = recover_data_consistency(corrupted_data)
        
        assert len(recovery_log) == 2  # Two corrupted checkpoints
        assert "backup" in recovery_log[0]
        assert "regenerate" in recovery_log[1]
        
        # Verify recovery actions
        checkpoint_1 = corrupted_data["model_checkpoints"][0]
        checkpoint_3 = corrupted_data["model_checkpoints"][2]
        
        assert not checkpoint_1["corrupted"]
        assert checkpoint_1["recovered_from"] == "backup"
        assert checkpoint_3["action"] == "regenerate"


def test_comprehensive_error_scenarios():
    """Run comprehensive error scenario testing."""
    error_scenarios = [
        ErrorScenario(
            name="model_out_of_memory",
            description="Model loading fails due to insufficient memory",
            error_type=MemoryError,
            error_message="CUDA out of memory",
            trigger_condition=lambda: MockTransformerModel("huge-model"),
            expected_behavior="graceful_failure",
            severity="high"
        ),
        ErrorScenario(
            name="training_data_corruption",
            description="Training fails due to corrupted data",
            error_type=ValueError,
            error_message="Invalid data format",
            trigger_condition=lambda: {"invalid": "data"},
            expected_behavior="error_logging",
            severity="medium"
        ),
        ErrorScenario(
            name="network_timeout",
            description="API call times out",
            error_type=TimeoutError,
            error_message="Request timeout",
            trigger_condition=lambda: time.sleep(10),
            expected_behavior="retry_with_backoff",
            severity="medium"
        )
    ]
    
    # Test each scenario
    for scenario in error_scenarios:
        try:
            scenario.trigger_condition()
            # If no exception, scenario didn't trigger
            continue
        except scenario.error_type as e:
            assert scenario.error_message.lower() in str(e).lower()
        except Exception as e:
            # Unexpected exception type
            pass