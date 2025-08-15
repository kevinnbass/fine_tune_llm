"""
Load and performance testing suite for comprehensive system stress testing.

This test module implements systematic load testing, performance benchmarking,
scalability validation, and stress testing across all platform components.
"""

import pytest
import asyncio
import time
import threading
import multiprocessing
import psutil
import gc
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple
import statistics
import json
import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment,
    MockTransformerModel, MockTokenizer, MockTrainingDataset
)


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    name: str
    duration_seconds: int
    concurrent_users: int
    requests_per_second: int
    ramp_up_time: int
    ramp_down_time: int
    target_operation: str
    payload_size: int = 1024
    success_rate_threshold: float = 0.95
    response_time_threshold: float = 5.0


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during load testing."""
    timestamp: datetime
    operation: str
    response_time: float
    success: bool
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    throughput: float = 0.0


class LoadGenerator:
    """Generate synthetic load for testing system capacity."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = []
        self.active_threads = 0
        self.stop_event = threading.Event()
        self.results_lock = threading.Lock()
        
    def start_load_test(self, target_function: Callable):
        """Start load test with specified configuration."""
        print(f"Starting load test: {self.config.name}")
        print(f"Duration: {self.config.duration_seconds}s, Concurrent users: {self.config.concurrent_users}")
        
        start_time = time.time()
        
        # Calculate timing parameters
        total_requests = self.config.requests_per_second * self.config.duration_seconds
        request_interval = 1.0 / self.config.requests_per_second if self.config.requests_per_second > 0 else 0.1
        
        def worker_thread(worker_id: int):
            """Worker thread that generates load."""
            thread_start = time.time()
            requests_sent = 0
            
            while not self.stop_event.is_set():
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if test duration exceeded
                if elapsed > self.config.duration_seconds:
                    break
                
                # Ramp-up logic
                if elapsed < self.config.ramp_up_time:
                    ramp_factor = elapsed / self.config.ramp_up_time
                    actual_interval = request_interval / ramp_factor if ramp_factor > 0 else request_interval * 10
                # Ramp-down logic
                elif elapsed > (self.config.duration_seconds - self.config.ramp_down_time):
                    remaining = self.config.duration_seconds - elapsed
                    ramp_factor = remaining / self.config.ramp_down_time
                    actual_interval = request_interval / ramp_factor if ramp_factor > 0 else request_interval * 10
                else:
                    actual_interval = request_interval
                
                # Execute target function and collect metrics
                metric = self._execute_and_measure(target_function, worker_id, requests_sent)
                
                with self.results_lock:
                    self.metrics.append(metric)
                
                requests_sent += 1
                
                # Wait for next request
                time.sleep(actual_interval)
        
        # Start worker threads
        threads = []
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = [
                executor.submit(worker_thread, worker_id)
                for worker_id in range(self.config.concurrent_users)
            ]
            
            # Wait for test duration
            time.sleep(self.config.duration_seconds + 1)
            self.stop_event.set()
            
            # Wait for all threads to complete
            for future in as_completed(futures, timeout=10):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker thread error: {e}")
        
        print(f"Load test completed: {len(self.metrics)} requests processed")
        return self._calculate_summary()
    
    def _execute_and_measure(self, target_function: Callable, worker_id: int, request_id: int) -> PerformanceMetrics:
        """Execute target function and measure performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        success = False
        error_message = None
        
        try:
            # Execute target function
            result = target_function(worker_id, request_id)
            success = True
        except Exception as e:
            error_message = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()
        
        response_time = end_time - start_time
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            operation=self.config.target_operation,
            response_time=response_time,
            success=success,
            error_message=error_message,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=(start_cpu + end_cpu) / 2,
            throughput=1.0 / response_time if response_time > 0 else 0
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _calculate_summary(self) -> Dict:
        """Calculate performance summary from collected metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        successful_requests = [m for m in self.metrics if m.success]
        failed_requests = [m for m in self.metrics if not m.success]
        
        response_times = [m.response_time for m in successful_requests]
        throughputs = [m.throughput for m in successful_requests]
        
        success_rate = len(successful_requests) / len(self.metrics)
        
        summary = {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": success_rate,
            "duration_seconds": self.config.duration_seconds,
            "concurrent_users": self.config.concurrent_users,
            "requests_per_second": len(self.metrics) / self.config.duration_seconds,
            "response_time": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "avg": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "p95": self._percentile(response_times, 95) if len(response_times) > 0 else 0,
                "p99": self._percentile(response_times, 99) if len(response_times) > 0 else 0
            },
            "throughput": {
                "avg": statistics.mean(throughputs) if throughputs else 0,
                "peak": max(throughputs) if throughputs else 0
            },
            "error_analysis": self._analyze_errors(failed_requests)
        }
        
        return summary
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _analyze_errors(self, failed_requests: List[PerformanceMetrics]) -> Dict:
        """Analyze error patterns in failed requests."""
        if not failed_requests:
            return {"error_count": 0}
        
        error_types = {}
        for request in failed_requests:
            error_type = type(Exception(request.error_message)).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "error_count": len(failed_requests),
            "error_types": error_types,
            "error_rate": len(failed_requests) / (len(failed_requests) + len([m for m in self.metrics if m.success]))
        }


class PerformanceBenchmark:
    """Benchmark system performance under various conditions."""
    
    def __init__(self):
        self.baseline_metrics = None
        self.benchmark_results = []
        
    def establish_baseline(self, target_function: Callable, iterations: int = 100):
        """Establish performance baseline."""
        print(f"Establishing performance baseline with {iterations} iterations...")
        
        metrics = []
        for i in range(iterations):
            start_time = time.time()
            try:
                target_function(0, i)
                response_time = time.time() - start_time
                metrics.append(response_time)
            except Exception:
                pass
        
        if metrics:
            self.baseline_metrics = {
                "iterations": len(metrics),
                "avg_response_time": statistics.mean(metrics),
                "median_response_time": statistics.median(metrics),
                "p95_response_time": self._percentile(metrics, 95),
                "min_response_time": min(metrics),
                "max_response_time": max(metrics)
            }
            print(f"Baseline established: {self.baseline_metrics['avg_response_time']:.3f}s avg response time")
        
        return self.baseline_metrics
    
    def run_performance_test(self, test_name: str, target_function: Callable, 
                           concurrent_users: int, duration: int) -> Dict:
        """Run performance test and compare against baseline."""
        config = LoadTestConfig(
            name=test_name,
            duration_seconds=duration,
            concurrent_users=concurrent_users,
            requests_per_second=10,  # Moderate rate
            ramp_up_time=5,
            ramp_down_time=5,
            target_operation=test_name
        )
        
        load_generator = LoadGenerator(config)
        results = load_generator.start_load_test(target_function)
        
        # Compare against baseline
        if self.baseline_metrics:
            baseline_avg = self.baseline_metrics["avg_response_time"]
            current_avg = results["response_time"]["avg"]
            
            performance_ratio = current_avg / baseline_avg if baseline_avg > 0 else float('inf')
            
            results["baseline_comparison"] = {
                "baseline_avg_response_time": baseline_avg,
                "current_avg_response_time": current_avg,
                "performance_ratio": performance_ratio,
                "performance_degradation": performance_ratio > 1.2,  # 20% threshold
                "performance_improvement": performance_ratio < 0.8   # 20% improvement
            }
        
        self.benchmark_results.append({
            "test_name": test_name,
            "timestamp": datetime.now(timezone.utc),
            "results": results
        })
        
        return results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class StressTester:
    """Perform stress testing to find system breaking points."""
    
    def __init__(self):
        self.stress_results = []
        
    def find_breaking_point(self, target_function: Callable, 
                          max_concurrent_users: int = 100,
                          step_size: int = 10,
                          test_duration: int = 30) -> Dict:
        """Find system breaking point by gradually increasing load."""
        print(f"Finding breaking point: testing up to {max_concurrent_users} concurrent users")
        
        breaking_point = None
        last_successful_load = 0
        
        for concurrent_users in range(step_size, max_concurrent_users + 1, step_size):
            print(f"Testing with {concurrent_users} concurrent users...")
            
            config = LoadTestConfig(
                name=f"stress_test_{concurrent_users}_users",
                duration_seconds=test_duration,
                concurrent_users=concurrent_users,
                requests_per_second=concurrent_users * 2,  # 2 RPS per user
                ramp_up_time=5,
                ramp_down_time=5,
                target_operation="stress_test"
            )
            
            load_generator = LoadGenerator(config)
            results = load_generator.start_load_test(target_function)
            
            success_rate = results["success_rate"]
            avg_response_time = results["response_time"]["avg"]
            
            self.stress_results.append({
                "concurrent_users": concurrent_users,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "results": results
            })
            
            # Check if system is breaking down
            if success_rate < 0.90 or avg_response_time > 10.0:  # 90% success rate or 10s response time
                breaking_point = concurrent_users
                print(f"Breaking point found at {concurrent_users} concurrent users")
                break
            else:
                last_successful_load = concurrent_users
        
        return {
            "breaking_point": breaking_point,
            "last_successful_load": last_successful_load,
            "max_tested_load": max_concurrent_users,
            "stress_results": self.stress_results
        }
    
    def memory_stress_test(self, target_function: Callable, 
                          memory_limit_mb: int = 1000) -> Dict:
        """Test system behavior under memory pressure."""
        print(f"Memory stress test: limit {memory_limit_mb}MB")
        
        initial_memory = self._get_memory_usage()
        memory_samples = []
        operations_completed = 0
        memory_exceeded = False
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < 60:  # 1-minute test
                current_memory = self._get_memory_usage()
                memory_samples.append(current_memory)
                
                if current_memory - initial_memory > memory_limit_mb:
                    memory_exceeded = True
                    print(f"Memory limit exceeded: {current_memory - initial_memory:.1f}MB")
                    break
                
                # Execute operation
                try:
                    target_function(0, operations_completed)
                    operations_completed += 1
                except MemoryError:
                    print("MemoryError encountered during stress test")
                    break
                except Exception:
                    # Other errors are acceptable during stress testing
                    pass
                
                # Small delay
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("Memory stress test interrupted")
        
        final_memory = self._get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": memory_growth,
            "peak_memory_mb": max(memory_samples) if memory_samples else final_memory,
            "memory_limit_mb": memory_limit_mb,
            "memory_exceeded": memory_exceeded,
            "operations_completed": operations_completed,
            "duration_seconds": time.time() - start_time
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class TestModelLoadTesting:
    """Load testing for model operations."""
    
    def test_model_inference_load(self):
        """Test model inference under load."""
        with mock_dependencies_context() as env:
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            def inference_operation(worker_id: int, request_id: int):
                """Single inference operation."""
                test_text = f"Load test sentence {request_id} from worker {worker_id}"
                
                # Tokenize
                encoding = tokenizer(test_text, max_length=128, padding=True, truncation=True)
                
                # Inference
                import torch
                input_ids = torch.tensor([encoding.input_ids])
                attention_mask = torch.tensor([encoding.attention_mask])
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)
                
                return prediction.item()
            
            # Establish baseline
            benchmark = PerformanceBenchmark()
            baseline = benchmark.establish_baseline(inference_operation, iterations=50)
            assert baseline is not None
            
            # Run load tests with increasing concurrency
            load_configs = [
                (5, 30),    # 5 users, 30 seconds
                (10, 30),   # 10 users, 30 seconds
                (20, 30),   # 20 users, 30 seconds
            ]
            
            for concurrent_users, duration in load_configs:
                results = benchmark.run_performance_test(
                    f"model_inference_{concurrent_users}_users",
                    inference_operation,
                    concurrent_users,
                    duration
                )
                
                # Validate performance
                assert results["success_rate"] > 0.90, f"Success rate too low: {results['success_rate']}"
                assert results["response_time"]["avg"] < 5.0, f"Average response time too high: {results['response_time']['avg']}"
                assert results["response_time"]["p95"] < 10.0, f"P95 response time too high: {results['response_time']['p95']}"
    
    def test_model_training_load(self):
        """Test model training under load."""
        with mock_dependencies_context() as env:
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            dataset = MockTrainingDataset("load-test-dataset", size=1000, tokenizer=tokenizer)
            
            def training_operation(worker_id: int, request_id: int):
                """Single training step operation."""
                batch_idx = request_id % len(dataset)
                batch = dataset[batch_idx]
                
                # Convert to tensors
                import torch
                input_ids = torch.tensor([batch["input_ids"]])
                attention_mask = torch.tensor([batch["attention_mask"]])
                labels = torch.tensor([batch["labels"]])
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Simulate backward pass
                time.sleep(0.001)
                
                return loss.item()
            
            # Load test configuration
            config = LoadTestConfig(
                name="training_load_test",
                duration_seconds=60,
                concurrent_users=8,  # Fewer for training
                requests_per_second=16,  # 2 per user
                ramp_up_time=10,
                ramp_down_time=10,
                target_operation="training_step"
            )
            
            load_generator = LoadGenerator(config)
            results = load_generator.start_load_test(training_operation)
            
            # Validate training load performance
            assert results["success_rate"] > 0.85, f"Training success rate too low: {results['success_rate']}"
            assert results["response_time"]["avg"] < 2.0, f"Training step too slow: {results['response_time']['avg']}"
            
            # Check for memory growth (training can use more memory)
            memory_metrics = [m.memory_usage_mb for m in load_generator.metrics if m.success]
            if memory_metrics:
                avg_memory_growth = statistics.mean(memory_metrics)
                assert avg_memory_growth < 10.0, f"Excessive memory growth during training: {avg_memory_growth:.2f}MB"


class TestDataPipelineLoadTesting:
    """Load testing for data pipeline operations."""
    
    def test_data_loading_load(self):
        """Test data loading under load."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            def data_loading_operation(worker_id: int, request_id: int):
                """Single data loading operation."""
                dataset_name = f"test-dataset-{request_id % 5}"  # Cycle through 5 datasets
                
                # Load dataset
                dataset = hf_api.load_dataset(dataset_name, split="train[:100]")
                
                # Process first item
                if len(dataset) > 0:
                    item = dataset[0]
                    return len(str(item))
                
                return 0
            
            # Stress test data loading
            stress_tester = StressTester()
            stress_results = stress_tester.find_breaking_point(
                data_loading_operation,
                max_concurrent_users=50,
                step_size=5,
                test_duration=20
            )
            
            # Should handle at least 10 concurrent data loading operations
            assert stress_results["last_successful_load"] >= 10, \
                f"Data loading broke too early: {stress_results['last_successful_load']} users"
    
    def test_data_preprocessing_load(self):
        """Test data preprocessing under load."""
        with mock_dependencies_context() as env:
            tokenizer = MockTokenizer()
            
            # Create test texts of varying lengths
            test_texts = [
                "Short text",
                "Medium length text with some additional content for processing",
                "Very long text " * 50 + " that requires more processing time and memory",
                "",  # Edge case: empty text
                "Text with special characters: éñüà¿¡@#$%^&*()",
            ]
            
            def preprocessing_operation(worker_id: int, request_id: int):
                """Single preprocessing operation."""
                text = test_texts[request_id % len(test_texts)]
                
                # Tokenize
                encoding = tokenizer(
                    text, 
                    max_length=512, 
                    padding=True, 
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Additional preprocessing
                processed_text = text.lower().strip()
                word_count = len(processed_text.split())
                
                return {
                    "input_ids_length": len(encoding.input_ids[0]),
                    "word_count": word_count,
                    "processed": True
                }
            
            # High-throughput load test
            config = LoadTestConfig(
                name="preprocessing_throughput_test",
                duration_seconds=45,
                concurrent_users=20,
                requests_per_second=100,  # High throughput
                ramp_up_time=5,
                ramp_down_time=5,
                target_operation="text_preprocessing"
            )
            
            load_generator = LoadGenerator(config)
            results = load_generator.start_load_test(preprocessing_operation)
            
            # Validate preprocessing performance
            assert results["success_rate"] > 0.95, f"Preprocessing success rate too low: {results['success_rate']}"
            assert results["response_time"]["avg"] < 0.5, f"Preprocessing too slow: {results['response_time']['avg']}"
            assert results["throughput"]["avg"] > 50, f"Preprocessing throughput too low: {results['throughput']['avg']}"


class TestSystemLoadTesting:
    """Load testing for system-level operations."""
    
    def test_configuration_system_load(self):
        """Test configuration system under load."""
        with mock_dependencies_context() as env:
            from src.fine_tune_llm.config.manager import ConfigManager
            
            def config_operation(worker_id: int, request_id: int):
                """Single configuration operation."""
                config_manager = ConfigManager()
                
                # Set operation
                key = f"load_test.worker_{worker_id}.param_{request_id % 100}"
                value = f"value_{request_id}"
                config_manager.set(key, value)
                
                # Get operation
                retrieved_value = config_manager.get(key)
                
                # Validation operation
                test_config = {
                    "model": {"name": "test-model", "size": 768},
                    "training": {"lr": 1e-4, "batch_size": 32}
                }
                validation_result = config_manager.validate_config(test_config)
                
                return retrieved_value == value and validation_result.get("valid", False)
            
            # High-concurrency configuration test
            config = LoadTestConfig(
                name="config_system_load",
                duration_seconds=30,
                concurrent_users=50,  # High concurrency
                requests_per_second=200,  # Very high throughput
                ramp_up_time=5,
                ramp_down_time=5,
                target_operation="config_operations"
            )
            
            load_generator = LoadGenerator(config)
            results = load_generator.start_load_test(config_operation)
            
            # Configuration system should handle high load efficiently
            assert results["success_rate"] > 0.98, f"Config system success rate too low: {results['success_rate']}"
            assert results["response_time"]["avg"] < 0.1, f"Config operations too slow: {results['response_time']['avg']}"
            assert results["response_time"]["p99"] < 0.5, f"Config P99 response time too high: {results['response_time']['p99']}"
    
    def test_event_system_load(self):
        """Test event system under load."""
        with mock_dependencies_context() as env:
            from src.fine_tune_llm.core.events import EventBus, Event, EventType
            
            event_bus = EventBus()
            events_processed = []
            
            def event_handler(event):
                """Event handler that processes events."""
                events_processed.append({
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "worker_id": event.data.get("worker_id"),
                    "request_id": event.data.get("request_id")
                })
            
            # Subscribe handlers
            event_bus.subscribe(EventType.TRAINING_STARTED, event_handler)
            event_bus.subscribe(EventType.TRAINING_COMPLETED, event_handler)
            event_bus.subscribe(EventType.MODEL_LOADED, event_handler)
            
            def event_operation(worker_id: int, request_id: int):
                """Single event operation."""
                event_types = [
                    EventType.TRAINING_STARTED,
                    EventType.TRAINING_COMPLETED,
                    EventType.MODEL_LOADED
                ]
                
                event_type = event_types[request_id % len(event_types)]
                
                event = Event(
                    event_type,
                    {
                        "worker_id": worker_id,
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "data": f"load_test_data_{request_id}"
                    },
                    f"load_tester_{worker_id}"
                )
                
                event_bus.publish(event)
                return True
            
            # Event system stress test
            stress_tester = StressTester()
            stress_results = stress_tester.find_breaking_point(
                event_operation,
                max_concurrent_users=100,
                step_size=10,
                test_duration=15
            )
            
            # Allow time for event processing
            time.sleep(1.0)
            
            # Event system should handle very high load
            assert stress_results["last_successful_load"] >= 50, \
                f"Event system broke too early: {stress_results['last_successful_load']} users"
            
            # Verify events were processed
            assert len(events_processed) > 0, "No events were processed during load test"
    
    def test_memory_stress_under_load(self):
        """Test system behavior under memory stress."""
        with mock_dependencies_context() as env:
            def memory_intensive_operation(worker_id: int, request_id: int):
                """Memory-intensive operation for stress testing."""
                # Create large data structures
                large_list = [f"data_item_{i}_{worker_id}_{request_id}" for i in range(1000)]
                large_dict = {f"key_{i}": f"value_{i}_{worker_id}_{request_id}" for i in range(500)}
                
                # Process data
                processed_list = [item.upper() for item in large_list]
                processed_dict = {k: v.upper() for k, v in large_dict.items()}
                
                # Force garbage collection periodically
                if request_id % 10 == 0:
                    gc.collect()
                
                return len(processed_list) + len(processed_dict)
            
            # Memory stress test
            stress_tester = StressTester()
            memory_results = stress_tester.memory_stress_test(
                memory_intensive_operation,
                memory_limit_mb=500  # 500MB limit
            )
            
            # Validate memory stress results
            assert memory_results["memory_growth_mb"] < 600, \
                f"Excessive memory growth: {memory_results['memory_growth_mb']:.1f}MB"
            
            assert memory_results["operations_completed"] > 0, \
                "No operations completed during memory stress test"
            
            # Memory should not exceed limit by too much
            if memory_results["memory_exceeded"]:
                assert memory_results["memory_growth_mb"] < memory_results["memory_limit_mb"] * 1.5, \
                    "Memory usage exceeded limit by too much"


class TestScalabilityValidation:
    """Validate system scalability characteristics."""
    
    def test_linear_scalability(self):
        """Test if system scales linearly with increased load."""
        with mock_dependencies_context() as env:
            def simple_operation(worker_id: int, request_id: int):
                """Simple operation for scalability testing."""
                # Simulate lightweight processing
                result = sum(range(100))
                time.sleep(0.001)  # 1ms processing time
                return result
            
            # Test different load levels
            load_levels = [1, 5, 10, 20]
            scalability_results = []
            
            benchmark = PerformanceBenchmark()
            
            for concurrent_users in load_levels:
                results = benchmark.run_performance_test(
                    f"scalability_test_{concurrent_users}_users",
                    simple_operation,
                    concurrent_users,
                    duration=20
                )
                
                scalability_results.append({
                    "concurrent_users": concurrent_users,
                    "throughput": results["throughput"]["avg"],
                    "avg_response_time": results["response_time"]["avg"],
                    "success_rate": results["success_rate"]
                })
            
            # Analyze scalability
            throughputs = [r["throughput"] for r in scalability_results]
            users = [r["concurrent_users"] for r in scalability_results]
            
            # Calculate scalability coefficient
            if len(throughputs) >= 2:
                # Linear regression to check if throughput scales with users
                n = len(users)
                sum_x = sum(users)
                sum_y = sum(throughputs)
                sum_xy = sum(x * y for x, y in zip(users, throughputs))
                sum_x2 = sum(x * x for x in users)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Positive slope indicates good scalability
                assert slope > 0, f"Negative scalability slope: {slope}"
                
                # Check that success rate remains high
                for result in scalability_results:
                    assert result["success_rate"] > 0.90, \
                        f"Success rate degraded at {result['concurrent_users']} users: {result['success_rate']}"
    
    def test_sustained_load_stability(self):
        """Test system stability under sustained load."""
        with mock_dependencies_context() as env:
            def sustained_operation(worker_id: int, request_id: int):
                """Operation for sustained load testing."""
                # Simulate realistic workload
                data = {"worker": worker_id, "request": request_id, "timestamp": time.time()}
                processed = json.dumps(data)
                parsed = json.loads(processed)
                return parsed["request"]
            
            # Long-duration sustained load test
            config = LoadTestConfig(
                name="sustained_load_test",
                duration_seconds=120,  # 2 minutes
                concurrent_users=15,
                requests_per_second=45,  # 3 RPS per user
                ramp_up_time=10,
                ramp_down_time=10,
                target_operation="sustained_operation"
            )
            
            load_generator = LoadGenerator(config)
            results = load_generator.start_load_test(sustained_operation)
            
            # Analyze stability over time
            metrics = load_generator.metrics
            successful_metrics = [m for m in metrics if m.success]
            
            # Split into time windows to check stability
            if successful_metrics:
                start_time = successful_metrics[0].timestamp
                window_size = timedelta(seconds=30)
                
                windows = []
                current_window = []
                
                for metric in successful_metrics:
                    if metric.timestamp - start_time > len(windows) * window_size:
                        if current_window:
                            windows.append(current_window)
                        current_window = [metric]
                    else:
                        current_window.append(metric)
                
                if current_window:
                    windows.append(current_window)
                
                # Check that performance is stable across windows
                if len(windows) >= 2:
                    window_avg_times = [
                        statistics.mean([m.response_time for m in window])
                        for window in windows
                    ]
                    
                    # Performance should not degrade significantly over time
                    first_window_avg = window_avg_times[0]
                    last_window_avg = window_avg_times[-1]
                    
                    performance_degradation = (last_window_avg - first_window_avg) / first_window_avg
                    
                    assert performance_degradation < 0.5, \
                        f"Performance degraded significantly over time: {performance_degradation:.2%}"
            
            # Overall stability checks
            assert results["success_rate"] > 0.95, \
                f"Sustained load success rate too low: {results['success_rate']}"
            
            assert results["response_time"]["avg"] < 1.0, \
                f"Average response time too high under sustained load: {results['response_time']['avg']}"