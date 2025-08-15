"""
Performance regression testing for detecting performance degradations across releases.

This test module validates that performance characteristics remain stable across
code changes and identifies potential performance regressions.
"""

import pytest
import time
import threading
import psutil
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev
from dataclasses import dataclass
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment,
    MockTransformerModel, MockTokenizer, MockTrainingDataset
)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    name: str
    duration: float
    memory_peak_mb: float
    cpu_avg_percent: float
    throughput: float
    timestamp: datetime
    version: str = "current"
    metadata: Dict = None


class PerformanceCollector:
    """Collects performance metrics during test execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval=0.1):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = True
        
        def monitor():
            process = psutil.Process()
            while self.monitoring:
                try:
                    # Memory usage in MB
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.memory_samples.append(memory_mb)
                    
                    # CPU percentage
                    cpu_percent = process.cpu_percent()
                    self.cpu_samples.append(cpu_percent)
                    
                    time.sleep(interval)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def get_metrics(self):
        """Get collected performance metrics."""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            "duration": duration,
            "memory_peak_mb": max(self.memory_samples) if self.memory_samples else 0,
            "memory_avg_mb": mean(self.memory_samples) if self.memory_samples else 0,
            "cpu_avg_percent": mean(self.cpu_samples) if self.cpu_samples else 0,
            "cpu_peak_percent": max(self.cpu_samples) if self.cpu_samples else 0
        }


class PerformanceRegression:
    """Performance regression testing framework."""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        self.baseline_file = baseline_file or Path("performance_baseline.json")
        self.baselines = self.load_baselines()
        self.current_results = {}
        
    def load_baselines(self) -> Dict:
        """Load baseline performance metrics."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_baselines(self):
        """Save current results as new baselines."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
        except Exception as e:
            print(f"Failed to save baselines: {e}")
    
    def record_benchmark(self, benchmark: PerformanceBenchmark):
        """Record a performance benchmark."""
        self.current_results[benchmark.name] = {
            "duration": benchmark.duration,
            "memory_peak_mb": benchmark.memory_peak_mb,
            "cpu_avg_percent": benchmark.cpu_avg_percent,
            "throughput": benchmark.throughput,
            "timestamp": benchmark.timestamp.isoformat(),
            "version": benchmark.version,
            "metadata": benchmark.metadata or {}
        }
    
    def check_regression(self, benchmark: PerformanceBenchmark, 
                        thresholds: Dict[str, float] = None) -> Dict:
        """Check for performance regression against baseline."""
        if thresholds is None:
            thresholds = {
                "duration": 1.2,     # 20% slower
                "memory_peak_mb": 1.3,  # 30% more memory
                "cpu_avg_percent": 1.25,  # 25% more CPU
                "throughput": 0.8    # 20% less throughput
            }
        
        baseline = self.baselines.get(benchmark.name)
        if not baseline:
            # No baseline - record this as the new baseline
            self.baselines[benchmark.name] = self.current_results[benchmark.name]
            return {"status": "baseline_created", "regressions": []}
        
        regressions = []
        
        # Check each metric
        for metric in ["duration", "memory_peak_mb", "cpu_avg_percent"]:
            current_value = getattr(benchmark, metric)
            baseline_value = baseline.get(metric, 0)
            
            if baseline_value > 0:
                ratio = current_value / baseline_value
                threshold = thresholds.get(metric, 1.2)
                
                if ratio > threshold:
                    regressions.append({
                        "metric": metric,
                        "current": current_value,
                        "baseline": baseline_value,
                        "ratio": ratio,
                        "threshold": threshold,
                        "regression_percent": (ratio - 1) * 100
                    })
        
        # Check throughput (inverse - lower is worse)
        if benchmark.throughput > 0 and baseline.get("throughput", 0) > 0:
            throughput_ratio = benchmark.throughput / baseline["throughput"]
            threshold = thresholds.get("throughput", 0.8)
            
            if throughput_ratio < threshold:
                regressions.append({
                    "metric": "throughput",
                    "current": benchmark.throughput,
                    "baseline": baseline["throughput"],
                    "ratio": throughput_ratio,
                    "threshold": threshold,
                    "regression_percent": (1 - throughput_ratio) * 100
                })
        
        status = "regression" if regressions else "pass"
        return {"status": status, "regressions": regressions}


class TestTrainingPerformance:
    """Test training performance characteristics."""
    
    def test_model_loading_performance(self):
        """Test model loading performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            # Test model loading performance
            collector.start_monitoring()
            
            start_time = time.time()
            
            # Load multiple models to simulate real workload
            models_loaded = 0
            hf_api = env.get_service('huggingface')
            
            for i in range(5):
                model = hf_api.from_pretrained(f"test-model-{i}")
                tokenizer = hf_api.load_tokenizer(f"test-model-{i}")
                models_loaded += 1
                
                # Force garbage collection between models
                gc.collect()
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = models_loaded / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="model_loading",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={"models_loaded": models_loaded}
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Performance regression detected: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 10.0, "Model loading too slow"
            assert benchmark.memory_peak_mb < 1000, "Memory usage too high"
            assert throughput > 0.1, "Throughput too low"
    
    def test_training_step_performance(self):
        """Test training step performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            # Set up training components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            dataset = MockTrainingDataset("test-dataset", size=100, tokenizer=tokenizer)
            
            # Test training step performance
            collector.start_monitoring()
            
            training_steps = 50
            start_time = time.time()
            
            for step in range(training_steps):
                # Simulate training step
                batch = dataset[step % len(dataset)]
                
                # Convert to tensors
                import torch
                input_ids = torch.tensor([batch["input_ids"]])
                attention_mask = torch.tensor([batch["attention_mask"]])
                labels = torch.tensor([batch["labels"]])
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Simulate backward pass
                time.sleep(0.001)  # Minimal processing time
                
                # Periodic cleanup
                if step % 10 == 0:
                    gc.collect()
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = training_steps / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="training_steps",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={"training_steps": training_steps, "batch_size": 1}
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Training performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 5.0, "Training too slow"
            assert throughput > 5.0, "Training throughput too low"
    
    def test_evaluation_performance(self):
        """Test evaluation performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            # Set up evaluation components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            test_dataset = MockTrainingDataset("test-dataset", size=200, tokenizer=tokenizer)
            
            # Test evaluation performance
            collector.start_monitoring()
            
            eval_samples = 100
            correct_predictions = 0
            start_time = time.time()
            
            # Simulate evaluation
            import torch
            model.eval()
            
            with torch.no_grad():
                for i in range(eval_samples):
                    batch = test_dataset[i % len(test_dataset)]
                    
                    input_ids = torch.tensor([batch["input_ids"]])
                    attention_mask = torch.tensor([batch["attention_mask"]])
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Simulate prediction accuracy calculation
                    predicted = torch.argmax(outputs.logits, dim=-1)
                    if predicted.item() == batch.get("labels", 0):
                        correct_predictions += 1
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = eval_samples / metrics["duration"] if metrics["duration"] > 0 else 0
            accuracy = correct_predictions / eval_samples
            
            benchmark = PerformanceBenchmark(
                name="evaluation",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "eval_samples": eval_samples,
                    "accuracy": accuracy,
                    "correct_predictions": correct_predictions
                }
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Evaluation performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 3.0, "Evaluation too slow"
            assert throughput > 20.0, "Evaluation throughput too low"


class TestInferencePerformance:
    """Test inference performance characteristics."""
    
    def test_single_inference_performance(self):
        """Test single inference performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            # Set up inference components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            # Test single inference performance
            test_texts = [
                "This is a test sentence for inference.",
                "Another test sentence with different content.",
                "A third example for performance testing."
            ]
            
            collector.start_monitoring()
            
            inferences = 100
            start_time = time.time()
            
            import torch
            model.eval()
            
            with torch.no_grad():
                for i in range(inferences):
                    text = test_texts[i % len(test_texts)]
                    
                    # Tokenize
                    encoding = tokenizer(text, max_length=128, padding=True, truncation=True)
                    input_ids = torch.tensor([encoding.input_ids])
                    attention_mask = torch.tensor([encoding.attention_mask])
                    
                    # Inference
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Process output
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    prediction = torch.argmax(probabilities, dim=-1)
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = inferences / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="single_inference",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={"inferences": inferences, "avg_latency_ms": (metrics["duration"] / inferences) * 1000}
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Inference performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 2.0, "Inference too slow"
            assert throughput > 30.0, "Inference throughput too low"
            
            # Latency should be reasonable
            avg_latency_ms = benchmark.metadata["avg_latency_ms"]
            assert avg_latency_ms < 50.0, "Average latency too high"
    
    def test_batch_inference_performance(self):
        """Test batch inference performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            # Set up inference components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            # Test batch inference performance
            test_texts = [
                f"This is test sentence number {i} for batch inference testing."
                for i in range(32)  # Batch size of 32
            ]
            
            collector.start_monitoring()
            
            num_batches = 20
            start_time = time.time()
            
            import torch
            model.eval()
            
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    # Tokenize batch
                    encoding = tokenizer(
                        test_texts,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # Batch inference
                    outputs = model(
                        input_ids=encoding.input_ids,
                        attention_mask=encoding.attention_mask
                    )
                    
                    # Process batch output
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            total_samples = num_batches * len(test_texts)
            throughput = total_samples / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="batch_inference",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "num_batches": num_batches,
                    "batch_size": len(test_texts),
                    "total_samples": total_samples
                }
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Batch inference performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 5.0, "Batch inference too slow"
            assert throughput > 100.0, "Batch inference throughput too low"
    
    def test_concurrent_inference_performance(self):
        """Test concurrent inference performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            # Set up inference components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            def single_inference(text):
                """Perform single inference."""
                import torch
                with torch.no_grad():
                    encoding = tokenizer(text, max_length=128, padding=True, truncation=True)
                    input_ids = torch.tensor([encoding.input_ids])
                    attention_mask = torch.tensor([encoding.attention_mask])
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    prediction = torch.argmax(probabilities, dim=-1)
                    
                    return prediction.item()
            
            # Test concurrent inference performance
            test_texts = [
                f"Concurrent inference test text number {i}"
                for i in range(50)
            ]
            
            collector.start_monitoring()
            
            start_time = time.time()
            
            # Run concurrent inferences
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(single_inference, text) for text in test_texts]
                results = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = len(test_texts) / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="concurrent_inference",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "concurrent_requests": len(test_texts),
                    "max_workers": 8,
                    "successful_inferences": len(results)
                }
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Concurrent inference performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 3.0, "Concurrent inference too slow"
            assert throughput > 15.0, "Concurrent inference throughput too low"
            assert len(results) == len(test_texts), "Some inferences failed"


class TestSystemPerformance:
    """Test overall system performance characteristics."""
    
    def test_configuration_loading_performance(self):
        """Test configuration loading performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            from src.fine_tune_llm.config.manager import ConfigManager
            
            # Test configuration loading performance
            collector.start_monitoring()
            
            config_operations = 1000
            start_time = time.time()
            
            config_manager = ConfigManager()
            
            # Simulate heavy configuration usage
            for i in range(config_operations):
                # Set configurations
                config_manager.set(f"test.param_{i % 100}", f"value_{i}")
                
                # Get configurations
                value = config_manager.get(f"test.param_{i % 100}")
                
                # Periodic cleanup
                if i % 100 == 0:
                    gc.collect()
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = config_operations / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="configuration_operations",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={"config_operations": config_operations}
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Configuration performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 2.0, "Configuration operations too slow"
            assert throughput > 500.0, "Configuration throughput too low"
    
    def test_event_system_performance(self):
        """Test event system performance regression."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            collector = PerformanceCollector()
            
            from src.fine_tune_llm.core.events import EventBus, Event, EventType
            
            # Test event system performance
            collector.start_monitoring()
            
            event_bus = EventBus()
            events_processed = 0
            
            # Set up event handlers
            def handler1(event):
                nonlocal events_processed
                events_processed += 1
            
            def handler2(event):
                nonlocal events_processed
                events_processed += 1
            
            # Subscribe handlers
            event_bus.subscribe(EventType.TRAINING_STARTED, handler1)
            event_bus.subscribe(EventType.TRAINING_STARTED, handler2)
            event_bus.subscribe(EventType.TRAINING_COMPLETED, handler1)
            
            num_events = 1000
            start_time = time.time()
            
            # Publish many events
            for i in range(num_events):
                event_type = EventType.TRAINING_STARTED if i % 2 == 0 else EventType.TRAINING_COMPLETED
                
                event = Event(
                    event_type,
                    {"step": i, "data": f"test_data_{i}"},
                    f"source_{i % 10}"
                )
                
                event_bus.publish(event)
                
                # Periodic cleanup
                if i % 100 == 0:
                    gc.collect()
            
            # Wait for all events to be processed
            time.sleep(0.1)
            
            end_time = time.time()
            collector.stop_monitoring()
            
            # Calculate performance metrics
            metrics = collector.get_metrics()
            throughput = num_events / metrics["duration"] if metrics["duration"] > 0 else 0
            
            benchmark = PerformanceBenchmark(
                name="event_system",
                duration=metrics["duration"],
                memory_peak_mb=metrics["memory_peak_mb"],
                cpu_avg_percent=metrics["cpu_avg_percent"],
                throughput=throughput,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "events_published": num_events,
                    "events_processed": events_processed,
                    "handlers_registered": 3
                }
            )
            
            performance.record_benchmark(benchmark)
            regression_result = performance.check_regression(benchmark)
            
            # Assert no significant regressions
            assert regression_result["status"] != "regression", \
                f"Event system performance regression: {regression_result['regressions']}"
            
            # Performance should be reasonable
            assert benchmark.duration < 1.0, "Event processing too slow"
            assert throughput > 1000.0, "Event throughput too low"
            assert events_processed >= num_events, "Some events were not processed"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during sustained operations."""
        with mock_dependencies_context() as env:
            performance = PerformanceRegression()
            
            # Monitor memory over sustained operations
            memory_samples = []
            iterations = 50
            
            for i in range(iterations):
                gc.collect()  # Force garbage collection
                
                # Measure memory before operation
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # Perform operation that should not leak memory
                model = MockTransformerModel(f"test-model-{i}")
                tokenizer = MockTokenizer()
                dataset = MockTrainingDataset(f"test-dataset-{i}", size=10, tokenizer=tokenizer)
                
                # Simulate some work
                for j in range(10):
                    batch = dataset[j]
                    encoding = tokenizer(f"test text {j}")
                
                # Explicitly delete objects
                del model, tokenizer, dataset
                
                # Force garbage collection
                gc.collect()
                
                # Measure memory after operation
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_after)
                
                time.sleep(0.01)  # Small delay
            
            # Analyze memory trend
            if len(memory_samples) >= 10:
                # Check if memory is consistently growing
                early_avg = mean(memory_samples[:10])
                late_avg = mean(memory_samples[-10:])
                memory_growth = late_avg - early_avg
                
                benchmark = PerformanceBenchmark(
                    name="memory_leak_test",
                    duration=iterations * 0.01,  # Approximate duration
                    memory_peak_mb=max(memory_samples),
                    cpu_avg_percent=0,  # Not measuring CPU
                    throughput=iterations / (iterations * 0.01),
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "iterations": iterations,
                        "memory_growth_mb": memory_growth,
                        "early_avg_mb": early_avg,
                        "late_avg_mb": late_avg
                    }
                )
                
                performance.record_benchmark(benchmark)
                
                # Assert no significant memory leaks
                assert memory_growth < 50.0, f"Potential memory leak detected: {memory_growth}MB growth"
                assert max(memory_samples) < 500.0, "Peak memory usage too high"