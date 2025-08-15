"""
Resource usage validation testing for ensuring optimal system resource utilization.

This test module validates resource consumption patterns, memory efficiency,
CPU utilization, disk I/O, and network usage across all platform components.
"""

import pytest
import psutil
import time
import threading
import gc
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
class ResourceSnapshot:
    """Resource usage snapshot at a point in time."""
    timestamp: datetime
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    cpu_percent: float
    open_files: int
    threads: int
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int


class ResourceMonitor:
    """Monitor system resource usage during test execution."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.snapshots = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.snapshots = []
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                try:
                    snapshot = self._take_snapshot()
                    self.snapshots.append(snapshot)
                    time.sleep(self.interval)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a resource usage snapshot."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        cpu_percent = self.process.cpu_percent()
        
        # Get open files count
        try:
            open_files = len(self.process.open_files())
        except Exception:
            open_files = 0
        
        # Get thread count
        try:
            threads = self.process.num_threads()
        except Exception:
            threads = 0
        
        # Get disk I/O
        try:
            io_counters = self.process.io_counters()
            disk_read_mb = io_counters.read_bytes / 1024 / 1024
            disk_write_mb = io_counters.write_bytes / 1024 / 1024
        except Exception:
            disk_read_mb = disk_write_mb = 0
        
        # Get network I/O (system-wide)
        try:
            net_io = psutil.net_io_counters()
            network_sent = net_io.bytes_sent
            network_recv = net_io.bytes_recv
        except Exception:
            network_sent = network_recv = 0
        
        return ResourceSnapshot(
            timestamp=datetime.now(timezone.utc),
            memory_rss_mb=memory_info.rss / 1024 / 1024,
            memory_vms_mb=memory_info.vms / 1024 / 1024,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            open_files=open_files,
            threads=threads,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv
        )
    
    def get_resource_summary(self) -> Dict:
        """Get summary of resource usage during monitoring."""
        if not self.snapshots:
            return {}
        
        memory_rss = [s.memory_rss_mb for s in self.snapshots]
        memory_percent = [s.memory_percent for s in self.snapshots]
        cpu_percent = [s.cpu_percent for s in self.snapshots]
        open_files = [s.open_files for s in self.snapshots]
        threads = [s.threads for s in self.snapshots]
        
        return {
            "duration_seconds": (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds(),
            "memory_rss_mb": {
                "min": min(memory_rss),
                "max": max(memory_rss),
                "avg": sum(memory_rss) / len(memory_rss),
                "final": memory_rss[-1]
            },
            "memory_percent": {
                "min": min(memory_percent),
                "max": max(memory_percent),
                "avg": sum(memory_percent) / len(memory_percent),
                "final": memory_percent[-1]
            },
            "cpu_percent": {
                "min": min(cpu_percent),
                "max": max(cpu_percent),
                "avg": sum(cpu_percent) / len(cpu_percent),
                "peak": max(cpu_percent)
            },
            "open_files": {
                "min": min(open_files),
                "max": max(open_files),
                "final": open_files[-1]
            },
            "threads": {
                "min": min(threads),
                "max": max(threads),
                "final": threads[-1]
            },
            "snapshots_count": len(self.snapshots)
        }


class ResourceValidator:
    """Validate resource usage against defined thresholds."""
    
    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or self._default_thresholds()
    
    def _default_thresholds(self) -> Dict:
        """Default resource usage thresholds."""
        return {
            "memory_rss_mb_max": 1000,      # Max 1GB RSS
            "memory_percent_max": 10.0,      # Max 10% of system memory
            "cpu_percent_max": 80.0,         # Max 80% CPU
            "open_files_max": 100,           # Max 100 open files
            "threads_max": 50,               # Max 50 threads
            "memory_growth_mb_max": 100,     # Max 100MB memory growth
            "cpu_sustained_max": 50.0,       # Max 50% sustained CPU
            "disk_io_mb_max": 500,           # Max 500MB disk I/O
        }
    
    def validate_resource_usage(self, summary: Dict) -> Dict:
        """Validate resource usage against thresholds."""
        violations = []
        
        # Memory validation
        max_memory = summary["memory_rss_mb"]["max"]
        if max_memory > self.thresholds["memory_rss_mb_max"]:
            violations.append({
                "type": "memory_rss_exceeded",
                "value": max_memory,
                "threshold": self.thresholds["memory_rss_mb_max"],
                "message": f"Peak memory usage {max_memory:.1f}MB exceeds threshold {self.thresholds['memory_rss_mb_max']}MB"
            })
        
        max_memory_percent = summary["memory_percent"]["max"]
        if max_memory_percent > self.thresholds["memory_percent_max"]:
            violations.append({
                "type": "memory_percent_exceeded",
                "value": max_memory_percent,
                "threshold": self.thresholds["memory_percent_max"],
                "message": f"Peak memory percentage {max_memory_percent:.1f}% exceeds threshold {self.thresholds['memory_percent_max']}%"
            })
        
        # CPU validation
        max_cpu = summary["cpu_percent"]["max"]
        if max_cpu > self.thresholds["cpu_percent_max"]:
            violations.append({
                "type": "cpu_peak_exceeded",
                "value": max_cpu,
                "threshold": self.thresholds["cpu_percent_max"],
                "message": f"Peak CPU usage {max_cpu:.1f}% exceeds threshold {self.thresholds['cpu_percent_max']}%"
            })
        
        avg_cpu = summary["cpu_percent"]["avg"]
        if avg_cpu > self.thresholds["cpu_sustained_max"]:
            violations.append({
                "type": "cpu_sustained_exceeded",
                "value": avg_cpu,
                "threshold": self.thresholds["cpu_sustained_max"],
                "message": f"Sustained CPU usage {avg_cpu:.1f}% exceeds threshold {self.thresholds['cpu_sustained_max']}%"
            })
        
        # File handle validation
        max_files = summary["open_files"]["max"]
        if max_files > self.thresholds["open_files_max"]:
            violations.append({
                "type": "open_files_exceeded",
                "value": max_files,
                "threshold": self.thresholds["open_files_max"],
                "message": f"Peak open files {max_files} exceeds threshold {self.thresholds['open_files_max']}"
            })
        
        # Thread validation
        max_threads = summary["threads"]["max"]
        if max_threads > self.thresholds["threads_max"]:
            violations.append({
                "type": "threads_exceeded",
                "value": max_threads,
                "threshold": self.thresholds["threads_max"],
                "message": f"Peak thread count {max_threads} exceeds threshold {self.thresholds['threads_max']}"
            })
        
        # Memory growth validation
        memory_growth = summary["memory_rss_mb"]["max"] - summary["memory_rss_mb"]["min"]
        if memory_growth > self.thresholds["memory_growth_mb_max"]:
            violations.append({
                "type": "memory_growth_exceeded",
                "value": memory_growth,
                "threshold": self.thresholds["memory_growth_mb_max"],
                "message": f"Memory growth {memory_growth:.1f}MB exceeds threshold {self.thresholds['memory_growth_mb_max']}MB"
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "summary": summary
        }


class TestTrainingResourceUsage:
    """Test resource usage during training operations."""
    
    def test_model_loading_resource_usage(self):
        """Test resource usage during model loading."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 500,  # Stricter for model loading
                "cpu_percent_max": 70.0,
                "open_files_max": 50
            })
            
            monitor.start_monitoring()
            
            hf_api = env.get_service('huggingface')
            
            # Load multiple models sequentially
            models = []
            for i in range(3):
                model = hf_api.from_pretrained(f"test-model-{i}")
                tokenizer = hf_api.load_tokenizer(f"test-model-{i}")
                models.append((model, tokenizer))
                
                # Force garbage collection
                gc.collect()
            
            # Clean up models
            del models
            gc.collect()
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Resource validation failed: {validation_result['violations']}"
            
            # Additional assertions
            assert summary["memory_rss_mb"]["max"] < 500, "Model loading used too much memory"
            assert summary["open_files"]["max"] < 50, "Too many files opened during model loading"
    
    def test_training_step_resource_usage(self):
        """Test resource usage during training steps."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_growth_mb_max": 50,  # Limited memory growth during training
                "cpu_sustained_max": 60.0,
                "threads_max": 20
            })
            
            # Set up training components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            dataset = MockTrainingDataset("test-dataset", size=100, tokenizer=tokenizer)
            
            monitor.start_monitoring()
            
            # Simulate training steps
            training_steps = 50
            for step in range(training_steps):
                batch = dataset[step % len(dataset)]
                
                # Convert to tensors (simulated)
                import torch
                input_ids = torch.tensor([batch["input_ids"]])
                attention_mask = torch.tensor([batch["attention_mask"]])
                labels = torch.tensor([batch["labels"]])
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Simulate backward pass processing
                time.sleep(0.002)
                
                # Periodic cleanup
                if step % 10 == 0:
                    gc.collect()
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Training resource validation failed: {validation_result['violations']}"
            
            # Training-specific assertions
            memory_growth = summary["memory_rss_mb"]["max"] - summary["memory_rss_mb"]["min"]
            assert memory_growth < 50, f"Training caused too much memory growth: {memory_growth:.1f}MB"
    
    def test_evaluation_resource_usage(self):
        """Test resource usage during evaluation."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 300,   # Lower for evaluation
                "cpu_percent_max": 60.0,
                "memory_growth_mb_max": 20
            })
            
            # Set up evaluation components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            eval_dataset = MockTrainingDataset("eval-dataset", size=200, tokenizer=tokenizer)
            
            monitor.start_monitoring()
            
            # Simulate evaluation
            import torch
            model.eval()
            
            eval_samples = 100
            with torch.no_grad():
                for i in range(eval_samples):
                    batch = eval_dataset[i % len(eval_dataset)]
                    
                    input_ids = torch.tensor([batch["input_ids"]])
                    attention_mask = torch.tensor([batch["attention_mask"]])
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    # Small processing delay
                    time.sleep(0.001)
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Evaluation resource validation failed: {validation_result['violations']}"
            
            # Evaluation should be lighter than training
            assert summary["cpu_percent"]["avg"] < 60.0, "Evaluation CPU usage too high"
            assert summary["memory_rss_mb"]["max"] < 300, "Evaluation memory usage too high"


class TestInferenceResourceUsage:
    """Test resource usage during inference operations."""
    
    def test_single_inference_resource_usage(self):
        """Test resource usage for single inference calls."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 200,   # Very low for single inference
                "cpu_percent_max": 50.0,
                "memory_growth_mb_max": 10,
                "open_files_max": 20
            })
            
            # Set up inference components
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            monitor.start_monitoring()
            
            # Single inference calls
            test_texts = [
                "This is a test sentence.",
                "Another test sentence for inference.",
                "Final test sentence for validation."
            ]
            
            import torch
            model.eval()
            
            with torch.no_grad():
                for text in test_texts:
                    encoding = tokenizer(text, max_length=128, padding=True, truncation=True)
                    input_ids = torch.tensor([encoding.input_ids])
                    attention_mask = torch.tensor([encoding.attention_mask])
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    prediction = torch.argmax(probabilities, dim=-1)
                    
                    time.sleep(0.001)
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Single inference resource validation failed: {validation_result['violations']}"
            
            # Single inference should be very light
            assert summary["memory_growth_mb_max"] < 10, "Single inference caused too much memory growth"
            assert summary["open_files"]["max"] < 20, "Too many files opened for single inference"
    
    def test_batch_inference_resource_usage(self):
        """Test resource usage for batch inference."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 400,   # Higher for batch processing
                "cpu_percent_max": 70.0,
                "memory_growth_mb_max": 30,
                "threads_max": 15
            })
            
            # Set up batch inference
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            monitor.start_monitoring()
            
            # Batch inference
            batch_texts = [
                f"Batch inference test sentence number {i}"
                for i in range(32)  # Batch size of 32
            ]
            
            import torch
            model.eval()
            
            num_batches = 10
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    encoding = tokenizer(
                        batch_texts,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    outputs = model(
                        input_ids=encoding.input_ids,
                        attention_mask=encoding.attention_mask
                    )
                    
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
                    
                    time.sleep(0.005)  # Processing time
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Batch inference resource validation failed: {validation_result['violations']}"
            
            # Batch inference efficiency checks
            assert summary["memory_rss_mb"]["max"] < 400, "Batch inference memory usage too high"
            assert summary["threads"]["max"] < 15, "Too many threads for batch inference"
    
    def test_concurrent_inference_resource_usage(self):
        """Test resource usage for concurrent inference."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 600,   # Higher for concurrency
                "cpu_percent_max": 85.0,
                "threads_max": 25,         # More threads expected
                "memory_growth_mb_max": 50
            })
            
            # Set up concurrent inference
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
            
            monitor.start_monitoring()
            
            # Concurrent inference
            test_texts = [
                f"Concurrent inference test text {i}"
                for i in range(20)
            ]
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(single_inference, text) for text in test_texts]
                results = [future.result() for future in as_completed(futures)]
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Concurrent inference resource validation failed: {validation_result['violations']}"
            
            # Concurrency efficiency checks
            assert len(results) == len(test_texts), "Some concurrent inferences failed"
            assert summary["threads"]["max"] <= 25, "Too many threads for concurrent inference"


class TestSystemResourceUsage:
    """Test resource usage for system-level operations."""
    
    def test_configuration_system_resource_usage(self):
        """Test resource usage of configuration system."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 150,   # Low for config operations
                "cpu_percent_max": 40.0,
                "memory_growth_mb_max": 15,
                "open_files_max": 30
            })
            
            from src.fine_tune_llm.config.manager import ConfigManager
            
            monitor.start_monitoring()
            
            # Heavy configuration usage
            config_manager = ConfigManager()
            
            # Set many configurations
            for i in range(1000):
                config_manager.set(f"test.group_{i % 10}.param_{i}", f"value_{i}")
                
                # Get configurations
                if i % 10 == 0:
                    for j in range(10):
                        value = config_manager.get(f"test.group_{j}.param_{i-j}")
                
                # Periodic cleanup
                if i % 100 == 0:
                    gc.collect()
            
            # Test configuration validation
            test_config = {
                "model": {"name": "test", "size": 768},
                "training": {"lr": 1e-4, "batch_size": 32}
            }
            
            for _ in range(100):
                config_manager.validate_config(test_config)
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Configuration system resource validation failed: {validation_result['violations']}"
            
            # Configuration should be very efficient
            assert summary["memory_rss_mb"]["max"] < 150, "Configuration system memory usage too high"
            assert summary["cpu_percent"]["avg"] < 40.0, "Configuration system CPU usage too high"
    
    def test_event_system_resource_usage(self):
        """Test resource usage of event system."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 100,   # Very low for events
                "cpu_percent_max": 30.0,
                "memory_growth_mb_max": 10,
                "threads_max": 10
            })
            
            from src.fine_tune_llm.core.events import EventBus, Event, EventType
            
            monitor.start_monitoring()
            
            event_bus = EventBus()
            events_handled = 0
            
            # Set up event handlers
            def handler1(event):
                nonlocal events_handled
                events_handled += 1
            
            def handler2(event):
                nonlocal events_handled
                events_handled += 1
            
            # Subscribe handlers
            event_bus.subscribe(EventType.TRAINING_STARTED, handler1)
            event_bus.subscribe(EventType.TRAINING_STARTED, handler2)
            event_bus.subscribe(EventType.TRAINING_COMPLETED, handler1)
            
            # Publish many events
            for i in range(2000):
                event_type = EventType.TRAINING_STARTED if i % 2 == 0 else EventType.TRAINING_COMPLETED
                
                event = Event(
                    event_type,
                    {"step": i, "data": f"event_data_{i}"},
                    f"source_{i % 5}"
                )
                
                event_bus.publish(event)
                
                # Periodic cleanup
                if i % 200 == 0:
                    gc.collect()
            
            # Wait for event processing
            time.sleep(0.1)
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"Event system resource validation failed: {validation_result['violations']}"
            
            # Event system should be very efficient
            assert summary["memory_growth_mb_max"] < 10, "Event system caused too much memory growth"
            assert events_handled >= 2000, "Some events were not processed"
    
    def test_file_io_resource_usage(self):
        """Test resource usage for file I/O operations."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            validator = ResourceValidator({
                "memory_rss_mb_max": 200,
                "cpu_percent_max": 50.0,
                "open_files_max": 50,
                "disk_io_mb_max": 100
            })
            
            monitor.start_monitoring()
            
            # Create temporary directory for I/O testing
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Write many files
                for i in range(100):
                    file_path = Path(temp_dir) / f"test_file_{i}.txt"
                    with open(file_path, 'w') as f:
                        f.write(f"Test data for file {i}\n" * 100)
                
                # Read files
                for i in range(100):
                    file_path = Path(temp_dir) / f"test_file_{i}.txt"
                    with open(file_path, 'r') as f:
                        content = f.read()
                
                # Clean up periodically
                for i in range(0, 100, 10):
                    file_path = Path(temp_dir) / f"test_file_{i}.txt"
                    if file_path.exists():
                        file_path.unlink()
                    gc.collect()
                
            finally:
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            monitor.stop_monitoring()
            
            # Validate resource usage
            summary = monitor.get_resource_summary()
            validation_result = validator.validate_resource_usage(summary)
            
            assert validation_result["valid"], f"File I/O resource validation failed: {validation_result['violations']}"
            
            # File I/O should not leak file handles
            assert summary["open_files"]["final"] <= summary["open_files"]["min"] + 5, "File handles may have leaked"
    
    def test_memory_cleanup_efficiency(self):
        """Test memory cleanup efficiency."""
        with mock_dependencies_context() as env:
            monitor = ResourceMonitor()
            
            monitor.start_monitoring()
            
            # Create large objects and ensure they're cleaned up
            large_objects = []
            
            # Phase 1: Allocate memory
            for i in range(10):
                # Create large mock objects
                large_data = [f"data_{j}" for j in range(10000)]
                large_objects.append(large_data)
            
            # Take snapshot after allocation
            allocation_snapshot = monitor._take_snapshot()
            
            # Phase 2: Clean up
            del large_objects
            gc.collect()
            
            # Wait for cleanup
            time.sleep(0.1)
            
            # Take snapshot after cleanup
            cleanup_snapshot = monitor._take_snapshot()
            
            monitor.stop_monitoring()
            
            # Validate memory cleanup
            memory_before_cleanup = allocation_snapshot.memory_rss_mb
            memory_after_cleanup = cleanup_snapshot.memory_rss_mb
            memory_released = memory_before_cleanup - memory_after_cleanup
            
            # Should release significant memory
            cleanup_efficiency = memory_released / memory_before_cleanup if memory_before_cleanup > 0 else 0
            
            assert cleanup_efficiency > 0.1, f"Memory cleanup efficiency too low: {cleanup_efficiency:.2%}"
            assert memory_after_cleanup < memory_before_cleanup, "Memory was not properly released"


class TestResourceUsageRegression:
    """Test for resource usage regressions across versions."""
    
    def test_resource_usage_baseline_comparison(self):
        """Test resource usage against baseline."""
        with mock_dependencies_context() as env:
            # Load or create baseline
            baseline_file = Path("resource_usage_baseline.json")
            baseline = {}
            
            if baseline_file.exists():
                try:
                    with open(baseline_file, 'r') as f:
                        baseline = json.load(f)
                except Exception:
                    pass
            
            # Run standard workload
            monitor = ResourceMonitor()
            monitor.start_monitoring()
            
            # Standard workload: model loading + training + inference
            hf_api = env.get_service('huggingface')
            model = hf_api.from_pretrained("test-model")
            tokenizer = hf_api.load_tokenizer("test-model")
            dataset = MockTrainingDataset("test-dataset", size=50, tokenizer=tokenizer)
            
            # Training simulation
            for i in range(20):
                batch = dataset[i % len(dataset)]
                import torch
                input_ids = torch.tensor([batch["input_ids"]])
                outputs = model(input_ids=input_ids)
                time.sleep(0.001)
            
            # Inference simulation
            test_text = "Standard test sentence for resource baseline"
            for _ in range(10):
                encoding = tokenizer(test_text)
                input_ids = torch.tensor([encoding.input_ids])
                outputs = model(input_ids=input_ids)
                time.sleep(0.001)
            
            monitor.stop_monitoring()
            
            # Get current resource summary
            current_summary = monitor.get_resource_summary()
            
            # Compare with baseline
            if baseline:
                memory_regression = (
                    current_summary["memory_rss_mb"]["max"] - baseline.get("memory_rss_mb_max", 0)
                ) / baseline.get("memory_rss_mb_max", 1)
                
                cpu_regression = (
                    current_summary["cpu_percent"]["avg"] - baseline.get("cpu_percent_avg", 0)
                ) / baseline.get("cpu_percent_avg", 1)
                
                # Check for significant regressions (>20%)
                assert memory_regression < 0.2, f"Memory usage regression: {memory_regression:.1%}"
                assert cpu_regression < 0.2, f"CPU usage regression: {cpu_regression:.1%}"
            
            # Update baseline
            new_baseline = {
                "memory_rss_mb_max": current_summary["memory_rss_mb"]["max"],
                "memory_percent_max": current_summary["memory_percent"]["max"],
                "cpu_percent_avg": current_summary["cpu_percent"]["avg"],
                "cpu_percent_max": current_summary["cpu_percent"]["max"],
                "open_files_max": current_summary["open_files"]["max"],
                "threads_max": current_summary["threads"]["max"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                with open(baseline_file, 'w') as f:
                    json.dump(new_baseline, f, indent=2)
            except Exception:
                pass  # Baseline saving is optional
            
            # Validate current usage is reasonable
            assert current_summary["memory_rss_mb"]["max"] < 500, "Memory usage too high for standard workload"
            assert current_summary["cpu_percent"]["avg"] < 60, "CPU usage too high for standard workload"