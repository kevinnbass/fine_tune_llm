"""
Performance Metrics Collection System.

This module provides comprehensive performance metrics collection with
advanced profiling, benchmarking, and analysis capabilities.
"""

import logging
import time
import threading
import functools
import gc
import tracemalloc
import cProfile
import pstats
import io
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import numpy as np
import psutil
import weakref
from contextlib import contextmanager

from ...core.events import EventBus, Event, EventType
from ...utils.logging import get_centralized_logger

logger = get_centralized_logger().get_logger("performance_metrics")


class MetricCategory(Enum):
    """Performance metric categories."""
    TIMING = "timing"
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    GPU = "gpu"
    CACHE = "cache"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


class AggregationType(Enum):
    """Metric aggregation types."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    name: str
    value: Union[int, float]
    category: MetricCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    component: str = ""
    operation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata,
            'component': self.component,
            'operation': self.operation
        }


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    component: str
    operation: str
    duration: float
    cpu_time: float
    memory_used: int
    memory_peak: int
    call_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    call_stack: Optional[str] = None
    profiling_data: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'operation': self.operation,
            'duration': self.duration,
            'cpu_time': self.cpu_time,
            'memory_used': self.memory_used,
            'memory_peak': self.memory_peak,
            'call_count': self.call_count,
            'timestamp': self.timestamp.isoformat(),
            'call_stack': self.call_stack,
            'profiling_data': self.profiling_data
        }


@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float
    memory_used: int
    cpu_percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'std_dev': self.std_dev,
            'throughput': self.throughput,
            'memory_used': self.memory_used,
            'cpu_percent': self.cpu_percent,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class MetricAggregator:
    """Aggregates metrics over time windows."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize metric aggregator."""
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.RLock()
    
    def add_metric(self, metric: PerformanceMetric):
        """Add metric to aggregation."""
        with self._lock:
            key = f"{metric.component}.{metric.name}"
            self.metrics[key].append(metric)
    
    def get_aggregated_stats(self, metric_key: str, aggregation: AggregationType) -> Optional[float]:
        """Get aggregated statistics for metric."""
        with self._lock:
            values = [m.value for m in self.metrics.get(metric_key, [])]
            
            if not values:
                return None
            
            if aggregation == AggregationType.SUM:
                return sum(values)
            elif aggregation == AggregationType.AVERAGE:
                return sum(values) / len(values)
            elif aggregation == AggregationType.MIN:
                return min(values)
            elif aggregation == AggregationType.MAX:
                return max(values)
            elif aggregation == AggregationType.COUNT:
                return len(values)
            elif aggregation == AggregationType.MEDIAN:
                return float(np.median(values))
            elif aggregation == AggregationType.P95:
                return float(np.percentile(values, 95))
            elif aggregation == AggregationType.P99:
                return float(np.percentile(values, 99))
            elif aggregation == AggregationType.STDDEV:
                return float(np.std(values))
            
            return None
    
    def get_all_stats(self, metric_key: str) -> Dict[str, float]:
        """Get all aggregated statistics for metric."""
        stats = {}
        for agg_type in AggregationType:
            value = self.get_aggregated_stats(metric_key, agg_type)
            if value is not None:
                stats[agg_type.value] = value
        return stats


class PerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.profile_history: deque = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    def start_profiling(self, component: str, operation: str) -> str:
        """Start profiling operation."""
        profile_id = f"{component}_{operation}_{int(time.time())}"
        
        with self._lock:
            # Start memory tracking
            tracemalloc.start()
            
            # Start CPU profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            self.active_profiles[profile_id] = {
                'component': component,
                'operation': operation,
                'start_time': time.time(),
                'start_cpu_time': time.process_time(),
                'start_memory': tracemalloc.get_traced_memory()[0],
                'profiler': profiler,
                'call_count': 0
            }
        
        return profile_id
    
    def stop_profiling(self, profile_id: str) -> Optional[PerformanceProfile]:
        """Stop profiling and get results."""
        with self._lock:
            if profile_id not in self.active_profiles:
                return None
            
            profile_data = self.active_profiles[profile_id]
            
            # Stop CPU profiling
            profiler = profile_data['profiler']
            profiler.disable()
            
            # Get CPU profiling data
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            profiling_output = s.getvalue()
            
            # Get memory data
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            end_time = time.time()
            end_cpu_time = time.process_time()
            
            duration = end_time - profile_data['start_time']
            cpu_time = end_cpu_time - profile_data['start_cpu_time']
            memory_used = current_memory - profile_data['start_memory']
            
            # Create performance profile
            performance_profile = PerformanceProfile(
                component=profile_data['component'],
                operation=profile_data['operation'],
                duration=duration,
                cpu_time=cpu_time,
                memory_used=memory_used,
                memory_peak=peak_memory,
                call_count=profile_data['call_count'],
                profiling_data=profiling_output
            )
            
            # Store in history
            self.profile_history.append(performance_profile)
            
            # Clean up
            del self.active_profiles[profile_id]
            
            return performance_profile
    
    def increment_call_count(self, profile_id: str):
        """Increment call count for active profile."""
        with self._lock:
            if profile_id in self.active_profiles:
                self.active_profiles[profile_id]['call_count'] += 1
    
    def get_profile_history(self, component: Optional[str] = None, limit: Optional[int] = None) -> List[PerformanceProfile]:
        """Get profiling history."""
        with self._lock:
            history = list(self.profile_history)
            
            if component:
                history = [p for p in history if p.component == component]
            
            if limit:
                history = history[-limit:]
            
            return history


class BenchmarkRunner:
    """Benchmark execution system."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.benchmark_results: List[BenchmarkResult] = []
        self._lock = threading.RLock()
    
    def run_benchmark(self, 
                     name: str,
                     func: Callable,
                     iterations: int = 100,
                     warmup_iterations: int = 10,
                     args: Tuple = (),
                     kwargs: Dict[str, Any] = None) -> BenchmarkResult:
        """
        Run benchmark on function.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Benchmark result
        """
        kwargs = kwargs or {}
        
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Benchmark runs
        times = []
        memory_start = psutil.Process().memory_info().rss
        cpu_start = time.process_time()
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            try:
                func(*args, **kwargs)
                success = True
            except Exception as e:
                logger.warning(f"Benchmark iteration failed: {e}")
                success = False
            
            end_time = time.perf_counter()
            
            if success:
                times.append(end_time - start_time)
        
        cpu_end = time.process_time()
        memory_end = psutil.Process().memory_info().rss
        
        # Calculate statistics
        if times:
            total_time = sum(times)
            avg_time = total_time / len(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = float(np.std(times))
            throughput = len(times) / total_time if total_time > 0 else 0
        else:
            total_time = avg_time = min_time = max_time = std_dev = throughput = 0
        
        memory_used = memory_end - memory_start
        cpu_time = cpu_end - cpu_start
        cpu_percent = (cpu_time / total_time * 100) if total_time > 0 else 0
        
        # Create benchmark result
        result = BenchmarkResult(
            name=name,
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_used=memory_used,
            cpu_percent=cpu_percent,
            metadata={
                'warmup_iterations': warmup_iterations,
                'requested_iterations': iterations,
                'successful_iterations': len(times)
            }
        )
        
        with self._lock:
            self.benchmark_results.append(result)
        
        return result
    
    def compare_benchmarks(self, names: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmarks."""
        with self._lock:
            results = {name: None for name in names}
            
            for result in self.benchmark_results:
                if result.name in names:
                    results[result.name] = result
            
            # Calculate comparisons
            comparisons = {}
            baseline = None
            
            for name, result in results.items():
                if result and baseline is None:
                    baseline = result
                    comparisons[name] = {'relative_performance': 1.0, 'is_baseline': True}
                elif result and baseline:
                    relative_perf = baseline.avg_time / result.avg_time
                    comparisons[name] = {'relative_performance': relative_perf, 'is_baseline': False}
                else:
                    comparisons[name] = {'relative_performance': 0.0, 'is_baseline': False}
            
            return {
                'results': {name: result.to_dict() if result else None for name, result in results.items()},
                'comparisons': comparisons
            }


class PerformanceMetricsCollector:
    """
    Advanced performance metrics collection system.
    
    Provides comprehensive performance monitoring, profiling,
    and benchmarking capabilities across all platform components.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize performance metrics collector.
        
        Args:
            event_bus: Event bus for notifications
        """
        self.event_bus = event_bus or EventBus()
        
        # Core components
        self.aggregator = MetricAggregator()
        self.profiler = PerformanceProfiler()
        self.benchmark_runner = BenchmarkRunner()
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=100000)
        
        # Collection configuration
        self.collection_enabled = True
        self.auto_profiling_enabled = False
        self.profiling_threshold = 1.0  # seconds
        
        # Component tracking
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'error_count': 0,
            'avg_memory': 0,
            'last_activity': None
        })
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background collection
        self.collection_thread: Optional[threading.Thread] = None
        self.collection_running = False
        
        logger.info("Initialized PerformanceMetricsCollector")
    
    def start_collection(self, interval: float = 5.0) -> bool:
        """
        Start background metrics collection.
        
        Args:
            interval: Collection interval in seconds
            
        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                if self.collection_running:
                    return True
                
                self.collection_running = True
                self.collection_thread = threading.Thread(
                    target=self._collection_loop,
                    args=(interval,),
                    name="PerformanceMetricsCollector"
                )
                self.collection_thread.daemon = True
                self.collection_thread.start()
                
                logger.info("Started performance metrics collection")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start metrics collection: {e}")
            return False
    
    def stop_collection(self):
        """Stop background metrics collection."""
        try:
            with self._lock:
                self.collection_running = False
                
                if self.collection_thread and self.collection_thread.is_alive():
                    self.collection_thread.join(timeout=5)
                
                logger.info("Stopped performance metrics collection")
                
        except Exception as e:
            logger.error(f"Error stopping metrics collection: {e}")
    
    def record_metric(self, 
                     name: str,
                     value: Union[int, float],
                     category: MetricCategory,
                     component: str = "",
                     operation: str = "",
                     **kwargs) -> bool:
        """
        Record performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            category: Metric category
            component: Component name
            operation: Operation name
            **kwargs: Additional metadata
            
        Returns:
            True if recorded successfully
        """
        try:
            if not self.collection_enabled:
                return False
            
            metric = PerformanceMetric(
                name=name,
                value=value,
                category=category,
                component=component,
                operation=operation,
                **kwargs
            )
            
            with self._lock:
                self.metrics.append(metric)
                self.aggregator.add_metric(metric)
                
                # Update component stats
                if component:
                    stats = self.component_stats[component]
                    stats['last_activity'] = datetime.now(timezone.utc)
                    
                    if category == MetricCategory.TIMING:
                        stats['call_count'] += 1
                        stats['total_time'] += value
            
            # Publish metric event
            self._publish_metric_event(metric)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
            return False
    
    def record_timing(self, component: str, operation: str, duration: float, **kwargs) -> bool:
        """Record timing metric."""
        return self.record_metric(
            name=f"{operation}.duration",
            value=duration,
            category=MetricCategory.TIMING,
            component=component,
            operation=operation,
            tags={'unit': 'seconds'},
            **kwargs
        )
    
    def record_memory(self, component: str, operation: str, memory_bytes: int, **kwargs) -> bool:
        """Record memory metric."""
        return self.record_metric(
            name=f"{operation}.memory",
            value=memory_bytes,
            category=MetricCategory.MEMORY,
            component=component,
            operation=operation,
            tags={'unit': 'bytes'},
            **kwargs
        )
    
    def record_throughput(self, component: str, operation: str, items_per_second: float, **kwargs) -> bool:
        """Record throughput metric."""
        return self.record_metric(
            name=f"{operation}.throughput",
            value=items_per_second,
            category=MetricCategory.THROUGHPUT,
            component=component,
            operation=operation,
            tags={'unit': 'items/sec'},
            **kwargs
        )
    
    def record_error(self, component: str, operation: str, **kwargs) -> bool:
        """Record error occurrence."""
        with self._lock:
            if component:
                self.component_stats[component]['error_count'] += 1
        
        return self.record_metric(
            name=f"{operation}.error",
            value=1,
            category=MetricCategory.ERROR_RATE,
            component=component,
            operation=operation,
            **kwargs
        )
    
    def start_profiling(self, component: str, operation: str) -> str:
        """Start performance profiling."""
        return self.profiler.start_profiling(component, operation)
    
    def stop_profiling(self, profile_id: str) -> Optional[PerformanceProfile]:
        """Stop performance profiling."""
        profile = self.profiler.stop_profiling(profile_id)
        
        if profile:
            # Record profiling metrics
            self.record_timing(profile.component, profile.operation, profile.duration, metadata={'profiled': True})
            self.record_memory(profile.component, profile.operation, profile.memory_used, metadata={'profiled': True})
            
            # Publish profiling event
            self._publish_profiling_event(profile)
        
        return profile
    
    def run_benchmark(self, 
                     name: str, 
                     func: Callable, 
                     iterations: int = 100,
                     **kwargs) -> BenchmarkResult:
        """Run performance benchmark."""
        result = self.benchmark_runner.run_benchmark(name, func, iterations, **kwargs)
        
        # Record benchmark metrics
        self.record_metric(
            name=f"benchmark.{name}.avg_time",
            value=result.avg_time,
            category=MetricCategory.TIMING,
            metadata={'benchmark': True, 'iterations': iterations}
        )
        
        self.record_metric(
            name=f"benchmark.{name}.throughput",
            value=result.throughput,
            category=MetricCategory.THROUGHPUT,
            metadata={'benchmark': True, 'iterations': iterations}
        )
        
        # Publish benchmark event
        self._publish_benchmark_event(result)
        
        return result
    
    def get_component_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get component performance statistics."""
        with self._lock:
            if component:
                stats = dict(self.component_stats.get(component, {}))
                
                # Add aggregated metrics
                for category in MetricCategory:
                    key = f"{component}.{category.value}"
                    agg_stats = self.aggregator.get_all_stats(key)
                    if agg_stats:
                        stats[f"{category.value}_stats"] = agg_stats
                
                return stats
            else:
                return {
                    comp: dict(stats) for comp, stats in self.component_stats.items()
                }
    
    def get_metric_history(self, 
                          component: Optional[str] = None,
                          category: Optional[MetricCategory] = None,
                          since: Optional[datetime] = None,
                          limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metric history with filtering."""
        with self._lock:
            metrics = list(self.metrics)
            
            # Apply filters
            if component:
                metrics = [m for m in metrics if m.component == component]
            
            if category:
                metrics = [m for m in metrics if m.category == category]
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            if limit:
                metrics = metrics[-limit:]
            
            return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            total_metrics = len(self.metrics)
            
            # Category breakdown
            category_counts = defaultdict(int)
            for metric in self.metrics:
                category_counts[metric.category.value] += 1
            
            # Component breakdown
            component_counts = defaultdict(int)
            for metric in self.metrics:
                if metric.component:
                    component_counts[metric.component] += 1
            
            # Recent activity
            recent_activity = {}
            for component, stats in self.component_stats.items():
                if stats['last_activity']:
                    recent_activity[component] = stats['last_activity'].isoformat()
            
            return {
                'overview': {
                    'total_metrics': total_metrics,
                    'collection_enabled': self.collection_enabled,
                    'auto_profiling_enabled': self.auto_profiling_enabled,
                    'active_profiles': len(self.profiler.active_profiles),
                    'benchmark_results': len(self.benchmark_runner.benchmark_results)
                },
                'categories': dict(category_counts),
                'components': dict(component_counts),
                'recent_activity': recent_activity,
                'top_components': sorted(
                    component_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }
    
    def _collection_loop(self, interval: float):
        """Background collection loop."""
        while self.collection_running:
            try:
                # Collect system performance metrics
                self._collect_system_performance()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Performance collection error: {e}")
    
    def _collect_system_performance(self):
        """Collect system-wide performance metrics."""
        try:
            process = psutil.Process()
            
            # CPU metrics
            cpu_percent = process.cpu_percent()
            self.record_metric(
                name="cpu.usage",
                value=cpu_percent,
                category=MetricCategory.CPU,
                component="system",
                tags={'unit': 'percent'}
            )
            
            # Memory metrics
            memory_info = process.memory_info()
            self.record_metric(
                name="memory.rss",
                value=memory_info.rss,
                category=MetricCategory.MEMORY,
                component="system",
                tags={'unit': 'bytes'}
            )
            
            # Thread metrics
            thread_count = threading.active_count()
            self.record_metric(
                name="threads.count",
                value=thread_count,
                category=MetricCategory.CUSTOM,
                component="system"
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system performance: {e}")
    
    def _publish_metric_event(self, metric: PerformanceMetric):
        """Publish metric event."""
        event = Event(
            type=EventType.PERFORMANCE_METRIC,
            data=metric.to_dict(),
            source="PerformanceMetricsCollector"
        )
        
        self.event_bus.publish(event)
    
    def _publish_profiling_event(self, profile: PerformanceProfile):
        """Publish profiling event."""
        event = Event(
            type=EventType.PERFORMANCE_PROFILE,
            data=profile.to_dict(),
            source="PerformanceMetricsCollector"
        )
        
        self.event_bus.publish(event)
    
    def _publish_benchmark_event(self, result: BenchmarkResult):
        """Publish benchmark event."""
        event = Event(
            type=EventType.BENCHMARK_RESULT,
            data=result.to_dict(),
            source="PerformanceMetricsCollector"
        )
        
        self.event_bus.publish(event)


# Global collector instance
_performance_collector = None

def get_performance_collector() -> PerformanceMetricsCollector:
    """Get global performance metrics collector instance."""
    global _performance_collector
    if _performance_collector is None:
        _performance_collector = PerformanceMetricsCollector()
    return _performance_collector


# Convenience functions

def record_timing_metric(component: str, operation: str, duration: float, **kwargs) -> bool:
    """Record timing metric."""
    collector = get_performance_collector()
    return collector.record_timing(component, operation, duration, **kwargs)


def record_memory_metric(component: str, operation: str, memory_bytes: int, **kwargs) -> bool:
    """Record memory metric."""
    collector = get_performance_collector()
    return collector.record_memory(component, operation, memory_bytes, **kwargs)


def record_throughput_metric(component: str, operation: str, items_per_second: float, **kwargs) -> bool:
    """Record throughput metric."""
    collector = get_performance_collector()
    return collector.record_throughput(component, operation, items_per_second, **kwargs)


# Decorators for automatic performance monitoring

def monitor_timing(component: str, operation: Optional[str] = None):
    """Decorator for automatic timing monitoring."""
    def decorator(func):
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception:
                success = False
                raise
            finally:
                duration = time.perf_counter() - start_time
                record_timing_metric(component, op_name, duration, success=success)
            
            return result
        return wrapper
    return decorator


def monitor_memory(component: str, operation: Optional[str] = None):
    """Decorator for automatic memory monitoring."""
    def decorator(func):
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception:
                success = False
                raise
            finally:
                memory_after = process.memory_info().rss
                memory_used = memory_after - memory_before
                record_memory_metric(component, op_name, memory_used, success=success)
            
            return result
        return wrapper
    return decorator


def profile_performance(component: str, operation: Optional[str] = None):
    """Decorator for automatic performance profiling."""
    def decorator(func):
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_performance_collector()
            profile_id = collector.start_profiling(component, op_name)
            
            try:
                result = func(*args, **kwargs)
            finally:
                collector.stop_profiling(profile_id)
            
            return result
        return wrapper
    return decorator


# Context managers

@contextmanager
def timing_context(component: str, operation: str, **metadata):
    """Context manager for timing measurement."""
    start_time = time.perf_counter()
    
    try:
        yield
        success = True
    except Exception:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        record_timing_metric(component, operation, duration, success=success, **metadata)


@contextmanager
def profiling_context(component: str, operation: str):
    """Context manager for performance profiling."""
    collector = get_performance_collector()
    profile_id = collector.start_profiling(component, operation)
    
    try:
        yield
    finally:
        collector.stop_profiling(profile_id)


# Benchmark utilities

def benchmark_function(name: str, func: Callable, iterations: int = 100, **kwargs) -> BenchmarkResult:
    """Benchmark a function."""
    collector = get_performance_collector()
    return collector.run_benchmark(name, func, iterations, **kwargs)


def compare_functions(functions: Dict[str, Callable], iterations: int = 100, **kwargs) -> Dict[str, Any]:
    """Compare performance of multiple functions."""
    collector = get_performance_collector()
    
    # Run benchmarks
    for name, func in functions.items():
        collector.run_benchmark(name, func, iterations, **kwargs)
    
    # Compare results
    return collector.benchmark_runner.compare_benchmarks(list(functions.keys()))