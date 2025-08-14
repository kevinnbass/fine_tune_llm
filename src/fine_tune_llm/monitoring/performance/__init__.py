"""
Performance Metrics Collection Module.

This module provides comprehensive performance monitoring, profiling,
and benchmarking capabilities across all platform components.
"""

from .metrics_collector import (
    MetricCategory,
    AggregationType,
    PerformanceMetric,
    PerformanceProfile,
    BenchmarkResult,
    PerformanceMetricsCollector,
    get_performance_collector,
    record_timing_metric,
    record_memory_metric,
    record_throughput_metric,
    monitor_timing,
    monitor_memory,
    profile_performance,
    timing_context,
    profiling_context,
    benchmark_function,
    compare_functions
)

__all__ = [
    'MetricCategory',
    'AggregationType',
    'PerformanceMetric',
    'PerformanceProfile',
    'BenchmarkResult',
    'PerformanceMetricsCollector',
    'get_performance_collector',
    'record_timing_metric',
    'record_memory_metric',
    'record_throughput_metric',
    'monitor_timing',
    'monitor_memory',
    'profile_performance',
    'timing_context',
    'profiling_context',
    'benchmark_function',
    'compare_functions'
]