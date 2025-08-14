"""Comprehensive monitoring and observability system."""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction."""
    timestamp: datetime
    request_id: str
    text_length: int
    decision: Optional[str]
    confidence: float
    abstained: bool
    tier: int
    total_latency_ms: float
    voter_latencies: Dict[str, float]
    voter_costs: Dict[str, float]
    llm_called: bool
    cache_hit: bool
    slice_name: Optional[str] = None
    true_label: Optional[str] = None
    feedback_score: Optional[float] = None
    error: Optional[str] = None


class MonitoringSystem:
    """Comprehensive monitoring for the ensemble system."""
    
    def __init__(
        self,
        service_name: str = "birdflu-ensemble",
        otlp_endpoint: Optional[str] = None,
        enable_prometheus: bool = True,
        enable_otel: bool = True,
        buffer_size: int = 10000
    ):
        """
        Initialize monitoring system.
        
        Args:
            service_name: Name of the service
            otlp_endpoint: OTLP endpoint for traces/metrics
            enable_prometheus: Enable Prometheus metrics
            enable_otel: Enable OpenTelemetry
            buffer_size: Size of metrics buffer
        """
        self.service_name = service_name
        self.buffer_size = buffer_size
        
        # Metrics buffer for analysis
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.slice_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize Prometheus metrics
        if enable_prometheus:
            self._init_prometheus_metrics()
        
        # Initialize OpenTelemetry
        if enable_otel and otlp_endpoint:
            self._init_opentelemetry(otlp_endpoint)
        
        # Performance tracking
        self.performance_window = deque(maxlen=1000)
        self.alert_thresholds = {
            'error_rate': 0.05,
            'p99_latency_ms': 1000,
            'abstention_rate': 0.20,
            'llm_call_rate': 0.15
        }
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.registry = CollectorRegistry()
        
        # Counters
        self.prediction_counter = Counter(
            'predictions_total',
            'Total number of predictions',
            ['decision', 'abstained', 'tier'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Histograms
        self.latency_histogram = Histogram(
            'prediction_latency_ms',
            'Prediction latency in milliseconds',
            ['tier'],
            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
            registry=self.registry
        )
        
        self.confidence_histogram = Histogram(
            'prediction_confidence',
            'Prediction confidence scores',
            ['decision'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
            registry=self.registry
        )
        
        # Gauges
        self.active_requests = Gauge(
            'active_requests',
            'Number of active requests',
            registry=self.registry
        )
        
        self.model_version = Gauge(
            'model_version_info',
            'Model version information',
            ['model_type', 'version'],
            registry=self.registry
        )
        
        # Summary
        self.cost_summary = Summary(
            'prediction_cost_cents',
            'Prediction cost in cents',
            ['model'],
            registry=self.registry
        )
        
    def _init_opentelemetry(self, endpoint: str):
        """Initialize OpenTelemetry tracing and metrics."""
        # Resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0"
        })
        
        # Tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Metrics
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        metric_reader = metrics.PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=60000
        )
        
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=[metric_reader])
        )
        
        meter = metrics.get_meter(__name__)
        
        # Create OTel metrics
        self.otel_prediction_counter = meter.create_counter(
            "predictions",
            description="Number of predictions"
        )
        
        self.otel_latency_histogram = meter.create_histogram(
            "latency_ms",
            description="Prediction latency",
            unit="ms"
        )
        
    def record_prediction(self, metrics: PredictionMetrics):
        """
        Record prediction metrics.
        
        Args:
            metrics: Prediction metrics object
        """
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        if metrics.slice_name:
            self.slice_metrics[metrics.slice_name].append(metrics)
        
        # Update Prometheus metrics
        if hasattr(self, 'prediction_counter'):
            self.prediction_counter.labels(
                decision=metrics.decision or "abstain",
                abstained=str(metrics.abstained),
                tier=str(metrics.tier)
            ).inc()
            
            self.latency_histogram.labels(tier=str(metrics.tier)).observe(
                metrics.total_latency_ms
            )
            
            if metrics.confidence > 0:
                self.confidence_histogram.labels(
                    decision=metrics.decision or "abstain"
                ).observe(metrics.confidence)
            
            # Record costs
            for model, cost in metrics.voter_costs.items():
                self.cost_summary.labels(model=model).observe(cost)
        
        # Update OpenTelemetry metrics
        if hasattr(self, 'otel_prediction_counter'):
            self.otel_prediction_counter.add(1, {
                "decision": metrics.decision or "abstain",
                "tier": str(metrics.tier)
            })
            
            self.otel_latency_histogram.record(
                metrics.total_latency_ms,
                {"tier": str(metrics.tier)}
            )
        
        # Log structured event
        logger.info(
            "prediction_recorded",
            request_id=metrics.request_id,
            decision=metrics.decision,
            confidence=metrics.confidence,
            abstained=metrics.abstained,
            tier=metrics.tier,
            latency_ms=metrics.total_latency_ms,
            llm_called=metrics.llm_called,
            cache_hit=metrics.cache_hit
        )
        
        # Check for alerts
        self._check_alerts()
        
    def record_error(self, error_type: str, error_msg: str, request_id: str):
        """Record an error."""
        if hasattr(self, 'error_counter'):
            self.error_counter.labels(error_type=error_type).inc()
        
        logger.error(
            "prediction_error",
            error_type=error_type,
            error_msg=error_msg,
            request_id=request_id
        )
        
    def _check_alerts(self):
        """Check if any metrics exceed alert thresholds."""
        if len(self.metrics_buffer) < 100:
            return  # Not enough data
        
        recent_metrics = list(self.metrics_buffer)[-100:]
        
        # Calculate rates
        error_rate = sum(1 for m in recent_metrics if m.error) / len(recent_metrics)
        abstention_rate = sum(1 for m in recent_metrics if m.abstained) / len(recent_metrics)
        llm_call_rate = sum(1 for m in recent_metrics if m.llm_called) / len(recent_metrics)
        
        # Calculate latency percentiles
        latencies = [m.total_latency_ms for m in recent_metrics]
        p99_latency = np.percentile(latencies, 99)
        
        # Check thresholds
        if error_rate > self.alert_thresholds['error_rate']:
            logger.warning(
                "alert_triggered",
                alert_type="high_error_rate",
                value=error_rate,
                threshold=self.alert_thresholds['error_rate']
            )
        
        if p99_latency > self.alert_thresholds['p99_latency_ms']:
            logger.warning(
                "alert_triggered",
                alert_type="high_latency",
                value=p99_latency,
                threshold=self.alert_thresholds['p99_latency_ms']
            )
        
        if abstention_rate > self.alert_thresholds['abstention_rate']:
            logger.warning(
                "alert_triggered",
                alert_type="high_abstention_rate",
                value=abstention_rate,
                threshold=self.alert_thresholds['abstention_rate']
            )
        
        if llm_call_rate > self.alert_thresholds['llm_call_rate']:
            logger.warning(
                "alert_triggered",
                alert_type="high_llm_call_rate",
                value=llm_call_rate,
                threshold=self.alert_thresholds['llm_call_rate']
            )
    
    def get_metrics_summary(self, window_size: int = 1000) -> Dict[str, Any]:
        """
        Get summary of recent metrics.
        
        Args:
            window_size: Number of recent predictions to analyze
            
        Returns:
            Dictionary of summary statistics
        """
        if not self.metrics_buffer:
            return {}
        
        recent = list(self.metrics_buffer)[-window_size:]
        
        # Basic statistics
        summary = {
            'total_predictions': len(recent),
            'abstention_rate': sum(1 for m in recent if m.abstained) / len(recent),
            'llm_call_rate': sum(1 for m in recent if m.llm_called) / len(recent),
            'cache_hit_rate': sum(1 for m in recent if m.cache_hit) / len(recent),
            'error_rate': sum(1 for m in recent if m.error) / len(recent)
        }
        
        # Latency statistics
        latencies = [m.total_latency_ms for m in recent]
        summary['latency'] = {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
        
        # Confidence statistics
        confidences = [m.confidence for m in recent if m.confidence > 0]
        if confidences:
            summary['confidence'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        # Decision distribution
        decisions = [m.decision for m in recent if m.decision]
        if decisions:
            from collections import Counter
            summary['decision_distribution'] = dict(Counter(decisions))
        
        # Tier distribution
        tiers = [m.tier for m in recent]
        summary['tier_distribution'] = dict(Counter(tiers))
        
        # Cost analysis
        total_cost = sum(
            sum(m.voter_costs.values())
            for m in recent
        )
        summary['cost'] = {
            'total_cents': total_cost,
            'per_prediction_cents': total_cost / len(recent)
        }
        
        return summary
    
    def get_slice_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics per slice."""
        slice_performance = {}
        
        for slice_name, metrics in self.slice_metrics.items():
            if not metrics:
                continue
            
            recent = list(metrics)
            
            # Calculate accuracy if we have labels
            labeled = [m for m in recent if m.true_label is not None]
            if labeled:
                correct = sum(
                    1 for m in labeled
                    if m.decision == m.true_label and not m.abstained
                )
                accuracy = correct / len(labeled) if labeled else 0
            else:
                accuracy = None
            
            slice_performance[slice_name] = {
                'n_predictions': len(recent),
                'abstention_rate': sum(1 for m in recent if m.abstained) / len(recent),
                'mean_confidence': np.mean([m.confidence for m in recent if m.confidence > 0]),
                'mean_latency_ms': np.mean([m.total_latency_ms for m in recent]),
                'accuracy': accuracy
            }
        
        return slice_performance
    
    def export_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        if hasattr(self, 'registry'):
            return generate_latest(self.registry)
        return b""
    
    def create_span(self, name: str) -> Any:
        """Create OpenTelemetry span for tracing."""
        if hasattr(self, 'tracer'):
            return self.tracer.start_as_current_span(name)
        return None


class PerformanceProfiler:
    """Profile performance of different components."""
    
    def __init__(self):
        """Initialize profiler."""
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        
    def profile_voter(self, voter_id: str):
        """Context manager to profile voter execution."""
        class VoterProfiler:
            def __init__(self, profiler, voter_id):
                self.profiler = profiler
                self.voter_id = voter_id
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = (time.perf_counter() - self.start_time) * 1000
                self.profiler.timings[self.voter_id].append(elapsed)
        
        return VoterProfiler(self, voter_id)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get profiling summary."""
        summary = {}
        
        for component, timings in self.timings.items():
            if timings:
                summary[component] = {
                    'mean_ms': np.mean(timings),
                    'p50_ms': np.percentile(timings, 50),
                    'p95_ms': np.percentile(timings, 95),
                    'p99_ms': np.percentile(timings, 99),
                    'total_calls': len(timings)
                }
        
        return summary


# Global monitoring instance
monitoring = MonitoringSystem()