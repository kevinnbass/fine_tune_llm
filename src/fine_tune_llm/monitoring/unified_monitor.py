"""
Unified Monitoring System.

This module provides comprehensive monitoring across all platform components
with real-time metrics collection, alerting, and visualization capabilities.
"""

import logging
import threading
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import json
import psutil
import traceback
from pathlib import Path
import weakref

from ..core.events import EventBus, Event, EventType
from ..utils.logging import get_centralized_logger, LogLevel
from .streaming.metrics_streamer import MetricsStreamer
from .collectors.base import BaseCollector

logger = get_centralized_logger().get_logger("unified_monitor")


class MonitoringLevel(Enum):
    """Monitoring levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    PERCENTAGE = "percentage"
    RATE = "rate"
    CUSTOM = "custom"


@dataclass
class MetricDefinition:
    """Definition of a monitoring metric."""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    tags: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    collection_interval: float = 1.0
    retention_period: int = 3600  # seconds
    alert_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricValue:
    """Single metric value with metadata."""
    name: str
    value: Union[int, float, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    component: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata,
            'component': self.component
        }


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: Union[int, float]
    threshold: Union[int, float]
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'metric': self.metric_name,
            'severity': self.severity.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_timestamp': self.resolved_timestamp.isoformat() if self.resolved_timestamp else None,
            'metadata': self.metadata
        }


class ComponentMonitor:
    """Monitor for individual component."""
    
    def __init__(self, component_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize component monitor."""
        self.component_name = component_name
        self.config = config or {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.last_collection = {}
        self.enabled = True
        self._lock = threading.RLock()
    
    def define_metric(self, metric: MetricDefinition):
        """Define a metric for this component."""
        with self._lock:
            self.metric_definitions[metric.name] = metric
    
    def record_metric(self, name: str, value: Union[int, float], **kwargs) -> bool:
        """Record metric value."""
        try:
            with self._lock:
                if not self.enabled:
                    return False
                
                metric_value = MetricValue(
                    name=name,
                    value=value,
                    component=self.component_name,
                    **kwargs
                )
                
                self.metrics[name].append(metric_value)
                self.last_collection[name] = time.time()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
            return False
    
    def get_metric_values(self, name: str, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get metric values."""
        with self._lock:
            values = list(self.metrics.get(name, []))
            
            if since:
                values = [v for v in values if v.timestamp >= since]
            
            return values
    
    def get_latest_metrics(self) -> Dict[str, MetricValue]:
        """Get latest value for each metric."""
        with self._lock:
            latest = {}
            for name, values in self.metrics.items():
                if values:
                    latest[name] = values[-1]
            return latest
    
    def clear_metrics(self, older_than: Optional[datetime] = None):
        """Clear old metrics."""
        with self._lock:
            if older_than is None:
                older_than = datetime.now(timezone.utc) - timedelta(hours=1)
            
            for name, values in self.metrics.items():
                # Remove old values
                while values and values[0].timestamp < older_than:
                    values.popleft()


class SystemMetricsCollector:
    """Collects system-wide metrics."""
    
    def __init__(self):
        """Initialize system metrics collector."""
        self.process = psutil.Process()
        self.last_cpu_times = None
        self.last_io_counters = None
        self.last_network_io = None
    
    def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics['system.cpu.usage'] = MetricValue('system.cpu.usage', cpu_percent, tags={'unit': 'percent'})
            
            cpu_count = psutil.cpu_count()
            metrics['system.cpu.count'] = MetricValue('system.cpu.count', cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['system.memory.total'] = MetricValue('system.memory.total', memory.total, tags={'unit': 'bytes'})
            metrics['system.memory.available'] = MetricValue('system.memory.available', memory.available, tags={'unit': 'bytes'})
            metrics['system.memory.used'] = MetricValue('system.memory.used', memory.used, tags={'unit': 'bytes'})
            metrics['system.memory.usage'] = MetricValue('system.memory.usage', memory.percent, tags={'unit': 'percent'})
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['system.disk.total'] = MetricValue('system.disk.total', disk.total, tags={'unit': 'bytes'})
            metrics['system.disk.used'] = MetricValue('system.disk.used', disk.used, tags={'unit': 'bytes'})
            metrics['system.disk.free'] = MetricValue('system.disk.free', disk.free, tags={'unit': 'bytes'})
            metrics['system.disk.usage'] = MetricValue('system.disk.usage', (disk.used / disk.total) * 100, tags={'unit': 'percent'})
            
            # Process metrics
            process_memory = self.process.memory_info()
            metrics['process.memory.rss'] = MetricValue('process.memory.rss', process_memory.rss, tags={'unit': 'bytes'})
            metrics['process.memory.vms'] = MetricValue('process.memory.vms', process_memory.vms, tags={'unit': 'bytes'})
            
            process_cpu = self.process.cpu_percent()
            metrics['process.cpu.usage'] = MetricValue('process.cpu.usage', process_cpu, tags={'unit': 'percent'})
            
            # Thread count
            thread_count = threading.active_count()
            metrics['process.threads.count'] = MetricValue('process.threads.count', thread_count)
            
            # File descriptors (if available)
            try:
                fd_count = self.process.num_fds()
                metrics['process.fd.count'] = MetricValue('process.fd.count', fd_count)
            except (AttributeError, psutil.AccessDenied):
                pass
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics


class AlertManager:
    """Manages monitoring alerts."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize alert manager."""
        self.event_bus = event_bus or EventBus()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_rules: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alert_handlers: List[Callable] = []
        self._lock = threading.RLock()
    
    def add_alert_rule(self, metric_name: str, rule: Dict[str, Any]):
        """Add alert rule for metric."""
        with self._lock:
            self.alert_rules[metric_name].append(rule)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metric: MetricValue) -> List[Alert]:
        """Check if metric triggers any alerts."""
        alerts = []
        
        with self._lock:
            rules = self.alert_rules.get(metric.name, [])
            
            for rule in rules:
                try:
                    alert = self._evaluate_rule(metric, rule)
                    if alert:
                        alerts.append(alert)
                        self._handle_alert(alert)
                except Exception as e:
                    logger.error(f"Failed to evaluate alert rule: {e}")
        
        return alerts
    
    def _evaluate_rule(self, metric: MetricValue, rule: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate single alert rule."""
        condition = rule.get('condition', 'gt')
        threshold = rule.get('threshold')
        severity = AlertSeverity(rule.get('severity', 'medium'))
        
        if threshold is None:
            return None
        
        triggered = False
        
        if condition == 'gt' and metric.value > threshold:
            triggered = True
        elif condition == 'lt' and metric.value < threshold:
            triggered = True
        elif condition == 'eq' and metric.value == threshold:
            triggered = True
        elif condition == 'ge' and metric.value >= threshold:
            triggered = True
        elif condition == 'le' and metric.value <= threshold:
            triggered = True
        
        if not triggered:
            return None
        
        alert_id = f"{metric.name}_{metric.component}_{condition}_{threshold}"
        
        # Check if alert is already active
        if alert_id in self.active_alerts:
            return None
        
        message = rule.get('message', f"{metric.name} {condition} {threshold}")
        
        alert = Alert(
            id=alert_id,
            metric_name=metric.name,
            severity=severity,
            message=message,
            value=metric.value,
            threshold=threshold,
            component=metric.component,
            metadata={'rule': rule, 'metric_tags': metric.tags}
        )
        
        return alert
    
    def _handle_alert(self, alert: Alert):
        """Handle triggered alert."""
        with self._lock:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Publish alert event
        self._publish_alert_event(alert)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = datetime.now(timezone.utc)
                del self.active_alerts[alert_id]
                return True
            return False
    
    def _publish_alert_event(self, alert: Alert):
        """Publish alert event."""
        event = Event(
            type=EventType.ALERT,
            data=alert.to_dict(),
            source="AlertManager"
        )
        
        self.event_bus.publish(event)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            history = list(self.alert_history)
            if limit:
                history = history[-limit:]
            return history


class UnifiedMonitor:
    """
    Unified monitoring system.
    
    Provides comprehensive monitoring across all platform components
    with real-time metrics collection, alerting, and visualization.
    """
    
    def __init__(self, 
                 level: MonitoringLevel = MonitoringLevel.STANDARD,
                 metrics_streamer: Optional[MetricsStreamer] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize unified monitor.
        
        Args:
            level: Monitoring level
            metrics_streamer: Metrics streaming system
            event_bus: Event bus for notifications
        """
        self.level = level
        self.metrics_streamer = metrics_streamer
        self.event_bus = event_bus or EventBus()
        
        # Component monitors
        self.component_monitors: Dict[str, ComponentMonitor] = {}
        
        # System metrics collector
        self.system_collector = SystemMetricsCollector()
        
        # Alert manager
        self.alert_manager = AlertManager(self.event_bus)
        
        # Collection configuration
        self.collection_interval = self._get_collection_interval()
        self.collection_enabled = True
        
        # Collection threads
        self.collection_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'collection_errors': 0,
            'start_time': datetime.now(timezone.utc),
            'last_collection': None
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup default metrics and alerts
        self._setup_default_monitoring()
        
        logger.info(f"Initialized UnifiedMonitor with level: {level.value}")
    
    def _get_collection_interval(self) -> float:
        """Get collection interval based on monitoring level."""
        intervals = {
            MonitoringLevel.MINIMAL: 60.0,
            MonitoringLevel.STANDARD: 10.0,
            MonitoringLevel.DETAILED: 5.0,
            MonitoringLevel.COMPREHENSIVE: 1.0,
            MonitoringLevel.DEBUG: 0.5
        }
        return intervals.get(self.level, 10.0)
    
    def _setup_default_monitoring(self):
        """Setup default monitoring configuration."""
        # System monitoring
        system_monitor = self.register_component("system")
        
        # Define system metrics
        cpu_metric = MetricDefinition(
            name="system.cpu.usage",
            metric_type=MetricType.PERCENTAGE,
            description="System CPU usage percentage",
            unit="percent",
            thresholds={'high': 80.0, 'critical': 95.0}
        )
        system_monitor.define_metric(cpu_metric)
        
        memory_metric = MetricDefinition(
            name="system.memory.usage",
            metric_type=MetricType.PERCENTAGE,
            description="System memory usage percentage",
            unit="percent",
            thresholds={'high': 85.0, 'critical': 95.0}
        )
        system_monitor.define_metric(memory_metric)
        
        disk_metric = MetricDefinition(
            name="system.disk.usage",
            metric_type=MetricType.PERCENTAGE,
            description="System disk usage percentage",
            unit="percent",
            thresholds={'high': 80.0, 'critical': 90.0}
        )
        system_monitor.define_metric(disk_metric)
        
        # Setup default alert rules
        self.alert_manager.add_alert_rule("system.cpu.usage", {
            'condition': 'gt',
            'threshold': 80.0,
            'severity': 'high',
            'message': 'High CPU usage detected'
        })
        
        self.alert_manager.add_alert_rule("system.cpu.usage", {
            'condition': 'gt',
            'threshold': 95.0,
            'severity': 'critical',
            'message': 'Critical CPU usage detected'
        })
        
        self.alert_manager.add_alert_rule("system.memory.usage", {
            'condition': 'gt',
            'threshold': 85.0,
            'severity': 'high',
            'message': 'High memory usage detected'
        })
        
        self.alert_manager.add_alert_rule("system.memory.usage", {
            'condition': 'gt',
            'threshold': 95.0,
            'severity': 'critical',
            'message': 'Critical memory usage detected'
        })
    
    def start_monitoring(self) -> bool:
        """Start monitoring system."""
        try:
            with self._lock:
                if self.running:
                    return True
                
                self.running = True
                self.collection_enabled = True
                
                # Start collection thread
                self.collection_thread = threading.Thread(
                    target=self._collection_loop,
                    name="UnifiedMonitor-Collection"
                )
                self.collection_thread.daemon = True
                self.collection_thread.start()
                
                logger.info("Unified monitoring started")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        try:
            with self._lock:
                self.running = False
                self.collection_enabled = False
                
                # Wait for collection thread to finish
                if self.collection_thread and self.collection_thread.is_alive():
                    self.collection_thread.join(timeout=5)
                
                logger.info("Unified monitoring stopped")
                
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def register_component(self, component_name: str, config: Optional[Dict[str, Any]] = None) -> ComponentMonitor:
        """
        Register component for monitoring.
        
        Args:
            component_name: Name of component
            config: Component-specific configuration
            
        Returns:
            Component monitor instance
        """
        with self._lock:
            if component_name not in self.component_monitors:
                monitor = ComponentMonitor(component_name, config)
                self.component_monitors[component_name] = monitor
                logger.info(f"Registered component for monitoring: {component_name}")
            
            return self.component_monitors[component_name]
    
    def unregister_component(self, component_name: str) -> bool:
        """
        Unregister component from monitoring.
        
        Args:
            component_name: Name of component
            
        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if component_name in self.component_monitors:
                del self.component_monitors[component_name]
                logger.info(f"Unregistered component from monitoring: {component_name}")
                return True
            return False
    
    def record_metric(self, 
                     component: str, 
                     metric_name: str, 
                     value: Union[int, float],
                     **kwargs) -> bool:
        """
        Record metric for component.
        
        Args:
            component: Component name
            metric_name: Metric name
            value: Metric value
            **kwargs: Additional metadata
            
        Returns:
            True if recorded successfully
        """
        try:
            # Get or create component monitor
            monitor = self.component_monitors.get(component)
            if not monitor:
                monitor = self.register_component(component)
            
            # Record metric
            success = monitor.record_metric(metric_name, value, **kwargs)
            
            if success:
                self.stats['metrics_collected'] += 1
                
                # Create metric value for alerting
                metric_value = MetricValue(
                    name=metric_name,
                    value=value,
                    component=component,
                    **kwargs
                )
                
                # Check alerts
                alerts = self.alert_manager.check_alerts(metric_value)
                if alerts:
                    self.stats['alerts_triggered'] += len(alerts)
                
                # Stream metric if available
                if self.metrics_streamer:
                    self.metrics_streamer.send_metric(
                        name=metric_name,
                        value=value,
                        step=int(time.time()),
                        metadata={'component': component, **kwargs}
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
            self.stats['collection_errors'] += 1
            return False
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                if self.collection_enabled:
                    self._collect_system_metrics()
                    self.stats['last_collection'] = datetime.now(timezone.utc)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                self.stats['collection_errors'] += 1
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            system_metrics = self.system_collector.collect_metrics()
            
            for metric_name, metric_value in system_metrics.items():
                self.record_metric(
                    component="system",
                    metric_name=metric_name,
                    value=metric_value.value,
                    tags=metric_value.tags,
                    metadata=metric_value.metadata
                )
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            self.stats['collection_errors'] += 1
    
    def get_component_metrics(self, 
                             component: str, 
                             metric_name: Optional[str] = None,
                             since: Optional[datetime] = None) -> Dict[str, List[MetricValue]]:
        """
        Get metrics for component.
        
        Args:
            component: Component name
            metric_name: Specific metric name (all if None)
            since: Only metrics since timestamp
            
        Returns:
            Dictionary of metric values
        """
        monitor = self.component_monitors.get(component)
        if not monitor:
            return {}
        
        if metric_name:
            return {metric_name: monitor.get_metric_values(metric_name, since)}
        else:
            metrics = {}
            for name in monitor.metric_definitions.keys():
                metrics[name] = monitor.get_metric_values(name, since)
            return metrics
    
    def get_latest_metrics(self, component: Optional[str] = None) -> Dict[str, Dict[str, MetricValue]]:
        """
        Get latest metrics.
        
        Args:
            component: Specific component (all if None)
            
        Returns:
            Dictionary of latest metric values
        """
        if component:
            monitor = self.component_monitors.get(component)
            return {component: monitor.get_latest_metrics()} if monitor else {}
        else:
            latest = {}
            for comp_name, monitor in self.component_monitors.items():
                latest[comp_name] = monitor.get_latest_metrics()
            return latest
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
            
            return {
                'general': {
                    **self.stats,
                    'uptime_seconds': uptime,
                    'collection_interval': self.collection_interval,
                    'monitoring_level': self.level.value,
                    'collection_enabled': self.collection_enabled,
                    'running': self.running
                },
                'components': {
                    name: {
                        'enabled': monitor.enabled,
                        'metric_count': len(monitor.metric_definitions),
                        'data_points': sum(len(values) for values in monitor.metrics.values())
                    }
                    for name, monitor in self.component_monitors.items()
                },
                'alerts': {
                    'active_count': len(self.alert_manager.active_alerts),
                    'total_triggered': self.stats['alerts_triggered'],
                    'rules_count': sum(len(rules) for rules in self.alert_manager.alert_rules.values())
                }
            }
    
    def add_alert_rule(self, metric_name: str, rule: Dict[str, Any]) -> bool:
        """
        Add alert rule.
        
        Args:
            metric_name: Metric name
            rule: Alert rule configuration
            
        Returns:
            True if added successfully
        """
        try:
            self.alert_manager.add_alert_rule(metric_name, rule)
            logger.info(f"Added alert rule for metric: {metric_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> bool:
        """
        Add alert handler.
        
        Args:
            handler: Alert handler function
            
        Returns:
            True if added successfully
        """
        try:
            self.alert_manager.add_alert_handler(handler)
            logger.info("Added alert handler")
            return True
        except Exception as e:
            logger.error(f"Failed to add alert handler: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert."""
        return self.alert_manager.resolve_alert(alert_id)
    
    def cleanup_old_data(self, older_than: Optional[datetime] = None):
        """Clean up old monitoring data."""
        if older_than is None:
            older_than = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for monitor in self.component_monitors.values():
            monitor.clear_metrics(older_than)
        
        logger.info(f"Cleaned up monitoring data older than {older_than}")


# Global monitor instance
_unified_monitor = None

def get_unified_monitor() -> UnifiedMonitor:
    """Get global unified monitor instance."""
    global _unified_monitor
    if _unified_monitor is None:
        _unified_monitor = UnifiedMonitor()
    return _unified_monitor


# Convenience functions

def start_monitoring(level: MonitoringLevel = MonitoringLevel.STANDARD) -> bool:
    """
    Start unified monitoring.
    
    Args:
        level: Monitoring level
        
    Returns:
        True if started successfully
    """
    monitor = get_unified_monitor()
    monitor.level = level
    return monitor.start_monitoring()


def record_component_metric(component: str, metric_name: str, value: Union[int, float], **kwargs) -> bool:
    """
    Record metric for component.
    
    Args:
        component: Component name
        metric_name: Metric name
        value: Metric value
        **kwargs: Additional metadata
        
    Returns:
        True if recorded successfully
    """
    monitor = get_unified_monitor()
    return monitor.record_metric(component, metric_name, value, **kwargs)


def get_component_monitor(component: str) -> ComponentMonitor:
    """
    Get monitor for component.
    
    Args:
        component: Component name
        
    Returns:
        Component monitor instance
    """
    monitor = get_unified_monitor()
    return monitor.register_component(component)


# Decorator for method performance monitoring
def monitor_performance(component: str, metric_prefix: str = ""):
    """
    Decorator for monitoring method performance.
    
    Args:
        component: Component name
        metric_prefix: Prefix for metric names
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metric_name = f"{metric_prefix}.{func.__name__}" if metric_prefix else func.__name__
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Record performance metrics
                record_component_metric(component, f"{metric_name}.duration", duration, tags={'unit': 'seconds'})
                record_component_metric(component, f"{metric_name}.success", 1 if success else 0)
                record_component_metric(component, f"{metric_name}.calls", 1)
            
            return result
        return wrapper
    return decorator


# Context manager for monitoring operations
class MonitoringContext:
    """Context manager for monitoring operations."""
    
    def __init__(self, component: str, operation: str, **metadata):
        """Initialize monitoring context."""
        self.component = component
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        record_component_metric(
            self.component, 
            f"{self.operation}.started", 
            1, 
            metadata=self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        duration = time.time() - self.start_time if self.start_time else 0
        success = exc_type is None
        
        record_component_metric(
            self.component, 
            f"{self.operation}.duration", 
            duration, 
            tags={'unit': 'seconds'},
            metadata=self.metadata
        )
        
        record_component_metric(
            self.component, 
            f"{self.operation}.completed", 
            1, 
            tags={'success': str(success)},
            metadata=self.metadata
        )


def create_monitoring_context(component: str, operation: str, **metadata) -> MonitoringContext:
    """
    Create monitoring context manager.
    
    Args:
        component: Component name
        operation: Operation name
        **metadata: Additional metadata
        
    Returns:
        Monitoring context manager
    """
    return MonitoringContext(component, operation, **metadata)