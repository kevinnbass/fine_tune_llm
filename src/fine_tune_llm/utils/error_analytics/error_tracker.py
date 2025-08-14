"""
Error analytics and monitoring system.

This module provides comprehensive error tracking, analysis, and alerting
capabilities for monitoring application health and identifying patterns.
"""

import time
import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import logging

from ...core.exceptions import FineTuneLLMError, SystemError

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"   # System-breaking errors
    HIGH = "high"          # Major functionality impacted  
    MEDIUM = "medium"      # Minor functionality impacted
    LOW = "low"           # Informational/warnings


class ErrorCategory(Enum):
    """Error categories for classification."""
    CONFIGURATION = "configuration"
    MODEL = "model"
    TRAINING = "training"
    INFERENCE = "inference"
    DATA = "data"
    INTEGRATION = "integration"
    SYSTEM = "system"
    SECURITY = "security"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class ErrorEvent:
    """Individual error event record."""
    error_id: str
    timestamp: float
    exception_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        data['timestamp_iso'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return data


@dataclass
class ErrorPattern:
    """Detected error pattern."""
    pattern_id: str
    error_type: str
    component: str
    operation: Optional[str]
    message_pattern: str
    frequency: int
    first_seen: float
    last_seen: float
    severity: ErrorSeverity
    impact_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring."""
    total_errors: int = 0
    errors_by_severity: Dict[ErrorSeverity, int] = None
    errors_by_category: Dict[ErrorCategory, int] = None
    errors_by_component: Dict[str, int] = None
    error_rate_per_minute: float = 0.0
    most_frequent_errors: List[Tuple[str, int]] = None
    
    def __post_init__(self):
        if self.errors_by_severity is None:
            self.errors_by_severity = {severity: 0 for severity in ErrorSeverity}
        if self.errors_by_category is None:
            self.errors_by_category = {category: 0 for category in ErrorCategory}
        if self.errors_by_component is None:
            self.errors_by_component = {}
        if self.most_frequent_errors is None:
            self.most_frequent_errors = []


class ErrorTracker:
    """
    Comprehensive error tracking and analytics system.
    
    Tracks all errors in the application, detects patterns, calculates metrics,
    and provides alerting capabilities for proactive error management.
    """
    
    def __init__(self, 
                 max_events: int = 10000,
                 pattern_detection_window: int = 300,  # 5 minutes
                 min_pattern_frequency: int = 3):
        """
        Initialize error tracker.
        
        Args:
            max_events: Maximum number of events to keep in memory
            pattern_detection_window: Time window in seconds for pattern detection
            min_pattern_frequency: Minimum frequency to consider as a pattern
        """
        self.max_events = max_events
        self.pattern_detection_window = pattern_detection_window
        self.min_pattern_frequency = min_pattern_frequency
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._events: deque = deque(maxlen=max_events)
        self._patterns: Dict[str, ErrorPattern] = {}
        
        # Metrics tracking
        self._metrics_cache: Optional[ErrorMetrics] = None
        self._last_metrics_update = 0.0
        self._metrics_cache_ttl = 60.0  # 1 minute cache TTL
        
        # Alert handlers
        self._alert_handlers: List[Callable[[ErrorEvent], None]] = []
        self._pattern_handlers: List[Callable[[ErrorPattern], None]] = []
        
        # Background processing
        self._last_pattern_detection = time.time()
        self._pattern_detection_interval = 30.0  # 30 seconds
        
        logger.info(f"Initialized ErrorTracker with max_events={max_events}")
    
    def track_error(self, 
                   exception: Exception,
                   component: str,
                   operation: str,
                   context: Optional[Dict[str, Any]] = None,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   correlation_id: Optional[str] = None) -> str:
        """
        Track an error event.
        
        Args:
            exception: Exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context information
            user_id: User identifier (if applicable)
            session_id: Session identifier
            correlation_id: Correlation ID for tracking related events
            
        Returns:
            Generated error ID
        """
        # Generate error ID
        error_id = self._generate_error_id(exception, component, operation)
        
        # Determine category and severity
        category = self._categorize_error(exception, component)
        severity = self._assess_severity(exception, category)
        
        # Create error event
        event = ErrorEvent(
            error_id=error_id,
            timestamp=time.time(),
            exception_type=type(exception).__name__,
            message=str(exception),
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            stack_trace=self._get_stack_trace(exception),
            context=context or {},
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )
        
        # Store event
        with self._lock:
            self._events.append(event)
            
            # Invalidate metrics cache
            self._metrics_cache = None
        
        # Trigger alerts
        self._trigger_alerts(event)
        
        # Check for pattern detection
        if time.time() - self._last_pattern_detection > self._pattern_detection_interval:
            self._detect_patterns()
        
        logger.debug(f"Tracked error: {error_id} [{severity.value}] {exception}")
        return error_id
    
    def get_recent_errors(self, 
                         limit: Optional[int] = None,
                         severity: Optional[ErrorSeverity] = None,
                         category: Optional[ErrorCategory] = None,
                         component: Optional[str] = None,
                         time_window: Optional[int] = None) -> List[ErrorEvent]:
        """
        Get recent errors with optional filtering.
        
        Args:
            limit: Maximum number of errors to return
            severity: Filter by severity level
            category: Filter by error category
            component: Filter by component name
            time_window: Time window in seconds (from now)
            
        Returns:
            List of filtered error events
        """
        with self._lock:
            events = list(self._events)
        
        # Apply time window filter
        if time_window:
            cutoff_time = time.time() - time_window
            events = [e for e in events if e.timestamp >= cutoff_time]
        
        # Apply filters
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if category:
            events = [e for e in events if e.category == category]
        
        if component:
            events = [e for e in events if e.component == component]
        
        # Sort by timestamp (newest first) and apply limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            events = events[:limit]
        
        return events
    
    def get_error_metrics(self, time_window: Optional[int] = None) -> ErrorMetrics:
        """
        Calculate error metrics.
        
        Args:
            time_window: Time window in seconds (default: all time)
            
        Returns:
            Calculated error metrics
        """
        # Check cache if no time window specified
        if not time_window and self._metrics_cache:
            if time.time() - self._last_metrics_update < self._metrics_cache_ttl:
                return self._metrics_cache
        
        with self._lock:
            events = list(self._events)
        
        # Apply time window
        if time_window:
            cutoff_time = time.time() - time_window
            events = [e for e in events if e.timestamp >= cutoff_time]
        
        # Calculate metrics
        metrics = ErrorMetrics()
        metrics.total_errors = len(events)
        
        # Count by severity
        metrics.errors_by_severity = {severity: 0 for severity in ErrorSeverity}
        for event in events:
            metrics.errors_by_severity[event.severity] += 1
        
        # Count by category
        metrics.errors_by_category = {category: 0 for category in ErrorCategory}
        for event in events:
            metrics.errors_by_category[event.category] += 1
        
        # Count by component
        component_counts = defaultdict(int)
        for event in events:
            component_counts[event.component] += 1
        metrics.errors_by_component = dict(component_counts)
        
        # Calculate error rate per minute
        if events and time_window:
            time_span_minutes = time_window / 60.0
            metrics.error_rate_per_minute = len(events) / time_span_minutes
        
        # Most frequent error types
        error_type_counts = defaultdict(int)
        for event in events:
            error_type_counts[event.exception_type] += 1
        
        metrics.most_frequent_errors = sorted(
            error_type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        # Cache metrics if no time window
        if not time_window:
            self._metrics_cache = metrics
            self._last_metrics_update = time.time()
        
        return metrics
    
    def get_error_patterns(self) -> List[ErrorPattern]:
        """Get detected error patterns."""
        with self._lock:
            return list(self._patterns.values())
    
    def add_alert_handler(self, handler: Callable[[ErrorEvent], None]):
        """Add alert handler for error events."""
        self._alert_handlers.append(handler)
        logger.info(f"Added error alert handler: {handler.__name__}")
    
    def add_pattern_handler(self, handler: Callable[[ErrorPattern], None]):
        """Add handler for detected error patterns."""
        self._pattern_handlers.append(handler)
        logger.info(f"Added pattern alert handler: {handler.__name__}")
    
    def _generate_error_id(self, exception: Exception, component: str, operation: str) -> str:
        """Generate unique error ID."""
        content = f"{type(exception).__name__}:{component}:{operation}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _categorize_error(self, exception: Exception, component: str) -> ErrorCategory:
        """Categorize error based on exception type and component."""
        exception_type = type(exception).__name__.lower()
        component_lower = component.lower()
        
        # Map exception types to categories
        if 'config' in exception_type or 'validation' in exception_type:
            return ErrorCategory.CONFIGURATION
        elif 'model' in exception_type or 'model' in component_lower:
            return ErrorCategory.MODEL
        elif 'training' in exception_type or 'training' in component_lower:
            return ErrorCategory.TRAINING
        elif 'inference' in exception_type or 'inference' in component_lower:
            return ErrorCategory.INFERENCE
        elif 'data' in exception_type or 'data' in component_lower:
            return ErrorCategory.DATA
        elif 'service' in exception_type or 'api' in exception_type or 'network' in exception_type:
            return ErrorCategory.INTEGRATION
        elif 'security' in exception_type or 'auth' in exception_type:
            return ErrorCategory.SECURITY
        elif 'connection' in exception_type or 'timeout' in exception_type:
            return ErrorCategory.NETWORK
        elif 'system' in exception_type or 'resource' in exception_type:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on exception and category."""
        exception_type = type(exception).__name__.lower()
        message = str(exception).lower()
        
        # Critical errors
        if (isinstance(exception, SystemError) or
            'critical' in message or
            'system' in exception_type or
            category == ErrorCategory.SECURITY):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if (category in [ErrorCategory.MODEL, ErrorCategory.TRAINING] or
            'failed' in message or
            'error' in exception_type):
            return ErrorSeverity.HIGH
        
        # Medium severity  
        if (category in [ErrorCategory.INFERENCE, ErrorCategory.INTEGRATION] or
            'warning' in message):
            return ErrorSeverity.MEDIUM
        
        # Low severity (default)
        return ErrorSeverity.LOW
    
    def _get_stack_trace(self, exception: Exception) -> Optional[str]:
        """Extract stack trace from exception."""
        import traceback
        
        try:
            if hasattr(exception, '__traceback__') and exception.__traceback__:
                return ''.join(traceback.format_tb(exception.__traceback__))
        except Exception:
            pass
        
        return None
    
    def _trigger_alerts(self, event: ErrorEvent):
        """Trigger alert handlers for error event."""
        for handler in self._alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")
    
    def _detect_patterns(self):
        """Detect error patterns in recent events."""
        self._last_pattern_detection = time.time()
        
        # Get events in detection window
        cutoff_time = time.time() - self.pattern_detection_window
        recent_events = [e for e in self._events if e.timestamp >= cutoff_time]
        
        if len(recent_events) < self.min_pattern_frequency:
            return
        
        # Group events by potential patterns
        pattern_groups = defaultdict(list)
        
        for event in recent_events:
            # Create pattern key
            pattern_key = f"{event.exception_type}:{event.component}:{event.operation}"
            pattern_groups[pattern_key].append(event)
        
        # Identify patterns that meet frequency threshold
        with self._lock:
            for pattern_key, events in pattern_groups.items():
                if len(events) >= self.min_pattern_frequency:
                    pattern_id = hashlib.sha256(pattern_key.encode()).hexdigest()[:16]
                    
                    # Update existing pattern or create new one
                    if pattern_id in self._patterns:
                        pattern = self._patterns[pattern_id]
                        pattern.frequency = len(events)
                        pattern.last_seen = max(e.timestamp for e in events)
                    else:
                        # Create new pattern
                        first_event = events[0]
                        pattern = ErrorPattern(
                            pattern_id=pattern_id,
                            error_type=first_event.exception_type,
                            component=first_event.component,
                            operation=first_event.operation,
                            message_pattern=self._extract_message_pattern(events),
                            frequency=len(events),
                            first_seen=min(e.timestamp for e in events),
                            last_seen=max(e.timestamp for e in events),
                            severity=max(e.severity for e in events),
                            impact_score=self._calculate_impact_score(events)
                        )
                        
                        self._patterns[pattern_id] = pattern
                        
                        # Trigger pattern handlers
                        for handler in self._pattern_handlers:
                            try:
                                handler(pattern)
                            except Exception as e:
                                logger.error(f"Error in pattern handler {handler.__name__}: {e}")
    
    def _extract_message_pattern(self, events: List[ErrorEvent]) -> str:
        """Extract common message pattern from events."""
        messages = [e.message for e in events]
        
        # For now, just return the most common message
        # Could be enhanced with more sophisticated pattern extraction
        message_counts = defaultdict(int)
        for msg in messages:
            message_counts[msg] += 1
        
        if message_counts:
            return max(message_counts.items(), key=lambda x: x[1])[0]
        
        return "Unknown pattern"
    
    def _calculate_impact_score(self, events: List[ErrorEvent]) -> float:
        """Calculate impact score for error pattern."""
        # Simple scoring based on frequency and severity
        severity_weights = {
            ErrorSeverity.CRITICAL: 4.0,
            ErrorSeverity.HIGH: 3.0,
            ErrorSeverity.MEDIUM: 2.0,
            ErrorSeverity.LOW: 1.0
        }
        
        total_weight = sum(severity_weights[e.severity] for e in events)
        frequency_multiplier = min(len(events) / 10.0, 2.0)  # Cap at 2x
        
        return total_weight * frequency_multiplier
    
    def export_events(self, 
                     file_path: str,
                     time_window: Optional[int] = None,
                     format: str = 'json') -> int:
        """
        Export error events to file.
        
        Args:
            file_path: Output file path
            time_window: Time window in seconds
            format: Export format ('json' or 'csv')
            
        Returns:
            Number of events exported
        """
        events = self.get_recent_errors(time_window=time_window)
        
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump([e.to_dict() for e in events], f, indent=2)
        elif format.lower() == 'csv':
            import csv
            with open(file_path, 'w', newline='') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].to_dict().keys())
                    writer.writeheader()
                    for event in events:
                        writer.writerow(event.to_dict())
        
        logger.info(f"Exported {len(events)} error events to {file_path}")
        return len(events)


# Global error tracker instance
_global_error_tracker = ErrorTracker()


def track_error(exception: Exception, 
               component: str,
               operation: str,
               context: Optional[Dict[str, Any]] = None,
               **kwargs) -> str:
    """Track error using global error tracker."""
    return _global_error_tracker.track_error(
        exception, component, operation, context, **kwargs
    )


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance."""
    return _global_error_tracker


def add_global_alert_handler(handler: Callable[[ErrorEvent], None]):
    """Add alert handler to global error tracker."""
    _global_error_tracker.add_alert_handler(handler)