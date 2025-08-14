"""
Error analytics and monitoring utilities.

This package provides comprehensive error tracking, pattern detection,
and analytics capabilities for monitoring application health.
"""

from .error_tracker import (
    ErrorTracker,
    ErrorEvent,
    ErrorPattern,
    ErrorMetrics,
    ErrorSeverity,
    ErrorCategory,
    track_error,
    get_error_tracker,
    add_global_alert_handler
)

__all__ = [
    "ErrorTracker",
    "ErrorEvent",
    "ErrorPattern", 
    "ErrorMetrics",
    "ErrorSeverity",
    "ErrorCategory",
    "track_error",
    "get_error_tracker",
    "add_global_alert_handler"
]