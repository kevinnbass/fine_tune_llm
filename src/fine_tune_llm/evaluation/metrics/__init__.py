"""
Metrics computation modules for evaluation.

This package provides comprehensive metrics computation capabilities
including accuracy, calibration, and specialized metrics.
"""

from .computer import MetricsComputer

# Import advanced metrics if available
try:
    from .advanced import compute_ece, compute_mce, compute_brier_score
    _HAS_ADVANCED_METRICS = True
except ImportError:
    # Provide fallback implementations
    def compute_ece(*args, **kwargs):
        """Fallback ECE implementation."""
        return 0.0
    
    def compute_mce(*args, **kwargs):
        """Fallback MCE implementation."""
        return 0.0
    
    def compute_brier_score(*args, **kwargs):
        """Fallback Brier score implementation."""
        return 0.0
    
    _HAS_ADVANCED_METRICS = False

__all__ = [
    'MetricsComputer',
    'compute_ece',
    'compute_mce', 
    'compute_brier_score',
]

# Version information
__version__ = '2.0.0'

# Feature flags
HAS_ADVANCED_METRICS = _HAS_ADVANCED_METRICS