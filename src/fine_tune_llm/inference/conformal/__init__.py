"""
Conformal prediction for uncertainty quantification.

This package provides conformal prediction methods that offer statistical
guarantees on prediction sets and coverage rates.
"""

from .predictor import (
    BaseConformalPredictor,
    LACConformalPredictor,
    APSConformalPredictor,
    RAPSConformalPredictor,
    ConformalPredictor
)

__all__ = [
    'BaseConformalPredictor',
    'LACConformalPredictor',
    'APSConformalPredictor',
    'RAPSConformalPredictor',
    'ConformalPredictor',
]

# Version information
__version__ = '2.0.0'