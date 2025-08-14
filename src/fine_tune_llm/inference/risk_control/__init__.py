"""
Risk-controlled prediction for safe AI deployment.

This package provides risk-aware prediction capabilities with statistical
guarantees on error rates and cost-sensitive decision making.
"""

from .controller import (
    BaseRiskController,
    ConfidenceRiskController,
    LossRiskController,
    EntropyRiskController,
    RiskControlledPredictor
)

__all__ = [
    'BaseRiskController',
    'ConfidenceRiskController',
    'LossRiskController',
    'EntropyRiskController',
    'RiskControlledPredictor',
]

# Version information
__version__ = '2.0.0'