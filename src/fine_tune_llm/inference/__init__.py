"""
Inference package for model prediction and uncertainty quantification.

This package provides comprehensive inference capabilities including
engines, predictors, conformal prediction, and risk control.
"""

from .engines import BaseInferenceEngine, LLMInferenceEngine, create_inference_engine
from .conformal import (
    BaseConformalPredictor,
    LACConformalPredictor,
    APSConformalPredictor,
    RAPSConformalPredictor,
    ConformalPredictor
)
from .risk_control import (
    BaseRiskController,
    ConfidenceRiskController,
    LossRiskController,
    EntropyRiskController,
    RiskControlledPredictor
)

__all__ = [
    # Engines
    'BaseInferenceEngine',
    'LLMInferenceEngine',
    'create_inference_engine',
    
    # Conformal Prediction
    'BaseConformalPredictor',
    'LACConformalPredictor',
    'APSConformalPredictor',
    'RAPSConformalPredictor',
    'ConformalPredictor',
    
    # Risk Control
    'BaseRiskController',
    'ConfidenceRiskController',
    'LossRiskController',
    'EntropyRiskController',
    'RiskControlledPredictor',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Comprehensive inference framework with uncertainty quantification'