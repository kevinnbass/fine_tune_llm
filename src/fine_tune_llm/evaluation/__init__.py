"""
Evaluation package for model evaluation and analysis.

This package provides comprehensive evaluation capabilities including
evaluators, metrics computation, visualization, and reporting.
"""

from .evaluators import BaseEvaluator, LLMEvaluator, create_evaluator
from .metrics import MetricsComputer, compute_ece, compute_mce, compute_brier_score
from .visualization import EvaluationPlotter
from .reporting import ReportGenerator

# Import auditing components
from .auditing import (
    BiasAuditor,
    ExplainableReasoning,
    ProceduralAlignment,
    VerifiableTraining,
    AdvancedHighStakesAuditor
)

__all__ = [
    # Evaluators
    'BaseEvaluator',
    'LLMEvaluator',
    'create_evaluator',
    
    # Metrics
    'MetricsComputer',
    'compute_ece',
    'compute_mce',
    'compute_brier_score',
    
    # Visualization
    'EvaluationPlotter',
    
    # Reporting
    'ReportGenerator',
    
    # Auditing
    'BiasAuditor',
    'ExplainableReasoning',
    'ProceduralAlignment',
    'VerifiableTraining',
    'AdvancedHighStakesAuditor',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Comprehensive evaluation framework for LLM fine-tuning'