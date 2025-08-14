"""
Evaluator modules for model evaluation.

This package provides various evaluator implementations for different
types of models and evaluation scenarios.
"""

from .base import BaseEvaluator
from .llm_evaluator import LLMEvaluator

__all__ = [
    'BaseEvaluator',
    'LLMEvaluator',
]

# Version information
__version__ = '2.0.0'

# Default evaluator factory
def create_evaluator(evaluator_type: str = 'llm', **kwargs) -> BaseEvaluator:
    """
    Factory function for creating evaluators.
    
    Args:
        evaluator_type: Type of evaluator to create
        **kwargs: Arguments to pass to evaluator
        
    Returns:
        Evaluator instance
    """
    if evaluator_type == 'llm':
        return LLMEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")