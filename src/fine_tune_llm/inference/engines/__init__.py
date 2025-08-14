"""
Inference engines for model prediction.

This package provides various inference engine implementations for
different types of models and prediction scenarios.
"""

from .base import BaseInferenceEngine
from .llm_engine import LLMInferenceEngine

__all__ = [
    'BaseInferenceEngine',
    'LLMInferenceEngine',
]

# Version information
__version__ = '2.0.0'

# Factory function for creating inference engines
def create_inference_engine(engine_type: str = 'llm', **kwargs) -> BaseInferenceEngine:
    """
    Factory function for creating inference engines.
    
    Args:
        engine_type: Type of engine to create
        **kwargs: Arguments to pass to engine
        
    Returns:
        Inference engine instance
    """
    if engine_type == 'llm':
        return LLMInferenceEngine(**kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")