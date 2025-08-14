"""
Data pipeline package for LLM training data processing.

This package provides comprehensive data processing capabilities including
loading, validation, processing, and transformation for LLM fine-tuning.
"""

from .processors import BaseDataProcessor, TextProcessor
from .validators import SchemaValidator
from .loaders import BaseDataLoader, JsonlDataLoader
from .transformers import BaseDataTransformer, InstructionFormatter

__all__ = [
    # Processors
    'BaseDataProcessor',
    'TextProcessor',
    
    # Validators
    'SchemaValidator',
    
    # Loaders
    'BaseDataLoader', 
    'JsonlDataLoader',
    
    # Transformers
    'BaseDataTransformer',
    'InstructionFormatter',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Comprehensive data pipeline for LLM training'