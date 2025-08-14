"""
Data transformers for format conversion and augmentation.

This package provides data transformation capabilities including
format conversion, data augmentation, and preprocessing pipelines.
"""

from .base import BaseDataTransformer
from .format_converter import FormatConverter
from .instruction_formatter import InstructionFormatter
from .augmentation_transformer import AugmentationTransformer

__all__ = [
    'BaseDataTransformer',
    'FormatConverter',
    'InstructionFormatter', 
    'AugmentationTransformer',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Data transformation utilities for LLM training datasets'