"""
Data loaders for various input formats.

This package provides data loading capabilities for different file formats
and data sources commonly used in LLM fine-tuning.
"""

from .base import BaseDataLoader
from .json_loader import JsonDataLoader
from .jsonl_loader import JsonlDataLoader
from .csv_loader import CsvDataLoader
from .huggingface_loader import HuggingFaceDataLoader

__all__ = [
    'BaseDataLoader',
    'JsonDataLoader', 
    'JsonlDataLoader',
    'CsvDataLoader',
    'HuggingFaceDataLoader',
]

# Version information
__version__ = '2.0.0'

# Package metadata
__author__ = 'Fine-Tune LLM Team'
__description__ = 'Data loading utilities for LLM training datasets'