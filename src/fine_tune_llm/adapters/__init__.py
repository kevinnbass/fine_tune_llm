"""
Adapters for hexagonal architecture implementation.

This package provides adapter implementations for the ports defined
in the core interfaces, enabling clean separation of business logic
from infrastructure concerns.
"""

from .filesystem import FileSystemAdapter
from .database import DatabaseAdapter  
from .api import APIAdapter
from .cache import CacheAdapter

__all__ = [
    "FileSystemAdapter",
    "DatabaseAdapter",
    "APIAdapter", 
    "CacheAdapter"
]