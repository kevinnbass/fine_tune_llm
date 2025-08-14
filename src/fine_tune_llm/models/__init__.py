"""
Model management system for fine-tune LLM library.

Provides unified model loading, saving, and management with support for
multiple architectures, LoRA adapters, and checkpoint management.
"""

# Public API - Core model management
from .factory import ModelFactory, AdapterFactory
from .registry import ModelRegistry
from .manager import ModelManager

# Public API - Model loaders (explicit imports for clarity)
from .loaders.base import BaseModelLoader
from .loaders.huggingface import HuggingFaceModelLoader
from .loaders.local import LocalModelLoader
from .loaders.checkpoint import CheckpointLoader

# Public API - Adapters
from .adapters.base import BaseAdapter
from .adapters.lora import LoRAAdapter
from .adapters.qlora import QLoRAAdapter
from .adapters.dora import DoRAAdapter

# Public API - Checkpoint management
from .checkpoints.manager import CheckpointManager
from .checkpoints.metadata import CheckpointMetadata
from .checkpoints.storage import CheckpointStorage

__all__ = [
    # Core management
    "ModelFactory",
    "AdapterFactory", 
    "ModelRegistry",
    "ModelManager",
    
    # Model loaders
    "BaseModelLoader",
    "HuggingFaceModelLoader",
    "LocalModelLoader", 
    "CheckpointLoader",
    
    # Adapters
    "BaseAdapter",
    "LoRAAdapter",
    "QLoRAAdapter",
    "DoRAAdapter",
    
    # Checkpoint management
    "CheckpointManager",
    "CheckpointMetadata",
    "CheckpointStorage"
]

# Private imports for internal module use only
from . import _internals