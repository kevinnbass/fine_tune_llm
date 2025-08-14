"""
Internal utilities for model management system.

This module contains internal helper functions and utilities
that are not part of the public API.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def _normalize_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model configuration for internal processing."""
    normalized = config.copy()
    
    # Normalize model name/path
    if 'model_name_or_path' in normalized:
        normalized['model_path'] = normalized.pop('model_name_or_path')
    
    # Normalize tokenizer
    if 'tokenizer_name' not in normalized and 'model_path' in normalized:
        normalized['tokenizer_name'] = normalized['model_path']
    
    # Default values
    normalized.setdefault('trust_remote_code', False)
    normalized.setdefault('use_auth_token', None)
    normalized.setdefault('revision', None)
    
    return normalized


def _validate_adapter_config(config: Dict[str, Any]) -> bool:
    """Validate adapter configuration for internal processing."""
    required_fields = ['r', 'lora_alpha', 'target_modules']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required adapter config field: {field}")
            return False
    
    # Validate ranges
    if not (1 <= config['r'] <= 1024):
        logger.error(f"Invalid LoRA rank: {config['r']} (must be 1-1024)")
        return False
    
    if not (1 <= config['lora_alpha'] <= 1024):
        logger.error(f"Invalid LoRA alpha: {config['lora_alpha']} (must be 1-1024)")
        return False
    
    return True


def _get_model_architecture_info(model) -> Dict[str, Any]:
    """Extract architecture information from a loaded model."""
    info = {}
    
    try:
        if hasattr(model, 'config'):
            config = model.config
            info['model_type'] = getattr(config, 'model_type', None)
            info['architecture'] = getattr(config, 'architectures', [])
            info['vocab_size'] = getattr(config, 'vocab_size', None)
            info['hidden_size'] = getattr(config, 'hidden_size', None)
            info['num_layers'] = getattr(config, 'num_hidden_layers', 
                                        getattr(config, 'n_layer', None))
            info['num_attention_heads'] = getattr(config, 'num_attention_heads', 
                                                 getattr(config, 'n_head', None))
    
    except Exception as e:
        logger.warning(f"Could not extract model architecture info: {e}")
    
    return info


def _estimate_model_memory_usage(model) -> Optional[float]:
    """Estimate model memory usage in MB."""
    try:
        if hasattr(model, 'get_memory_footprint'):
            return model.get_memory_footprint() / (1024 * 1024)  # Convert to MB
        
        # Fallback: count parameters
        if hasattr(model, 'parameters'):
            num_params = sum(p.numel() for p in model.parameters())
            # Rough estimate: 4 bytes per parameter (float32) + overhead
            return (num_params * 4 + num_params * 0.2) / (1024 * 1024)
    
    except Exception as e:
        logger.warning(f"Could not estimate model memory usage: {e}")
    
    return None


def _cleanup_model_cache():
    """Clean up model cache and temporary files."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Could not clean up CUDA cache: {e}")


# Internal constants
_DEFAULT_MAX_MEMORY_MB = 8192
_DEFAULT_DEVICE_MAP = "auto"
_SUPPORTED_MODEL_TYPES = {
    'llama', 'mistral', 'qwen', 'glm', 'baichuan', 'chatglm'
}
_ADAPTER_TYPE_MAP = {
    'lora': 'LoRAAdapter',
    'qlora': 'QLoRAAdapter', 
    'dora': 'DoRAAdapter'
}