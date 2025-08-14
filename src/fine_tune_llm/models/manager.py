"""
Model manager for fine-tune LLM library.

Provides high-level model management by coordinating the factory and registry
systems for seamless model operations.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from ..core.interfaces import BaseComponent
from ..core.exceptions import ModelError, ConfigurationError
from ..config.schemas import ModelConfig, LoRAConfig
from .factory import ModelFactory, AdapterFactory, ModelLoader
from .registry import ModelRegistry, ModelMetadata, AdapterMetadata, ModelStatus

logger = logging.getLogger(__name__)

class ModelManager(BaseComponent):
    """Unified model management system."""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry = ModelRegistry(registry_path)
        self.model_factory = ModelFactory()
        self.adapter_factory = AdapterFactory()
        self.model_loader = ModelLoader()
        
        # Active models cache
        self._active_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Model manager initialized")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize model manager with configuration."""
        self._config = config
        
        # Configure model cache directory
        cache_dir = config.get("cache_dir", "cache/models")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Model manager initialized with configuration")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clear active models from memory
        for name in list(self._active_models.keys()):
            self.unload_model(name)
        
        logger.info("Model manager cleaned up")
    
    @property
    def name(self) -> str:
        """Component name."""
        return "ModelManager"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def list_available_models(self) -> List[ModelMetadata]:
        """List all available models."""
        return self.registry.list_models(status=ModelStatus.AVAILABLE)
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded models."""
        return list(self._active_models.keys())
    
    def get_model_info(self, name: str) -> Optional[ModelMetadata]:
        """Get model information."""
        return self.registry.get_model(name)
    
    def load_model_for_training(self, model_name: str, model_config: ModelConfig, 
                              lora_config: LoRAConfig, cache_name: Optional[str] = None) -> str:
        """Load model configured for training and return cache key."""
        try:
            cache_key = cache_name or f"{model_name}_training"
            
            # Check if already loaded
            if cache_key in self._active_models:
                logger.info(f"Model already loaded: {cache_key}")
                return cache_key
            
            # Update registry with model info
            model_metadata = self.registry.get_model(model_name)
            if model_metadata:
                # Sync target modules from registry to config
                if not lora_config.target_modules and model_metadata.target_modules:
                    lora_config.target_modules = model_metadata.target_modules
            
            # Load model using model loader
            model, tokenizer = self.model_loader.load_model_for_training(model_config, lora_config)
            
            # Cache the model
            self._active_models[cache_key] = (model, tokenizer)
            self._model_configs[cache_key] = {
                "model_config": model_config.to_dict(),
                "lora_config": lora_config.to_dict(),
                "model_name": model_name,
                "mode": "training"
            }
            
            # Update registry status
            if model_metadata:
                self.registry.update_model_status(model_name, ModelStatus.CACHED)
            
            logger.info(f"Model loaded for training: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to load model for training: {e}")
            raise ModelError(f"Training model loading failed: {e}")
    
    def load_model_for_inference(self, model_path: str, adapter_path: Optional[str] = None,
                                cache_name: Optional[str] = None) -> str:
        """Load model configured for inference and return cache key."""
        try:
            cache_key = cache_name or f"inference_{Path(model_path).name}"
            
            # Check if already loaded
            if cache_key in self._active_models:
                logger.info(f"Model already loaded: {cache_key}")
                return cache_key
            
            # Load model using model loader
            model, tokenizer = self.model_loader.load_model_for_inference(model_path, adapter_path)
            
            # Cache the model
            self._active_models[cache_key] = (model, tokenizer)
            self._model_configs[cache_key] = {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "mode": "inference"
            }
            
            logger.info(f"Model loaded for inference: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to load model for inference: {e}")
            raise ModelError(f"Inference model loading failed: {e}")
    
    def get_model(self, cache_key: str) -> Optional[Tuple[PreTrainedModel, PreTrainedTokenizer]]:
        """Get loaded model by cache key."""
        return self._active_models.get(cache_key)
    
    def unload_model(self, cache_key: str) -> None:
        """Unload model from memory."""
        if cache_key in self._active_models:
            model, tokenizer = self._active_models[cache_key]
            
            # Clear GPU memory
            if hasattr(model, 'cpu'):
                model.cpu()
            
            del self._active_models[cache_key]
            del self._model_configs[cache_key]
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model unloaded: {cache_key}")
    
    def save_model(self, cache_key: str, output_dir: str, save_adapter_only: bool = True) -> None:
        """Save model to directory."""
        if cache_key not in self._active_models:
            raise ModelError(f"Model not loaded: {cache_key}")
        
        try:
            model, tokenizer = self._active_models[cache_key]
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save tokenizer
            tokenizer.save_pretrained(output_path)
            
            # Save model
            if isinstance(model, PeftModel) and save_adapter_only:
                # Save only adapter weights for PEFT models
                model.save_pretrained(output_path)
                logger.info(f"Saved adapter to: {output_path}")
            else:
                # Save full model
                model.save_pretrained(output_path)
                logger.info(f"Saved full model to: {output_path}")
            
            # Update registry if this is a registered model
            config = self._model_configs.get(cache_key, {})
            model_name = config.get("model_name")
            if model_name:
                self.registry.update_model_status(model_name, ModelStatus.CACHED, str(output_path))
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Model saving failed: {e}")
    
    def create_adapter(self, base_model_key: str, lora_config: LoRAConfig, 
                      adapter_name: str) -> str:
        """Create and apply adapter to loaded base model."""
        if base_model_key not in self._active_models:
            raise ModelError(f"Base model not loaded: {base_model_key}")
        
        try:
            base_model, tokenizer = self._active_models[base_model_key]
            
            # Create adapter config
            adapter_config = self.adapter_factory.create_adapter_config(lora_config.method, lora_config)
            
            # Apply adapter
            peft_model = self.adapter_factory.apply_adapter(base_model, adapter_config)
            
            # Cache the adapter model
            adapter_cache_key = f"{base_model_key}_{adapter_name}"
            self._active_models[adapter_cache_key] = (peft_model, tokenizer)
            self._model_configs[adapter_cache_key] = {
                **self._model_configs[base_model_key],
                "adapter_name": adapter_name,
                "lora_config": lora_config.to_dict()
            }
            
            logger.info(f"Adapter created and applied: {adapter_cache_key}")
            return adapter_cache_key
            
        except Exception as e:
            logger.error(f"Failed to create adapter: {e}")
            raise ModelError(f"Adapter creation failed: {e}")
    
    def merge_adapter(self, peft_model_key: str, output_dir: str) -> str:
        """Merge adapter weights with base model."""
        if peft_model_key not in self._active_models:
            raise ModelError(f"PEFT model not loaded: {peft_model_key}")
        
        try:
            peft_model, tokenizer = self._active_models[peft_model_key]
            
            if not isinstance(peft_model, PeftModel):
                raise ModelError(f"Model is not a PEFT model: {peft_model_key}")
            
            # Merge adapter weights
            merged_model = peft_model.merge_and_unload()
            
            # Save merged model
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            merged_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Cache merged model
            merged_cache_key = f"{peft_model_key}_merged"
            self._active_models[merged_cache_key] = (merged_model, tokenizer)
            self._model_configs[merged_cache_key] = {
                **self._model_configs[peft_model_key],
                "merged": True,
                "merged_path": str(output_path)
            }
            
            logger.info(f"Adapter merged and saved: {output_path}")
            return merged_cache_key
            
        except Exception as e:
            logger.error(f"Failed to merge adapter: {e}")
            raise ModelError(f"Adapter merging failed: {e}")
    
    def register_custom_model(self, metadata: ModelMetadata) -> None:
        """Register custom model with the registry."""
        self.registry.register_model(metadata)
        logger.info(f"Custom model registered: {metadata.name}")
    
    def register_custom_adapter(self, metadata: AdapterMetadata) -> None:
        """Register custom adapter with the registry."""
        self.registry.register_adapter(metadata)
        logger.info(f"Custom adapter registered: {metadata.name}")
    
    def get_memory_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get memory usage statistics for loaded models."""
        memory_stats = {}
        
        for cache_key, (model, tokenizer) in self._active_models.items():
            stats = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": next(model.parameters()).device if list(model.parameters()) else "cpu",
                "dtype": next(model.parameters()).dtype if list(model.parameters()) else None
            }
            
            # Add GPU memory usage if available
            if torch.cuda.is_available() and "cuda" in str(stats["device"]):
                stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            
            memory_stats[cache_key] = stats
        
        return memory_stats
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up unused models."""
        initial_memory = 0
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = 0
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
        
        freed_memory = initial_memory - final_memory
        
        logger.info(f"Memory optimization freed {freed_memory / (1024 * 1024):.1f} MB")
        
        return {
            "initial_memory_mb": initial_memory / (1024 * 1024),
            "final_memory_mb": final_memory / (1024 * 1024),
            "freed_memory_mb": freed_memory / (1024 * 1024),
            "active_models": len(self._active_models)
        }
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status."""
        registry_stats = self.registry.get_registry_stats()
        memory_stats = self.get_memory_usage()
        
        return {
            "name": self.name,
            "version": self.version,
            "registry": registry_stats,
            "active_models": {
                "count": len(self._active_models),
                "models": list(self._active_models.keys()),
                "memory_usage": memory_stats
            },
            "supported_models": self.model_factory.list_types(),
            "supported_adapters": self.adapter_factory.list_types()
        }

# Global model manager instance
_global_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager

# Convenience functions
def load_model_for_training(model_name: str, model_config: ModelConfig, 
                           lora_config: LoRAConfig) -> str:
    """Load model for training using global manager."""
    manager = get_model_manager()
    return manager.load_model_for_training(model_name, model_config, lora_config)

def load_model_for_inference(model_path: str, adapter_path: Optional[str] = None) -> str:
    """Load model for inference using global manager."""
    manager = get_model_manager()
    return manager.load_model_for_inference(model_path, adapter_path)

def get_loaded_model(cache_key: str) -> Optional[Tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """Get loaded model using global manager."""
    manager = get_model_manager()
    return manager.get_model(cache_key)