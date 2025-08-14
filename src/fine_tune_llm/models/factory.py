"""
Model and adapter factory system for fine-tune LLM library.

Provides factory patterns for creating models, tokenizers, and adapters
with proper configuration and dependency management.
"""

from typing import Dict, Any, Type, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    PreTrainedModel, PreTrainedTokenizer
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from ..core.interfaces import BaseFactory, BaseComponent
from ..core.exceptions import ModelError, ConfigurationError
from ..core.protocols import ModelProtocol, TokenizerProtocol
from ..config.schemas import ModelConfig, LoRAConfig

logger = logging.getLogger(__name__)

class ModelFactory(BaseFactory):
    """Factory for creating models and tokenizers."""
    
    def __init__(self):
        self.supported_models = {
            "glm-4.5-air": {
                "model_class": AutoModelForCausalLM,
                "model_id": "ZHIPU-AI/glm-4-9b-chat",
                "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
                "architecture": "GLMForCausalLM"
            },
            "qwen2.5-7b": {
                "model_class": AutoModelForCausalLM,
                "model_id": "Qwen/Qwen2.5-7B",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture": "Qwen2ForCausalLM"
            },
            "llama-3-8b": {
                "model_class": AutoModelForCausalLM,
                "model_id": "meta-llama/Meta-Llama-3-8B",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture": "LlamaForCausalLM"
            },
            "mistral-7b": {
                "model_class": AutoModelForCausalLM,
                "model_id": "mistralai/Mistral-7B-v0.1",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture": "MistralForCausalLM"
            }
        }
        
        self.registered_models: Dict[str, Type] = {}
    
    def create(self, component_type: str, config: Dict[str, Any]) -> BaseComponent:
        """Create model or tokenizer based on component type."""
        if component_type == "model":
            return self.create_model(config)
        elif component_type == "tokenizer":
            return self.create_tokenizer(config)
        else:
            raise ConfigurationError(f"Unknown component type: {component_type}")
    
    def register(self, component_type: str, component_class: type) -> None:
        """Register custom model type."""
        self.registered_models[component_type] = component_class
        logger.info(f"Registered custom model: {component_type}")
    
    def list_types(self) -> list:
        """List available model types."""
        return list(self.supported_models.keys()) + list(self.registered_models.keys())
    
    def create_model(self, config: Union[ModelConfig, Dict[str, Any]]) -> PreTrainedModel:
        """Create model instance."""
        if isinstance(config, dict):
            model_config = ModelConfig.from_dict(config)
        else:
            model_config = config
        
        try:
            # Get model information
            model_info = self._get_model_info(model_config.selected_model)
            model_id = model_config.model_id or model_info["model_id"]
            
            # Prepare model arguments
            model_kwargs = {
                "pretrained_model_name_or_path": model_id,
                "torch_dtype": getattr(torch, model_config.torch_dtype),
                "device_map": model_config.device_map,
                "trust_remote_code": model_config.trust_remote_code,
                "use_cache": model_config.use_cache
            }
            
            # Load model
            model_class = model_info["model_class"]
            model = model_class.from_pretrained(**model_kwargs)
            
            logger.info(f"Created model: {model_id} ({model.__class__.__name__})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise ModelError(f"Model creation failed: {e}")
    
    def create_tokenizer(self, config: Union[ModelConfig, Dict[str, Any]]) -> PreTrainedTokenizer:
        """Create tokenizer instance."""
        if isinstance(config, dict):
            model_config = ModelConfig.from_dict(config)
        else:
            model_config = config
        
        try:
            # Get tokenizer ID
            tokenizer_id = model_config.tokenizer_id or model_config.model_id
            
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True
            )
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
            
            logger.info(f"Created tokenizer: {tokenizer_id}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to create tokenizer: {e}")
            raise ModelError(f"Tokenizer creation failed: {e}")
    
    def create_model_tokenizer_pair(self, config: Union[ModelConfig, Dict[str, Any]]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Create model and tokenizer pair."""
        model = self.create_model(config)
        tokenizer = self.create_tokenizer(config)
        return model, tokenizer
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        if model_name in self.supported_models:
            return self.supported_models[model_name]
        elif model_name in self.registered_models:
            return {"model_class": self.registered_models[model_name]}
        else:
            raise ConfigurationError(f"Unknown model: {model_name}")

class AdapterFactory(BaseFactory):
    """Factory for creating LoRA and other adapters."""
    
    def __init__(self):
        self.adapter_types = {
            "lora": self._create_lora_config,
            "qlora": self._create_qlora_config, 
            "dora": self._create_dora_config
        }
    
    def create(self, component_type: str, config: Dict[str, Any]) -> BaseComponent:
        """Create adapter configuration."""
        return self.create_adapter_config(component_type, config)
    
    def register(self, component_type: str, component_class: type) -> None:
        """Register custom adapter type."""
        self.adapter_types[component_type] = component_class
        logger.info(f"Registered adapter type: {component_type}")
    
    def list_types(self) -> list:
        """List available adapter types."""
        return list(self.adapter_types.keys())
    
    def create_adapter_config(self, adapter_type: str, config: Union[LoRAConfig, Dict[str, Any]]) -> LoraConfig:
        """Create adapter configuration."""
        if isinstance(config, dict):
            lora_config = LoRAConfig.from_dict(config)
        else:
            lora_config = config
        
        if adapter_type not in self.adapter_types:
            raise ConfigurationError(f"Unknown adapter type: {adapter_type}")
        
        try:
            config_func = self.adapter_types[adapter_type]
            peft_config = config_func(lora_config)
            logger.info(f"Created {adapter_type} adapter config")
            return peft_config
            
        except Exception as e:
            logger.error(f"Failed to create adapter config: {e}")
            raise ModelError(f"Adapter creation failed: {e}")
    
    def apply_adapter(self, model: PreTrainedModel, adapter_config: LoraConfig) -> PeftModel:
        """Apply adapter to model."""
        try:
            peft_model = get_peft_model(model, adapter_config)
            logger.info(f"Applied adapter to model: {adapter_config.peft_type}")
            return peft_model
            
        except Exception as e:
            logger.error(f"Failed to apply adapter: {e}")
            raise ModelError(f"Adapter application failed: {e}")
    
    def _create_lora_config(self, config: LoRAConfig) -> LoraConfig:
        """Create standard LoRA configuration."""
        return LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.target_modules,
            modules_to_save=None
        )
    
    def _create_qlora_config(self, config: LoRAConfig) -> LoraConfig:
        """Create QLoRA configuration with quantization."""
        # QLoRA uses the same LoraConfig but with quantized base model
        lora_config = self._create_lora_config(config)
        
        # The quantization is handled during model loading, not in LoRA config
        # This is here for consistency with the factory pattern
        return lora_config
    
    def _create_dora_config(self, config: LoRAConfig) -> LoraConfig:
        """Create DoRA configuration."""
        # DoRA extends LoRA with additional parameters
        lora_config = LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.target_modules,
            use_dora=config.dora_settings.get("decompose_both", True),
            use_rslora=config.dora_settings.get("use_rslora", False)
        )
        
        return lora_config

class QuantizationFactory:
    """Factory for quantization configurations."""
    
    @staticmethod
    def create_quantization_config(config: Dict[str, Any]) -> Optional[Any]:
        """Create quantization configuration."""
        if not config.get("enabled", False):
            return None
        
        try:
            from transformers import BitsAndBytesConfig
            
            return BitsAndBytesConfig(
                load_in_4bit=config.get("bits", 4) == 4,
                load_in_8bit=config.get("bits", 4) == 8,
                bnb_4bit_quant_type=config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, config.get("compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=config.get("use_double_quant", True)
            )
            
        except ImportError:
            logger.warning("BitsAndBytesConfig not available, skipping quantization")
            return None
        except Exception as e:
            logger.error(f"Failed to create quantization config: {e}")
            raise ModelError(f"Quantization config creation failed: {e}")

class ModelLoader:
    """High-level model loader with integrated factories."""
    
    def __init__(self):
        self.model_factory = ModelFactory()
        self.adapter_factory = AdapterFactory()
        self.quantization_factory = QuantizationFactory()
    
    def load_model_for_training(self, model_config: ModelConfig, lora_config: LoRAConfig) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """Load model and tokenizer configured for training."""
        try:
            # Create quantization config if needed
            if lora_config.quantization.get("enabled", False):
                quant_config = self.quantization_factory.create_quantization_config(lora_config.quantization)
                model_config_dict = model_config.to_dict()
                model_config_dict["quantization_config"] = quant_config
            else:
                model_config_dict = model_config.to_dict()
            
            # Create model and tokenizer
            model = self.model_factory.create_model(model_config_dict)
            tokenizer = self.model_factory.create_tokenizer(model_config)
            
            # Apply LoRA adapter
            adapter_config = self.adapter_factory.create_adapter_config(lora_config.method, lora_config)
            peft_model = self.adapter_factory.apply_adapter(model, adapter_config)
            
            # Enable training mode
            peft_model.train()
            
            logger.info("Model loaded and configured for training")
            return peft_model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model for training: {e}")
            raise ModelError(f"Training model loading failed: {e}")
    
    def load_model_for_inference(self, model_path: str, adapter_path: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer configured for inference."""
        try:
            if adapter_path:
                # Load model with adapter
                model = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(model_path),
                    adapter_path
                )
            else:
                # Load regular model
                model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Set evaluation mode
            model.eval()
            
            logger.info("Model loaded and configured for inference")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model for inference: {e}")
            raise ModelError(f"Inference model loading failed: {e}")

# Global factory instances
_model_factory: Optional[ModelFactory] = None
_adapter_factory: Optional[AdapterFactory] = None

def get_model_factory() -> ModelFactory:
    """Get global model factory instance."""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory

def get_adapter_factory() -> AdapterFactory:
    """Get global adapter factory instance."""
    global _adapter_factory
    if _adapter_factory is None:
        _adapter_factory = AdapterFactory()
    return _adapter_factory

# Convenience functions
def create_model(config: Union[ModelConfig, Dict[str, Any]]) -> PreTrainedModel:
    """Convenience function to create model."""
    factory = get_model_factory()
    return factory.create_model(config)

def create_tokenizer(config: Union[ModelConfig, Dict[str, Any]]) -> PreTrainedTokenizer:
    """Convenience function to create tokenizer."""
    factory = get_model_factory()
    return factory.create_tokenizer(config)

def create_adapter_config(adapter_type: str, config: Union[LoRAConfig, Dict[str, Any]]) -> LoraConfig:
    """Convenience function to create adapter config."""
    factory = get_adapter_factory()
    return factory.create_adapter_config(adapter_type, config)