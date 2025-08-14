"""
Model registry system for fine-tune LLM library.

Provides centralized model registration, discovery, and metadata management
for all supported models and adapters.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
from datetime import datetime

from ..core.exceptions import ModelError, ConfigurationError

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model registration status."""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    CACHED = "cached"
    ERROR = "error"
    DEPRECATED = "deprecated"

class AdapterType(Enum):
    """Adapter types."""
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    ADALORA = "adalora"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    
    # Basic information
    name: str
    model_id: str
    architecture: str
    parameter_count: Optional[int] = None
    
    # Configuration
    target_modules: List[str] = field(default_factory=list)
    max_sequence_length: int = 2048
    vocab_size: Optional[int] = None
    
    # Model capabilities
    supports_chat: bool = False
    supports_instruct: bool = False
    supports_code: bool = False
    supports_multilingual: bool = False
    
    # Technical details
    torch_dtype: str = "bfloat16"
    device_requirements: Dict[str, Any] = field(default_factory=dict)
    memory_requirements: Dict[str, int] = field(default_factory=dict)  # in MB
    
    # Status and availability
    status: ModelStatus = ModelStatus.AVAILABLE
    local_path: Optional[str] = None
    remote_url: Optional[str] = None
    checksum: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    license: Optional[str] = None
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_date: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    # Performance benchmarks
    benchmarks: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "target_modules": self.target_modules,
            "max_sequence_length": self.max_sequence_length,
            "vocab_size": self.vocab_size,
            "supports_chat": self.supports_chat,
            "supports_instruct": self.supports_instruct,
            "supports_code": self.supports_code,
            "supports_multilingual": self.supports_multilingual,
            "torch_dtype": self.torch_dtype,
            "device_requirements": self.device_requirements,
            "memory_requirements": self.memory_requirements,
            "status": self.status.value,
            "local_path": self.local_path,
            "remote_url": self.remote_url,
            "checksum": self.checksum,
            "description": self.description,
            "license": self.license,
            "created_date": self.created_date,
            "updated_date": self.updated_date,
            "tags": self.tags,
            "benchmarks": self.benchmarks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        # Handle enum conversion
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ModelStatus(data["status"])
        
        return cls(**data)

@dataclass
class AdapterMetadata:
    """Metadata for registered adapters."""
    
    # Basic information
    name: str
    adapter_type: AdapterType
    base_model: str
    
    # Configuration
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=list)
    
    # Training information
    dataset_name: Optional[str] = None
    training_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    # Performance metrics
    final_loss: Optional[float] = None
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Status and storage
    status: ModelStatus = ModelStatus.AVAILABLE
    local_path: Optional[str] = None
    size_mb: Optional[int] = None
    
    # Metadata
    description: Optional[str] = None
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_date: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "adapter_type": self.adapter_type.value,
            "base_model": self.base_model,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "dataset_name": self.dataset_name,
            "training_epochs": self.training_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "final_loss": self.final_loss,
            "eval_metrics": self.eval_metrics,
            "status": self.status.value,
            "local_path": self.local_path,
            "size_mb": self.size_mb,
            "description": self.description,
            "created_date": self.created_date,
            "updated_date": self.updated_date,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdapterMetadata':
        """Create from dictionary."""
        # Handle enum conversions
        if "adapter_type" in data and isinstance(data["adapter_type"], str):
            data["adapter_type"] = AdapterType(data["adapter_type"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ModelStatus(data["status"])
        
        return cls(**data)

class ModelRegistry:
    """Central registry for models and adapters."""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path) if registry_path else Path("registry/models.json")
        self.models: Dict[str, ModelMetadata] = {}
        self.adapters: Dict[str, AdapterMetadata] = {}
        
        # Load built-in models
        self._register_builtin_models()
        
        # Load from file if exists
        if self.registry_path.exists():
            self.load_from_file()
    
    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a model."""
        metadata.updated_date = datetime.now().isoformat()
        self.models[metadata.name] = metadata
        logger.info(f"Registered model: {metadata.name}")
        
        # Auto-save
        self.save_to_file()
    
    def register_adapter(self, metadata: AdapterMetadata) -> None:
        """Register an adapter."""
        metadata.updated_date = datetime.now().isoformat()
        self.adapters[metadata.name] = metadata
        logger.info(f"Registered adapter: {metadata.name}")
        
        # Auto-save
        self.save_to_file()
    
    def get_model(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name."""
        return self.models.get(name)
    
    def get_adapter(self, name: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata by name."""
        return self.adapters.get(name)
    
    def list_models(self, status: Optional[ModelStatus] = None, 
                   architecture: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if architecture:
            models = [m for m in models if m.architecture == architecture]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        return models
    
    def list_adapters(self, base_model: Optional[str] = None,
                     adapter_type: Optional[AdapterType] = None,
                     status: Optional[ModelStatus] = None) -> List[AdapterMetadata]:
        """List adapters with optional filtering."""
        adapters = list(self.adapters.values())
        
        if base_model:
            adapters = [a for a in adapters if a.base_model == base_model]
        
        if adapter_type:
            adapters = [a for a in adapters if a.adapter_type == adapter_type]
        
        if status:
            adapters = [a for a in adapters if a.status == status]
        
        return adapters
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """Search models by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for model in self.models.values():
            if (query_lower in model.name.lower() or
                (model.description and query_lower in model.description.lower()) or
                any(query_lower in tag.lower() for tag in model.tags)):
                results.append(model)
        
        return results
    
    def get_compatible_adapters(self, model_name: str) -> List[AdapterMetadata]:
        """Get adapters compatible with a model."""
        model = self.get_model(model_name)
        if not model:
            return []
        
        return [a for a in self.adapters.values() if a.base_model == model.name]
    
    def get_model_recommendations(self, task_type: str, 
                                max_memory_mb: Optional[int] = None) -> List[ModelMetadata]:
        """Get model recommendations for a task type."""
        models = list(self.models.values())
        
        # Filter by capabilities
        if task_type == "chat":
            models = [m for m in models if m.supports_chat]
        elif task_type == "instruct":
            models = [m for m in models if m.supports_instruct]
        elif task_type == "code":
            models = [m for m in models if m.supports_code]
        
        # Filter by memory requirements
        if max_memory_mb:
            models = [m for m in models 
                     if m.memory_requirements.get("inference", 0) <= max_memory_mb]
        
        # Sort by parameter count (smaller first for efficiency)
        models.sort(key=lambda m: m.parameter_count or float('inf'))
        
        return models
    
    def update_model_status(self, name: str, status: ModelStatus, 
                           local_path: Optional[str] = None) -> None:
        """Update model status."""
        if name in self.models:
            self.models[name].status = status
            self.models[name].updated_date = datetime.now().isoformat()
            
            if local_path:
                self.models[name].local_path = local_path
            
            self.save_to_file()
            logger.info(f"Updated model status: {name} -> {status.value}")
    
    def update_adapter_status(self, name: str, status: ModelStatus,
                             local_path: Optional[str] = None) -> None:
        """Update adapter status."""
        if name in self.adapters:
            self.adapters[name].status = status
            self.adapters[name].updated_date = datetime.now().isoformat()
            
            if local_path:
                self.adapters[name].local_path = local_path
            
            self.save_to_file()
            logger.info(f"Updated adapter status: {name} -> {status.value}")
    
    def save_to_file(self) -> None:
        """Save registry to file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            registry_data = {
                "models": {name: model.to_dict() for name, model in self.models.items()},
                "adapters": {name: adapter.to_dict() for name, adapter in self.adapters.items()},
                "metadata": {
                    "version": "1.0",
                    "updated": datetime.now().isoformat()
                }
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Registry saved to: {self.registry_path}")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def load_from_file(self) -> None:
        """Load registry from file."""
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load models
            for name, model_data in registry_data.get("models", {}).items():
                self.models[name] = ModelMetadata.from_dict(model_data)
            
            # Load adapters
            for name, adapter_data in registry_data.get("adapters", {}).items():
                self.adapters[name] = AdapterMetadata.from_dict(adapter_data)
            
            logger.info(f"Registry loaded from: {self.registry_path}")
            logger.info(f"Loaded {len(self.models)} models, {len(self.adapters)} adapters")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _register_builtin_models(self) -> None:
        """Register built-in model definitions."""
        
        # GLM-4.5-Air
        glm_metadata = ModelMetadata(
            name="glm-4.5-air",
            model_id="ZHIPU-AI/glm-4-9b-chat",
            architecture="GLMForCausalLM",
            parameter_count=9_000_000_000,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            max_sequence_length=8192,
            supports_chat=True,
            supports_instruct=True,
            supports_multilingual=True,
            torch_dtype="bfloat16",
            memory_requirements={"training": 24000, "inference": 18000},
            device_requirements={"min_gpu_memory_gb": 16},
            description="GLM-4.5-Air: Advanced bilingual conversational model",
            license="Custom",
            tags=["chat", "instruct", "bilingual", "chinese", "english"]
        )
        self.models["glm-4.5-air"] = glm_metadata
        
        # Qwen2.5-7B
        qwen_metadata = ModelMetadata(
            name="qwen2.5-7b",
            model_id="Qwen/Qwen2.5-7B",
            architecture="Qwen2ForCausalLM",
            parameter_count=7_000_000_000,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            max_sequence_length=32768,
            supports_chat=True,
            supports_instruct=True,
            supports_code=True,
            supports_multilingual=True,
            torch_dtype="bfloat16",
            memory_requirements={"training": 20000, "inference": 14000},
            device_requirements={"min_gpu_memory_gb": 12},
            description="Qwen2.5-7B: Powerful multilingual model with long context",
            license="Apache 2.0",
            tags=["chat", "instruct", "code", "multilingual", "long-context"]
        )
        self.models["qwen2.5-7b"] = qwen_metadata
        
        # Llama-3-8B
        llama_metadata = ModelMetadata(
            name="llama-3-8b",
            model_id="meta-llama/Meta-Llama-3-8B",
            architecture="LlamaForCausalLM",
            parameter_count=8_000_000_000,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            max_sequence_length=8192,
            supports_chat=True,
            supports_instruct=True,
            torch_dtype="bfloat16",
            memory_requirements={"training": 22000, "inference": 16000},
            device_requirements={"min_gpu_memory_gb": 14},
            description="Llama 3 8B: Meta's advanced instruction-following model",
            license="Llama 3 Community License",
            tags=["chat", "instruct", "general"]
        )
        self.models["llama-3-8b"] = llama_metadata
        
        # Mistral-7B
        mistral_metadata = ModelMetadata(
            name="mistral-7b",
            model_id="mistralai/Mistral-7B-v0.1",
            architecture="MistralForCausalLM",
            parameter_count=7_000_000_000,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            max_sequence_length=32768,
            supports_chat=True,
            supports_instruct=True,
            supports_code=True,
            torch_dtype="bfloat16",
            memory_requirements={"training": 20000, "inference": 14000},
            device_requirements={"min_gpu_memory_gb": 12},
            description="Mistral 7B: Efficient and powerful instruction model",
            license="Apache 2.0",
            tags=["chat", "instruct", "code", "efficient"]
        )
        self.models["mistral-7b"] = mistral_metadata
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        model_stats = {
            "total": len(self.models),
            "available": len([m for m in self.models.values() if m.status == ModelStatus.AVAILABLE]),
            "cached": len([m for m in self.models.values() if m.status == ModelStatus.CACHED]),
            "architectures": len(set(m.architecture for m in self.models.values()))
        }
        
        adapter_stats = {
            "total": len(self.adapters),
            "available": len([a for a in self.adapters.values() if a.status == ModelStatus.AVAILABLE]),
            "types": len(set(a.adapter_type for a in self.adapters.values()))
        }
        
        return {
            "models": model_stats,
            "adapters": adapter_stats,
            "registry_path": str(self.registry_path),
            "last_updated": datetime.now().isoformat()
        }

# Global registry instance
_global_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry

# Convenience functions
def register_model(metadata: ModelMetadata) -> None:
    """Register model with global registry."""
    registry = get_model_registry()
    registry.register_model(metadata)

def get_model_info(name: str) -> Optional[ModelMetadata]:
    """Get model information from global registry."""
    registry = get_model_registry()
    return registry.get_model(name)

def list_available_models() -> List[ModelMetadata]:
    """List all available models."""
    registry = get_model_registry()
    return registry.list_models(status=ModelStatus.AVAILABLE)