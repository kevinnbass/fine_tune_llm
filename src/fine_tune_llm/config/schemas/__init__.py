"""
Configuration schemas for fine-tune LLM library.

Defines structured configuration schemas with validation rules,
defaults, and documentation for all system components.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

class ModelType(Enum):
    """Supported model types."""
    GLM_4_5_AIR = "glm-4.5-air"
    QWEN_2_5_7B = "qwen2.5-7b"
    LLAMA_3_8B = "llama-3-8b"
    MISTRAL_7B = "mistral-7b"

class LoRAMethod(Enum):
    """LoRA fine-tuning methods."""
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"

class SchedulerType(Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"

@dataclass
class BaseConfig:
    """Base configuration class with common properties."""
    
    version: str = "2.0.0"
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create configuration from dictionary."""
        # Filter out unknown fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

@dataclass
class ModelConfig(BaseConfig):
    """Model configuration schema."""
    
    # Model selection
    selected_model: str = ModelType.GLM_4_5_AIR.value
    model_id: str = "ZHIPU-AI/glm-4-9b-chat"
    tokenizer_id: Optional[str] = None  # Defaults to model_id if None
    
    # Model loading options
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = False
    
    # Model-specific options
    model_options: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "glm-4.5-air": {
            "model_id": "ZHIPU-AI/glm-4-9b-chat",
            "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "attention_type": "grouped_query"
        },
        "qwen2.5-7b": {
            "model_id": "Qwen/Qwen2.5-7B",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attention_type": "multi_head"
        },
        "llama-3-8b": {
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attention_type": "multi_head"
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-v0.1",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attention_type": "grouped_query"
        }
    })

@dataclass
class LoRAConfig(BaseConfig):
    """LoRA configuration schema."""
    
    # LoRA method and parameters
    method: str = LoRAMethod.LORA.value
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.1
    bias: str = "none"  # "none", "all", or "lora_only"
    
    # Target modules (model-specific)
    target_modules: List[str] = field(default_factory=list)
    
    # Quantization settings
    quantization: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "bits": 4,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quant": True
    })
    
    # DoRA specific settings
    dora_settings: Dict[str, Any] = field(default_factory=lambda: {
        "decompose_both": True,
        "use_rslora": False
    })
    
    # Task type for PEFT
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration schema."""
    
    # Basic training parameters
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    
    # Optimization
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "type": SchedulerType.COSINE.value,
        "warmup_ratio": 0.03,
        "warmup_steps": 0,
        "num_cycles": 0.5
    })
    
    # Training precision
    fp16: bool = False
    bf16: bool = True
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Early stopping
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "patience": 3,
        "threshold": 0.001,
        "metric": "eval_loss",
        "greater_is_better": False
    })
    
    # Advanced training features
    advanced_features: Dict[str, Any] = field(default_factory=lambda: {
        "calibration_monitoring": True,
        "abstention_loss": True,
        "conformal_prediction": True,
        "risk_controlled_training": True
    })

@dataclass
class AbstentionLossConfig(BaseConfig):
    """Abstention loss configuration."""
    
    enabled: bool = True
    confidence_threshold: float = 0.7
    abstention_penalty: float = 0.3
    uncertainty_weight: float = 0.1
    temperature_scaling: bool = True
    
    # Loss function weights
    classification_weight: float = 1.0
    confidence_weight: float = 0.1
    entropy_weight: float = 0.05

@dataclass
class CalibrationConfig(BaseConfig):
    """Calibration monitoring configuration."""
    
    enabled: bool = True
    adjustment_threshold: float = 0.05
    lr_reduction_factor: float = 0.8
    calibration_frequency: int = 100
    
    # Metrics to monitor
    metrics: List[str] = field(default_factory=lambda: ["ece", "mce", "brier_score"])
    
    # Calibration methods
    calibration_methods: List[str] = field(default_factory=lambda: ["temperature_scaling", "platt_scaling"])

@dataclass
class DataConfig(BaseConfig):
    """Data configuration schema."""
    
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    cache_dir: str = "data/cache"
    
    # Data processing
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "remove_duplicates": True,
        "filter_length": True,
        "min_length": 10,
        "max_length": 2048,
        "shuffle": True,
        "seed": 42
    })
    
    # Data splitting
    split_config: Dict[str, Any] = field(default_factory=lambda: {
        "train_ratio": 0.8,
        "eval_ratio": 0.1,
        "test_ratio": 0.1,
        "stratify": False,
        "seed": 42
    })
    
    # Data validation
    validation: Dict[str, Any] = field(default_factory=lambda: {
        "required_fields": ["text", "output"],
        "max_errors": 100,
        "error_handling": "skip"  # "skip", "stop", "fix"
    })

@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration schema."""
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Batch processing
    batch_size: int = 1
    max_batch_size: int = 16
    
    # Conformal prediction
    conformal_prediction: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "alpha": 0.1,  # 90% confidence
        "method": "lac",  # "lac", "aps", "raps"
        "calibration_size": 0.2
    })
    
    # Risk control
    risk_control: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "risk_level": 0.1,
        "cost_matrix": None,  # Will be set based on task
        "abstention_threshold": 0.7
    })

@dataclass
class EvaluationConfig(BaseConfig):
    """Evaluation configuration schema."""
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1",
        "ece", "mce", "brier_score",
        "abstention_rate", "coverage", "risk"
    ])
    
    # Calibration assessment
    calibration: Dict[str, Any] = field(default_factory=lambda: {
        "n_bins": 10,
        "strategy": "uniform",  # "uniform", "quantile"
        "confidence_intervals": True
    })
    
    # High-stakes auditing
    auditing: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "fairness_assessment": True,
        "bias_detection": True,
        "robustness_testing": True,
        "interpretability_analysis": True
    })
    
    # Reporting
    reporting: Dict[str, Any] = field(default_factory=lambda: {
        "format": "json",  # "json", "html", "pdf"
        "include_plots": True,
        "save_predictions": True,
        "confidence_analysis": True
    })

@dataclass
class MonitoringConfig(BaseConfig):
    """Monitoring configuration schema."""
    
    # Dashboard settings
    dashboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "port": 8501,
        "host": "localhost",
        "update_interval": 5,
        "max_history": 1000
    })
    
    # Metrics collection
    metrics_collection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "collection_interval": 30,
        "storage_backend": "file",  # "file", "redis", "prometheus"
        "retention_days": 30
    })
    
    # Alerting
    alerting: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "channels": ["log"],  # "log", "email", "slack"
        "thresholds": {
            "high_loss": 5.0,
            "low_accuracy": 0.5,
            "high_error_rate": 0.1
        }
    })

@dataclass
class LLMLoRAConfig(BaseConfig):
    """Complete LLM LoRA configuration combining all components."""
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Advanced configurations
    abstention_loss: AbstentionLossConfig = field(default_factory=AbstentionLossConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    
    # Global settings
    experiment_name: str = "llm_lora_experiment"
    output_dir: str = "artifacts/models/llm_lora"
    cache_dir: str = "cache"
    log_level: str = "INFO"
    
    # Resource management
    resources: Dict[str, Any] = field(default_factory=lambda: {
        "gpu_memory_fraction": 0.9,
        "cpu_threads": None,  # Auto-detect
        "mixed_precision": True,
        "gradient_checkpointing": True
    })
    
    def __post_init__(self):
        """Post-initialization to sync model-specific settings."""
        # Sync target modules based on selected model
        if self.model.selected_model in self.model.model_options:
            model_info = self.model.model_options[self.model.selected_model]
            self.lora.target_modules = model_info.get("target_modules", [])
            
            # Update model_id if not explicitly set
            if not self.model.model_id or self.model.model_id == "ZHIPU-AI/glm-4-9b-chat":
                self.model.model_id = model_info.get("model_id", self.model.model_id)

# JSON Schema definitions for validation
LLM_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "selected_model": {
                    "type": "string",
                    "enum": [model.value for model in ModelType]
                },
                "model_id": {
                    "type": "string",
                    "minLength": 1
                },
                "torch_dtype": {
                    "type": "string",
                    "enum": ["float32", "float16", "bfloat16"]
                }
            },
            "required": ["selected_model", "model_id"]
        },
        "lora": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [method.value for method in LoRAMethod]
                },
                "r": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 256
                },
                "lora_alpha": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 512
                },
                "lora_dropout": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["r", "lora_alpha"]
        },
        "training": {
            "type": "object",
            "properties": {
                "num_epochs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100
                },
                "learning_rate": {
                    "type": "number",
                    "minimum": 1e-8,
                    "maximum": 1.0
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1024
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192
                }
            },
            "required": ["num_epochs", "learning_rate", "batch_size"]
        }
    },
    "required": ["model", "lora", "training"]
}

# Export all configuration classes and schemas
__all__ = [
    "BaseConfig",
    "ModelConfig",
    "LoRAConfig", 
    "TrainingConfig",
    "AbstentionLossConfig",
    "CalibrationConfig",
    "DataConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "MonitoringConfig",
    "LLMLoRAConfig",
    "ModelType",
    "LoRAMethod",
    "SchedulerType",
    "LLM_CONFIG_SCHEMA"
]