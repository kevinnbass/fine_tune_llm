"""
Type protocols for fine-tune LLM library.

Defines structural typing interfaces using Python protocols for
type checking and documentation without requiring inheritance.
"""

from typing import Protocol, Dict, Any, List, Optional, Union, runtime_checkable
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for configuration objects."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration."""
        ...

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects."""
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass."""
        ...
    
    def parameters(self) -> Any:
        """Get model parameters."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        ...
    
    def train(self, mode: bool = True) -> Any:
        """Set training mode."""
        ...
    
    def eval(self) -> Any:
        """Set evaluation mode."""
        ...

@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizer objects."""
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        ...
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Tokenize text."""
        ...
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...

@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for dataset objects."""
    
    def __len__(self) -> int:
        """Dataset length."""
        ...
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        ...

@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loader objects."""
    
    def __iter__(self):
        """Iterate over batches."""
        ...
    
    def __len__(self) -> int:
        """Number of batches."""
        ...

@runtime_checkable
class MetricsProtocol(Protocol):
    """Protocol for metrics objects."""
    
    def compute(self, predictions: Any, references: Any) -> Dict[str, float]:
        """Compute metrics."""
        ...
    
    def add_batch(self, predictions: Any, references: Any) -> None:
        """Add batch for computation."""
        ...
    
    def reset(self) -> None:
        """Reset metrics state."""
        ...

@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for trainer objects."""
    
    def train(self) -> Dict[str, Any]:
        """Train model."""
        ...
    
    def evaluate(self, eval_dataset: Optional[DatasetProtocol] = None) -> Dict[str, float]:
        """Evaluate model."""
        ...
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save model."""
        ...
    
    def save_state(self) -> None:
        """Save training state."""
        ...

@runtime_checkable
class PredictorProtocol(Protocol):
    """Protocol for predictor objects."""
    
    def predict(self, inputs: Any) -> Dict[str, Any]:
        """Make predictions."""
        ...
    
    def predict_proba(self, inputs: Any) -> Dict[str, Any]:
        """Make probabilistic predictions."""
        ...

@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for evaluator objects."""
    
    def evaluate(self, model: ModelProtocol, dataset: DatasetProtocol) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        ...
    
    def compute_metrics(self, predictions: Any, references: Any) -> Dict[str, float]:
        """Compute evaluation metrics."""
        ...

@runtime_checkable
class AuditorProtocol(Protocol):
    """Protocol for auditor objects."""
    
    def audit(self, model: ModelProtocol, dataset: DatasetProtocol, 
             config: ConfigProtocol) -> Dict[str, Any]:
        """Perform model audit."""
        ...
    
    def generate_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate audit report."""
        ...

@runtime_checkable
class DashboardProtocol(Protocol):
    """Protocol for dashboard objects."""
    
    def start(self) -> None:
        """Start dashboard."""
        ...
    
    def stop(self) -> None:
        """Stop dashboard."""
        ...
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update dashboard metrics."""
        ...

@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logger objects."""
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ...
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        ...
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ...
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        ...

@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache objects."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete cached value."""
        ...
    
    def clear(self) -> None:
        """Clear all cached values."""
        ...

@runtime_checkable
class StorageProtocol(Protocol):
    """Protocol for storage objects."""
    
    def save(self, data: Any, path: Union[str, Path]) -> None:
        """Save data to path."""
        ...
    
    def load(self, path: Union[str, Path]) -> Any:
        """Load data from path."""
        ...
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if path exists."""
        ...
    
    def delete(self, path: Union[str, Path]) -> None:
        """Delete path."""
        ...

@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for event bus objects."""
    
    async def publish(self, event: Any) -> None:
        """Publish event."""
        ...
    
    def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe to event type."""
        ...
    
    def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe from event type."""
        ...

@runtime_checkable
class ServiceProtocol(Protocol):
    """Protocol for service objects."""
    
    async def start(self) -> None:
        """Start service."""
        ...
    
    async def stop(self) -> None:
        """Stop service."""
        ...
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        ...
    
    @property
    def name(self) -> str:
        """Service name."""
        ...

@runtime_checkable
class PluginProtocol(Protocol):
    """Protocol for plugin objects."""
    
    def initialize(self, config: ConfigProtocol) -> None:
        """Initialize plugin."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...
    
    @property
    def name(self) -> str:
        """Plugin name."""
        ...
    
    @property
    def version(self) -> str:
        """Plugin version."""
        ...

@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for validator objects."""
    
    def validate(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate data and return errors."""
        ...
    
    def is_valid(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Check if data is valid."""
        ...

@runtime_checkable
class ProcessorProtocol(Protocol):
    """Protocol for data processor objects."""
    
    def process(self, data: Any) -> Any:
        """Process data."""
        ...
    
    def batch_process(self, batch: List[Any]) -> List[Any]:
        """Process batch of data."""
        ...

@runtime_checkable
class LoaderProtocol(Protocol):
    """Protocol for data loader objects."""
    
    def load(self, source: Union[str, Path]) -> Any:
        """Load data from source."""
        ...
    
    def supports_format(self, format_type: str) -> bool:
        """Check if format is supported."""
        ...

# Extended protocols for ML-specific types
@runtime_checkable
class TensorProtocol(Protocol):
    """Protocol for tensor objects."""
    
    @property
    def shape(self) -> tuple:
        """Tensor shape."""
        ...
    
    @property
    def dtype(self) -> Any:
        """Tensor data type."""
        ...
    
    def to(self, device: Any) -> Any:
        """Move tensor to device."""
        ...
    
    def cpu(self) -> Any:
        """Move tensor to CPU."""
        ...
    
    def numpy(self) -> Any:
        """Convert to numpy array."""
        ...

@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for optimizer objects."""
    
    def step(self) -> None:
        """Perform optimization step."""
        ...
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        ...
    
    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Parameter groups."""
        ...

@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for learning rate scheduler objects."""
    
    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate."""
        ...
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rate."""
        ...

@runtime_checkable
class CallbackProtocol(Protocol):
    """Protocol for training callback objects."""
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at beginning of training."""
        ...
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at end of training."""
        ...
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at beginning of epoch."""
        ...
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at end of epoch."""
        ...

# Type aliases for commonly used union types
ModelType = Union[PreTrainedModel, ModelProtocol]
TokenizerType = Union[PreTrainedTokenizer, TokenizerProtocol]
TensorType = Union[torch.Tensor, TensorProtocol]
DataType = Union[Dict[str, Any], List[Dict[str, Any]], DatasetProtocol]

# Export all protocols
__all__ = [
    # Basic protocols
    "ConfigProtocol",
    "ModelProtocol", 
    "TokenizerProtocol",
    "DatasetProtocol",
    "DataLoaderProtocol",
    "MetricsProtocol",
    
    # ML protocols
    "TrainerProtocol",
    "PredictorProtocol",
    "EvaluatorProtocol",
    "AuditorProtocol",
    "TensorProtocol",
    "OptimizerProtocol",
    "SchedulerProtocol",
    "CallbackProtocol",
    
    # Infrastructure protocols
    "DashboardProtocol",
    "LoggerProtocol", 
    "CacheProtocol",
    "StorageProtocol",
    "EventBusProtocol",
    "ServiceProtocol",
    "PluginProtocol",
    
    # Data protocols
    "ValidatorProtocol",
    "ProcessorProtocol",
    "LoaderProtocol",
    
    # Type aliases
    "ModelType",
    "TokenizerType",
    "TensorType",
    "DataType"
]