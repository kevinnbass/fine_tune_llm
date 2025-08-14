"""
Abstract base classes and interfaces for fine-tune LLM components.

This module defines the contracts that all components must implement,
ensuring consistent APIs and enabling dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Base component interface
class BaseComponent(ABC):
    """Root component interface for all fine-tune LLM components."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    @abstractmethod 
    def name(self) -> str:
        """Component name identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Component version."""
        pass

# Service layer interface
class BaseService(ABC):
    """Service layer interface for business logic components."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the service."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the service."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check service health status."""
        pass

# Factory pattern interface
class BaseFactory(ABC):
    """Factory pattern interface for component creation."""
    
    @abstractmethod
    def create(self, component_type: str, config: Dict[str, Any]) -> BaseComponent:
        """Create component instance."""
        pass
    
    @abstractmethod
    def register(self, component_type: str, component_class: type) -> None:
        """Register component type."""
        pass
    
    @abstractmethod
    def list_types(self) -> List[str]:
        """List available component types."""
        pass

# Strategy pattern interface
class BaseStrategy(ABC):
    """Strategy pattern interface for algorithm selection."""
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute strategy with context."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

# Observer pattern interface  
class BaseObserver(ABC):
    """Observer pattern interface for event handling."""
    
    @abstractmethod
    def update(self, event: Dict[str, Any]) -> None:
        """Handle event notification."""
        pass
    
    @property
    @abstractmethod
    def event_types(self) -> List[str]:
        """Event types this observer handles."""
        pass

# Domain-specific interfaces

class BaseTrainer(BaseComponent):
    """Training interface for model fine-tuning."""
    
    @abstractmethod
    def train(self, dataset: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model with dataset."""
        pass
    
    @abstractmethod
    def evaluate(self, dataset: Any) -> Dict[str, Any]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        pass

class BasePredictor(BaseComponent):
    """Prediction interface for model inference."""
    
    @abstractmethod
    def predict(self, inputs: Any) -> Dict[str, Any]:
        """Make predictions on inputs."""
        pass
    
    @abstractmethod
    def predict_proba(self, inputs: Any) -> Dict[str, Any]:
        """Make probabilistic predictions."""
        pass
    
    @abstractmethod
    def calibrate(self, calibration_data: Any) -> None:
        """Calibrate predictor with data."""
        pass

class BaseEvaluator(BaseComponent):
    """Evaluation interface for model assessment."""
    
    @abstractmethod
    def evaluate(self, predictions: Any, ground_truth: Any) -> Dict[str, Any]:
        """Evaluate predictions against ground truth."""
        pass
    
    @abstractmethod
    def compute_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        pass
    
    @abstractmethod
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate evaluation report."""
        pass

class BaseAuditor(BaseComponent):
    """Auditing interface for high-stakes evaluation."""
    
    @abstractmethod
    def audit(self, model: Any, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive model audit."""
        pass
    
    @abstractmethod
    def assess_fairness(self, predictions: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model fairness."""
        pass
    
    @abstractmethod
    def detect_bias(self, predictions: Any, groups: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential bias in predictions."""
        pass

class BaseMetric(BaseComponent):
    """Metric computation interface."""
    
    @abstractmethod
    def compute(self, predictions: Any, ground_truth: Any) -> float:
        """Compute metric value."""
        pass
    
    @abstractmethod
    def batch_compute(self, predictions: List[Any], ground_truth: List[Any]) -> List[float]:
        """Compute metric for batch of examples."""
        pass
    
    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        pass

class BaseLoader(BaseComponent):
    """Data loading interface."""
    
    @abstractmethod
    def load(self, source: Union[str, Path]) -> Any:
        """Load data from source."""
        pass
    
    @abstractmethod
    def save(self, data: Any, destination: Union[str, Path]) -> None:
        """Save data to destination."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data format."""
        pass

class BaseProcessor(BaseComponent):
    """Data processing interface."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data."""
        pass
    
    @abstractmethod
    def batch_process(self, data: List[Any]) -> List[Any]:
        """Process batch of data."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get expected data schema."""
        pass

class BaseValidator(BaseComponent):
    """Data validation interface."""
    
    @abstractmethod
    def validate(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate data against schema."""
        pass
    
    @abstractmethod
    def is_valid(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Check if data is valid."""
        pass
    
    @abstractmethod
    def get_validation_errors(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get detailed validation errors."""
        pass

# Hexagonal architecture ports

class FileSystemPort(ABC):
    """Port for file system operations."""
    
    @abstractmethod
    def read_file(self, file_path: Union[str, Path], mode: str = 'r') -> Union[str, bytes]:
        """Read file content."""
        pass
    
    @abstractmethod
    def write_file(self, file_path: Union[str, Path], content: Union[str, bytes], mode: str = 'w') -> bool:
        """Write content to file."""
        pass
    
    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if path exists."""
        pass
    
    @abstractmethod
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete file."""
        pass

class DatabasePort(ABC):
    """Port for database operations."""
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query."""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, params: Optional[tuple] = None) -> int:
        """Execute INSERT/UPDATE/DELETE command."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

class APIPort(ABC):
    """Port for API operations."""
    
    @abstractmethod
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make GET request."""
        pass
    
    @abstractmethod
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make POST request."""
        pass
    
    @abstractmethod
    def health_check(self, endpoint: str = '/health') -> Dict[str, Any]:
        """Perform health check."""
        pass

class CachePort(ABC):
    """Port for caching operations."""
    
    @abstractmethod
    def get(self, key: Union[str, tuple, dict]) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: Union[str, tuple, dict], value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: Union[str, tuple, dict]) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: Union[str, tuple, dict]) -> bool:
        """Check if key exists."""
        pass

# Export all interfaces
__all__ = [
    "BaseComponent",
    "BaseService", 
    "BaseFactory",
    "BaseStrategy",
    "BaseObserver",
    "BaseTrainer",
    "BasePredictor",
    "BaseEvaluator",
    "BaseAuditor",
    "BaseMetric",
    "BaseLoader",
    "BaseProcessor", 
    "BaseValidator",
    # Hexagonal architecture ports
    "FileSystemPort",
    "DatabasePort", 
    "APIPort",
    "CachePort"
]