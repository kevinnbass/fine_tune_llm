"""
Fine-Tune LLM: Advanced LLM Fine-tuning with Statistical Guarantees

A comprehensive framework for fine-tuning large language models with:
- LoRA (Low-Rank Adaptation) parameter-efficient training
- Calibration-aware training with ECE/MCE monitoring
- Conformal prediction with statistical guarantees
- Risk-controlled predictions with abstention
- Advanced metrics and uncertainty quantification
- Real-time monitoring and interactive interfaces

Core Components:
- config: Configuration management and validation
- models: Model loading, management, and factories
- training: Training pipelines and calibration-aware trainers
- inference: Inference engines with risk control and conformal prediction
- evaluation: Advanced metrics, calibration, and audit systems
- monitoring: Real-time dashboards and monitoring tools
- utils: Shared utilities and helper functions
"""

__version__ = "1.0.0"
__author__ = "Fine-Tune LLM Team"

# Core API exports
from .config import ConfigManager, ValidationError
from .models import ModelFactory, ModelManager
from .training import TrainerFactory, CalibratedTrainer
from .inference import PredictorFactory, RiskControlledInference
from .evaluation import EvaluatorFactory, AdvancedMetrics
from .monitoring import DashboardManager, MonitoringSystem

__all__ = [
    # Configuration
    "ConfigManager",
    "ValidationError",
    
    # Models
    "ModelFactory", 
    "ModelManager",
    
    # Training
    "TrainerFactory",
    "CalibratedTrainer",
    
    # Inference
    "PredictorFactory",
    "RiskControlledInference",
    
    # Evaluation
    "EvaluatorFactory", 
    "AdvancedMetrics",
    
    # Monitoring
    "DashboardManager",
    "MonitoringSystem",
]