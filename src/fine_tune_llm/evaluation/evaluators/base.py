"""
Base evaluator interface for model evaluation.

This module provides the abstract base class for all evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...core.interfaces import BaseComponent


class BaseEvaluator(BaseComponent, ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, 
                 model: Optional[PreTrainedModel] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base evaluator.
        
        Args:
            model: Pre-trained model to evaluate
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device if provided
        if self.model:
            self.model = self.model.to(self.device)
    
    @abstractmethod
    def evaluate_single(self, 
                       text: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single text input.
        
        Args:
            text: Input text to evaluate
            metadata: Optional metadata for the input
            
        Returns:
            Evaluation results dictionary
        """
        pass
    
    @abstractmethod
    def evaluate_dataset(self, 
                        dataset: Union[List[Dict], Any],
                        batch_size: int = 1,
                        output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Evaluate an entire dataset.
        
        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size for evaluation
            output_path: Optional path to save results
            
        Returns:
            List of evaluation results
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, 
                       predictions: List[Any],
                       labels: List[Any],
                       **kwargs) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            **kwargs: Additional metric parameters
            
        Returns:
            Dictionary of computed metrics
        """
        pass
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        
        # Re-initialize model if needed
        if 'model_path' in config and not self.model:
            self._load_model(config['model_path'])
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            torch.cuda.empty_cache()
    
    @property
    def name(self) -> str:
        """Component name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def _load_model(self, model_path: str) -> None:
        """Load model from path."""
        # Implementation would load model
        pass