"""
Shared Evaluation Pipelines System.

This module provides reusable evaluation pipelines that can be composed
and configured for different evaluation scenarios across the platform.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from datetime import datetime
import numpy as np
from pathlib import Path
import json

from .metrics.unified_metrics import UnifiedMetricsComputer, MetricResult, MetricConfig
from ..data.processors.base import BaseProcessor
from ..inference.engines.base import BaseInferenceEngine
from ..core.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Evaluation pipeline stages."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    METRIC_COMPUTATION = "metric_computation"
    POSTPROCESSING = "postprocessing"
    REPORTING = "reporting"
    VALIDATION = "validation"


class PipelineMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class PipelineConfig:
    """Configuration for evaluation pipeline."""
    name: str
    mode: PipelineMode = PipelineMode.SEQUENTIAL
    batch_size: int = 32
    num_workers: int = 1
    cache_results: bool = True
    save_predictions: bool = False
    output_dir: Optional[Path] = None
    metrics: List[str] = field(default_factory=list)
    stages: List[PipelineStage] = field(default_factory=list)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    pipeline_name: str
    metrics: Dict[str, MetricResult]
    predictions: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BasePipelineStage(ABC):
    """
    Base class for pipeline stages.
    
    Provides interface for custom pipeline stages.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline stage."""
        self.name = name
        self.config = config or {}
        self.metrics = {}
    
    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute pipeline stage.
        
        Args:
            data: Input data
            context: Execution context
            
        Returns:
            Tuple of (processed_data, updated_context)
        """
        pass
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        return True
    
    def cleanup(self):
        """Cleanup stage resources."""
        pass


class DataLoadingStage(BasePipelineStage):
    """Data loading stage implementation."""
    
    def __init__(self, data_loader: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize data loading stage."""
        super().__init__("data_loading", config)
        self.data_loader = data_loader
    
    def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Load data."""
        try:
            # Load data using provided loader
            if hasattr(self.data_loader, 'load'):
                loaded_data = self.data_loader.load(**self.config)
            else:
                loaded_data = self.data_loader(data, **self.config)
            
            # Update context
            context['data_loaded'] = True
            context['data_shape'] = getattr(loaded_data, 'shape', None)
            context['data_size'] = len(loaded_data) if hasattr(loaded_data, '__len__') else None
            
            return loaded_data, context
            
        except Exception as e:
            raise EvaluationError(f"Data loading failed: {e}")


class PreprocessingStage(BasePipelineStage):
    """Preprocessing stage implementation."""
    
    def __init__(self, preprocessor: BaseProcessor, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing stage."""
        super().__init__("preprocessing", config)
        self.preprocessor = preprocessor
    
    def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Preprocess data."""
        try:
            # Apply preprocessing
            processed_data = self.preprocessor.process(data, **self.config)
            
            # Update context
            context['preprocessed'] = True
            context['preprocessing_method'] = self.preprocessor.__class__.__name__
            
            return processed_data, context
            
        except Exception as e:
            raise EvaluationError(f"Preprocessing failed: {e}")


class InferenceStage(BasePipelineStage):
    """Inference stage implementation."""
    
    def __init__(self, inference_engine: BaseInferenceEngine, config: Optional[Dict[str, Any]] = None):
        """Initialize inference stage."""
        super().__init__("inference", config)
        self.inference_engine = inference_engine
    
    def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Run inference."""
        try:
            # Run inference
            predictions = self.inference_engine.predict(data, **self.config)
            
            # Update context
            context['inference_completed'] = True
            context['model_name'] = getattr(self.inference_engine, 'model_name', 'unknown')
            context['predictions_shape'] = getattr(predictions, 'shape', None)
            
            # Store both data and predictions
            return {'data': data, 'predictions': predictions}, context
            
        except Exception as e:
            raise EvaluationError(f"Inference failed: {e}")


class MetricComputationStage(BasePipelineStage):
    """Metric computation stage implementation."""
    
    def __init__(self, metrics_computer: UnifiedMetricsComputer, config: Optional[Dict[str, Any]] = None):
        """Initialize metric computation stage."""
        super().__init__("metric_computation", config)
        self.metrics_computer = metrics_computer
    
    def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Compute metrics."""
        try:
            # Extract predictions and targets
            if isinstance(data, dict):
                predictions = data.get('predictions')
                targets = data.get('targets') or data.get('labels')
                original_data = data.get('data')
            else:
                raise ValueError("Expected dict with predictions and targets")
            
            if predictions is None or targets is None:
                raise ValueError("Missing predictions or targets for metric computation")
            
            # Compute metrics
            metric_names = self.config.get('metrics', context.get('metrics', []))
            
            for metric_name in metric_names:
                if metric_name not in self.metrics_computer.active_metrics:
                    self.metrics_computer.activate_metric(metric_name)
            
            self.metrics_computer.update(predictions, targets)
            metric_results = self.metrics_computer.compute(metrics=metric_names)
            
            # Update context
            context['metrics_computed'] = True
            context['metric_results'] = metric_results
            
            # Pass through data with metrics
            data['metrics'] = metric_results
            return data, context
            
        except Exception as e:
            raise EvaluationError(f"Metric computation failed: {e}")


class EvaluationPipeline:
    """
    Composable evaluation pipeline.
    
    Provides flexible pipeline construction with pluggable stages
    and comprehensive evaluation capabilities.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize evaluation pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Pipeline stages
        self.stages: List[BasePipelineStage] = []
        
        # Execution context
        self.context: Dict[str, Any] = {}
        
        # Results cache
        self.results_cache: Dict[str, PipelineResult] = {}
        
        # Metrics computer
        self.metrics_computer = UnifiedMetricsComputer()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize output directory if needed
        if config.output_dir:
            config.output_dir = Path(config.output_dir)
            config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EvaluationPipeline: {config.name}")
    
    def add_stage(self, stage: BasePipelineStage) -> 'EvaluationPipeline':
        """
        Add stage to pipeline.
        
        Args:
            stage: Pipeline stage
            
        Returns:
            Self for chaining
        """
        with self._lock:
            self.stages.append(stage)
            logger.info(f"Added stage {stage.name} to pipeline {self.config.name}")
        
        return self
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove stage from pipeline.
        
        Args:
            stage_name: Name of stage to remove
            
        Returns:
            True if removed successfully
        """
        with self._lock:
            for i, stage in enumerate(self.stages):
                if stage.name == stage_name:
                    self.stages.pop(i)
                    logger.info(f"Removed stage {stage_name} from pipeline {self.config.name}")
                    return True
        return False
    
    def execute(self, 
               data: Any,
               targets: Optional[Any] = None,
               context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute evaluation pipeline.
        
        Args:
            data: Input data
            targets: Ground truth targets (optional)
            context: Additional context
            
        Returns:
            Pipeline execution result
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        
        # Initialize context
        exec_context = self.context.copy()
        if context:
            exec_context.update(context)
        
        # Add targets to context if provided
        if targets is not None:
            if isinstance(data, dict):
                data['targets'] = targets
            else:
                data = {'data': data, 'targets': targets}
        
        # Add metrics configuration
        exec_context['metrics'] = self.config.metrics
        
        try:
            # Execute stages sequentially
            current_data = data
            
            for stage in self.stages:
                logger.debug(f"Executing stage: {stage.name}")
                
                # Validate input
                if not stage.validate_input(current_data):
                    raise EvaluationError(f"Input validation failed for stage {stage.name}")
                
                # Execute stage
                try:
                    current_data, exec_context = stage.execute(current_data, exec_context)
                except Exception as e:
                    if self.config.mode == PipelineMode.SEQUENTIAL:
                        raise
                    else:
                        errors.append(f"Stage {stage.name} failed: {e}")
                        logger.error(f"Stage {stage.name} failed: {e}")
            
            # Extract results
            metric_results = exec_context.get('metric_results', {})
            predictions = None
            
            if isinstance(current_data, dict):
                predictions = current_data.get('predictions')
            
            # Save predictions if configured
            if self.config.save_predictions and predictions is not None and self.config.output_dir:
                self._save_predictions(predictions, exec_context)
            
            # Create result
            result = PipelineResult(
                pipeline_name=self.config.name,
                metrics=metric_results,
                predictions=predictions if self.config.save_predictions else None,
                metadata=exec_context,
                execution_time=(datetime.now() - start_time).total_seconds(),
                errors=errors,
                warnings=warnings
            )
            
            # Cache result if configured
            if self.config.cache_results:
                cache_key = self._generate_cache_key(data, exec_context)
                self.results_cache[cache_key] = result
            
            logger.info(f"Pipeline {self.config.name} executed successfully in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline {self.config.name} execution failed: {e}")
            
            # Return partial result with error
            return PipelineResult(
                pipeline_name=self.config.name,
                metrics={},
                metadata=exec_context,
                execution_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)],
                warnings=warnings
            )
        
        finally:
            # Cleanup stages
            for stage in self.stages:
                try:
                    stage.cleanup()
                except Exception as e:
                    logger.warning(f"Stage {stage.name} cleanup failed: {e}")
    
    def execute_batch(self, 
                     data_batches: List[Any],
                     targets_batches: Optional[List[Any]] = None) -> List[PipelineResult]:
        """
        Execute pipeline on multiple batches.
        
        Args:
            data_batches: List of data batches
            targets_batches: List of target batches (optional)
            
        Returns:
            List of pipeline results
        """
        results = []
        
        if targets_batches is None:
            targets_batches = [None] * len(data_batches)
        
        for i, (data, targets) in enumerate(zip(data_batches, targets_batches)):
            logger.debug(f"Processing batch {i+1}/{len(data_batches)}")
            
            result = self.execute(data, targets, context={'batch_index': i})
            results.append(result)
        
        return results
    
    def _save_predictions(self, predictions: np.ndarray, context: Dict[str, Any]):
        """Save predictions to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.name}_predictions_{timestamp}.npy"
            filepath = self.config.output_dir / filename
            
            np.save(filepath, predictions)
            
            # Save metadata
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'pipeline': self.config.name,
                    'timestamp': timestamp,
                    'shape': predictions.shape,
                    'context': {k: str(v) for k, v in context.items()}
                }, f, indent=2)
            
            logger.info(f"Saved predictions to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    
    def _generate_cache_key(self, data: Any, context: Dict[str, Any]) -> str:
        """Generate cache key for results."""
        # Simple hash-based key (can be improved)
        import hashlib
        
        key_parts = [
            self.config.name,
            str(type(data)),
            str(context.get('model_name', '')),
            str(self.config.metrics)
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear results cache."""
        with self._lock:
            self.results_cache.clear()
    
    def get_stage(self, stage_name: str) -> Optional[BasePipelineStage]:
        """Get stage by name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def list_stages(self) -> List[str]:
        """List all stage names."""
        return [stage.name for stage in self.stages]


class PipelineBuilder:
    """
    Builder for creating evaluation pipelines.
    
    Provides fluent interface for pipeline construction.
    """
    
    def __init__(self, name: str):
        """Initialize pipeline builder."""
        self.config = PipelineConfig(name=name)
        self.stages = []
    
    def with_mode(self, mode: PipelineMode) -> 'PipelineBuilder':
        """Set pipeline mode."""
        self.config.mode = mode
        return self
    
    def with_batch_size(self, batch_size: int) -> 'PipelineBuilder':
        """Set batch size."""
        self.config.batch_size = batch_size
        return self
    
    def with_metrics(self, metrics: List[str]) -> 'PipelineBuilder':
        """Set metrics to compute."""
        self.config.metrics = metrics
        return self
    
    def with_output_dir(self, output_dir: Union[str, Path]) -> 'PipelineBuilder':
        """Set output directory."""
        self.config.output_dir = Path(output_dir)
        return self
    
    def add_data_loader(self, data_loader: Any, **kwargs) -> 'PipelineBuilder':
        """Add data loading stage."""
        stage = DataLoadingStage(data_loader, config=kwargs)
        self.stages.append(stage)
        self.config.stages.append(PipelineStage.DATA_LOADING)
        return self
    
    def add_preprocessor(self, preprocessor: BaseProcessor, **kwargs) -> 'PipelineBuilder':
        """Add preprocessing stage."""
        stage = PreprocessingStage(preprocessor, config=kwargs)
        self.stages.append(stage)
        self.config.stages.append(PipelineStage.PREPROCESSING)
        return self
    
    def add_inference_engine(self, inference_engine: BaseInferenceEngine, **kwargs) -> 'PipelineBuilder':
        """Add inference stage."""
        stage = InferenceStage(inference_engine, config=kwargs)
        self.stages.append(stage)
        self.config.stages.append(PipelineStage.INFERENCE)
        return self
    
    def add_metrics_computer(self, metrics_computer: UnifiedMetricsComputer = None, **kwargs) -> 'PipelineBuilder':
        """Add metric computation stage."""
        if metrics_computer is None:
            metrics_computer = UnifiedMetricsComputer()
        
        stage = MetricComputationStage(metrics_computer, config={'metrics': self.config.metrics, **kwargs})
        self.stages.append(stage)
        self.config.stages.append(PipelineStage.METRIC_COMPUTATION)
        return self
    
    def add_custom_stage(self, stage: BasePipelineStage) -> 'PipelineBuilder':
        """Add custom stage."""
        self.stages.append(stage)
        return self
    
    def save_predictions(self, save: bool = True) -> 'PipelineBuilder':
        """Enable/disable prediction saving."""
        self.config.save_predictions = save
        return self
    
    def cache_results(self, cache: bool = True) -> 'PipelineBuilder':
        """Enable/disable result caching."""
        self.config.cache_results = cache
        return self
    
    def build(self) -> EvaluationPipeline:
        """Build evaluation pipeline."""
        pipeline = EvaluationPipeline(self.config)
        
        for stage in self.stages:
            pipeline.add_stage(stage)
        
        return pipeline


# Predefined pipeline templates

def create_standard_evaluation_pipeline(
    name: str,
    model: Any,
    metrics: List[str],
    output_dir: Optional[Path] = None
) -> EvaluationPipeline:
    """
    Create standard evaluation pipeline.
    
    Args:
        name: Pipeline name
        model: Model for inference
        metrics: Metrics to compute
        output_dir: Output directory for results
        
    Returns:
        Configured evaluation pipeline
    """
    from ..inference.engines.base import BaseInferenceEngine
    
    # Wrap model if needed
    if not isinstance(model, BaseInferenceEngine):
        from ..inference.engines.model_wrapper import ModelWrapper
        model = ModelWrapper(model)
    
    builder = PipelineBuilder(name)
    
    if output_dir:
        builder.with_output_dir(output_dir)
    
    pipeline = (builder
                .with_metrics(metrics)
                .add_inference_engine(model)
                .add_metrics_computer()
                .save_predictions(output_dir is not None)
                .build())
    
    return pipeline


def create_calibration_evaluation_pipeline(
    name: str,
    model: Any,
    n_bins: int = 10,
    output_dir: Optional[Path] = None
) -> EvaluationPipeline:
    """
    Create calibration-focused evaluation pipeline.
    
    Args:
        name: Pipeline name
        model: Model for inference
        n_bins: Number of calibration bins
        output_dir: Output directory
        
    Returns:
        Calibration evaluation pipeline
    """
    metrics = ['accuracy', 'calibration', 'entropy', 'precision_recall_f1']
    
    pipeline = create_standard_evaluation_pipeline(
        name=name,
        model=model,
        metrics=metrics,
        output_dir=output_dir
    )
    
    # Configure calibration metric
    metrics_computer = pipeline.stages[-1].metrics_computer
    metrics_computer.activate_metric(
        'calibration',
        config=MetricConfig(
            name='calibration',
            type='calibration',
            custom_params={'n_bins': n_bins}
        )
    )
    
    return pipeline