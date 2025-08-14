"""
Training service for orchestrating model fine-tuning operations.

This service coordinates training workflows, manages training state,
and provides high-level training operations.
"""

from typing import Dict, List, Optional, Any, Type, Union
from pathlib import Path
import logging
import time
import threading

from .base import BaseService
from ..training.trainers import BaseTrainer, CalibratedTrainer
from ..models import ModelManager, ModelFactory
from ..data import BaseDataLoader, JsonlDataLoader
from ..evaluation.metrics import MetricsAggregator
from ..monitoring.collectors import TrainingMetricsCollector
from ..core.exceptions import TrainingError, ConfigurationError

logger = logging.getLogger(__name__)


class TrainingService(BaseService):
    """
    Service for managing model training operations.
    
    Provides high-level training orchestration, state management,
    and integration with monitoring and evaluation systems.
    """
    
    def __init__(self, **kwargs):
        """Initialize training service."""
        super().__init__(**kwargs)
        
        # Training components (injected)
        self.model_manager: Optional[ModelManager] = None
        self.model_factory: Optional[ModelFactory] = None
        self.metrics_aggregator: Optional[MetricsAggregator] = None
        self.metrics_collector: Optional[TrainingMetricsCollector] = None
        
        # Training state
        self.current_training_job: Optional[Dict[str, Any]] = None
        self.training_history: List[Dict[str, Any]] = []
        self.training_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.default_trainer_config = self.service_config.get('trainer', {})
        self.checkpoint_interval = self.service_config.get('checkpoint_interval', 500)
        self.max_concurrent_jobs = self.service_config.get('max_concurrent_jobs', 1)
    
    def get_required_dependencies(self) -> Dict[str, Type]:
        """Get required dependencies."""
        return {
            'model_manager': ModelManager,
            'model_factory': ModelFactory,
            'metrics_aggregator': MetricsAggregator,
            'metrics_collector': TrainingMetricsCollector
        }
    
    def get_event_handlers(self) -> Dict[str, callable]:
        """Get event handlers."""
        return {
            'TrainingStarted': self._on_training_started,
            'TrainingCompleted': self._on_training_completed,
            'TrainingFailed': self._on_training_failed,
            'CheckpointSaved': self._on_checkpoint_saved
        }
    
    def _initialize_service(self) -> None:
        """Initialize training service components."""
        # Validate required dependencies
        required_deps = ['model_manager', 'model_factory', 'metrics_aggregator']
        for dep in required_deps:
            if dep not in self.dependencies:
                raise ConfigurationError(f"Missing required dependency: {dep}")
        
        # Set up dependencies
        self.model_manager = self.dependencies['model_manager']
        self.model_factory = self.dependencies['model_factory']
        self.metrics_aggregator = self.dependencies['metrics_aggregator']
        self.metrics_collector = self.dependencies.get('metrics_collector')
        
        logger.info("Training service initialized with all dependencies")
    
    def _cleanup_service(self) -> None:
        """Clean up training service resources."""
        # Stop any running training
        if self.is_training():
            self.stop_training()
        
        # Clear state
        self.current_training_job = None
        self.training_history.clear()
    
    def _start_service(self) -> None:
        """Start training service operations."""
        logger.info("Training service started and ready to accept jobs")
    
    def _stop_service(self) -> None:
        """Stop training service operations."""
        # Stop any active training
        if self.is_training():
            self.stop_training()
        
        logger.info("Training service stopped")
    
    def start_training(self, 
                      training_config: Dict[str, Any],
                      model_config: Dict[str, Any],
                      data_config: Dict[str, Any],
                      job_id: Optional[str] = None) -> str:
        """
        Start a training job.
        
        Args:
            training_config: Training configuration
            model_config: Model configuration
            data_config: Data configuration
            job_id: Optional job ID
            
        Returns:
            Job ID
        """
        if not self.is_running:
            raise RuntimeError("Training service not running")
        
        if self.is_training():
            raise TrainingError("Training job already in progress")
        
        # Generate job ID if not provided
        if job_id is None:
            job_id = self._generate_job_id()
        
        # Create training job
        training_job = {
            'job_id': job_id,
            'status': 'initializing',
            'start_time': time.time(),
            'training_config': training_config,
            'model_config': model_config,
            'data_config': data_config,
            'metrics': {},
            'checkpoints': []
        }
        
        self.current_training_job = training_job
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(training_job,),
            daemon=False
        )
        self.training_thread.start()
        
        logger.info(f"Started training job {job_id}")
        
        # Publish event
        self._publish_event('TrainingJobStarted', {
            'job_id': job_id,
            'config': training_config
        })
        
        return job_id
    
    def stop_training(self) -> None:
        """Stop current training job."""
        if not self.is_training():
            logger.warning("No training job in progress")
            return
        
        # Update job status
        if self.current_training_job:
            self.current_training_job['status'] = 'stopping'
            
            # Signal training to stop (implementation would depend on trainer)
            logger.info(f"Stopping training job {self.current_training_job['job_id']}")
            
            # Wait for thread to complete
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=30)
                
                if self.training_thread.is_alive():
                    logger.error("Training thread did not stop gracefully")
            
            # Mark as stopped
            self.current_training_job['status'] = 'stopped'
            self.current_training_job['end_time'] = time.time()
            
            # Add to history and clear current job
            self.training_history.append(self.current_training_job)
            self.current_training_job = None
            
            # Publish event
            self._publish_event('TrainingJobStopped', {
                'job_id': self.current_training_job['job_id'] if self.current_training_job else 'unknown'
            })
    
    def _run_training(self, training_job: Dict[str, Any]) -> None:
        """Run training job (internal method)."""
        job_id = training_job['job_id']
        
        try:
            # Update status
            training_job['status'] = 'preparing'
            
            # Prepare model
            model = self._prepare_model(training_job['model_config'])
            training_job['model'] = model
            
            # Prepare data
            train_data, eval_data = self._prepare_data(training_job['data_config'])
            training_job['data_info'] = {
                'train_samples': len(train_data),
                'eval_samples': len(eval_data) if eval_data else 0
            }
            
            # Create trainer
            trainer = self._create_trainer(
                training_job['training_config'],
                model,
                train_data,
                eval_data
            )
            training_job['trainer'] = trainer
            
            # Start training
            training_job['status'] = 'training'
            self._publish_event('TrainingStarted', {'job_id': job_id})
            
            # Run training
            training_result = trainer.train()
            
            # Update job with results
            training_job['status'] = 'completed'
            training_job['end_time'] = time.time()
            training_job['result'] = training_result
            training_job['final_metrics'] = training_result.get('metrics', {})
            
            # Save final model
            if 'model_path' in training_job['training_config']:
                model_path = Path(training_job['training_config']['model_path'])
                self.model_manager.save_model(model, model_path)
                training_job['final_model_path'] = str(model_path)
            
            logger.info(f"Training job {job_id} completed successfully")
            
            # Publish completion event
            self._publish_event('TrainingCompleted', {
                'job_id': job_id,
                'result': training_result
            })
            
        except Exception as e:
            # Handle training failure
            training_job['status'] = 'failed'
            training_job['end_time'] = time.time()
            training_job['error'] = str(e)
            
            logger.error(f"Training job {job_id} failed: {e}")
            
            # Publish failure event
            self._publish_event('TrainingFailed', {
                'job_id': job_id,
                'error': str(e)
            })
        
        finally:
            # Clean up and move to history
            self.training_history.append(training_job)
            
            if self.current_training_job and self.current_training_job['job_id'] == job_id:
                self.current_training_job = None
    
    def _prepare_model(self, model_config: Dict[str, Any]) -> Any:
        """Prepare model for training."""
        model_type = model_config.get('model_type', 'llm_lora')
        
        # Load or create model
        if 'model_path' in model_config:
            model = self.model_manager.load_model(Path(model_config['model_path']))
        else:
            model = self.model_factory.create_model(model_type, model_config)
        
        return model
    
    def _prepare_data(self, data_config: Dict[str, Any]) -> tuple:
        """Prepare training and evaluation data."""
        # Load training data
        train_loader = JsonlDataLoader(data_config.get('loader', {}))
        train_data = train_loader.load(data_config['train_path'])
        
        # Load evaluation data if provided
        eval_data = None
        if 'eval_path' in data_config:
            eval_data = train_loader.load(data_config['eval_path'])
        
        return train_data, eval_data
    
    def _create_trainer(self, 
                       training_config: Dict[str, Any],
                       model: Any,
                       train_data: List[Dict[str, Any]],
                       eval_data: Optional[List[Dict[str, Any]]]) -> BaseTrainer:
        """Create trainer instance."""
        trainer_type = training_config.get('trainer_type', 'calibrated')
        
        # Merge with default config
        merged_config = {**self.default_trainer_config, **training_config}
        
        if trainer_type == 'calibrated':
            trainer = CalibratedTrainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=eval_data,
                config=merged_config,
                metrics_aggregator=self.metrics_aggregator
            )
        else:
            raise ConfigurationError(f"Unknown trainer type: {trainer_type}")
        
        return trainer
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        import uuid
        return f"train_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def is_training(self) -> bool:
        """Check if training is currently in progress."""
        return (self.current_training_job is not None and 
                self.current_training_job.get('status') in ['initializing', 'preparing', 'training'])
    
    def get_training_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get training status.
        
        Args:
            job_id: Optional job ID (current job if None)
            
        Returns:
            Training status information
        """
        if job_id is None:
            # Return current job status
            if self.current_training_job:
                return self._format_job_status(self.current_training_job)
            else:
                return {'status': 'idle', 'message': 'No training job in progress'}
        else:
            # Find job in history or current job
            target_job = None
            
            if self.current_training_job and self.current_training_job['job_id'] == job_id:
                target_job = self.current_training_job
            else:
                for job in self.training_history:
                    if job['job_id'] == job_id:
                        target_job = job
                        break
            
            if target_job:
                return self._format_job_status(target_job)
            else:
                return {'error': f'Job {job_id} not found'}
    
    def _format_job_status(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Format job status for external consumption."""
        status = {
            'job_id': job['job_id'],
            'status': job['status'],
            'start_time': job['start_time']
        }
        
        if 'end_time' in job:
            status['end_time'] = job['end_time']
            status['duration'] = job['end_time'] - job['start_time']
        elif job['status'] in ['training', 'preparing']:
            status['elapsed_time'] = time.time() - job['start_time']
        
        if 'data_info' in job:
            status['data_info'] = job['data_info']
        
        if 'metrics' in job and job['metrics']:
            status['current_metrics'] = job['metrics']
        
        if 'final_metrics' in job:
            status['final_metrics'] = job['final_metrics']
        
        if 'error' in job:
            status['error'] = job['error']
        
        if 'checkpoints' in job:
            status['checkpoints_count'] = len(job['checkpoints'])
        
        return status
    
    def get_training_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get training job history.
        
        Args:
            limit: Optional limit on number of jobs to return
            
        Returns:
            List of historical training jobs
        """
        jobs = self.training_history.copy()
        jobs.reverse()  # Most recent first
        
        if limit:
            jobs = jobs[:limit]
        
        return [self._format_job_status(job) for job in jobs]
    
    def _on_training_started(self, event_data: Dict[str, Any]) -> None:
        """Handle training started event."""
        job_id = event_data.get('job_id')
        logger.info(f"Received training started event for job {job_id}")
        
        # Start metrics collection if available
        if self.metrics_collector:
            self.metrics_collector.start_collection(job_id)
    
    def _on_training_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle training completed event."""
        job_id = event_data.get('job_id')
        logger.info(f"Training completed for job {job_id}")
        
        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.stop_collection(job_id)
    
    def _on_training_failed(self, event_data: Dict[str, Any]) -> None:
        """Handle training failed event."""
        job_id = event_data.get('job_id')
        error = event_data.get('error')
        logger.error(f"Training failed for job {job_id}: {error}")
        
        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.stop_collection(job_id)
    
    def _on_checkpoint_saved(self, event_data: Dict[str, Any]) -> None:
        """Handle checkpoint saved event."""
        job_id = event_data.get('job_id')
        checkpoint_path = event_data.get('checkpoint_path')
        
        # Add checkpoint to current job
        if (self.current_training_job and 
            self.current_training_job['job_id'] == job_id):
            self.current_training_job.setdefault('checkpoints', []).append({
                'path': checkpoint_path,
                'timestamp': time.time()
            })
    
    def _check_service_health(self) -> Dict[str, Any]:
        """Check training service health."""
        health = {
            'status': 'healthy',
            'details': {
                'current_training': self.current_training_job is not None,
                'training_history_count': len(self.training_history)
            }
        }
        
        # Check if training thread is stuck
        if (self.current_training_job and 
            self.training_thread and 
            not self.training_thread.is_alive()):
            health['status'] = 'unhealthy'
            health['details']['training_thread_issue'] = True
        
        return health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training service metrics."""
        metrics = super().get_metrics()
        
        metrics.update({
            'total_jobs': len(self.training_history),
            'current_job_active': self.is_training(),
            'successful_jobs': len([j for j in self.training_history if j.get('status') == 'completed']),
            'failed_jobs': len([j for j in self.training_history if j.get('status') == 'failed'])
        })
        
        if self.current_training_job:
            metrics['current_job'] = {
                'id': self.current_training_job['job_id'],
                'status': self.current_training_job['status'],
                'elapsed_time': time.time() - self.current_training_job['start_time']
            }
        
        return metrics