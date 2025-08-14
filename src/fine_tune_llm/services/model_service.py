"""
Model service for managing model lifecycle and operations.

This service handles model loading, saving, versioning, and
provides high-level model management operations.
"""

from typing import Dict, List, Optional, Any, Type, Union
from pathlib import Path
import logging
import time
import hashlib
import json

from .base import BaseService
from ..models import ModelManager, ModelFactory, ModelRegistry
from ..core.exceptions import ModelError, ValidationError

logger = logging.getLogger(__name__)


class ModelService(BaseService):
    """
    Service for managing model operations and lifecycle.
    
    Provides high-level model management including loading, saving,
    versioning, validation, and metadata management.
    """
    
    def __init__(self, **kwargs):
        """Initialize model service."""
        super().__init__(**kwargs)
        
        # Model management components (injected)
        self.model_manager: Optional[ModelManager] = None
        self.model_factory: Optional[ModelFactory] = None
        self.model_registry: Optional[ModelRegistry] = None
        
        # Model registry and caching
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.models_directory = Path(self.service_config.get('models_directory', 'artifacts/models'))
        self.auto_cleanup_interval = self.service_config.get('auto_cleanup_interval', 3600)  # 1 hour
        self.max_loaded_models = self.service_config.get('max_loaded_models', 5)
        self.enable_model_versioning = self.service_config.get('enable_versioning', True)
        
        # Ensure models directory exists
        self.models_directory.mkdir(parents=True, exist_ok=True)
    
    def get_required_dependencies(self) -> Dict[str, Type]:
        """Get required dependencies."""
        return {
            'model_manager': ModelManager,
            'model_factory': ModelFactory,
            'model_registry': ModelRegistry
        }
    
    def get_event_handlers(self) -> Dict[str, callable]:
        """Get event handlers."""
        return {
            'ModelLoaded': self._on_model_loaded,
            'ModelSaved': self._on_model_saved,
            'ModelDeleted': self._on_model_deleted,
            'TrainingCompleted': self._on_training_completed
        }
    
    def _initialize_service(self) -> None:
        """Initialize model service components."""
        # Set up dependencies
        self.model_manager = self.dependencies['model_manager']
        self.model_factory = self.dependencies['model_factory']
        self.model_registry = self.dependencies['model_registry']
        
        # Load existing model metadata
        self._load_model_metadata()
        
        logger.info("Model service initialized")
    
    def _cleanup_service(self) -> None:
        """Clean up model service resources."""
        # Unload all models
        self._unload_all_models()
        
        # Save metadata
        self._save_model_metadata()
        
        # Clear state
        self.loaded_models.clear()
        self.model_metadata.clear()
    
    def _start_service(self) -> None:
        """Start model service operations."""
        # Start auto-cleanup if configured
        if self.auto_cleanup_interval > 0:
            # In a real implementation, you'd set up a scheduled task
            logger.info("Model service started with auto-cleanup enabled")
        else:
            logger.info("Model service started")
    
    def _stop_service(self) -> None:
        """Stop model service operations."""
        # Stop any background tasks
        logger.info("Model service stopped")
    
    def load_model(self, 
                   model_path: Union[str, Path],
                   model_id: Optional[str] = None,
                   load_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to model files
            model_id: Optional model identifier
            load_config: Optional loading configuration
            
        Returns:
            Model ID for the loaded model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ModelError(f"Model path not found: {model_path}")
        
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(model_path)
        
        # Check if already loaded
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return model_id
        
        # Check model cache limit
        if len(self.loaded_models) >= self.max_loaded_models:
            self._evict_least_used_model()
        
        try:
            # Load model using model manager
            model = self.model_manager.load_model(model_path, load_config or {})
            
            # Create model entry
            model_entry = {
                'model': model,
                'model_id': model_id,
                'model_path': str(model_path),
                'load_time': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'config': load_config or {},
                'metadata': self._extract_model_metadata(model, model_path)
            }
            
            # Store in loaded models
            self.loaded_models[model_id] = model_entry
            
            # Update registry
            self.model_registry.register_model(model_id, {
                'path': str(model_path),
                'type': model_entry['metadata'].get('model_type'),
                'load_time': model_entry['load_time']
            })
            
            logger.info(f"Successfully loaded model {model_id} from {model_path}")
            
            # Publish event
            self._publish_event('ModelLoaded', {
                'model_id': model_id,
                'model_path': str(model_path),
                'metadata': model_entry['metadata']
            })
            
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def save_model(self, 
                   model_id: str,
                   save_path: Union[str, Path],
                   save_config: Optional[Dict[str, Any]] = None,
                   version: Optional[str] = None) -> Path:
        """
        Save a loaded model to disk.
        
        Args:
            model_id: ID of loaded model
            save_path: Path to save model
            save_config: Optional saving configuration
            version: Optional version string
            
        Returns:
            Path where model was saved
        """
        if model_id not in self.loaded_models:
            raise ModelError(f"Model {model_id} not loaded")
        
        model_entry = self.loaded_models[model_id]
        model = model_entry['model']
        save_path = Path(save_path)
        
        # Create versioned path if versioning enabled
        if self.enable_model_versioning and version:
            save_path = save_path / f"v{version}"
        
        try:
            # Save model using model manager
            actual_save_path = self.model_manager.save_model(
                model, 
                save_path, 
                save_config or {}
            )
            
            # Update model metadata
            metadata = {
                'model_id': model_id,
                'original_path': model_entry['model_path'],
                'save_path': str(actual_save_path),
                'save_time': time.time(),
                'version': version,
                'config': save_config or {},
                'model_type': model_entry['metadata'].get('model_type'),
                'model_hash': self._compute_model_hash(actual_save_path)
            }
            
            # Store metadata
            self.model_metadata[model_id] = metadata
            
            # Save metadata file
            metadata_path = actual_save_path / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Successfully saved model {model_id} to {actual_save_path}")
            
            # Publish event
            self._publish_event('ModelSaved', {
                'model_id': model_id,
                'save_path': str(actual_save_path),
                'version': version,
                'metadata': metadata
            })
            
            return actual_save_path
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise ModelError(f"Failed to save model: {e}")
    
    def create_model(self, 
                     model_type: str,
                     model_config: Dict[str, Any],
                     model_id: Optional[str] = None) -> str:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create
            model_config: Model configuration
            model_id: Optional model identifier
            
        Returns:
            Model ID for the created model
        """
        if model_id is None:
            model_id = self._generate_model_id_from_config(model_type, model_config)
        
        # Check if model already exists
        if model_id in self.loaded_models:
            raise ModelError(f"Model {model_id} already exists")
        
        try:
            # Create model using factory
            model = self.model_factory.create_model(model_type, model_config)
            
            # Create model entry
            model_entry = {
                'model': model,
                'model_id': model_id,
                'model_path': None,  # Not loaded from disk
                'load_time': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'config': model_config,
                'metadata': {
                    'model_type': model_type,
                    'created_time': time.time(),
                    'created_from_config': True
                }
            }
            
            # Store in loaded models
            self.loaded_models[model_id] = model_entry
            
            # Register model
            self.model_registry.register_model(model_id, {
                'type': model_type,
                'config': model_config,
                'created_time': model_entry['load_time']
            })
            
            logger.info(f"Successfully created model {model_id} of type {model_type}")
            
            # Publish event
            self._publish_event('ModelCreated', {
                'model_id': model_id,
                'model_type': model_type,
                'config': model_config
            })
            
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            raise ModelError(f"Failed to create model: {e}")
    
    def get_model(self, model_id: str) -> Any:
        """
        Get a loaded model instance.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model instance
        """
        if model_id not in self.loaded_models:
            raise ModelError(f"Model {model_id} not loaded")
        
        model_entry = self.loaded_models[model_id]
        
        # Update access statistics
        model_entry['access_count'] += 1
        model_entry['last_accessed'] = time.time()
        
        return model_entry['model']
    
    def unload_model(self, model_id: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model identifier
        """
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded")
            return
        
        # Remove from loaded models
        model_entry = self.loaded_models.pop(model_id)
        
        # Unregister from registry
        self.model_registry.unregister_model(model_id)
        
        logger.info(f"Unloaded model {model_id}")
        
        # Publish event
        self._publish_event('ModelUnloaded', {
            'model_id': model_id,
            'access_count': model_entry.get('access_count', 0)
        })
    
    def list_models(self, 
                   include_loaded_only: bool = False,
                   include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Args:
            include_loaded_only: Only return loaded models
            include_metadata: Include detailed metadata
            
        Returns:
            List of model information
        """
        models = []
        
        # Add loaded models
        for model_id, model_entry in self.loaded_models.items():
            model_info = {
                'model_id': model_id,
                'status': 'loaded',
                'load_time': model_entry['load_time'],
                'access_count': model_entry['access_count'],
                'last_accessed': model_entry['last_accessed']
            }
            
            if include_metadata:
                model_info['metadata'] = model_entry['metadata'].copy()
                model_info['config'] = model_entry['config'].copy()
                model_info['model_path'] = model_entry['model_path']
            
            models.append(model_info)
        
        # Add unloaded models if requested
        if not include_loaded_only:
            # Scan models directory for saved models
            saved_models = self._scan_saved_models()
            for model_info in saved_models:
                if model_info['model_id'] not in self.loaded_models:
                    model_info['status'] = 'saved'
                    models.append(model_info)
        
        return models
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary
        """
        if model_id in self.loaded_models:
            model_entry = self.loaded_models[model_id]
            return {
                'model_id': model_id,
                'status': 'loaded',
                'model_path': model_entry['model_path'],
                'load_time': model_entry['load_time'],
                'access_count': model_entry['access_count'],
                'last_accessed': model_entry['last_accessed'],
                'config': model_entry['config'].copy(),
                'metadata': model_entry['metadata'].copy()
            }
        elif model_id in self.model_metadata:
            metadata = self.model_metadata[model_id]
            return {
                'model_id': model_id,
                'status': 'saved',
                'metadata': metadata.copy()
            }
        else:
            raise ModelError(f"Model {model_id} not found")
    
    def delete_model(self, 
                    model_id: str,
                    delete_files: bool = False) -> None:
        """
        Delete a model.
        
        Args:
            model_id: Model identifier
            delete_files: Whether to delete model files from disk
        """
        # Unload if loaded
        if model_id in self.loaded_models:
            self.unload_model(model_id)
        
        # Delete files if requested
        if delete_files and model_id in self.model_metadata:
            metadata = self.model_metadata[model_id]
            if 'save_path' in metadata:
                model_path = Path(metadata['save_path'])
                if model_path.exists():
                    import shutil
                    shutil.rmtree(model_path)
                    logger.info(f"Deleted model files at {model_path}")
        
        # Remove metadata
        if model_id in self.model_metadata:
            del self.model_metadata[model_id]
        
        logger.info(f"Deleted model {model_id}")
        
        # Publish event
        self._publish_event('ModelDeleted', {
            'model_id': model_id,
            'deleted_files': delete_files
        })
    
    def _generate_model_id(self, model_path: Path) -> str:
        """Generate model ID from path."""
        # Use path and timestamp to create unique ID
        path_hash = hashlib.md5(str(model_path).encode()).hexdigest()[:8]
        return f"model_{path_hash}_{int(time.time())}"
    
    def _generate_model_id_from_config(self, model_type: str, config: Dict[str, Any]) -> str:
        """Generate model ID from type and config."""
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{model_type}_{config_hash}_{int(time.time())}"
    
    def _extract_model_metadata(self, model: Any, model_path: Path) -> Dict[str, Any]:
        """Extract metadata from loaded model."""
        metadata = {
            'model_path': str(model_path),
            'file_size': self._get_path_size(model_path),
            'load_timestamp': time.time()
        }
        
        # Try to extract model-specific metadata
        try:
            if hasattr(model, 'config'):
                metadata['model_config'] = model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
            
            if hasattr(model, 'num_parameters'):
                metadata['num_parameters'] = model.num_parameters()
            
            if hasattr(model, 'dtype'):
                metadata['dtype'] = str(model.dtype)
                
        except Exception as e:
            logger.warning(f"Could not extract full metadata: {e}")
        
        return metadata
    
    def _get_path_size(self, path: Path) -> int:
        """Get total size of path (file or directory)."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0
    
    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute hash of model files."""
        import hashlib
        
        hash_md5 = hashlib.md5()
        
        if model_path.is_file():
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        elif model_path.is_dir():
            # Hash all files in directory
            for file_path in sorted(model_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _evict_least_used_model(self) -> None:
        """Evict least recently used model."""
        if not self.loaded_models:
            return
        
        # Find least recently used model
        lru_model_id = min(
            self.loaded_models.keys(),
            key=lambda mid: self.loaded_models[mid]['last_accessed']
        )
        
        logger.info(f"Evicting least used model: {lru_model_id}")
        self.unload_model(lru_model_id)
    
    def _unload_all_models(self) -> None:
        """Unload all models."""
        model_ids = list(self.loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
    
    def _load_model_metadata(self) -> None:
        """Load model metadata from disk."""
        metadata_file = self.models_directory / 'models_metadata.json'
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.model_metadata)} models")
            except Exception as e:
                logger.error(f"Failed to load model metadata: {e}")
    
    def _save_model_metadata(self) -> None:
        """Save model metadata to disk."""
        metadata_file = self.models_directory / 'models_metadata.json'
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.model_metadata, f, indent=2, default=str)
            logger.info("Saved model metadata")
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
    
    def _scan_saved_models(self) -> List[Dict[str, Any]]:
        """Scan models directory for saved models."""
        saved_models = []
        
        try:
            for model_dir in self.models_directory.iterdir():
                if model_dir.is_dir():
                    metadata_file = model_dir / 'model_metadata.json'
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            model_info = {
                                'model_id': metadata.get('model_id', model_dir.name),
                                'model_path': str(model_dir),
                                'metadata': metadata
                            }
                            saved_models.append(model_info)
                            
                        except Exception as e:
                            logger.warning(f"Could not load metadata for {model_dir}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning saved models: {e}")
        
        return saved_models
    
    def _on_model_loaded(self, event_data: Dict[str, Any]) -> None:
        """Handle model loaded event."""
        model_id = event_data.get('model_id')
        logger.debug(f"Model loaded event received for {model_id}")
    
    def _on_model_saved(self, event_data: Dict[str, Any]) -> None:
        """Handle model saved event."""
        model_id = event_data.get('model_id')
        logger.debug(f"Model saved event received for {model_id}")
    
    def _on_model_deleted(self, event_data: Dict[str, Any]) -> None:
        """Handle model deleted event."""
        model_id = event_data.get('model_id')
        logger.debug(f"Model deleted event received for {model_id}")
    
    def _on_training_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle training completed event."""
        job_id = event_data.get('job_id')
        logger.info(f"Training completed event received for job {job_id}")
        
        # Auto-save trained model if configured
        auto_save_config = self.service_config.get('auto_save_trained_models', {})
        if auto_save_config.get('enabled', False):
            # Implementation would depend on training service integration
            logger.info("Auto-saving trained model (implementation needed)")
    
    def _check_service_health(self) -> Dict[str, Any]:
        """Check model service health."""
        health = {
            'status': 'healthy',
            'details': {
                'loaded_models_count': len(self.loaded_models),
                'registered_models_count': len(self.model_metadata),
                'models_directory_accessible': self.models_directory.exists()
            }
        }
        
        # Check if models directory is accessible
        if not self.models_directory.exists():
            health['status'] = 'unhealthy'
            health['details']['models_directory_issue'] = True
        
        return health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model service metrics."""
        metrics = super().get_metrics()
        
        metrics.update({
            'loaded_models': len(self.loaded_models),
            'registered_models': len(self.model_metadata),
            'total_model_accesses': sum(
                entry.get('access_count', 0) 
                for entry in self.loaded_models.values()
            )
        })
        
        # Model type distribution
        model_types = {}
        for entry in self.loaded_models.values():
            model_type = entry['metadata'].get('model_type', 'unknown')
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        metrics['model_types'] = model_types
        
        return metrics