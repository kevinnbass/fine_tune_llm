"""
Base data loader interface for loading training datasets.

This module provides the abstract base class for all data loaders
with common loading functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Iterator
from pathlib import Path
import logging

from ...core.interfaces import BaseComponent
from ...core.exceptions import DataError

logger = logging.getLogger(__name__)


class BaseDataLoader(BaseComponent, ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.loader_config = self.config.get('loader', {})
        
        # Loading parameters
        self.batch_size = self.loader_config.get('batch_size', 1000)
        self.shuffle = self.loader_config.get('shuffle', False)
        self.cache_data = self.loader_config.get('cache_data', True)
        
        # Validation and filtering
        self.validate_schema = self.loader_config.get('validate_schema', True)
        self.skip_invalid = self.loader_config.get('skip_invalid', True)
        self.max_samples = self.loader_config.get('max_samples', None)
        
        # Loading statistics
        self.stats = {
            'total_files': 0,
            'loaded_samples': 0,
            'skipped_samples': 0,
            'loading_errors': 0,
            'total_time': 0.0
        }
        
        # Cache for loaded data
        self._cache = {} if self.cache_data else None
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.loader_config}")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.loader_config = self.config.get('loader', {})
    
    def cleanup(self) -> None:
        """Clean up loading resources."""
        if self._cache:
            self._cache.clear()
        self._reset_stats()
    
    @property
    def name(self) -> str:
        """Component name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    @abstractmethod
    def load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a single file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of loaded data samples
        """
        pass
    
    @abstractmethod
    def load_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from all files in a directory.
        
        Args:
            directory_path: Path to input directory
            
        Returns:
            List of loaded data samples
        """
        pass
    
    def load(self, 
             source: Union[str, Path, List[Union[str, Path]]],
             **kwargs) -> List[Dict[str, Any]]:
        """
        Load data from source(s).
        
        Args:
            source: File path, directory path, or list of paths
            **kwargs: Additional loading parameters
            
        Returns:
            List of loaded data samples
        """
        import time
        start_time = time.time()
        
        # Handle different source types
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            
            if source_path.is_file():
                results = self._load_with_cache(source_path, is_file=True)
            elif source_path.is_dir():
                results = self._load_with_cache(source_path, is_file=False)
            else:
                raise DataError(f"Source path does not exist: {source_path}")
                
        elif isinstance(source, list):
            results = []
            for path in source:
                path_obj = Path(path)
                if path_obj.is_file():
                    file_results = self._load_with_cache(path_obj, is_file=True)
                    results.extend(file_results)
                else:
                    logger.warning(f"Skipping non-existent file: {path}")
        else:
            raise DataError(f"Unsupported source type: {type(source)}")
        
        # Apply sample limit
        if self.max_samples and len(results) > self.max_samples:
            if self.shuffle:
                import random
                results = random.sample(results, self.max_samples)
            else:
                results = results[:self.max_samples]
        
        # Shuffle if requested
        elif self.shuffle:
            import random
            random.shuffle(results)
        
        # Update timing stats
        self.stats['total_time'] = time.time() - start_time
        self.stats['loaded_samples'] = len(results)
        
        logger.info(f"Loaded {len(results)} samples in {self.stats['total_time']:.2f}s")
        
        return results
    
    def _load_with_cache(self, path: Path, is_file: bool) -> List[Dict[str, Any]]:
        """Load data with caching support."""
        cache_key = str(path)
        
        # Check cache first
        if self._cache is not None and cache_key in self._cache:
            logger.debug(f"Loading from cache: {path}")
            return self._cache[cache_key]
        
        try:
            # Load data
            if is_file:
                data = self.load_file(path)
                self.stats['total_files'] += 1
            else:
                data = self.load_directory(path)
            
            # Validate if requested
            if self.validate_schema:
                data = self._validate_loaded_data(data)
            
            # Cache results
            if self._cache is not None:
                self._cache[cache_key] = data
            
            return data
            
        except Exception as e:
            self.stats['loading_errors'] += 1
            logger.error(f"Error loading from {path}: {e}")
            
            if self.skip_invalid:
                logger.warning(f"Skipping invalid source: {path}")
                return []
            else:
                raise DataError(f"Failed to load from {path}: {e}")
    
    def _validate_loaded_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate loaded data samples."""
        valid_data = []
        
        for i, sample in enumerate(data):
            try:
                self._validate_sample(sample)
                valid_data.append(sample)
            except Exception as e:
                self.stats['skipped_samples'] += 1
                logger.warning(f"Skipping invalid sample {i}: {e}")
                
                if not self.skip_invalid:
                    raise
        
        return valid_data
    
    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        """
        Validate a single data sample.
        
        Args:
            sample: Data sample to validate
            
        Raises:
            ValueError: If sample is invalid
        """
        # Basic validation - can be overridden
        if not isinstance(sample, dict):
            raise ValueError(f"Sample must be dictionary, got {type(sample)}")
        
        if not sample:
            raise ValueError("Sample cannot be empty")
    
    def create_iterator(self, 
                       source: Union[str, Path, List[Union[str, Path]]],
                       **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Create iterator for streaming data loading.
        
        Args:
            source: Data source
            **kwargs: Additional parameters
            
        Yields:
            Individual data samples
        """
        # Load all data first (can be optimized for true streaming)
        data = self.load(source, **kwargs)
        
        # Yield in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            for sample in batch:
                yield sample
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        # Default implementation - should be overridden
        return ['.json', '.jsonl']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        stats = self.stats.copy()
        
        if stats['loaded_samples'] > 0 and stats['total_time'] > 0:
            stats['loading_rate'] = stats['loaded_samples'] / stats['total_time']
        
        if self._cache:
            stats['cache_size'] = len(self._cache)
        
        return stats
    
    def _reset_stats(self):
        """Reset loading statistics."""
        self.stats = {
            'total_files': 0,
            'loaded_samples': 0,
            'skipped_samples': 0,
            'loading_errors': 0,
            'total_time': 0.0
        }
    
    def clear_cache(self):
        """Clear loading cache."""
        if self._cache:
            self._cache.clear()
    
    def set_batch_size(self, batch_size: int):
        """Update batch size."""
        self.batch_size = batch_size
    
    def set_validation(self, validate_schema: bool = True, skip_invalid: bool = True):
        """Update validation settings."""
        self.validate_schema = validate_schema
        self.skip_invalid = skip_invalid