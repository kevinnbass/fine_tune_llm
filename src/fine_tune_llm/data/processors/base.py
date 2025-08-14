"""
Base data processor interface for data pipeline.

This module provides the abstract base class for all data processors
with common data processing functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Iterator
from pathlib import Path
import logging

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class BaseDataProcessor(BaseComponent, ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.processor_config = self.config.get('processor', {})
        
        # Processing parameters
        self.batch_size = self.processor_config.get('batch_size', 1000)
        self.num_workers = self.processor_config.get('num_workers', 1)
        self.cache_processed = self.processor_config.get('cache_processed', True)
        
        # Data validation
        self.validate_input = self.processor_config.get('validate_input', True)
        self.validate_output = self.processor_config.get('validate_output', True)
        self.skip_invalid = self.processor_config.get('skip_invalid', False)
        
        # Processing statistics
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_time': 0.0
        }
        
        # Cache for processed data
        self._cache = {} if self.cache_processed else None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.processor_config = self.config.get('processor', {})
    
    def cleanup(self) -> None:
        """Clean up processing resources."""
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
    def process_single(self, data: Any) -> Any:
        """
        Process a single data sample.
        
        Args:
            data: Input data sample
            
        Returns:
            Processed data sample
        """
        pass
    
    def process_batch(self, data_batch: List[Any]) -> List[Any]:
        """
        Process a batch of data samples.
        
        Args:
            data_batch: List of input data samples
            
        Returns:
            List of processed data samples
        """
        results = []
        
        for item in data_batch:
            try:
                if self.validate_input:
                    self._validate_input(item)
                
                # Check cache first
                if self._cache is not None:
                    cache_key = self._get_cache_key(item)
                    if cache_key in self._cache:
                        result = self._cache[cache_key]
                        results.append(result)
                        continue
                
                # Process item
                result = self.process_single(item)
                
                if self.validate_output:
                    self._validate_output(result)
                
                # Cache result
                if self._cache is not None:
                    cache_key = self._get_cache_key(item)
                    self._cache[cache_key] = result
                
                results.append(result)
                self.stats['processed'] += 1
                
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"Error processing item: {e}")
                
                if self.skip_invalid:
                    self.stats['skipped'] += 1
                    continue
                else:
                    raise
        
        return results
    
    def process_dataset(self, 
                       dataset: Union[List[Any], Iterator[Any]],
                       output_path: Optional[Path] = None) -> Union[List[Any], None]:
        """
        Process an entire dataset.
        
        Args:
            dataset: Input dataset (list or iterator)
            output_path: Optional path to save processed data
            
        Returns:
            Processed dataset or None if saved to file
        """
        import time
        start_time = time.time()
        
        # Convert to list if iterator
        if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
            dataset = list(dataset)
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Log progress
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)} / {len(dataset)} samples")
        
        # Update timing stats
        self.stats['total_time'] = time.time() - start_time
        
        # Save to file if path provided
        if output_path:
            self._save_processed_data(all_results, output_path)
            logger.info(f"Saved processed data to {output_path}")
            return None
        
        return all_results
    
    def _validate_input(self, data: Any) -> None:
        """
        Validate input data format.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        # Default implementation - can be overridden
        if data is None:
            raise ValueError("Input data cannot be None")
    
    def _validate_output(self, data: Any) -> None:
        """
        Validate output data format.
        
        Args:
            data: Output data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        # Default implementation - can be overridden
        if data is None:
            raise ValueError("Output data cannot be None")
    
    def _get_cache_key(self, data: Any) -> str:
        """
        Generate cache key for data item.
        
        Args:
            data: Data item
            
        Returns:
            Cache key string
        """
        # Simple hash-based key
        return str(hash(str(data)))
    
    def _save_processed_data(self, data: List[Any], output_path: Path):
        """
        Save processed data to file.
        
        Args:
            data: Processed data
            output_path: Output file path
        """
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif output_path.suffix == '.jsonl':
            with open(output_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item, default=str) + '\n')
        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['processed'] > 0:
            stats['avg_time_per_item'] = stats['total_time'] / stats['processed']
            stats['throughput'] = stats['processed'] / stats['total_time'] if stats['total_time'] > 0 else 0
        
        if self._cache:
            stats['cache_size'] = len(self._cache)
        
        return stats
    
    def _reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    def clear_cache(self):
        """Clear processing cache."""
        if self._cache:
            self._cache.clear()
    
    def set_batch_size(self, batch_size: int):
        """Update batch size."""
        self.batch_size = batch_size
    
    def set_validation(self, validate_input: bool = True, validate_output: bool = True):
        """Update validation settings."""
        self.validate_input = validate_input
        self.validate_output = validate_output