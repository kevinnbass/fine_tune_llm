"""
Base data transformer interface for data transformation pipeline.

This module provides the abstract base class for all data transformers
with common transformation functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Iterator
from pathlib import Path
import logging

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class BaseDataTransformer(BaseComponent, ABC):
    """Abstract base class for data transformers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base data transformer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.transformer_config = self.config.get('transformer', {})
        
        # Transformation parameters
        self.batch_size = self.transformer_config.get('batch_size', 1000)
        self.in_place = self.transformer_config.get('in_place', False)
        self.parallel_processing = self.transformer_config.get('parallel_processing', False)
        
        # Validation and error handling
        self.validate_input = self.transformer_config.get('validate_input', True)
        self.validate_output = self.transformer_config.get('validate_output', True)
        self.skip_invalid = self.transformer_config.get('skip_invalid', False)
        
        # Transformation statistics
        self.stats = {
            'transformed': 0,
            'skipped': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.transformer_config = self.config.get('transformer', {})
    
    def cleanup(self) -> None:
        """Clean up transformation resources."""
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
    def transform_single(self, data: Any) -> Any:
        """
        Transform a single data sample.
        
        Args:
            data: Input data sample
            
        Returns:
            Transformed data sample
        """
        pass
    
    def transform_batch(self, data_batch: List[Any]) -> List[Any]:
        """
        Transform a batch of data samples.
        
        Args:
            data_batch: List of input data samples
            
        Returns:
            List of transformed data samples
        """
        results = []
        
        for item in data_batch:
            try:
                if self.validate_input:
                    self._validate_input(item)
                
                # Transform item
                if self.in_place:
                    transformed_item = item
                    self._transform_in_place(transformed_item)
                else:
                    transformed_item = self.transform_single(item)
                
                if self.validate_output:
                    self._validate_output(transformed_item)
                
                results.append(transformed_item)
                self.stats['transformed'] += 1
                
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"Error transforming item: {e}")
                
                if self.skip_invalid:
                    self.stats['skipped'] += 1
                    continue
                else:
                    raise
        
        return results
    
    def transform_dataset(self, 
                         dataset: Union[List[Any], Iterator[Any]],
                         output_path: Optional[Path] = None) -> Union[List[Any], None]:
        """
        Transform an entire dataset.
        
        Args:
            dataset: Input dataset (list or iterator)
            output_path: Optional path to save transformed data
            
        Returns:
            Transformed dataset or None if saved to file
        """
        import time
        start_time = time.time()
        
        # Convert to list if iterator
        if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
            dataset = list(dataset)
        
        # Transform in batches
        all_results = []
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            
            if self.parallel_processing:
                batch_results = self._transform_batch_parallel(batch)
            else:
                batch_results = self.transform_batch(batch)
            
            all_results.extend(batch_results)
            
            # Log progress
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Transformed {i + len(batch)} / {len(dataset)} samples")
        
        # Update timing stats
        self.stats['total_time'] = time.time() - start_time
        
        # Save to file if path provided
        if output_path:
            self._save_transformed_data(all_results, output_path)
            logger.info(f"Saved transformed data to {output_path}")
            return None
        
        return all_results
    
    def _transform_batch_parallel(self, data_batch: List[Any]) -> List[Any]:
        """Transform batch using parallel processing."""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = [None] * len(data_batch)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit transformation tasks
                future_to_index = {
                    executor.submit(self._safe_transform_single, item): i 
                    for i, item in enumerate(data_batch)
                }
                
                # Collect results
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[index] = result
                            self.stats['transformed'] += 1
                        else:
                            self.stats['skipped'] += 1
                    except Exception as e:
                        self.stats['errors'] += 1
                        logger.error(f"Error in parallel transformation: {e}")
                        
                        if not self.skip_invalid:
                            raise
            
            # Filter out None results (failed transformations)
            return [r for r in results if r is not None]
            
        except ImportError:
            logger.warning("concurrent.futures not available, falling back to sequential processing")
            return self.transform_batch(data_batch)
    
    def _safe_transform_single(self, data: Any) -> Optional[Any]:
        """Safely transform single item with error handling."""
        try:
            if self.validate_input:
                self._validate_input(data)
            
            if self.in_place:
                transformed_item = data
                self._transform_in_place(transformed_item)
            else:
                transformed_item = self.transform_single(data)
            
            if self.validate_output:
                self._validate_output(transformed_item)
            
            return transformed_item
            
        except Exception as e:
            if self.skip_invalid:
                logger.warning(f"Skipping invalid item: {e}")
                return None
            else:
                raise
    
    def _transform_in_place(self, data: Any) -> None:
        """
        Transform data in-place (default implementation calls transform_single).
        
        Args:
            data: Data to transform in-place
        """
        # Default implementation - override for true in-place transformation
        transformed = self.transform_single(data)
        if hasattr(data, 'update') and hasattr(transformed, 'keys'):
            data.clear()
            data.update(transformed)
        else:
            raise NotImplementedError("In-place transformation not supported for this data type")
    
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
    
    def _save_transformed_data(self, data: List[Any], output_path: Path):
        """
        Save transformed data to file.
        
        Args:
            data: Transformed data
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
        """Get transformation statistics."""
        stats = self.stats.copy()
        
        if stats['transformed'] > 0 and stats['total_time'] > 0:
            stats['throughput'] = stats['transformed'] / stats['total_time']
            stats['avg_time_per_item'] = stats['total_time'] / stats['transformed']
        
        return stats
    
    def _reset_stats(self):
        """Reset transformation statistics."""
        self.stats = {
            'transformed': 0,
            'skipped': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    def set_batch_size(self, batch_size: int):
        """Update batch size."""
        self.batch_size = batch_size
    
    def set_validation(self, validate_input: bool = True, validate_output: bool = True):
        """Update validation settings."""
        self.validate_input = validate_input
        self.validate_output = validate_output
    
    def enable_parallel_processing(self, enabled: bool = True):
        """Enable or disable parallel processing."""
        self.parallel_processing = enabled
    
    def chain(self, other_transformer: 'BaseDataTransformer') -> 'ChainedTransformer':
        """
        Chain this transformer with another transformer.
        
        Args:
            other_transformer: Transformer to chain with this one
            
        Returns:
            ChainedTransformer that applies both transformations
        """
        return ChainedTransformer([self, other_transformer])


class ChainedTransformer(BaseDataTransformer):
    """Transformer that chains multiple transformers together."""
    
    def __init__(self, transformers: List[BaseDataTransformer]):
        """
        Initialize chained transformer.
        
        Args:
            transformers: List of transformers to chain
        """
        super().__init__()
        self.transformers = transformers
        
        # Aggregate configuration from all transformers
        for transformer in transformers:
            self.config.update(transformer.config)
    
    def transform_single(self, data: Any) -> Any:
        """Apply all transformers in sequence."""
        result = data
        
        for transformer in self.transformers:
            result = transformer.transform_single(result)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all transformers."""
        aggregated_stats = super().get_stats()
        
        # Add individual transformer stats
        transformer_stats = {}
        for i, transformer in enumerate(self.transformers):
            transformer_stats[f'transformer_{i}_{transformer.__class__.__name__}'] = transformer.get_stats()
        
        aggregated_stats['individual_transformers'] = transformer_stats
        
        return aggregated_stats