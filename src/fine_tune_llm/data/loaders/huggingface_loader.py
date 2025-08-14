"""
HuggingFace datasets loader for LLM training datasets.

This module provides loading capabilities for HuggingFace datasets
with support for streaming, caching, and preprocessing.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from .base import BaseDataLoader
from ...core.exceptions import DataError

logger = logging.getLogger(__name__)


class HuggingFaceDataLoader(BaseDataLoader):
    """
    HuggingFace datasets loader for large-scale datasets.
    
    Supports loading datasets from HuggingFace Hub with streaming,
    caching, filtering, and preprocessing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HuggingFace data loader.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # HuggingFace-specific configuration
        self.hf_config = self.config.get('huggingface', {})
        
        # Dataset loading options
        self.streaming = self.hf_config.get('streaming', False)
        self.cache_dir = self.hf_config.get('cache_dir', None)
        self.trust_remote_code = self.hf_config.get('trust_remote_code', False)
        
        # Dataset processing options
        self.split = self.hf_config.get('split', 'train')
        self.num_proc = self.hf_config.get('num_proc', None)
        self.column_mapping = self.hf_config.get('column_mapping', {})
        
        # Filtering and sampling
        self.max_samples = self.hf_config.get('max_samples', None)
        self.filter_conditions = self.hf_config.get('filter_conditions', [])
        self.shuffle_seed = self.hf_config.get('shuffle_seed', None)
        
        # Check if datasets library is available
        self._check_datasets_availability()
        
        logger.info(f"Initialized HuggingFaceDataLoader with streaming={self.streaming}")
    
    def _check_datasets_availability(self):
        """Check if datasets library is available."""
        try:
            import datasets
            self._datasets = datasets
        except ImportError:
            logger.error("HuggingFace datasets library not available. Install with: pip install datasets")
            self._datasets = None
    
    def load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a local dataset file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            List of loaded data samples
        """
        if self._datasets is None:
            raise DataError("HuggingFace datasets library not available")
        
        if not file_path.exists():
            raise DataError(f"Dataset file not found: {file_path}")
        
        try:
            # Determine file format and load accordingly
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.json':
                dataset = self._datasets.load_dataset('json', data_files=str(file_path))
            elif file_extension == '.jsonl':
                dataset = self._datasets.load_dataset('json', data_files=str(file_path), split='train')
            elif file_extension == '.csv':
                dataset = self._datasets.load_dataset('csv', data_files=str(file_path))
            elif file_extension in ['.txt', '.text']:
                dataset = self._datasets.load_dataset('text', data_files=str(file_path))
            elif file_extension == '.parquet':
                dataset = self._datasets.load_dataset('parquet', data_files=str(file_path))
            else:
                raise DataError(f"Unsupported file format: {file_extension}")
            
            # Convert to our format
            if isinstance(dataset, dict):
                dataset = dataset[list(dataset.keys())[0]]
            
            data = self._process_dataset(dataset)
            
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
            
        except Exception as e:
            raise DataError(f"Error loading HuggingFace dataset from {file_path}: {e}")
    
    def load_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from directory containing dataset files.
        
        Args:
            directory_path: Path to directory with dataset files
            
        Returns:
            List of loaded data samples
        """
        if self._datasets is None:
            raise DataError("HuggingFace datasets library not available")
        
        if not directory_path.exists():
            raise DataError(f"Directory not found: {directory_path}")
        
        # Find supported files
        supported_extensions = ['.json', '.jsonl', '.csv', '.txt', '.text', '.parquet']
        data_files = []
        
        for ext in supported_extensions:
            data_files.extend(directory_path.glob(f'*{ext}'))
        
        if not data_files:
            logger.warning(f"No supported dataset files found in {directory_path}")
            return []
        
        try:
            # Load all files as a single dataset
            data_files_str = [str(f) for f in data_files]
            
            # Group by extension and load
            all_data = []
            
            # JSON files
            json_files = [f for f in data_files_str if f.endswith('.json') or f.endswith('.jsonl')]
            if json_files:
                dataset = self._datasets.load_dataset('json', data_files=json_files, split='train')
                all_data.extend(self._process_dataset(dataset))
            
            # CSV files
            csv_files = [f for f in data_files_str if f.endswith('.csv')]
            if csv_files:
                dataset = self._datasets.load_dataset('csv', data_files=csv_files, split='train')
                all_data.extend(self._process_dataset(dataset))
            
            # Text files
            text_files = [f for f in data_files_str if f.endswith('.txt') or f.endswith('.text')]
            if text_files:
                dataset = self._datasets.load_dataset('text', data_files=text_files, split='train')
                all_data.extend(self._process_dataset(dataset))
            
            # Parquet files
            parquet_files = [f for f in data_files_str if f.endswith('.parquet')]
            if parquet_files:
                dataset = self._datasets.load_dataset('parquet', data_files=parquet_files, split='train')
                all_data.extend(self._process_dataset(dataset))
            
            logger.info(f"Loaded total {len(all_data)} samples from directory")
            return all_data
            
        except Exception as e:
            raise DataError(f"Error loading HuggingFace datasets from {directory_path}: {e}")
    
    def load_from_hub(self, dataset_name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load dataset from HuggingFace Hub.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            List of loaded data samples
        """
        if self._datasets is None:
            raise DataError("HuggingFace datasets library not available")
        
        try:
            # Merge configuration with kwargs
            load_kwargs = {
                'streaming': self.streaming,
                'cache_dir': self.cache_dir,
                'trust_remote_code': self.trust_remote_code,
                **kwargs
            }
            
            # Remove None values
            load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
            
            # Load dataset
            dataset = self._datasets.load_dataset(dataset_name, **load_kwargs)
            
            # Select split
            if isinstance(dataset, dict):
                if self.split in dataset:
                    dataset = dataset[self.split]
                else:
                    available_splits = list(dataset.keys())
                    logger.warning(f"Split '{self.split}' not available. Using '{available_splits[0]}'")
                    dataset = dataset[available_splits[0]]
            
            # Process dataset
            data = self._process_dataset(dataset)
            
            logger.info(f"Loaded {len(data)} samples from HuggingFace Hub: {dataset_name}")
            return data
            
        except Exception as e:
            raise DataError(f"Error loading dataset {dataset_name} from HuggingFace Hub: {e}")
    
    def _process_dataset(self, dataset) -> List[Dict[str, Any]]:
        """Process HuggingFace dataset into our format."""
        try:
            # Apply filtering
            if self.filter_conditions:
                dataset = self._apply_filters(dataset)
            
            # Shuffle if requested
            if self.shuffle_seed is not None:
                dataset = dataset.shuffle(seed=self.shuffle_seed)
            
            # Limit samples if requested
            if self.max_samples:
                if self.streaming:
                    dataset = dataset.take(self.max_samples)
                else:
                    dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            
            # Convert to list of dictionaries
            if self.streaming:
                data = []
                for i, item in enumerate(dataset):
                    processed_item = self._process_item(item)
                    if processed_item:
                        data.append(processed_item)
                    
                    # Break if we've reached max samples
                    if self.max_samples and i >= self.max_samples - 1:
                        break
            else:
                # Non-streaming: convert all at once
                data = [self._process_item(item) for item in dataset]
                data = [item for item in data if item is not None]
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return []
    
    def _apply_filters(self, dataset):
        """Apply filtering conditions to dataset."""
        for condition in self.filter_conditions:
            try:
                if isinstance(condition, dict):
                    column = condition.get('column')
                    operator = condition.get('operator', 'eq')
                    value = condition.get('value')
                    
                    if column and value is not None:
                        if operator == 'eq':
                            dataset = dataset.filter(lambda x: x.get(column) == value)
                        elif operator == 'ne':
                            dataset = dataset.filter(lambda x: x.get(column) != value)
                        elif operator == 'gt':
                            dataset = dataset.filter(lambda x: x.get(column, 0) > value)
                        elif operator == 'lt':
                            dataset = dataset.filter(lambda x: x.get(column, 0) < value)
                        elif operator == 'contains':
                            dataset = dataset.filter(lambda x: value in str(x.get(column, '')))
                
                elif callable(condition):
                    # Custom filter function
                    dataset = dataset.filter(condition)
                    
            except Exception as e:
                logger.warning(f"Error applying filter condition: {e}")
        
        return dataset
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single dataset item."""
        if not isinstance(item, dict):
            return None
        
        processed_item = {}
        
        # Apply column mapping
        for original_key, value in item.items():
            mapped_key = self.column_mapping.get(original_key, original_key)
            processed_item[mapped_key] = value
        
        return processed_item
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.json', '.jsonl', '.csv', '.txt', '.text', '.parquet']
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information dictionary
        """
        if self._datasets is None:
            return {'error': 'HuggingFace datasets library not available'}
        
        try:
            # Load dataset info without downloading data
            dataset_info = self._datasets.get_dataset_infos(dataset_name)
            
            if not dataset_info:
                # Try to load dataset config info
                dataset_builder = self._datasets.load_dataset_builder(dataset_name)
                info = dataset_builder.info
                
                return {
                    'name': dataset_name,
                    'description': info.description,
                    'features': str(info.features),
                    'homepage': info.homepage,
                    'license': info.license,
                    'citation': info.citation,
                    'splits': list(info.splits.keys()) if info.splits else []
                }
            else:
                # Return info for all configurations
                return {
                    'name': dataset_name,
                    'configurations': {
                        config_name: {
                            'description': config_info.description,
                            'features': str(config_info.features),
                            'splits': list(config_info.splits.keys()) if config_info.splits else []
                        }
                        for config_name, config_info in dataset_info.items()
                    }
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def create_streaming_loader(self, dataset_name: str, **kwargs):
        """
        Create streaming data loader for large datasets.
        
        Args:
            dataset_name: Name of the dataset
            **kwargs: Additional arguments
            
        Returns:
            Streaming dataset iterator
        """
        if self._datasets is None:
            raise DataError("HuggingFace datasets library not available")
        
        load_kwargs = {
            'streaming': True,
            'cache_dir': self.cache_dir,
            'trust_remote_code': self.trust_remote_code,
            **kwargs
        }
        
        # Remove None values
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        
        try:
            dataset = self._datasets.load_dataset(dataset_name, **load_kwargs)
            
            # Select split
            if isinstance(dataset, dict):
                dataset = dataset.get(self.split, dataset[list(dataset.keys())[0]])
            
            return dataset
            
        except Exception as e:
            raise DataError(f"Error creating streaming loader for {dataset_name}: {e}")
    
    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        """
        Validate a loaded dataset sample.
        
        Args:
            sample: Dataset sample to validate
        """
        # Call parent validation
        super()._validate_sample(sample)
        
        # HuggingFace-specific validation
        required_columns = self.hf_config.get('required_columns', [])
        for column in required_columns:
            if column not in sample:
                raise ValueError(f"Missing required column: {column}")
    
    def save_to_hub(self, 
                   data: List[Dict[str, Any]], 
                   dataset_name: str, 
                   private: bool = False,
                   **kwargs):
        """
        Save dataset to HuggingFace Hub.
        
        Args:
            data: Data to save
            dataset_name: Name for the dataset on Hub
            private: Whether to make dataset private
            **kwargs: Additional arguments for push_to_hub
        """
        if self._datasets is None:
            raise DataError("HuggingFace datasets library not available")
        
        try:
            # Create dataset from data
            dataset = self._datasets.Dataset.from_list(data)
            
            # Push to hub
            dataset.push_to_hub(
                dataset_name,
                private=private,
                **kwargs
            )
            
            logger.info(f"Saved dataset to HuggingFace Hub: {dataset_name}")
            
        except Exception as e:
            raise DataError(f"Error saving dataset to Hub: {e}")