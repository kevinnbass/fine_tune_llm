"""
Shared Data Transformation Utilities.

This module provides unified data transformation capabilities across all
platform components with consistent interfaces and extensible design.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Type, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import json
import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import gzip
import base64

from ...core.events import EventBus, Event, EventType
from ..validation.centralized_validator import get_centralized_validator

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of data transformations."""
    NORMALIZATION = "normalization"
    TOKENIZATION = "tokenization"
    ENCODING = "encoding"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    MAPPING = "mapping"
    VALIDATION = "validation"
    FORMATTING = "formatting"
    SERIALIZATION = "serialization"
    COMPRESSION = "compression"
    ENCRYPTION = "encryption"
    CUSTOM = "custom"


class TransformationScope(Enum):
    """Scope of transformation application."""
    SINGLE = "single"
    BATCH = "batch"
    STREAM = "stream"
    PIPELINE = "pipeline"


@dataclass
class TransformationResult:
    """Result of transformation operation."""
    success: bool
    data: Any
    original_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    transformation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str):
        """Add transformation error."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add transformation warning."""
        self.warnings.append(warning)


@dataclass
class TransformationConfig:
    """Configuration for data transformation."""
    name: str
    transformation_type: TransformationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    scope: TransformationScope = TransformationScope.SINGLE
    validate_input: bool = True
    validate_output: bool = True
    cache_results: bool = False
    parallel: bool = False
    max_workers: int = 4
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTransformer:
    """Base class for data transformers."""
    
    def __init__(self, config: TransformationConfig):
        """Initialize transformer."""
        self.config = config
        self.enabled = True
        self.stats = {
            'transforms_applied': 0,
            'transforms_failed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._cache = {} if config.cache_results else None
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform data."""
        raise NotImplementedError
    
    def validate_input(self, data: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input data."""
        return True
    
    def validate_output(self, data: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate output data."""
        return True
    
    def _generate_cache_key(self, data: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for data."""
        try:
            data_str = str(data) if not isinstance(data, (dict, list)) else json.dumps(data, sort_keys=True)
            context_str = json.dumps(context or {}, sort_keys=True)
            combined = f"{data_str}:{context_str}"
            return hashlib.md5(combined.encode()).hexdigest()
        except Exception:
            return str(hash((str(data), str(context))))


class TextTransformer(BaseTransformer):
    """Transformer for text data."""
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform text data."""
        result = TransformationResult(success=True, data=data, original_data=data)
        
        try:
            if not isinstance(data, str):
                result.add_error(f"Expected string, got {type(data)}")
                return result
            
            # Apply text transformations based on parameters
            transformed_text = data
            
            # Normalization
            if self.config.parameters.get('normalize_whitespace'):
                transformed_text = re.sub(r'\s+', ' ', transformed_text).strip()
            
            if self.config.parameters.get('lowercase'):
                transformed_text = transformed_text.lower()
            
            if self.config.parameters.get('remove_special_chars'):
                pattern = self.config.parameters.get('special_chars_pattern', r'[^\w\s]')
                transformed_text = re.sub(pattern, '', transformed_text)
            
            # Truncation
            max_length = self.config.parameters.get('max_length')
            if max_length and len(transformed_text) > max_length:
                transformed_text = transformed_text[:max_length]
                result.add_warning(f"Text truncated to {max_length} characters")
            
            # Padding
            min_length = self.config.parameters.get('min_length')
            if min_length and len(transformed_text) < min_length:
                padding_char = self.config.parameters.get('padding_char', ' ')
                transformed_text = transformed_text.ljust(min_length, padding_char)
            
            # Encoding
            encoding_type = self.config.parameters.get('encoding')
            if encoding_type == 'base64':
                transformed_text = base64.b64encode(transformed_text.encode()).decode()
            elif encoding_type == 'url':
                import urllib.parse
                transformed_text = urllib.parse.quote(transformed_text)
            
            result.data = transformed_text
            result.metadata.update({
                'original_length': len(data),
                'transformed_length': len(transformed_text),
                'transformations_applied': list(self.config.parameters.keys())
            })
            
            self.stats['transforms_applied'] += 1
            
        except Exception as e:
            result.add_error(f"Text transformation failed: {e}")
            self.stats['transforms_failed'] += 1
        
        return result


class NumericTransformer(BaseTransformer):
    """Transformer for numeric data."""
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform numeric data."""
        result = TransformationResult(success=True, data=data, original_data=data)
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, (list, tuple)):
                data_array = np.array(data)
            elif isinstance(data, (int, float)):
                data_array = np.array([data])
            elif isinstance(data, np.ndarray):
                data_array = data
            else:
                result.add_error(f"Cannot convert {type(data)} to numeric array")
                return result
            
            transformed_data = data_array.copy()
            
            # Normalization
            normalization = self.config.parameters.get('normalization')
            if normalization == 'minmax':
                min_val = self.config.parameters.get('min_val', data_array.min())
                max_val = self.config.parameters.get('max_val', data_array.max())
                if max_val > min_val:
                    transformed_data = (transformed_data - min_val) / (max_val - min_val)
            elif normalization == 'zscore':
                mean_val = self.config.parameters.get('mean', data_array.mean())
                std_val = self.config.parameters.get('std', data_array.std())
                if std_val > 0:
                    transformed_data = (transformed_data - mean_val) / std_val
            elif normalization == 'robust':
                median_val = np.median(data_array)
                mad_val = np.median(np.abs(data_array - median_val))
                if mad_val > 0:
                    transformed_data = (transformed_data - median_val) / mad_val
            
            # Clipping
            clip_min = self.config.parameters.get('clip_min')
            clip_max = self.config.parameters.get('clip_max')
            if clip_min is not None or clip_max is not None:
                transformed_data = np.clip(transformed_data, clip_min, clip_max)
            
            # Rounding
            decimals = self.config.parameters.get('round_decimals')
            if decimals is not None:
                transformed_data = np.round(transformed_data, decimals)
            
            # Scaling
            scale_factor = self.config.parameters.get('scale_factor')
            if scale_factor is not None:
                transformed_data = transformed_data * scale_factor
            
            # Output format
            output_format = self.config.parameters.get('output_format', 'array')
            if output_format == 'list':
                result.data = transformed_data.tolist()
            elif output_format == 'scalar' and transformed_data.size == 1:
                result.data = transformed_data.item()
            else:
                result.data = transformed_data
            
            result.metadata.update({
                'original_shape': data_array.shape,
                'original_dtype': str(data_array.dtype),
                'transformed_shape': transformed_data.shape,
                'transformed_dtype': str(transformed_data.dtype),
                'transformations_applied': list(self.config.parameters.keys())
            })
            
            self.stats['transforms_applied'] += 1
            
        except Exception as e:
            result.add_error(f"Numeric transformation failed: {e}")
            self.stats['transforms_failed'] += 1
        
        return result


class DataFrameTransformer(BaseTransformer):
    """Transformer for pandas DataFrame data."""
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform DataFrame data."""
        result = TransformationResult(success=True, data=data, original_data=data)
        
        try:
            if not isinstance(data, pd.DataFrame):
                result.add_error(f"Expected DataFrame, got {type(data)}")
                return result
            
            transformed_df = data.copy()
            
            # Column selection
            columns = self.config.parameters.get('columns')
            if columns:
                missing_cols = set(columns) - set(transformed_df.columns)
                if missing_cols:
                    result.add_warning(f"Missing columns: {missing_cols}")
                    columns = [col for col in columns if col in transformed_df.columns]
                transformed_df = transformed_df[columns]
            
            # Row filtering
            filter_condition = self.config.parameters.get('filter_condition')
            if filter_condition:
                if callable(filter_condition):
                    mask = filter_condition(transformed_df)
                    transformed_df = transformed_df[mask]
                elif isinstance(filter_condition, str):
                    transformed_df = transformed_df.query(filter_condition)
            
            # Null handling
            drop_nulls = self.config.parameters.get('drop_nulls')
            if drop_nulls:
                if drop_nulls == 'any':
                    transformed_df = transformed_df.dropna()
                elif drop_nulls == 'all':
                    transformed_df = transformed_df.dropna(how='all')
            
            fill_nulls = self.config.parameters.get('fill_nulls')
            if fill_nulls is not None:
                transformed_df = transformed_df.fillna(fill_nulls)
            
            # Sorting
            sort_by = self.config.parameters.get('sort_by')
            if sort_by:
                ascending = self.config.parameters.get('sort_ascending', True)
                transformed_df = transformed_df.sort_values(sort_by, ascending=ascending)
            
            # Grouping and aggregation
            group_by = self.config.parameters.get('group_by')
            if group_by:
                agg_func = self.config.parameters.get('agg_func', 'mean')
                transformed_df = transformed_df.groupby(group_by).agg(agg_func).reset_index()
            
            # Index reset
            if self.config.parameters.get('reset_index'):
                transformed_df = transformed_df.reset_index(drop=True)
            
            result.data = transformed_df
            result.metadata.update({
                'original_shape': data.shape,
                'transformed_shape': transformed_df.shape,
                'original_columns': list(data.columns),
                'transformed_columns': list(transformed_df.columns),
                'transformations_applied': list(self.config.parameters.keys())
            })
            
            self.stats['transforms_applied'] += 1
            
        except Exception as e:
            result.add_error(f"DataFrame transformation failed: {e}")
            self.stats['transforms_failed'] += 1
        
        return result


class JSONTransformer(BaseTransformer):
    """Transformer for JSON data."""
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform JSON data."""
        result = TransformationResult(success=True, data=data, original_data=data)
        
        try:
            # Parse JSON if string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    result.add_error(f"Invalid JSON: {e}")
                    return result
            
            transformed_data = data
            
            # Field extraction
            extract_fields = self.config.parameters.get('extract_fields')
            if extract_fields and isinstance(data, dict):
                transformed_data = {field: data.get(field) for field in extract_fields}
            
            # Field mapping/renaming
            field_mapping = self.config.parameters.get('field_mapping')
            if field_mapping and isinstance(transformed_data, dict):
                mapped_data = {}
                for old_key, new_key in field_mapping.items():
                    if old_key in transformed_data:
                        mapped_data[new_key] = transformed_data[old_key]
                    else:
                        mapped_data[new_key] = None
                transformed_data = mapped_data
            
            # Default values
            default_values = self.config.parameters.get('default_values')
            if default_values and isinstance(transformed_data, dict):
                for field, default_val in default_values.items():
                    if field not in transformed_data or transformed_data[field] is None:
                        transformed_data[field] = default_val
            
            # Type conversion
            type_conversions = self.config.parameters.get('type_conversions')
            if type_conversions and isinstance(transformed_data, dict):
                for field, target_type in type_conversions.items():
                    if field in transformed_data and transformed_data[field] is not None:
                        try:
                            if target_type == 'int':
                                transformed_data[field] = int(transformed_data[field])
                            elif target_type == 'float':
                                transformed_data[field] = float(transformed_data[field])
                            elif target_type == 'str':
                                transformed_data[field] = str(transformed_data[field])
                            elif target_type == 'bool':
                                transformed_data[field] = bool(transformed_data[field])
                        except (ValueError, TypeError) as e:
                            result.add_warning(f"Type conversion failed for {field}: {e}")
            
            # Flattening
            if self.config.parameters.get('flatten') and isinstance(transformed_data, dict):
                transformed_data = self._flatten_dict(transformed_data)
            
            # Serialization
            serialize = self.config.parameters.get('serialize')
            if serialize:
                if serialize == 'json':
                    transformed_data = json.dumps(transformed_data, default=str)
                elif serialize == 'compact':
                    transformed_data = json.dumps(transformed_data, separators=(',', ':'))
            
            result.data = transformed_data
            result.metadata.update({
                'original_type': type(data).__name__,
                'transformed_type': type(transformed_data).__name__,
                'transformations_applied': list(self.config.parameters.keys())
            })
            
            self.stats['transforms_applied'] += 1
            
        except Exception as e:
            result.add_error(f"JSON transformation failed: {e}")
            self.stats['transforms_failed'] += 1
        
        return result
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', separator: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, separator).items())
            else:
                items.append((new_key, value))
        return dict(items)


class CompressionTransformer(BaseTransformer):
    """Transformer for data compression."""
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform data with compression."""
        result = TransformationResult(success=True, data=data, original_data=data)
        
        try:
            compression_type = self.config.parameters.get('compression_type', 'gzip')
            
            # Serialize data first
            if isinstance(data, (dict, list)):
                serialized_data = json.dumps(data, default=str).encode()
            elif isinstance(data, str):
                serialized_data = data.encode()
            elif isinstance(data, bytes):
                serialized_data = data
            else:
                serialized_data = pickle.dumps(data)
            
            # Compress based on type
            if compression_type == 'gzip':
                compressed_data = gzip.compress(serialized_data)
            elif compression_type == 'base64':
                compressed_data = base64.b64encode(serialized_data)
            else:
                result.add_error(f"Unsupported compression type: {compression_type}")
                return result
            
            # Encode to string if requested
            if self.config.parameters.get('encode_to_string'):
                if isinstance(compressed_data, bytes):
                    compressed_data = base64.b64encode(compressed_data).decode()
            
            result.data = compressed_data
            result.metadata.update({
                'compression_type': compression_type,
                'original_size': len(serialized_data),
                'compressed_size': len(compressed_data),
                'compression_ratio': len(compressed_data) / len(serialized_data)
            })
            
            self.stats['transforms_applied'] += 1
            
        except Exception as e:
            result.add_error(f"Compression transformation failed: {e}")
            self.stats['transforms_failed'] += 1
        
        return result


class SharedTransformationEngine:
    """
    Shared data transformation engine.
    
    Provides unified transformation capabilities across all platform
    components with consistent interfaces and extensible design.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize transformation engine.
        
        Args:
            event_bus: Event bus for notifications
        """
        self.event_bus = event_bus or EventBus()
        
        # Transformer registry
        self.transformers: Dict[str, BaseTransformer] = {}
        
        # Transformation chains
        self.chains: Dict[str, List[str]] = {}
        
        # Global configuration
        self.global_config = {
            'validate_input': True,
            'validate_output': True,
            'cache_results': False,
            'parallel_processing': False,
            'max_workers': 4,
            'timeout_seconds': 300
        }
        
        # Statistics
        self.stats = {
            'total_transformations': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize built-in transformers
        self._initialize_builtin_transformers()
        
        logger.info("Initialized SharedTransformationEngine")
    
    def _initialize_builtin_transformers(self):
        """Initialize built-in transformers."""
        # Text transformer
        text_config = TransformationConfig(
            name="text_normalizer",
            transformation_type=TransformationType.NORMALIZATION,
            parameters={
                'normalize_whitespace': True,
                'lowercase': False,
                'remove_special_chars': False
            }
        )
        self.register_transformer("text_normalizer", TextTransformer(text_config))
        
        # Numeric transformer
        numeric_config = TransformationConfig(
            name="numeric_normalizer",
            transformation_type=TransformationType.NORMALIZATION,
            parameters={
                'normalization': 'minmax',
                'output_format': 'array'
            }
        )
        self.register_transformer("numeric_normalizer", NumericTransformer(numeric_config))
        
        # DataFrame transformer
        df_config = TransformationConfig(
            name="dataframe_processor",
            transformation_type=TransformationType.FILTERING,
            parameters={
                'drop_nulls': False,
                'reset_index': True
            }
        )
        self.register_transformer("dataframe_processor", DataFrameTransformer(df_config))
        
        # JSON transformer
        json_config = TransformationConfig(
            name="json_processor",
            transformation_type=TransformationType.FORMATTING,
            parameters={
                'serialize': False,
                'flatten': False
            }
        )
        self.register_transformer("json_processor", JSONTransformer(json_config))
        
        # Compression transformer
        compression_config = TransformationConfig(
            name="data_compressor",
            transformation_type=TransformationType.COMPRESSION,
            parameters={
                'compression_type': 'gzip',
                'encode_to_string': False
            }
        )
        self.register_transformer("data_compressor", CompressionTransformer(compression_config))
    
    def register_transformer(self, name: str, transformer: BaseTransformer) -> bool:
        """
        Register transformer.
        
        Args:
            name: Transformer name
            transformer: Transformer instance
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.transformers[name] = transformer
            
            logger.info(f"Registered transformer: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register transformer {name}: {e}")
            return False
    
    def unregister_transformer(self, name: str) -> bool:
        """
        Unregister transformer.
        
        Args:
            name: Transformer name
            
        Returns:
            True if unregistered successfully
        """
        try:
            with self._lock:
                if name in self.transformers:
                    del self.transformers[name]
                    logger.info(f"Unregistered transformer: {name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister transformer {name}: {e}")
            return False
    
    def create_transformation_chain(self, 
                                   name: str, 
                                   transformer_names: List[str]) -> bool:
        """
        Create transformation chain.
        
        Args:
            name: Chain name
            transformer_names: List of transformer names in order
            
        Returns:
            True if created successfully
        """
        try:
            with self._lock:
                # Validate transformers exist
                missing_transformers = set(transformer_names) - set(self.transformers.keys())
                if missing_transformers:
                    logger.error(f"Missing transformers: {missing_transformers}")
                    return False
                
                self.chains[name] = transformer_names
            
            logger.info(f"Created transformation chain: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create chain {name}: {e}")
            return False
    
    def transform(self, 
                 data: Any,
                 transformer_name: str,
                 context: Optional[Dict[str, Any]] = None,
                 validate: bool = None) -> TransformationResult:
        """
        Apply single transformation.
        
        Args:
            data: Data to transform
            transformer_name: Name of transformer
            context: Transformation context
            validate: Whether to validate (uses global config if None)
            
        Returns:
            Transformation result
        """
        try:
            start_time = datetime.now()
            
            # Get transformer
            transformer = self.transformers.get(transformer_name)
            if not transformer:
                result = TransformationResult(success=False, data=data, original_data=data)
                result.add_error(f"Transformer not found: {transformer_name}")
                return result
            
            # Check if transformer is enabled
            if not transformer.enabled:
                result = TransformationResult(success=False, data=data, original_data=data)
                result.add_error(f"Transformer disabled: {transformer_name}")
                return result
            
            # Validate input if enabled
            validate_input = validate if validate is not None else self.global_config.get('validate_input', True)
            if validate_input:
                validator = get_centralized_validator()
                validation_result = validator.validate(data, profile="comprehensive")
                if not validation_result.is_valid:
                    result = TransformationResult(success=False, data=data, original_data=data)
                    for error in validation_result.errors:
                        result.add_error(f"Input validation failed: {error.message}")
                    return result
            
            # Apply transformation
            result = transformer.transform(data, context)
            
            # Validate output if enabled and transformation succeeded
            validate_output = validate if validate is not None else self.global_config.get('validate_output', True)
            if result.success and validate_output:
                validator = get_centralized_validator()
                validation_result = validator.validate(result.data, profile="comprehensive")
                if not validation_result.is_valid:
                    for error in validation_result.errors:
                        result.add_warning(f"Output validation failed: {error.message}")
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(result.success, processing_time)
            
            # Add timing metadata
            result.metadata['processing_time'] = processing_time
            result.transformation_id = transformer_name
            
            # Publish transformation event
            self._publish_transformation_event(transformer_name, result)
            
            return result
            
        except Exception as e:
            result = TransformationResult(success=False, data=data, original_data=data)
            result.add_error(f"Transformation failed: {e}")
            logger.error(f"Transformation error in {transformer_name}: {e}")
            return result
    
    def transform_chain(self, 
                       data: Any,
                       chain_name: str,
                       context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """
        Apply transformation chain.
        
        Args:
            data: Data to transform
            chain_name: Name of transformation chain
            context: Transformation context
            
        Returns:
            Final transformation result
        """
        try:
            # Get chain
            chain = self.chains.get(chain_name)
            if not chain:
                result = TransformationResult(success=False, data=data, original_data=data)
                result.add_error(f"Chain not found: {chain_name}")
                return result
            
            # Apply transformations in sequence
            current_data = data
            final_result = TransformationResult(success=True, data=data, original_data=data)
            transformation_history = []
            
            for transformer_name in chain:
                result = self.transform(current_data, transformer_name, context)
                
                if not result.success:
                    final_result.success = False
                    final_result.errors.extend(result.errors)
                    final_result.data = current_data  # Keep last successful data
                    break
                
                current_data = result.data
                transformation_history.append({
                    'transformer': transformer_name,
                    'metadata': result.metadata,
                    'warnings': result.warnings
                })
                
                # Accumulate warnings
                final_result.warnings.extend(result.warnings)
            
            # Set final data and metadata
            if final_result.success:
                final_result.data = current_data
            
            final_result.metadata['chain_name'] = chain_name
            final_result.metadata['transformation_history'] = transformation_history
            final_result.transformation_id = chain_name
            
            return final_result
            
        except Exception as e:
            result = TransformationResult(success=False, data=data, original_data=data)
            result.add_error(f"Chain transformation failed: {e}")
            logger.error(f"Chain transformation error: {e}")
            return result
    
    def transform_batch(self, 
                       data_batch: List[Any],
                       transformer_name: str,
                       context: Optional[Dict[str, Any]] = None,
                       parallel: bool = None) -> List[TransformationResult]:
        """
        Apply transformation to batch of data.
        
        Args:
            data_batch: List of data items to transform
            transformer_name: Name of transformer
            context: Transformation context
            parallel: Whether to use parallel processing
            
        Returns:
            List of transformation results
        """
        use_parallel = parallel if parallel is not None else self.global_config.get('parallel_processing', False)
        
        if use_parallel:
            return self._transform_batch_parallel(data_batch, transformer_name, context)
        else:
            return [self.transform(data_item, transformer_name, context) for data_item in data_batch]
    
    def _transform_batch_parallel(self, 
                                 data_batch: List[Any],
                                 transformer_name: str,
                                 context: Optional[Dict[str, Any]] = None) -> List[TransformationResult]:
        """Apply transformation to batch in parallel."""
        import concurrent.futures
        
        max_workers = self.global_config.get('max_workers', 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.transform, data_item, transformer_name, context)
                for data_item in data_batch
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=self.global_config.get('timeout_seconds', 300))
                    results.append(result)
                except Exception as e:
                    error_result = TransformationResult(success=False, data=None, original_data=None)
                    error_result.add_error(f"Parallel transformation failed: {e}")
                    results.append(error_result)
            
            return results
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update transformation statistics."""
        with self._lock:
            self.stats['total_transformations'] += 1
            
            if success:
                self.stats['successful_transformations'] += 1
            else:
                self.stats['failed_transformations'] += 1
            
            # Update average processing time
            total = self.stats['total_transformations']
            current_avg = self.stats['average_processing_time']
            self.stats['average_processing_time'] = (current_avg * (total - 1) + processing_time) / total
    
    def _publish_transformation_event(self, transformer_name: str, result: TransformationResult):
        """Publish transformation event."""
        event = Event(
            type=EventType.DATA_TRANSFORMATION,
            data={
                'transformer': transformer_name,
                'success': result.success,
                'errors': len(result.errors),
                'warnings': len(result.warnings),
                'processing_time': result.metadata.get('processing_time', 0)
            },
            source="SharedTransformationEngine"
        )
        
        self.event_bus.publish(event)
    
    def get_transformer_stats(self, transformer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get transformer statistics.
        
        Args:
            transformer_name: Specific transformer name (all if None)
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            if transformer_name:
                transformer = self.transformers.get(transformer_name)
                if transformer:
                    return {
                        'transformer': transformer_name,
                        'enabled': transformer.enabled,
                        'stats': transformer.stats.copy(),
                        'config': {
                            'type': transformer.config.transformation_type.value,
                            'scope': transformer.config.scope.value,
                            'parameters': transformer.config.parameters
                        }
                    }
                return {}
            else:
                return {
                    'global_stats': self.stats.copy(),
                    'transformer_count': len(self.transformers),
                    'chain_count': len(self.chains),
                    'transformers': {
                        name: {
                            'enabled': transformer.enabled,
                            'stats': transformer.stats.copy()
                        }
                        for name, transformer in self.transformers.items()
                    }
                }
    
    def configure_transformer(self, 
                             transformer_name: str, 
                             parameters: Dict[str, Any]) -> bool:
        """
        Configure transformer parameters.
        
        Args:
            transformer_name: Transformer name
            parameters: New parameters
            
        Returns:
            True if configured successfully
        """
        try:
            with self._lock:
                transformer = self.transformers.get(transformer_name)
                if transformer:
                    transformer.config.parameters.update(parameters)
                    logger.info(f"Configured transformer {transformer_name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to configure transformer {transformer_name}: {e}")
            return False
    
    def enable_transformer(self, transformer_name: str) -> bool:
        """Enable transformer."""
        return self._set_transformer_state(transformer_name, True)
    
    def disable_transformer(self, transformer_name: str) -> bool:
        """Disable transformer."""
        return self._set_transformer_state(transformer_name, False)
    
    def _set_transformer_state(self, transformer_name: str, enabled: bool) -> bool:
        """Set transformer enabled state."""
        try:
            with self._lock:
                transformer = self.transformers.get(transformer_name)
                if transformer:
                    transformer.enabled = enabled
                    logger.info(f"{'Enabled' if enabled else 'Disabled'} transformer: {transformer_name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to set transformer state {transformer_name}: {e}")
            return False


# Global transformation engine instance
_transformation_engine = None

def get_transformation_engine() -> SharedTransformationEngine:
    """Get global transformation engine instance."""
    global _transformation_engine
    if _transformation_engine is None:
        _transformation_engine = SharedTransformationEngine()
    return _transformation_engine


# Convenience functions

def transform_text(text: str, **kwargs) -> TransformationResult:
    """
    Transform text data.
    
    Args:
        text: Text to transform
        **kwargs: Transformation parameters
        
    Returns:
        Transformation result
    """
    engine = get_transformation_engine()
    
    if kwargs:
        engine.configure_transformer("text_normalizer", kwargs)
    
    return engine.transform(text, "text_normalizer")


def transform_numeric(data: Union[List, np.ndarray, int, float], **kwargs) -> TransformationResult:
    """
    Transform numeric data.
    
    Args:
        data: Numeric data to transform
        **kwargs: Transformation parameters
        
    Returns:
        Transformation result
    """
    engine = get_transformation_engine()
    
    if kwargs:
        engine.configure_transformer("numeric_normalizer", kwargs)
    
    return engine.transform(data, "numeric_normalizer")


def transform_dataframe(df: pd.DataFrame, **kwargs) -> TransformationResult:
    """
    Transform DataFrame data.
    
    Args:
        df: DataFrame to transform
        **kwargs: Transformation parameters
        
    Returns:
        Transformation result
    """
    engine = get_transformation_engine()
    
    if kwargs:
        engine.configure_transformer("dataframe_processor", kwargs)
    
    return engine.transform(df, "dataframe_processor")


def transform_json(data: Union[str, Dict, List], **kwargs) -> TransformationResult:
    """
    Transform JSON data.
    
    Args:
        data: JSON data to transform
        **kwargs: Transformation parameters
        
    Returns:
        Transformation result
    """
    engine = get_transformation_engine()
    
    if kwargs:
        engine.configure_transformer("json_processor", kwargs)
    
    return engine.transform(data, "json_processor")


def compress_data(data: Any, **kwargs) -> TransformationResult:
    """
    Compress data.
    
    Args:
        data: Data to compress
        **kwargs: Compression parameters
        
    Returns:
        Transformation result
    """
    engine = get_transformation_engine()
    
    if kwargs:
        engine.configure_transformer("data_compressor", kwargs)
    
    return engine.transform(data, "data_compressor")