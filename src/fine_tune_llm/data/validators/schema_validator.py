"""
Schema validation for training data.

This module provides comprehensive data validation capabilities
to ensure data quality and consistency for model training.
"""

import json
import jsonschema
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ...core.interfaces import BaseComponent
from ...core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class SchemaValidator(BaseComponent):
    """
    Schema validator for training data validation.
    
    Provides JSON schema validation, custom validation rules,
    and data quality checks for training datasets.
    """
    
    def __init__(self, 
                 schema: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize schema validator.
        
        Args:
            schema: JSON schema for validation
            config: Configuration dictionary
        """
        self.config = config or {}
        self.validation_config = self.config.get('validation', {})
        
        # Schema configuration
        self.schema = schema or self._get_default_schema()
        self.strict_mode = self.validation_config.get('strict_mode', True)
        self.allow_additional_fields = self.validation_config.get('allow_additional_fields', False)
        
        # Validation rules
        self.custom_validators = {}
        self.required_fields = self.validation_config.get('required_fields', [])
        self.field_types = self.validation_config.get('field_types', {})
        
        # Quality checks
        self.enable_quality_checks = self.validation_config.get('enable_quality_checks', True)
        self.quality_thresholds = self.validation_config.get('quality_thresholds', {})
        
        # Statistics
        self.validation_stats = {
            'total_validated': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors_by_type': {}
        }
        
        # Compile schema validator
        self._compile_schema()
        
        logger.info(f"Initialized SchemaValidator with schema validation enabled")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.validation_config = self.config.get('validation', {})
    
    def cleanup(self) -> None:
        """Clean up validation resources."""
        self.validation_stats = {
            'total_validated': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors_by_type': {}
        }
    
    @property
    def name(self) -> str:
        """Component name."""
        return "SchemaValidator"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default JSON schema for LLM training data."""
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10000
                },
                "label": {
                    "type": ["string", "integer"]
                },
                "input": {
                    "type": "string",
                    "minLength": 1
                },
                "output": {
                    "type": "string",
                    "minLength": 1
                },
                "metadata": {
                    "type": "object"
                }
            },
            "anyOf": [
                {"required": ["text", "label"]},
                {"required": ["input", "output"]},
                {"required": ["text"]}
            ],
            "additionalProperties": True
        }
    
    def _compile_schema(self):
        """Compile JSON schema validator."""
        try:
            self.validator = jsonschema.Draft7Validator(self.schema)
        except Exception as e:
            logger.error(f"Error compiling schema: {e}")
            raise ValidationError(f"Invalid schema: {e}")
    
    def validate_sample(self, 
                       sample: Dict[str, Any],
                       raise_on_error: bool = True) -> Dict[str, Any]:
        """
        Validate a single data sample.
        
        Args:
            sample: Data sample to validate
            raise_on_error: Whether to raise exception on validation failure
            
        Returns:
            Validation result dictionary
        """
        self.validation_stats['total_validated'] += 1
        
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0
        }
        
        try:
            # JSON Schema validation
            errors = list(self.validator.iter_errors(sample))
            if errors:
                result['valid'] = False
                for error in errors:
                    error_msg = f"Schema error at {'.'.join(str(p) for p in error.path)}: {error.message}"
                    result['errors'].append(error_msg)
                    self._update_error_stats(error.validator)
            
            # Custom field validation
            field_errors = self._validate_fields(sample)
            if field_errors:
                result['valid'] = False
                result['errors'].extend(field_errors)
            
            # Quality checks
            if self.enable_quality_checks:
                quality_result = self._check_quality(sample)
                result['quality_score'] = quality_result['score']
                result['warnings'].extend(quality_result['warnings'])
                
                # Mark as invalid if quality is too low
                min_quality = self.quality_thresholds.get('min_quality_score', 0.3)
                if quality_result['score'] < min_quality:
                    result['valid'] = False
                    result['errors'].append(f"Quality score {quality_result['score']:.2f} below threshold {min_quality}")
            
            # Custom validators
            for name, validator_func in self.custom_validators.items():
                try:
                    custom_result = validator_func(sample)
                    if not custom_result.get('valid', True):
                        result['valid'] = False
                        result['errors'].extend(custom_result.get('errors', []))
                        result['warnings'].extend(custom_result.get('warnings', []))
                except Exception as e:
                    result['warnings'].append(f"Custom validator {name} failed: {e}")
            
            # Update statistics
            if result['valid']:
                self.validation_stats['passed'] += 1
            else:
                self.validation_stats['failed'] += 1
            
            if result['warnings']:
                self.validation_stats['warnings'] += 1
            
            # Raise exception if requested and validation failed
            if raise_on_error and not result['valid']:
                error_msg = f"Validation failed: {'; '.join(result['errors'])}"
                raise ValidationError(error_msg)
            
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            if raise_on_error:
                raise ValidationError(f"Validation error: {e}")
            
            result['valid'] = False
            result['errors'].append(f"Validation exception: {e}")
            return result
    
    def validate_dataset(self, 
                        dataset: List[Dict[str, Any]],
                        sample_size: Optional[int] = None,
                        raise_on_error: bool = False) -> Dict[str, Any]:
        """
        Validate an entire dataset.
        
        Args:
            dataset: List of data samples
            sample_size: Optional sample size for validation (None for all)
            raise_on_error: Whether to raise exception on any validation failure
            
        Returns:
            Dataset validation summary
        """
        # Sample dataset if requested
        if sample_size and len(dataset) > sample_size:
            import random
            sampled_data = random.sample(dataset, sample_size)
        else:
            sampled_data = dataset
        
        # Validate samples
        results = []
        for i, sample in enumerate(sampled_data):
            try:
                result = self.validate_sample(sample, raise_on_error=False)
                result['sample_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error validating sample {i}: {e}")
                if raise_on_error:
                    raise
        
        # Compute summary statistics
        summary = self._compute_dataset_summary(results, len(dataset))
        
        # Raise exception if requested and any validation failed
        if raise_on_error and summary['failed_count'] > 0:
            raise ValidationError(f"Dataset validation failed: {summary['failed_count']} samples failed")
        
        return summary
    
    def _validate_fields(self, sample: Dict[str, Any]) -> List[str]:
        """Validate specific fields with custom rules."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in sample:
                errors.append(f"Missing required field: {field}")
            elif sample[field] is None or (isinstance(sample[field], str) and not sample[field].strip()):
                errors.append(f"Required field {field} is empty")
        
        # Check field types
        for field, expected_type in self.field_types.items():
            if field in sample:
                value = sample[field]
                if expected_type == 'string' and not isinstance(value, str):
                    errors.append(f"Field {field} must be string, got {type(value).__name__}")
                elif expected_type == 'integer' and not isinstance(value, int):
                    errors.append(f"Field {field} must be integer, got {type(value).__name__}")
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    errors.append(f"Field {field} must be number, got {type(value).__name__}")
                elif expected_type == 'array' and not isinstance(value, list):
                    errors.append(f"Field {field} must be array, got {type(value).__name__}")
                elif expected_type == 'object' and not isinstance(value, dict):
                    errors.append(f"Field {field} must be object, got {type(value).__name__}")
        
        return errors
    
    def _check_quality(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality checks on sample."""
        warnings = []
        quality_factors = []
        
        # Text length checks
        text_fields = ['text', 'input', 'output', 'content']
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                text = sample[field]
                length = len(text)
                
                # Length quality factor
                min_len = self.quality_thresholds.get(f'{field}_min_length', 10)
                max_len = self.quality_thresholds.get(f'{field}_max_length', 5000)
                
                if length < min_len:
                    warnings.append(f"{field} too short ({length} < {min_len})")
                    quality_factors.append(0.3)
                elif length > max_len:
                    warnings.append(f"{field} too long ({length} > {max_len})")
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(1.0)
                
                # Word count quality
                words = text.split()
                min_words = self.quality_thresholds.get(f'{field}_min_words', 3)
                if len(words) < min_words:
                    warnings.append(f"{field} has too few words ({len(words)} < {min_words})")
                    quality_factors.append(0.4)
                else:
                    quality_factors.append(1.0)
                
                # Repetition check
                if self._has_excessive_repetition(text):
                    warnings.append(f"{field} has excessive repetition")
                    quality_factors.append(0.5)
                else:
                    quality_factors.append(1.0)
        
        # Compute overall quality score
        if quality_factors:
            quality_score = sum(quality_factors) / len(quality_factors)
        else:
            quality_score = 1.0
        
        return {
            'score': quality_score,
            'warnings': warnings
        }
    
    def _has_excessive_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text has excessive repetition."""
        if len(text) < 20:
            return False
        
        # Check for repeated 4-grams
        ngrams = [text[i:i+4] for i in range(len(text) - 3)]
        unique_ngrams = set(ngrams)
        
        if len(ngrams) == 0:
            return False
        
        repetition_ratio = 1 - (len(unique_ngrams) / len(ngrams))
        return repetition_ratio > threshold
    
    def _compute_dataset_summary(self, 
                                results: List[Dict[str, Any]],
                                total_size: int) -> Dict[str, Any]:
        """Compute dataset validation summary."""
        total_validated = len(results)
        passed_count = sum(1 for r in results if r['valid'])
        failed_count = total_validated - passed_count
        warning_count = sum(1 for r in results if r['warnings'])
        
        # Quality statistics
        quality_scores = [r['quality_score'] for r in results]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Error analysis
        all_errors = []
        for r in results:
            all_errors.extend(r['errors'])
        
        error_types = {}
        for error in all_errors:
            error_type = error.split(':')[0] if ':' in error else 'unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_dataset_size': total_size,
            'validated_count': total_validated,
            'passed_count': passed_count,
            'failed_count': failed_count,
            'warning_count': warning_count,
            'success_rate': passed_count / total_validated if total_validated > 0 else 0.0,
            'average_quality_score': avg_quality,
            'error_types': error_types,
            'validation_coverage': total_validated / total_size if total_size > 0 else 0.0
        }
    
    def _update_error_stats(self, error_type: str):
        """Update error statistics."""
        if error_type not in self.validation_stats['errors_by_type']:
            self.validation_stats['errors_by_type'][error_type] = 0
        self.validation_stats['errors_by_type'][error_type] += 1
    
    def add_custom_validator(self, 
                           name: str,
                           validator_func: callable):
        """
        Add custom validation function.
        
        Args:
            name: Validator name
            validator_func: Function that takes sample and returns validation result
        """
        self.custom_validators[name] = validator_func
        logger.info(f"Added custom validator: {name}")
    
    def set_schema(self, schema: Dict[str, Any]):
        """Update validation schema."""
        self.schema = schema
        self._compile_schema()
        logger.info("Updated validation schema")
    
    def set_required_fields(self, fields: List[str]):
        """Update required fields list."""
        self.required_fields = fields
    
    def set_quality_thresholds(self, thresholds: Dict[str, Any]):
        """Update quality check thresholds."""
        self.quality_thresholds.update(thresholds)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        
        if stats['total_validated'] > 0:
            stats['success_rate'] = stats['passed'] / stats['total_validated']
            stats['failure_rate'] = stats['failed'] / stats['total_validated']
            stats['warning_rate'] = stats['warnings'] / stats['total_validated']
        
        return stats
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validated': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors_by_type': {}
        }
    
    def export_schema(self, output_path: Path):
        """Export current schema to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.schema, f, indent=2)
        
        logger.info(f"Exported schema to {output_path}")
    
    def load_schema(self, schema_path: Path):
        """Load schema from file."""
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        self.set_schema(schema)
        logger.info(f"Loaded schema from {schema_path}")