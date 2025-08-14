"""
JSON data loader for structured training datasets.

This module provides loading capabilities for JSON files commonly
used in machine learning datasets with nested data structures.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .base import BaseDataLoader
from ...core.exceptions import DataError

logger = logging.getLogger(__name__)


class JsonDataLoader(BaseDataLoader):
    """
    JSON data loader for structured data files.
    
    Supports loading from single JSON files or directories containing
    multiple JSON files with robust error handling and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON data loader.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # JSON-specific configuration
        self.json_config = self.config.get('json', {})
        
        # Encoding and parsing options
        self.encoding = self.json_config.get('encoding', 'utf-8')
        self.strict_parsing = self.json_config.get('strict_parsing', False)
        self.allow_comments = self.json_config.get('allow_comments', False)
        
        # Data structure handling
        self.flatten_nested = self.json_config.get('flatten_nested', False)
        self.max_depth = self.json_config.get('max_depth', 10)
        
        logger.info(f"Initialized JsonDataLoader with encoding={self.encoding}")
    
    def load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a single JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of loaded JSON objects
        """
        if not file_path.exists():
            raise DataError(f"JSON file not found: {file_path}")
        
        if file_path.suffix.lower() != '.json':
            logger.warning(f"File may not be JSON format: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
                
                # Remove comments if allowed
                if self.allow_comments:
                    content = self._remove_json_comments(content)
                
                # Parse JSON
                data = json.loads(content)
                
                # Convert to list format if necessary
                if isinstance(data, dict):
                    # Single object - wrap in list
                    data = [data]
                elif isinstance(data, list):
                    # List of objects - use as is
                    pass
                else:
                    # Other types - convert to list
                    data = [{'value': data}]
                
                # Flatten nested structures if requested
                if self.flatten_nested:
                    data = [self._flatten_dict(item) for item in data if isinstance(item, dict)]
                
                # Validate data structure
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        if self.strict_parsing:
                            raise DataError(f"Item {i} is not a dictionary: {type(item)}")
                        else:
                            logger.warning(f"Converting non-dict item {i} to dict")
                            data[i] = {'value': item}
                
                logger.info(f"Loaded {len(data)} items from {file_path}")
                return data
                
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error in {file_path}: {e}"
            if self.strict_parsing:
                raise DataError(error_msg)
            else:
                logger.error(error_msg)
                return []
        
        except UnicodeDecodeError as e:
            raise DataError(f"Encoding error in {file_path}: {e}")
        except IOError as e:
            raise DataError(f"IO error reading {file_path}: {e}")
    
    def load_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from all JSON files in a directory.
        
        Args:
            directory_path: Path to directory containing JSON files
            
        Returns:
            List of loaded JSON objects from all files
        """
        if not directory_path.exists():
            raise DataError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DataError(f"Path is not a directory: {directory_path}")
        
        # Find JSON files
        json_files = list(directory_path.glob('*.json'))
        
        if not json_files:
            logger.warning(f"No JSON files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(json_files)} JSON files in {directory_path}")
        
        # Load all files
        all_data = []
        for file_path in sorted(json_files):
            try:
                file_data = self.load_file(file_path)
                
                # Add source file information to each item
                for item in file_data:
                    if isinstance(item, dict):
                        item['_source_file'] = str(file_path)
                
                all_data.extend(file_data)
                
            except Exception as e:
                if self.skip_invalid:
                    logger.error(f"Skipping file {file_path}: {e}")
                    continue
                else:
                    raise
        
        logger.info(f"Loaded total {len(all_data)} items from directory")
        return all_data
    
    def _remove_json_comments(self, content: str) -> str:
        """Remove comments from JSON content (non-standard JSON)."""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove line comments (// style)
            if '//' in line:
                comment_pos = line.find('//')
                # Make sure it's not inside a string
                in_string = False
                escape_next = False
                
                for i, char in enumerate(line):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    if i == comment_pos and not in_string:
                        line = line[:comment_pos]
                        break
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict) and len(str(new_key).split(sep)) < self.max_depth:
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Handle list of dictionaries
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        list_key = f"{new_key}[{i}]"
                        items.extend(self._flatten_dict(item, list_key, sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        """
        Validate a loaded JSON object.
        
        Args:
            sample: JSON object to validate
        """
        # Call parent validation
        super()._validate_sample(sample)
        
        # JSON-specific validation
        if not isinstance(sample, dict):
            raise ValueError(f"JSON sample must be dictionary, got {type(sample)}")
        
        # Check maximum nesting depth
        if self._get_nesting_depth(sample) > self.max_depth:
            raise ValueError(f"JSON object exceeds maximum nesting depth: {self.max_depth}")
        
        # Check for required fields (if configured)
        required_fields = self.json_config.get('required_fields', [])
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Missing required field: {field}")
    
    def _get_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of an object."""
        if not isinstance(obj, (dict, list)):
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(
                self._get_nesting_depth(value, current_depth + 1)
                for value in obj.values()
            )
        
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(
                self._get_nesting_depth(item, current_depth + 1)
                for item in obj
            )
        
        return current_depth
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.json']
    
    def write_json(self, data: List[Dict[str, Any]], output_path: Path, indent: int = 2):
        """
        Write data to JSON file.
        
        Args:
            data: List of dictionaries to write
            output_path: Output file path
            indent: JSON indentation level
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            logger.info(f"Written {len(data)} items to {output_path}")
            
        except Exception as e:
            raise DataError(f"Error writing JSON file {output_path}: {e}")
    
    def validate_json_structure(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate JSON file structure without loading all data.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Validation result dictionary
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                # Try to parse the JSON
                data = json.load(f)
                
                return {
                    'valid': True,
                    'type': type(data).__name__,
                    'length': len(data) if hasattr(data, '__len__') else 1,
                    'max_depth': self._get_nesting_depth(data),
                    'sample_keys': list(data.keys()) if isinstance(data, dict) else [],
                    'file_size': file_path.stat().st_size
                }
                
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'error': str(e),
                'error_type': 'JSONDecodeError',
                'file_size': file_path.stat().st_size
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'file_size': file_path.stat().st_size
            }