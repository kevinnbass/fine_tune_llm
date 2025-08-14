"""
JSONL (JSON Lines) data loader for LLM training datasets.

This module provides loading capabilities for JSONL files commonly
used in machine learning datasets.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .base import BaseDataLoader
from ...core.exceptions import DataError

logger = logging.getLogger(__name__)


class JsonlDataLoader(BaseDataLoader):
    """
    JSONL data loader for line-delimited JSON files.
    
    Supports loading from single JSONL files or directories containing
    multiple JSONL files with robust error handling and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize JSONL data loader.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # JSONL-specific configuration
        self.jsonl_config = self.config.get('jsonl', {})
        
        # Encoding and parsing options
        self.encoding = self.jsonl_config.get('encoding', 'utf-8')
        self.strict_parsing = self.jsonl_config.get('strict_parsing', False)
        self.skip_empty_lines = self.jsonl_config.get('skip_empty_lines', True)
        self.skip_comments = self.jsonl_config.get('skip_comments', True)
        
        # Line processing options
        self.max_line_length = self.jsonl_config.get('max_line_length', 1000000)  # 1MB
        self.strip_lines = self.jsonl_config.get('strip_lines', True)
        
        logger.info(f"Initialized JsonlDataLoader with encoding={self.encoding}")
    
    def load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a single JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of loaded JSON objects
        """
        if not file_path.exists():
            raise DataError(f"JSONL file not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.jsonl', '.json']:
            logger.warning(f"File may not be JSONL format: {file_path}")
        
        data = []
        line_number = 0
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                for line in f:
                    line_number += 1
                    
                    # Process line
                    processed_line = self._process_line(line, line_number)
                    if processed_line is None:
                        continue
                    
                    # Parse JSON
                    try:
                        json_obj = json.loads(processed_line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        error_msg = f"JSON parse error at line {line_number}: {e}"
                        
                        if self.strict_parsing:
                            raise DataError(error_msg)
                        else:
                            logger.warning(error_msg)
                            self.stats['skipped_samples'] += 1
                            continue
                
        except UnicodeDecodeError as e:
            raise DataError(f"Encoding error in {file_path}: {e}")
        except IOError as e:
            raise DataError(f"IO error reading {file_path}: {e}")
        
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def load_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from all JSONL files in a directory.
        
        Args:
            directory_path: Path to directory containing JSONL files
            
        Returns:
            List of loaded JSON objects from all files
        """
        if not directory_path.exists():
            raise DataError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DataError(f"Path is not a directory: {directory_path}")
        
        # Find JSONL files
        jsonl_files = list(directory_path.glob('*.jsonl')) + list(directory_path.glob('*.json'))
        
        if not jsonl_files:
            logger.warning(f"No JSONL files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(jsonl_files)} JSONL files in {directory_path}")
        
        # Load all files
        all_data = []
        for file_path in sorted(jsonl_files):
            try:
                file_data = self.load_file(file_path)
                all_data.extend(file_data)
            except Exception as e:
                if self.skip_invalid:
                    logger.error(f"Skipping file {file_path}: {e}")
                    continue
                else:
                    raise
        
        logger.info(f"Loaded total {len(all_data)} samples from directory")
        return all_data
    
    def _process_line(self, line: str, line_number: int) -> Optional[str]:
        """
        Process a single line from JSONL file.
        
        Args:
            line: Raw line from file
            line_number: Line number for error reporting
            
        Returns:
            Processed line or None if line should be skipped
        """
        # Strip whitespace if requested
        if self.strip_lines:
            line = line.strip()
        
        # Skip empty lines
        if self.skip_empty_lines and not line:
            return None
        
        # Skip comment lines (starting with # or //)
        if self.skip_comments and (line.startswith('#') or line.startswith('//')):
            return None
        
        # Check line length
        if len(line) > self.max_line_length:
            error_msg = f"Line {line_number} exceeds maximum length ({len(line)} > {self.max_line_length})"
            
            if self.strict_parsing:
                raise DataError(error_msg)
            else:
                logger.warning(error_msg)
                return None
        
        return line
    
    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        """
        Validate a loaded JSON object.
        
        Args:
            sample: JSON object to validate
        """
        # Call parent validation
        super()._validate_sample(sample)
        
        # JSONL-specific validation
        if not isinstance(sample, dict):
            raise ValueError(f"JSONL sample must be JSON object, got {type(sample)}")
        
        # Check for common required fields (can be configured)
        required_fields = self.jsonl_config.get('required_fields', [])
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Missing required field: {field}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.jsonl', '.json']
    
    def write_jsonl(self, data: List[Dict[str, Any]], output_path: Path):
        """
        Write data to JSONL file.
        
        Args:
            data: List of JSON objects to write
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding=self.encoding) as f:
                for item in data:
                    json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
            
            logger.info(f"Written {len(data)} samples to {output_path}")
            
        except Exception as e:
            raise DataError(f"Error writing JSONL file {output_path}: {e}")
    
    def append_jsonl(self, data: List[Dict[str, Any]], output_path: Path):
        """
        Append data to existing JSONL file.
        
        Args:
            data: List of JSON objects to append
            output_path: Output file path
        """
        try:
            with open(output_path, 'a', encoding=self.encoding) as f:
                for item in data:
                    json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
            
            logger.info(f"Appended {len(data)} samples to {output_path}")
            
        except Exception as e:
            raise DataError(f"Error appending to JSONL file {output_path}: {e}")
    
    def sample_jsonl(self, file_path: Path, n_samples: int, random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample random lines from JSONL file without loading entire file.
        
        Args:
            file_path: Path to JSONL file
            n_samples: Number of samples to extract
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled JSON objects
        """
        import random
        if random_seed is not None:
            random.seed(random_seed)
        
        # First pass: count total lines
        total_lines = 0
        with open(file_path, 'r', encoding=self.encoding) as f:
            for _ in f:
                total_lines += 1
        
        if n_samples >= total_lines:
            # Return all data if sampling more than available
            return self.load_file(file_path)
        
        # Select random line numbers
        selected_lines = set(random.sample(range(total_lines), n_samples))
        
        # Second pass: load selected lines
        sampled_data = []
        line_number = 0
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                if line_number in selected_lines:
                    processed_line = self._process_line(line, line_number)
                    if processed_line:
                        try:
                            json_obj = json.loads(processed_line)
                            sampled_data.append(json_obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_number}: {e}")
                
                line_number += 1
        
        logger.info(f"Sampled {len(sampled_data)} samples from {file_path}")
        return sampled_data