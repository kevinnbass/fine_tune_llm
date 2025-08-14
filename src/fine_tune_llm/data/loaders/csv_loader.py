"""
CSV data loader for tabular training datasets.

This module provides loading capabilities for CSV files commonly
used in machine learning with robust parsing and type inference.
"""

import csv
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from .base import BaseDataLoader
from ...core.exceptions import DataError

logger = logging.getLogger(__name__)


class CsvDataLoader(BaseDataLoader):
    """
    CSV data loader for tabular data files.
    
    Supports loading from CSV files with automatic type inference,
    flexible delimiter detection, and robust error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV data loader.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # CSV-specific configuration
        self.csv_config = self.config.get('csv', {})
        
        # Parsing options
        self.delimiter = self.csv_config.get('delimiter', None)  # Auto-detect if None
        self.encoding = self.csv_config.get('encoding', 'utf-8')
        self.has_header = self.csv_config.get('has_header', True)
        self.skip_rows = self.csv_config.get('skip_rows', 0)
        
        # Data processing options
        self.infer_types = self.csv_config.get('infer_types', True)
        self.convert_numbers = self.csv_config.get('convert_numbers', True)
        self.convert_booleans = self.csv_config.get('convert_booleans', True)
        self.strip_whitespace = self.csv_config.get('strip_whitespace', True)
        
        # Column handling
        self.column_names = self.csv_config.get('column_names', None)
        self.selected_columns = self.csv_config.get('selected_columns', None)
        self.column_mapping = self.csv_config.get('column_mapping', {})
        
        logger.info(f"Initialized CsvDataLoader with encoding={self.encoding}")
    
    def load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a single CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of dictionaries representing rows
        """
        if not file_path.exists():
            raise DataError(f"CSV file not found: {file_path}")
        
        if file_path.suffix.lower() != '.csv':
            logger.warning(f"File may not be CSV format: {file_path}")
        
        try:
            # Detect delimiter if not specified
            delimiter = self._detect_delimiter(file_path) if self.delimiter is None else self.delimiter
            
            data = []
            with open(file_path, 'r', encoding=self.encoding, newline='') as f:
                # Skip initial rows if specified
                for _ in range(self.skip_rows):
                    next(f, None)
                
                # Set up CSV reader
                reader = csv.DictReader(f, delimiter=delimiter, fieldnames=self.column_names)
                
                # Skip header row if using custom fieldnames
                if not self.has_header and self.column_names is None:
                    # Generate default column names
                    first_row = next(reader, None)
                    if first_row:
                        num_cols = len(first_row)
                        column_names = [f"column_{i}" for i in range(num_cols)]
                        f.seek(0)
                        for _ in range(self.skip_rows):
                            next(f, None)
                        reader = csv.DictReader(f, delimiter=delimiter, fieldnames=column_names)
                
                # Read all rows
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Process row
                        processed_row = self._process_row(row, row_num)
                        if processed_row:
                            data.append(processed_row)
                    except Exception as e:
                        if self.skip_invalid:
                            logger.warning(f"Skipping invalid row {row_num}: {e}")
                            continue
                        else:
                            raise DataError(f"Error processing row {row_num}: {e}")
            
            logger.info(f"Loaded {len(data)} rows from {file_path}")
            return data
            
        except UnicodeDecodeError as e:
            raise DataError(f"Encoding error in {file_path}: {e}")
        except IOError as e:
            raise DataError(f"IO error reading {file_path}: {e}")
    
    def load_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from all CSV files in a directory.
        
        Args:
            directory_path: Path to directory containing CSV files
            
        Returns:
            List of dictionaries from all files
        """
        if not directory_path.exists():
            raise DataError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DataError(f"Path is not a directory: {directory_path}")
        
        # Find CSV files
        csv_files = list(directory_path.glob('*.csv'))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(csv_files)} CSV files in {directory_path}")
        
        # Load all files
        all_data = []
        for file_path in sorted(csv_files):
            try:
                file_data = self.load_file(file_path)
                
                # Add source file information
                for row in file_data:
                    row['_source_file'] = str(file_path)
                
                all_data.extend(file_data)
                
            except Exception as e:
                if self.skip_invalid:
                    logger.error(f"Skipping file {file_path}: {e}")
                    continue
                else:
                    raise
        
        logger.info(f"Loaded total {len(all_data)} rows from directory")
        return all_data
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """Detect CSV delimiter by sampling the file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                # Read first few lines for detection
                sample = ""
                for i, line in enumerate(f):
                    sample += line
                    if i >= 10:  # Sample first 10 lines
                        break
                
                # Use csv.Sniffer to detect delimiter
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample, delimiters=',;\t|').delimiter
                
                logger.info(f"Detected delimiter: '{delimiter}'")
                return delimiter
                
        except Exception as e:
            logger.warning(f"Could not detect delimiter, using comma: {e}")
            return ','
    
    def _process_row(self, row: Dict[str, str], row_num: int) -> Optional[Dict[str, Any]]:
        """
        Process a single CSV row.
        
        Args:
            row: Raw CSV row as dictionary
            row_num: Row number for error reporting
            
        Returns:
            Processed row or None if row should be skipped
        """
        if not row:
            return None
        
        processed_row = {}
        
        for original_key, value in row.items():
            # Skip None keys (can happen with malformed CSV)
            if original_key is None:
                continue
            
            # Apply column mapping
            key = self.column_mapping.get(original_key, original_key)
            
            # Skip columns not in selected list
            if self.selected_columns and original_key not in self.selected_columns:
                continue
            
            # Strip whitespace if requested
            if self.strip_whitespace and isinstance(value, str):
                value = value.strip()
            
            # Skip empty values if they represent missing data
            if value == '' or value is None:
                processed_row[key] = None
                continue
            
            # Type inference and conversion
            if self.infer_types:
                value = self._infer_and_convert_type(value)
            
            processed_row[key] = value
        
        # Validate processed row
        if not processed_row:
            return None
        
        return processed_row
    
    def _infer_and_convert_type(self, value: str) -> Any:
        """
        Infer and convert string value to appropriate type.
        
        Args:
            value: String value from CSV
            
        Returns:
            Converted value
        """
        if not isinstance(value, str):
            return value
        
        original_value = value
        value = value.strip()
        
        # Handle empty string
        if not value:
            return None
        
        # Boolean conversion
        if self.convert_booleans:
            if value.lower() in ('true', 'yes', '1', 'on'):
                return True
            elif value.lower() in ('false', 'no', '0', 'off'):
                return False
        
        # Numeric conversion
        if self.convert_numbers:
            try:
                # Try integer first
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return int(value)
                
                # Try float
                return float(value)
                
            except (ValueError, OverflowError):
                pass
        
        # Return as string
        return original_value
    
    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        """
        Validate a loaded CSV row.
        
        Args:
            sample: CSV row to validate
        """
        # Call parent validation
        super()._validate_sample(sample)
        
        # CSV-specific validation
        if not isinstance(sample, dict):
            raise ValueError(f"CSV sample must be dictionary, got {type(sample)}")
        
        # Check for required columns
        required_columns = self.csv_config.get('required_columns', [])
        for column in required_columns:
            if column not in sample:
                raise ValueError(f"Missing required column: {column}")
            if sample[column] is None:
                raise ValueError(f"Required column {column} has null value")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.csv']
    
    def write_csv(self, data: List[Dict[str, Any]], output_path: Path):
        """
        Write data to CSV file.
        
        Args:
            data: List of dictionaries to write
            output_path: Output file path
        """
        if not data:
            logger.warning("No data to write to CSV")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get all unique column names
            all_columns = set()
            for row in data:
                all_columns.update(row.keys())
            
            # Remove internal columns
            columns = [col for col in sorted(all_columns) if not col.startswith('_')]
            
            with open(output_path, 'w', encoding=self.encoding, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                
                if self.has_header:
                    writer.writeheader()
                
                for row in data:
                    # Filter out internal columns
                    filtered_row = {k: v for k, v in row.items() if k in columns}
                    writer.writerow(filtered_row)
            
            logger.info(f"Written {len(data)} rows to {output_path}")
            
        except Exception as e:
            raise DataError(f"Error writing CSV file {output_path}: {e}")
    
    def get_column_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about columns in CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Column information dictionary
        """
        try:
            # Load a sample of the data
            sample_data = []
            delimiter = self._detect_delimiter(file_path) if self.delimiter is None else self.delimiter
            
            with open(file_path, 'r', encoding=self.encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # Read first 100 rows for analysis
                for i, row in enumerate(reader):
                    if i >= 100:
                        break
                    sample_data.append(row)
            
            if not sample_data:
                return {'columns': [], 'total_rows': 0}
            
            # Analyze columns
            column_info = {}
            column_names = list(sample_data[0].keys())
            
            for col_name in column_names:
                values = [row.get(col_name, '') for row in sample_data]
                non_empty_values = [v for v in values if v and v.strip()]
                
                column_info[col_name] = {
                    'total_values': len(values),
                    'non_empty_values': len(non_empty_values),
                    'empty_ratio': 1 - (len(non_empty_values) / len(values)),
                    'sample_values': non_empty_values[:5],
                    'inferred_type': self._infer_column_type(non_empty_values)
                }
            
            return {
                'columns': column_names,
                'total_columns': len(column_names),
                'sample_rows': len(sample_data),
                'column_info': column_info,
                'delimiter': delimiter,
                'encoding': self.encoding
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _infer_column_type(self, values: List[str]) -> str:
        """Infer the most likely type for a column based on its values."""
        if not values:
            return 'empty'
        
        # Count type occurrences
        type_counts = {
            'integer': 0,
            'float': 0,
            'boolean': 0,
            'string': 0
        }
        
        for value in values[:50]:  # Sample first 50 values
            value = str(value).strip().lower()
            
            # Check boolean
            if value in ('true', 'false', 'yes', 'no', '1', '0', 'on', 'off'):
                type_counts['boolean'] += 1
            # Check integer
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                type_counts['integer'] += 1
            # Check float
            else:
                try:
                    float(value)
                    type_counts['float'] += 1
                except ValueError:
                    type_counts['string'] += 1
        
        # Return most common type
        return max(type_counts, key=type_counts.get)