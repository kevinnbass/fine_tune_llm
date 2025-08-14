"""
File system adapter for hexagonal architecture.

This module provides file system operations through the adapter pattern,
allowing the core business logic to remain independent of specific
file system implementations.
"""

import os
import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO, TextIO
from datetime import datetime
import logging

from ..core.interfaces import FileSystemPort
from ..core.exceptions import SystemError, DataError
from ..utils.resilience import circuit_breaker, retry

logger = logging.getLogger(__name__)


class FileSystemAdapter(FileSystemPort):
    """
    File system adapter implementing file system operations.
    
    Provides resilient file operations with circuit breaker protection
    and retry mechanisms for handling transient file system issues.
    """
    
    def __init__(self, 
                 base_path: Optional[Path] = None,
                 enable_resilience: bool = True):
        """
        Initialize file system adapter.
        
        Args:
            base_path: Base directory path for all operations
            enable_resilience: Enable circuit breaker and retry patterns
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.enable_resilience = enable_resilience
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FileSystemAdapter with base_path: {self.base_path}")
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve relative path against base path."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path
    
    @circuit_breaker("filesystem_operations")
    @retry(max_attempts=3, base_delay=0.1)
    def read_file(self, file_path: Union[str, Path], mode: str = 'r') -> Union[str, bytes]:
        """
        Read file content.
        
        Args:
            file_path: Path to file
            mode: File open mode ('r', 'rb', etc.)
            
        Returns:
            File content as string or bytes
            
        Raises:
            DataError: If file cannot be read
        """
        resolved_path = self._resolve_path(file_path)
        
        try:
            if not resolved_path.exists():
                raise DataError(f"File not found: {resolved_path}")
            
            with open(resolved_path, mode) as f:
                content = f.read()
            
            logger.debug(f"Read file: {resolved_path} ({len(str(content))} chars)")
            return content
            
        except Exception as e:
            raise DataError(f"Failed to read file {resolved_path}: {e}")
    
    @circuit_breaker("filesystem_operations")
    @retry(max_attempts=3, base_delay=0.1)
    def write_file(self, 
                   file_path: Union[str, Path], 
                   content: Union[str, bytes], 
                   mode: str = 'w') -> bool:
        """
        Write content to file.
        
        Args:
            file_path: Path to file
            content: Content to write
            mode: File open mode ('w', 'wb', etc.)
            
        Returns:
            True if successful
            
        Raises:
            DataError: If file cannot be written
        """
        resolved_path = self._resolve_path(file_path)
        
        try:
            # Ensure parent directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write using temporary file
            temp_path = resolved_path.with_suffix(resolved_path.suffix + '.tmp')
            
            with open(temp_path, mode) as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic move
            temp_path.replace(resolved_path)
            
            logger.debug(f"Wrote file: {resolved_path} ({len(str(content))} chars)")
            return True
            
        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise DataError(f"Failed to write file {resolved_path}: {e}")
    
    def read_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        try:
            content = self.read_file(file_path, 'r')
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise DataError(f"Invalid JSON in file {file_path}: {e}")
    
    def write_json(self, 
                   file_path: Union[str, Path], 
                   data: Dict[str, Any], 
                   indent: int = 2) -> bool:
        """
        Write data as JSON file.
        
        Args:
            file_path: Path to JSON file
            data: Data to serialize
            indent: JSON indentation
            
        Returns:
            True if successful
        """
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=False)
            return self.write_file(file_path, content, 'w')
        except (TypeError, ValueError) as e:
            raise DataError(f"Cannot serialize data to JSON: {e}")
    
    def read_pickle(self, file_path: Union[str, Path]) -> Any:
        """
        Read pickled object from file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Unpickled object
        """
        try:
            content = self.read_file(file_path, 'rb')
            return pickle.loads(content)
        except pickle.PickleError as e:
            raise DataError(f"Cannot unpickle file {file_path}: {e}")
    
    def write_pickle(self, file_path: Union[str, Path], obj: Any) -> bool:
        """
        Write object as pickle file.
        
        Args:
            file_path: Path to pickle file
            obj: Object to pickle
            
        Returns:
            True if successful
        """
        try:
            content = pickle.dumps(obj)
            return self.write_file(file_path, content, 'wb')
        except pickle.PickleError as e:
            raise DataError(f"Cannot pickle object: {e}")
    
    @circuit_breaker("filesystem_operations")
    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if path exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if path exists
        """
        resolved_path = self._resolve_path(path)
        return resolved_path.exists()
    
    @circuit_breaker("filesystem_operations")
    def is_file(self, path: Union[str, Path]) -> bool:
        """
        Check if path is a file.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a file
        """
        resolved_path = self._resolve_path(path)
        return resolved_path.is_file()
    
    @circuit_breaker("filesystem_operations")
    def is_directory(self, path: Union[str, Path]) -> bool:
        """
        Check if path is a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a directory
        """
        resolved_path = self._resolve_path(path)
        return resolved_path.is_dir()
    
    @circuit_breaker("filesystem_operations")
    @retry(max_attempts=3, base_delay=0.1)
    def create_directory(self, path: Union[str, Path], parents: bool = True) -> bool:
        """
        Create directory.
        
        Args:
            path: Directory path to create
            parents: Create parent directories if needed
            
        Returns:
            True if successful
            
        Raises:
            SystemError: If directory cannot be created
        """
        resolved_path = self._resolve_path(path)
        
        try:
            resolved_path.mkdir(parents=parents, exist_ok=True)
            logger.debug(f"Created directory: {resolved_path}")
            return True
        except Exception as e:
            raise SystemError(f"Failed to create directory {resolved_path}: {e}")
    
    @circuit_breaker("filesystem_operations")
    @retry(max_attempts=3, base_delay=0.1)
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful
            
        Raises:
            SystemError: If file cannot be deleted
        """
        resolved_path = self._resolve_path(file_path)
        
        try:
            if resolved_path.exists():
                resolved_path.unlink()
                logger.debug(f"Deleted file: {resolved_path}")
            return True
        except Exception as e:
            raise SystemError(f"Failed to delete file {resolved_path}: {e}")
    
    @circuit_breaker("filesystem_operations")
    @retry(max_attempts=3, base_delay=0.1)
    def delete_directory(self, dir_path: Union[str, Path], recursive: bool = False) -> bool:
        """
        Delete directory.
        
        Args:
            dir_path: Path to directory to delete
            recursive: Delete recursively if not empty
            
        Returns:
            True if successful
            
        Raises:
            SystemError: If directory cannot be deleted
        """
        resolved_path = self._resolve_path(dir_path)
        
        try:
            if resolved_path.exists():
                if recursive:
                    shutil.rmtree(resolved_path)
                else:
                    resolved_path.rmdir()
                logger.debug(f"Deleted directory: {resolved_path}")
            return True
        except Exception as e:
            raise SystemError(f"Failed to delete directory {resolved_path}: {e}")
    
    def list_files(self, 
                   dir_path: Union[str, Path], 
                   pattern: Optional[str] = None,
                   recursive: bool = False) -> List[Path]:
        """
        List files in directory.
        
        Args:
            dir_path: Directory path
            pattern: Glob pattern to filter files
            recursive: Search recursively
            
        Returns:
            List of file paths
        """
        resolved_path = self._resolve_path(dir_path)
        
        if not resolved_path.is_dir():
            return []
        
        try:
            if recursive:
                if pattern:
                    files = list(resolved_path.rglob(pattern))
                else:
                    files = list(resolved_path.rglob('*'))
            else:
                if pattern:
                    files = list(resolved_path.glob(pattern))
                else:
                    files = list(resolved_path.glob('*'))
            
            # Filter to only files
            files = [f for f in files if f.is_file()]
            
            logger.debug(f"Listed {len(files)} files in {resolved_path}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files in {resolved_path}: {e}")
            return []
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
            
        Raises:
            DataError: If file size cannot be determined
        """
        resolved_path = self._resolve_path(file_path)
        
        try:
            return resolved_path.stat().st_size
        except Exception as e:
            raise DataError(f"Cannot get file size for {resolved_path}: {e}")
    
    def get_file_mtime(self, file_path: Union[str, Path]) -> datetime:
        """
        Get file modification time.
        
        Args:
            file_path: Path to file
            
        Returns:
            File modification time
        """
        resolved_path = self._resolve_path(file_path)
        
        try:
            mtime = resolved_path.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        except Exception as e:
            raise DataError(f"Cannot get modification time for {resolved_path}: {e}")
    
    @circuit_breaker("filesystem_operations")
    def copy_file(self, 
                  src_path: Union[str, Path], 
                  dst_path: Union[str, Path],
                  preserve_metadata: bool = True) -> bool:
        """
        Copy file.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            preserve_metadata: Preserve file metadata
            
        Returns:
            True if successful
        """
        src_resolved = self._resolve_path(src_path)
        dst_resolved = self._resolve_path(dst_path)
        
        try:
            # Ensure destination directory exists
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            if preserve_metadata:
                shutil.copy2(src_resolved, dst_resolved)
            else:
                shutil.copy(src_resolved, dst_resolved)
            
            logger.debug(f"Copied file: {src_resolved} -> {dst_resolved}")
            return True
            
        except Exception as e:
            raise SystemError(f"Failed to copy file {src_resolved} to {dst_resolved}: {e}")
    
    @circuit_breaker("filesystem_operations")  
    def move_file(self, 
                  src_path: Union[str, Path], 
                  dst_path: Union[str, Path]) -> bool:
        """
        Move/rename file.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            
        Returns:
            True if successful
        """
        src_resolved = self._resolve_path(src_path)
        dst_resolved = self._resolve_path(dst_path)
        
        try:
            # Ensure destination directory exists
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(src_resolved, dst_resolved)
            
            logger.debug(f"Moved file: {src_resolved} -> {dst_resolved}")
            return True
            
        except Exception as e:
            raise SystemError(f"Failed to move file {src_resolved} to {dst_resolved}: {e}")
    
    def get_disk_usage(self, path: Union[str, Path] = None) -> Dict[str, int]:
        """
        Get disk usage statistics.
        
        Args:
            path: Path to check (defaults to base path)
            
        Returns:
            Dictionary with 'total', 'used', 'free' in bytes
        """
        check_path = self._resolve_path(path) if path else self.base_path
        
        try:
            usage = shutil.disk_usage(check_path)
            return {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free
            }
        except Exception as e:
            logger.error(f"Cannot get disk usage for {check_path}: {e}")
            return {'total': 0, 'used': 0, 'free': 0}