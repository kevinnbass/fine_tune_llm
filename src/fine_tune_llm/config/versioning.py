"""
Configuration versioning and rollback system.

This module provides versioning capabilities for configuration changes
with the ability to rollback to previous versions and track changes.
"""

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of configuration changes."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"
    MIGRATED = "migrated"


@dataclass
class ConfigVersion:
    """Configuration version metadata."""
    version_id: str
    timestamp: datetime
    change_type: ChangeType
    description: str
    config_hash: str
    user: Optional[str] = None
    tags: Optional[List[str]] = None
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['change_type'] = self.change_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['change_type'] = ChangeType(data['change_type'])
        return cls(**data)


class ConfigVersionManager:
    """
    Manages configuration versions with rollback capabilities.
    
    Provides versioning, change tracking, and rollback functionality
    for configuration management with full audit trail.
    """
    
    def __init__(self, 
                 config_dir: Path,
                 versions_dir: Optional[Path] = None,
                 max_versions: int = 50,
                 auto_cleanup: bool = True):
        """
        Initialize configuration version manager.
        
        Args:
            config_dir: Main configuration directory
            versions_dir: Directory to store version history (default: config_dir/.versions)
            max_versions: Maximum number of versions to keep
            auto_cleanup: Whether to automatically cleanup old versions
        """
        self.config_dir = Path(config_dir)
        self.versions_dir = versions_dir or (self.config_dir / ".versions")
        self.max_versions = max_versions
        self.auto_cleanup = auto_cleanup
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Version metadata file
        self.metadata_file = self.versions_dir / "versions.json"
        
        # Load existing version history
        self._versions: List[ConfigVersion] = []
        self._load_version_history()
        
        logger.info(f"Initialized ConfigVersionManager with {len(self._versions)} versions")
    
    def _load_version_history(self):
        """Load version history from metadata file."""
        if not self.metadata_file.exists():
            self._versions = []
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            self._versions = [ConfigVersion.from_dict(v) for v in data.get('versions', [])]
            
            # Sort by timestamp
            self._versions.sort(key=lambda v: v.timestamp)
            
            logger.debug(f"Loaded {len(self._versions)} versions from history")
            
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            self._versions = []
    
    def _save_version_history(self):
        """Save version history to metadata file."""
        try:
            metadata = {
                'versions': [v.to_dict() for v in self._versions],
                'last_updated': datetime.utcnow().isoformat(),
                'total_versions': len(self._versions)
            }
            
            # Atomic write
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            temp_file.replace(self.metadata_file)
            
            logger.debug("Saved version history")
            
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")
            raise ConfigurationError(f"Failed to save version history: {e}")
    
    def _generate_version_id(self, config_hash: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_hash = config_hash[:8]
        return f"v{timestamp}_{short_hash}"
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate hash of configuration data."""
        # Create deterministic JSON string
        json_str = json.dumps(config_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_version_file_path(self, version_id: str) -> Path:
        """Get file path for a specific version."""
        return self.versions_dir / f"{version_id}.json"
    
    def create_version(self, 
                      config_data: Dict[str, Any],
                      change_type: ChangeType,
                      description: str,
                      user: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> ConfigVersion:
        """
        Create a new configuration version.
        
        Args:
            config_data: Configuration data to version
            change_type: Type of change
            description: Human-readable description of changes
            user: User making the change
            tags: Tags to associate with this version
            
        Returns:
            Created configuration version
        """
        # Calculate configuration hash
        config_hash = self._calculate_config_hash(config_data)
        
        # Check if this exact configuration already exists
        existing_version = self._find_version_by_hash(config_hash)
        if existing_version:
            logger.warning(f"Configuration unchanged, using existing version {existing_version.version_id}")
            return existing_version
        
        # Generate version ID
        version_id = self._generate_version_id(config_hash)
        
        # Get parent version (latest version)
        parent_version = self._versions[-1].version_id if self._versions else None
        
        # Create version object
        version = ConfigVersion(
            version_id=version_id,
            timestamp=datetime.utcnow(),
            change_type=change_type,
            description=description,
            config_hash=config_hash,
            user=user,
            tags=tags or [],
            parent_version=parent_version
        )
        
        # Save configuration data
        version_file = self._get_version_file_path(version_id)
        with open(version_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Add to version list
        self._versions.append(version)
        
        # Cleanup old versions if needed
        if self.auto_cleanup and len(self._versions) > self.max_versions:
            self._cleanup_old_versions()
        
        # Save version history
        self._save_version_history()
        
        logger.info(f"Created configuration version {version_id}: {description}")
        return version
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """
        Get version metadata by ID.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Version metadata or None if not found
        """
        for version in self._versions:
            if version.version_id == version_id:
                return version
        return None
    
    def get_latest_version(self) -> Optional[ConfigVersion]:
        """
        Get the latest version.
        
        Returns:
            Latest version or None if no versions exist
        """
        return self._versions[-1] if self._versions else None
    
    def list_versions(self, 
                     limit: Optional[int] = None,
                     tags: Optional[List[str]] = None,
                     user: Optional[str] = None,
                     change_type: Optional[ChangeType] = None) -> List[ConfigVersion]:
        """
        List configuration versions with optional filtering.
        
        Args:
            limit: Maximum number of versions to return
            tags: Filter by tags (any tag matches)
            user: Filter by user
            change_type: Filter by change type
            
        Returns:
            List of matching versions (newest first)
        """
        # Filter versions
        filtered_versions = []
        
        for version in reversed(self._versions):  # Newest first
            # Apply filters
            if tags and not any(tag in (version.tags or []) for tag in tags):
                continue
            if user and version.user != user:
                continue
            if change_type and version.change_type != change_type:
                continue
            
            filtered_versions.append(version)
            
            # Apply limit
            if limit and len(filtered_versions) >= limit:
                break
        
        return filtered_versions
    
    def load_version_data(self, version_id: str) -> Dict[str, Any]:
        """
        Load configuration data for a specific version.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Configuration data
            
        Raises:
            ConfigurationError: If version not found or data cannot be loaded
        """
        version = self.get_version(version_id)
        if not version:
            raise ConfigurationError(f"Version {version_id} not found")
        
        version_file = self._get_version_file_path(version_id)
        if not version_file.exists():
            raise ConfigurationError(f"Version data file not found: {version_file}")
        
        try:
            with open(version_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load version data: {e}")
    
    def rollback_to_version(self, 
                           version_id: str,
                           description: Optional[str] = None,
                           user: Optional[str] = None) -> ConfigVersion:
        """
        Rollback to a specific version by creating a new version with restored data.
        
        Args:
            version_id: Version to rollback to
            description: Description of rollback operation
            user: User performing rollback
            
        Returns:
            New version created for the rollback
            
        Raises:
            ConfigurationError: If rollback fails
        """
        # Load version data
        config_data = self.load_version_data(version_id)
        
        # Create description if not provided
        if not description:
            description = f"Rollback to version {version_id}"
        
        # Create new version with restored data
        rollback_version = self.create_version(
            config_data=config_data,
            change_type=ChangeType.RESTORED,
            description=description,
            user=user,
            tags=["rollback", f"restored_from_{version_id}"]
        )
        
        logger.info(f"Rolled back to version {version_id}, created new version {rollback_version.version_id}")
        return rollback_version
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        Compare two versions and return differences.
        
        Args:
            version1_id: First version to compare
            version2_id: Second version to compare
            
        Returns:
            Dictionary describing differences
        """
        # Load both versions
        data1 = self.load_version_data(version1_id)
        data2 = self.load_version_data(version2_id)
        
        # Calculate differences
        differences = self._calculate_differences(data1, data2)
        
        return {
            'version1': version1_id,
            'version2': version2_id,
            'differences': differences,
            'identical': len(differences) == 0
        }
    
    def _calculate_differences(self, data1: Dict[str, Any], data2: Dict[str, Any], 
                              path: str = "") -> List[Dict[str, Any]]:
        """Calculate differences between two configuration dictionaries."""
        differences = []
        
        # Find keys in data1 but not in data2 (deleted)
        for key in data1:
            current_path = f"{path}.{key}" if path else key
            
            if key not in data2:
                differences.append({
                    'type': 'deleted',
                    'path': current_path,
                    'old_value': data1[key]
                })
            elif isinstance(data1[key], dict) and isinstance(data2[key], dict):
                # Recursively compare nested dictionaries
                differences.extend(self._calculate_differences(data1[key], data2[key], current_path))
            elif data1[key] != data2[key]:
                # Value changed
                differences.append({
                    'type': 'modified',
                    'path': current_path,
                    'old_value': data1[key],
                    'new_value': data2[key]
                })
        
        # Find keys in data2 but not in data1 (added)
        for key in data2:
            if key not in data1:
                current_path = f"{path}.{key}" if path else key
                differences.append({
                    'type': 'added',
                    'path': current_path,
                    'new_value': data2[key]
                })
        
        return differences
    
    def get_version_history(self, config_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get version history with optional filtering by configuration path.
        
        Args:
            config_path: Optional path within configuration to track
            
        Returns:
            List of version history entries
        """
        history = []
        
        for version in reversed(self._versions):  # Newest first
            entry = {
                'version_id': version.version_id,
                'timestamp': version.timestamp.isoformat(),
                'change_type': version.change_type.value,
                'description': version.description,
                'user': version.user,
                'tags': version.tags or [],
                'config_hash': version.config_hash
            }
            
            # Add difference information if not the first version
            if version.parent_version:
                try:
                    comparison = self.compare_versions(version.parent_version, version.version_id)
                    entry['changes_count'] = len(comparison['differences'])
                    
                    # If filtering by config_path, only include relevant changes
                    if config_path:
                        relevant_changes = [
                            diff for diff in comparison['differences']
                            if diff['path'].startswith(config_path)
                        ]
                        entry['relevant_changes'] = relevant_changes
                        entry['relevant_changes_count'] = len(relevant_changes)
                    
                except Exception as e:
                    logger.warning(f"Failed to compare versions: {e}")
            
            history.append(entry)
        
        return history
    
    def _find_version_by_hash(self, config_hash: str) -> Optional[ConfigVersion]:
        """Find version by configuration hash."""
        for version in self._versions:
            if version.config_hash == config_hash:
                return version
        return None
    
    def _cleanup_old_versions(self):
        """Clean up old versions beyond max_versions limit."""
        if len(self._versions) <= self.max_versions:
            return
        
        # Calculate how many versions to remove
        versions_to_remove = len(self._versions) - self.max_versions
        
        # Remove oldest versions (but keep at least one)
        for i in range(min(versions_to_remove, len(self._versions) - 1)):
            old_version = self._versions[i]
            
            # Remove version data file
            version_file = self._get_version_file_path(old_version.version_id)
            if version_file.exists():
                version_file.unlink()
            
            logger.debug(f"Cleaned up old version {old_version.version_id}")
        
        # Remove from version list
        self._versions = self._versions[versions_to_remove:]
        
        logger.info(f"Cleaned up {versions_to_remove} old versions")
    
    def export_version(self, version_id: str, export_path: Path) -> Path:
        """
        Export a version to a file.
        
        Args:
            version_id: Version to export
            export_path: Path to export to
            
        Returns:
            Path to exported file
        """
        version_data = self.load_version_data(version_id)
        version_metadata = self.get_version(version_id)
        
        export_data = {
            'metadata': version_metadata.to_dict(),
            'configuration': version_data,
            'exported_at': datetime.utcnow().isoformat(),
            'export_format_version': '1.0'
        }
        
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported version {version_id} to {export_path}")
        return export_path
    
    def import_version(self, import_path: Path, user: Optional[str] = None) -> ConfigVersion:
        """
        Import a version from an exported file.
        
        Args:
            import_path: Path to imported file
            user: User performing import
            
        Returns:
            Imported version
        """
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        # Extract data
        metadata = import_data['metadata']
        config_data = import_data['configuration']
        
        # Create new version with imported data
        imported_version = self.create_version(
            config_data=config_data,
            change_type=ChangeType.MIGRATED,
            description=f"Imported: {metadata['description']}",
            user=user or metadata.get('user'),
            tags=(metadata.get('tags', []) + ["imported"])
        )
        
        logger.info(f"Imported version from {import_path}, created {imported_version.version_id}")
        return imported_version
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get version manager statistics.
        
        Returns:
            Dictionary with version statistics
        """
        if not self._versions:
            return {
                'total_versions': 0,
                'disk_usage_mb': 0,
                'date_range': None
            }
        
        # Calculate disk usage
        total_size = 0
        for version_file in self.versions_dir.glob("*.json"):
            if version_file != self.metadata_file:
                total_size += version_file.stat().st_size
        
        # Change type statistics
        change_type_counts = {}
        for change_type in ChangeType:
            change_type_counts[change_type.value] = sum(
                1 for v in self._versions if v.change_type == change_type
            )
        
        # User statistics
        users = set(v.user for v in self._versions if v.user)
        
        return {
            'total_versions': len(self._versions),
            'disk_usage_mb': round(total_size / (1024 * 1024), 2),
            'date_range': {
                'earliest': self._versions[0].timestamp.isoformat(),
                'latest': self._versions[-1].timestamp.isoformat()
            },
            'change_type_counts': change_type_counts,
            'unique_users': len(users),
            'total_tags': len(set(
                tag for v in self._versions for tag in (v.tags or [])
            ))
        }