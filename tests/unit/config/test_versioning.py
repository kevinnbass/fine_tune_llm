"""
Tests for configuration versioning and rollback system.

This module tests the ConfigVersionManager and related functionality
for tracking configuration changes and rollback capabilities.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from fine_tune_llm.config.versioning import (
        ConfigVersionManager, ConfigVersion, ChangeType
    )
    from fine_tune_llm.core.exceptions import ConfigurationError
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False


@pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Versioning system not available")
class TestConfigVersionManager:
    """Test ConfigVersionManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.versions_dir = self.temp_dir / ".versions"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_version_manager_initialization(self):
        """Test ConfigVersionManager initialization."""
        manager = ConfigVersionManager(
            config_dir=self.config_dir,
            versions_dir=self.versions_dir
        )
        
        assert manager.config_dir == self.config_dir
        assert manager.versions_dir == self.versions_dir
        assert manager.max_versions == 50
        assert manager.auto_cleanup is True
        
        # Check directories were created
        assert self.config_dir.exists()
        assert self.versions_dir.exists()
    
    def test_create_version(self):
        """Test creating configuration versions."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        config_data = {
            "model": {
                "name": "test_model",
                "parameters": {"learning_rate": 0.001}
            }
        }
        
        version = manager.create_version(
            config_data=config_data,
            change_type=ChangeType.CREATED,
            description="Initial configuration",
            user="test_user",
            tags=["initial", "test"]
        )
        
        assert version.change_type == ChangeType.CREATED
        assert version.description == "Initial configuration"
        assert version.user == "test_user"
        assert version.tags == ["initial", "test"]
        assert version.parent_version is None
        
        # Check that version data was saved
        version_file = manager._get_version_file_path(version.version_id)
        assert version_file.exists()
        
        with open(version_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == config_data
    
    def test_get_version(self):
        """Test retrieving version by ID."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        config_data = {"test": "data"}
        version = manager.create_version(
            config_data=config_data,
            change_type=ChangeType.CREATED,
            description="Test version"
        )
        
        retrieved_version = manager.get_version(version.version_id)
        assert retrieved_version is not None
        assert retrieved_version.version_id == version.version_id
        assert retrieved_version.description == "Test version"
        
        # Test non-existent version
        assert manager.get_version("non_existent") is None
    
    def test_get_latest_version(self):
        """Test getting the latest version."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # No versions initially
        assert manager.get_latest_version() is None
        
        # Create first version
        version1 = manager.create_version(
            config_data={"version": 1},
            change_type=ChangeType.CREATED,
            description="First version"
        )
        
        assert manager.get_latest_version().version_id == version1.version_id
        
        # Create second version
        version2 = manager.create_version(
            config_data={"version": 2},
            change_type=ChangeType.UPDATED,
            description="Second version"
        )
        
        assert manager.get_latest_version().version_id == version2.version_id
    
    def test_load_version_data(self):
        """Test loading version configuration data."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        config_data = {"test_key": "test_value", "nested": {"key": "value"}}
        version = manager.create_version(
            config_data=config_data,
            change_type=ChangeType.CREATED,
            description="Test data"
        )
        
        loaded_data = manager.load_version_data(version.version_id)
        assert loaded_data == config_data
        
        # Test non-existent version
        with pytest.raises(ConfigurationError):
            manager.load_version_data("non_existent")
    
    def test_list_versions(self):
        """Test listing versions with filtering."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # Create multiple versions
        version1 = manager.create_version(
            config_data={"v": 1},
            change_type=ChangeType.CREATED,
            description="V1",
            user="user1",
            tags=["tag1", "initial"]
        )
        
        version2 = manager.create_version(
            config_data={"v": 2},
            change_type=ChangeType.UPDATED,
            description="V2",
            user="user2",
            tags=["tag2", "update"]
        )
        
        version3 = manager.create_version(
            config_data={"v": 3},
            change_type=ChangeType.UPDATED,
            description="V3",
            user="user1",
            tags=["tag1", "update"]
        )
        
        # Test listing all versions
        all_versions = manager.list_versions()
        assert len(all_versions) == 3
        # Should be newest first
        assert all_versions[0].version_id == version3.version_id
        
        # Test limit
        limited = manager.list_versions(limit=2)
        assert len(limited) == 2
        
        # Test filter by user
        user1_versions = manager.list_versions(user="user1")
        assert len(user1_versions) == 2
        
        # Test filter by tags
        tag1_versions = manager.list_versions(tags=["tag1"])
        assert len(tag1_versions) == 2
        
        # Test filter by change type
        updated_versions = manager.list_versions(change_type=ChangeType.UPDATED)
        assert len(updated_versions) == 2
    
    def test_rollback_to_version(self):
        """Test rolling back to a previous version."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # Create initial version
        original_config = {"setting": "original", "value": 100}
        version1 = manager.create_version(
            config_data=original_config,
            change_type=ChangeType.CREATED,
            description="Original"
        )
        
        # Create modified version
        modified_config = {"setting": "modified", "value": 200}
        version2 = manager.create_version(
            config_data=modified_config,
            change_type=ChangeType.UPDATED,
            description="Modified"
        )
        
        # Rollback to original
        rollback_version = manager.rollback_to_version(
            version_id=version1.version_id,
            description="Rollback to original",
            user="admin"
        )
        
        assert rollback_version.change_type == ChangeType.RESTORED
        assert rollback_version.user == "admin"
        assert "rollback" in rollback_version.tags
        
        # Check that rollback data matches original
        rollback_data = manager.load_version_data(rollback_version.version_id)
        assert rollback_data == original_config
    
    def test_compare_versions(self):
        """Test comparing two versions."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # Create first version
        config1 = {
            "model": {"name": "test", "lr": 0.001},
            "data": {"batch_size": 32}
        }
        version1 = manager.create_version(
            config_data=config1,
            change_type=ChangeType.CREATED,
            description="V1"
        )
        
        # Create second version with changes
        config2 = {
            "model": {"name": "test", "lr": 0.002},  # Modified
            "data": {"batch_size": 32, "shuffle": True},  # Added field
            # Removed entire "data" -> "batch_size" stays, added "shuffle"
        }
        version2 = manager.create_version(
            config_data=config2,
            change_type=ChangeType.UPDATED,
            description="V2"
        )
        
        comparison = manager.compare_versions(version1.version_id, version2.version_id)
        
        assert comparison['version1'] == version1.version_id
        assert comparison['version2'] == version2.version_id
        assert 'differences' in comparison
        assert comparison['identical'] is False
        
        # Check specific differences
        differences = comparison['differences']
        assert len(differences) >= 1  # At least the learning rate change
        
        # Find the learning rate change
        lr_changes = [d for d in differences if d['path'] == 'model.lr']
        assert len(lr_changes) == 1
        assert lr_changes[0]['type'] == 'modified'
        assert lr_changes[0]['old_value'] == 0.001
        assert lr_changes[0]['new_value'] == 0.002
    
    def test_get_version_history(self):
        """Test getting version history."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # Create versions
        version1 = manager.create_version(
            config_data={"v": 1},
            change_type=ChangeType.CREATED,
            description="V1"
        )
        
        version2 = manager.create_version(
            config_data={"v": 2},
            change_type=ChangeType.UPDATED,
            description="V2"
        )
        
        history = manager.get_version_history()
        
        assert len(history) == 2
        # Should be newest first
        assert history[0]['version_id'] == version2.version_id
        assert history[1]['version_id'] == version1.version_id
        
        # Check that change counts are included for versions with parents
        assert 'changes_count' in history[0]  # V2 has parent V1
        assert 'changes_count' not in history[1]  # V1 has no parent
    
    def test_duplicate_configuration_handling(self):
        """Test that identical configurations reuse existing versions."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        config_data = {"same": "data"}
        
        # Create first version
        version1 = manager.create_version(
            config_data=config_data,
            change_type=ChangeType.CREATED,
            description="First"
        )
        
        # Try to create identical configuration
        version2 = manager.create_version(
            config_data=config_data,
            change_type=ChangeType.UPDATED,
            description="Second"
        )
        
        # Should return the same version
        assert version1.version_id == version2.version_id
    
    def test_export_import_version(self):
        """Test exporting and importing versions."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # Create version
        config_data = {"export": "test", "nested": {"key": "value"}}
        original_version = manager.create_version(
            config_data=config_data,
            change_type=ChangeType.CREATED,
            description="Export test",
            user="test_user",
            tags=["export", "test"]
        )
        
        # Export version
        export_path = self.temp_dir / "exported_version.json"
        manager.export_version(original_version.version_id, export_path)
        
        assert export_path.exists()
        
        # Verify export structure
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert 'metadata' in export_data
        assert 'configuration' in export_data
        assert export_data['configuration'] == config_data
        
        # Create new manager to test import
        new_manager = ConfigVersionManager(
            config_dir=self.temp_dir / "new_config"
        )
        
        # Import version
        imported_version = new_manager.import_version(export_path, user="import_user")
        
        assert imported_version.change_type == ChangeType.MIGRATED
        assert imported_version.user == "import_user"
        assert "imported" in imported_version.tags
        
        # Verify imported data
        imported_data = new_manager.load_version_data(imported_version.version_id)
        assert imported_data == config_data
    
    def test_version_cleanup(self):
        """Test automatic cleanup of old versions."""
        manager = ConfigVersionManager(
            config_dir=self.config_dir,
            max_versions=3,
            auto_cleanup=True
        )
        
        # Create more versions than the limit
        versions = []
        for i in range(5):
            version = manager.create_version(
                config_data={"version": i},
                change_type=ChangeType.CREATED if i == 0 else ChangeType.UPDATED,
                description=f"Version {i}"
            )
            versions.append(version)
        
        # Should only keep the last 3 versions
        all_versions = manager.list_versions()
        assert len(all_versions) == 3
        
        # Should keep the newest versions
        kept_ids = {v.version_id for v in all_versions}
        assert versions[-1].version_id in kept_ids
        assert versions[-2].version_id in kept_ids
        assert versions[-3].version_id in kept_ids
        
        # Older versions should be removed
        assert versions[0].version_id not in kept_ids
        assert versions[1].version_id not in kept_ids
    
    def test_get_statistics(self):
        """Test getting version manager statistics."""
        manager = ConfigVersionManager(config_dir=self.config_dir)
        
        # Empty manager
        stats = manager.get_statistics()
        assert stats['total_versions'] == 0
        assert stats['disk_usage_mb'] == 0
        assert stats['date_range'] is None
        
        # Create some versions
        manager.create_version(
            config_data={"v": 1},
            change_type=ChangeType.CREATED,
            description="V1",
            user="user1",
            tags=["tag1"]
        )
        
        manager.create_version(
            config_data={"v": 2},
            change_type=ChangeType.UPDATED,
            description="V2",
            user="user2",
            tags=["tag2"]
        )
        
        stats = manager.get_statistics()
        assert stats['total_versions'] == 2
        assert stats['disk_usage_mb'] > 0
        assert stats['date_range'] is not None
        assert stats['unique_users'] == 2
        assert stats['total_tags'] == 2
        assert stats['change_type_counts']['created'] == 1
        assert stats['change_type_counts']['updated'] == 1
    
    def test_persistence_across_instances(self):
        """Test that versions persist across manager instances."""
        # Create first manager and add version
        manager1 = ConfigVersionManager(config_dir=self.config_dir)
        version = manager1.create_version(
            config_data={"persistent": "data"},
            change_type=ChangeType.CREATED,
            description="Persistent version"
        )
        
        # Create second manager instance
        manager2 = ConfigVersionManager(config_dir=self.config_dir)
        
        # Should load existing versions
        loaded_version = manager2.get_version(version.version_id)
        assert loaded_version is not None
        assert loaded_version.description == "Persistent version"
        
        # Should be able to load data
        data = manager2.load_version_data(version.version_id)
        assert data == {"persistent": "data"}


@pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Versioning system not available")
class TestConfigVersion:
    """Test ConfigVersion dataclass functionality."""
    
    def test_config_version_serialization(self):
        """Test ConfigVersion to_dict and from_dict."""
        version = ConfigVersion(
            version_id="v20231201_12345678",
            timestamp=datetime(2023, 12, 1, 12, 0, 0),
            change_type=ChangeType.UPDATED,
            description="Test version",
            config_hash="abcd1234",
            user="test_user",
            tags=["tag1", "tag2"],
            parent_version="v20231130_87654321"
        )
        
        # Test serialization
        data = version.to_dict()
        assert data['version_id'] == "v20231201_12345678"
        assert data['timestamp'] == "2023-12-01T12:00:00"
        assert data['change_type'] == "updated"
        assert data['description'] == "Test version"
        
        # Test deserialization
        restored_version = ConfigVersion.from_dict(data)
        assert restored_version.version_id == version.version_id
        assert restored_version.timestamp == version.timestamp
        assert restored_version.change_type == version.change_type
        assert restored_version.description == version.description
        assert restored_version.user == version.user
        assert restored_version.tags == version.tags


if __name__ == '__main__':
    pytest.main([__file__])