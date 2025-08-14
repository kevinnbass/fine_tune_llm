"""
Tests for encrypted secret management system.

This module tests the SecretManager class and related functionality
for secure storage and retrieval of sensitive configuration data.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, Mock

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from fine_tune_llm.config.secrets import (
        SecretManager, SecretConfigMixin,
        get_global_secret_manager, set_secret, get_secret,
        delete_secret, has_secret
    )
    from fine_tune_llm.core.exceptions import ConfigurationError, SecurityError
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False


@pytest.mark.skipif(not SECRETS_AVAILABLE, reason="Secret management system not available")
class TestSecretManager:
    """Test SecretManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.secrets_path = self.temp_dir / "test_secrets.enc"
        self.test_master_key = "test_master_key_123456"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_secret_manager_initialization(self):
        """Test SecretManager initialization."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        assert manager.secrets_path == self.secrets_path
        assert manager.key_derivation_iterations == 100000
        assert manager._fernet is not None
        assert manager._salt is not None
    
    def test_secret_manager_no_master_key_error(self):
        """Test SecretManager raises error without master key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError):
                SecretManager(secrets_path=self.secrets_path)
    
    def test_secret_manager_env_var_master_key(self):
        """Test SecretManager uses environment variable for master key."""
        with patch.dict(os.environ, {'FINE_TUNE_LLM_SECRET_KEY': self.test_master_key}):
            manager = SecretManager(secrets_path=self.secrets_path)
            assert manager._fernet is not None
    
    def test_set_and_get_secret(self):
        """Test setting and getting secrets."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        # Test string secret
        manager.set_secret("api_key", "secret_key_value")
        assert manager.get_secret("api_key") == "secret_key_value"
        
        # Test dict secret
        config_secret = {"username": "admin", "password": "secret123"}
        manager.set_secret("database_config", config_secret)
        assert manager.get_secret("database_config") == config_secret
        
        # Test list secret
        list_secret = ["item1", "item2", "item3"]
        manager.set_secret("api_endpoints", list_secret)
        assert manager.get_secret("api_endpoints") == list_secret
    
    def test_get_secret_default(self):
        """Test getting secret with default value."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        # Non-existent secret with default
        assert manager.get_secret("non_existent", "default_value") == "default_value"
        
        # Non-existent secret without default should raise KeyError
        with pytest.raises(KeyError):
            manager.get_secret("non_existent")
    
    def test_delete_secret(self):
        """Test deleting secrets."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        # Set a secret
        manager.set_secret("temp_secret", "temp_value")
        assert manager.has_secret("temp_secret")
        
        # Delete the secret
        assert manager.delete_secret("temp_secret") is True
        assert not manager.has_secret("temp_secret")
        
        # Try to delete non-existent secret
        assert manager.delete_secret("non_existent") is False
    
    def test_has_secret(self):
        """Test checking secret existence."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        assert not manager.has_secret("test_secret")
        
        manager.set_secret("test_secret", "test_value")
        assert manager.has_secret("test_secret")
    
    def test_list_secrets(self):
        """Test listing secret keys."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        # Initially empty
        assert manager.list_secrets() == []
        
        # Add secrets
        manager.set_secret("secret1", "value1")
        manager.set_secret("secret2", "value2")
        
        secrets = manager.list_secrets()
        assert len(secrets) == 2
        assert "secret1" in secrets
        assert "secret2" in secrets
    
    def test_get_secret_metadata(self):
        """Test getting secret metadata."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        manager.set_secret("test_secret", "test_value")
        
        metadata = manager.get_secret_metadata("test_secret")
        assert metadata["key"] == "test_secret"
        assert "created_at" in metadata
        assert "updated_at" in metadata
        assert metadata["value_type"] == "str"
        
        # Test non-existent secret
        with pytest.raises(KeyError):
            manager.get_secret_metadata("non_existent")
    
    def test_update_secret(self):
        """Test updating existing secrets."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        # Set initial secret
        manager.set_secret("test_secret", "initial_value")
        
        # Update secret
        manager.update_secret("test_secret", "updated_value")
        assert manager.get_secret("test_secret") == "updated_value"
        
        # Test updating non-existent secret
        with pytest.raises(KeyError):
            manager.update_secret("non_existent", "value")
    
    def test_import_export_secrets(self):
        """Test importing and exporting secrets."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        # Test import
        secrets_to_import = {
            "api_key": "secret123",
            "config": {"host": "localhost", "port": 5432}
        }
        
        count = manager.import_secrets(secrets_to_import)
        assert count == 2
        assert manager.get_secret("api_key") == "secret123"
        assert manager.get_secret("config") == {"host": "localhost", "port": 5432}
        
        # Test export
        exported = manager.export_secrets()
        assert "api_key" in exported
        assert "config" in exported
        assert exported["api_key"] == "secret123"
        assert exported["config"] == {"host": "localhost", "port": 5432}
        
        # Test export with metadata
        exported_with_metadata = manager.export_secrets(include_metadata=True)
        assert "created_at" in exported_with_metadata["api_key"]
        assert exported_with_metadata["api_key"]["value"] == "secret123"
    
    def test_persistence_across_instances(self):
        """Test that secrets persist across SecretManager instances."""
        # First instance
        manager1 = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        manager1.set_secret("persistent_secret", "persistent_value")
        
        # Second instance should load the same secret
        manager2 = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        assert manager2.get_secret("persistent_secret") == "persistent_value"
    
    def test_wrong_master_key_fails(self):
        """Test that wrong master key fails to decrypt."""
        # Set secret with one key
        manager1 = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        manager1.set_secret("test_secret", "test_value")
        
        # Try to load with different key
        with pytest.raises(SecurityError):
            SecretManager(
                secrets_path=self.secrets_path,
                master_key="wrong_master_key"
            )
    
    def test_clear_cache(self):
        """Test clearing secrets cache."""
        manager = SecretManager(
            secrets_path=self.secrets_path,
            master_key=self.test_master_key
        )
        
        manager.set_secret("test_secret", "test_value")
        assert manager.has_secret("test_secret")
        
        manager.clear_cache()
        assert manager.has_secret("test_secret")  # Should still be there after reload


@pytest.mark.skipif(not SECRETS_AVAILABLE, reason="Secret management system not available")
class TestSecretConfigMixin:
    """Test SecretConfigMixin functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.secrets_path = self.temp_dir / "test_secrets.enc"
        self.test_master_key = "test_master_key_123456"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_secret_config_mixin(self):
        """Test SecretConfigMixin integration."""
        
        class TestConfig(SecretConfigMixin):
            def __init__(self):
                secret_manager = SecretManager(
                    secrets_path=self.secrets_path,
                    master_key=self.test_master_key
                )
                super().__init__(secret_manager=secret_manager)
        
        config = TestConfig()
        
        # Test setting and getting secrets
        config.set_secret_value("api_key", "secret123")
        assert config.get_secret_value("api_key") == "secret123"
    
    def test_resolve_secret_placeholders(self):
        """Test resolving secret placeholders in configuration."""
        
        class TestConfig(SecretConfigMixin):
            def __init__(self):
                secret_manager = SecretManager(
                    secrets_path=self.secrets_path,
                    master_key=self.test_master_key
                )
                super().__init__(secret_manager=secret_manager)
        
        config = TestConfig()
        config.set_secret_value("api_key", "secret123")
        config.set_secret_value("db_password", "dbpass456")
        
        # Test configuration with placeholders
        test_config = {
            "api": {
                "key": "${secret:api_key}",
                "endpoint": "https://api.example.com"
            },
            "database": {
                "host": "localhost",
                "password": "${secret:db_password}"
            },
            "normal_value": "not_a_secret"
        }
        
        resolved = config.resolve_secret_placeholders(test_config)
        
        assert resolved["api"]["key"] == "secret123"
        assert resolved["api"]["endpoint"] == "https://api.example.com"
        assert resolved["database"]["host"] == "localhost"  
        assert resolved["database"]["password"] == "dbpass456"
        assert resolved["normal_value"] == "not_a_secret"
    
    def test_resolve_missing_secret_placeholder(self):
        """Test resolving placeholder for missing secret."""
        
        class TestConfig(SecretConfigMixin):
            def __init__(self):
                secret_manager = SecretManager(
                    secrets_path=self.secrets_path,
                    master_key=self.test_master_key
                )
                super().__init__(secret_manager=secret_manager)
        
        config = TestConfig()
        
        test_config = {
            "api_key": "${secret:missing_secret}"
        }
        
        resolved = config.resolve_secret_placeholders(test_config)
        
        # Should keep placeholder if secret not found
        assert resolved["api_key"] == "${secret:missing_secret}"


@pytest.mark.skipif(not SECRETS_AVAILABLE, reason="Secret management system not available")
class TestGlobalSecretFunctions:
    """Test global secret management functions."""
    
    def setup_method(self):
        """Set up test environment."""
        # Reset global secret manager
        import fine_tune_llm.config.secrets
        fine_tune_llm.config.secrets._global_secret_manager = None
        
    def test_global_secret_functions(self):
        """Test global secret management functions."""
        with patch.dict(os.environ, {'FINE_TUNE_LLM_SECRET_KEY': 'test_key_123456'}):
            # Test setting and getting
            set_secret("global_test", "global_value")
            assert get_secret("global_test") == "global_value"
            
            # Test has_secret
            assert has_secret("global_test") is True
            assert has_secret("non_existent") is False
            
            # Test delete
            assert delete_secret("global_test") is True
            assert has_secret("global_test") is False
            
            # Test get with default
            assert get_secret("non_existent", "default") == "default"


if __name__ == '__main__':
    pytest.main([__file__])