"""
Encrypted secret management system.

This module provides secure storage and retrieval of sensitive configuration
data such as API keys, tokens, and passwords with encryption at rest.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

from ..core.exceptions import ConfigurationError, SecurityError

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Manages encrypted secrets with secure storage and retrieval.
    
    Provides encryption at rest for sensitive configuration data
    with password-based encryption and secure key derivation.
    """
    
    def __init__(self, 
                 secrets_path: Optional[Path] = None,
                 master_key: Optional[str] = None,
                 key_derivation_iterations: int = 100000):
        """
        Initialize secret manager.
        
        Args:
            secrets_path: Path to encrypted secrets file
            master_key: Master key for encryption (if not provided, uses env var)
            key_derivation_iterations: Number of PBKDF2 iterations
        """
        self.secrets_path = secrets_path or Path.home() / ".fine_tune_llm" / "secrets.enc"
        self.key_derivation_iterations = key_derivation_iterations
        
        # Ensure secrets directory exists
        self.secrets_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption key
        self._fernet = None
        self._salt = None
        self._initialize_encryption(master_key)
        
        # Load existing secrets
        self._secrets_cache = {}
        self._load_secrets()
        
        logger.info(f"Initialized SecretManager with secrets at {self.secrets_path}")
    
    def _initialize_encryption(self, master_key: Optional[str] = None):
        """Initialize encryption system with key derivation."""
        # Get master key from parameter, environment, or generate
        if master_key is None:
            master_key = os.getenv('FINE_TUNE_LLM_SECRET_KEY')
        
        if master_key is None:
            raise ConfigurationError(
                "Master key required. Set FINE_TUNE_LLM_SECRET_KEY environment variable "
                "or provide master_key parameter"
            )
        
        # Load or generate salt
        salt_path = self.secrets_path.with_suffix('.salt')
        if salt_path.exists():
            with open(salt_path, 'rb') as f:
                self._salt = f.read()
        else:
            self._salt = os.urandom(16)
            with open(salt_path, 'wb') as f:
                f.write(self._salt)
            
            # Set restrictive permissions on salt file
            os.chmod(salt_path, 0o600)
        
        # Derive encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=self.key_derivation_iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._fernet = Fernet(key)
        
        logger.debug("Encryption system initialized")
    
    def _load_secrets(self):
        """Load secrets from encrypted file."""
        if not self.secrets_path.exists():
            self._secrets_cache = {}
            return
        
        try:
            with open(self.secrets_path, 'rb') as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                self._secrets_cache = {}
                return
            
            # Decrypt data
            decrypted_data = self._fernet.decrypt(encrypted_data)
            self._secrets_cache = json.loads(decrypted_data.decode())
            
            logger.debug(f"Loaded {len(self._secrets_cache)} secrets")
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            raise SecurityError(f"Failed to decrypt secrets file: {e}")
    
    def _save_secrets(self):
        """Save secrets to encrypted file."""
        try:
            # Serialize secrets
            data = json.dumps(self._secrets_cache, indent=2)
            
            # Encrypt data
            encrypted_data = self._fernet.encrypt(data.encode())
            
            # Write to file with atomic operation
            temp_path = self.secrets_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(temp_path, 0o600)
            
            # Atomic move
            temp_path.replace(self.secrets_path)
            
            logger.debug(f"Saved {len(self._secrets_cache)} secrets")
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise SecurityError(f"Failed to encrypt and save secrets: {e}")
    
    def set_secret(self, key: str, value: Union[str, dict, list]) -> None:
        """
        Store a secret securely.
        
        Args:
            key: Secret identifier
            value: Secret value (string, dict, or list)
        """
        if not isinstance(key, str) or not key:
            raise ValueError("Secret key must be a non-empty string")
        
        # Store in cache
        self._secrets_cache[key] = {
            'value': value,
            'created_at': self._get_timestamp(),
            'updated_at': self._get_timestamp()
        }
        
        # Persist to disk
        self._save_secrets()
        
        logger.info(f"Secret '{key}' stored securely")
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a secret.
        
        Args:
            key: Secret identifier
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        if key not in self._secrets_cache:
            if default is not None:
                return default
            raise KeyError(f"Secret '{key}' not found")
        
        secret_data = self._secrets_cache[key]
        return secret_data['value']
    
    def delete_secret(self, key: str) -> bool:
        """
        Delete a secret.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if secret was deleted, False if not found
        """
        if key not in self._secrets_cache:
            return False
        
        del self._secrets_cache[key]
        self._save_secrets()
        
        logger.info(f"Secret '{key}' deleted")
        return True
    
    def has_secret(self, key: str) -> bool:
        """
        Check if a secret exists.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if secret exists
        """
        return key in self._secrets_cache
    
    def list_secrets(self) -> list:
        """
        List all secret keys (not values).
        
        Returns:
            List of secret keys
        """
        return list(self._secrets_cache.keys())
    
    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata about a secret.
        
        Args:
            key: Secret identifier
            
        Returns:
            Metadata dictionary with creation/update times
        """
        if key not in self._secrets_cache:
            raise KeyError(f"Secret '{key}' not found")
        
        secret_data = self._secrets_cache[key]
        return {
            'key': key,
            'created_at': secret_data.get('created_at'),
            'updated_at': secret_data.get('updated_at'),
            'value_type': type(secret_data['value']).__name__
        }
    
    def update_secret(self, key: str, value: Union[str, dict, list]) -> None:
        """
        Update an existing secret.
        
        Args:
            key: Secret identifier  
            value: New secret value
        """
        if key not in self._secrets_cache:
            raise KeyError(f"Secret '{key}' not found. Use set_secret() to create new secrets")
        
        # Update value and timestamp
        self._secrets_cache[key]['value'] = value
        self._secrets_cache[key]['updated_at'] = self._get_timestamp()
        
        # Persist to disk
        self._save_secrets()
        
        logger.info(f"Secret '{key}' updated")
    
    def import_secrets(self, secrets: Dict[str, Any], overwrite: bool = False) -> int:
        """
        Import multiple secrets at once.
        
        Args:
            secrets: Dictionary of key-value pairs to import
            overwrite: Whether to overwrite existing secrets
            
        Returns:
            Number of secrets imported
        """
        imported_count = 0
        timestamp = self._get_timestamp()
        
        for key, value in secrets.items():
            if not overwrite and key in self._secrets_cache:
                logger.warning(f"Skipping existing secret '{key}' (overwrite=False)")
                continue
            
            self._secrets_cache[key] = {
                'value': value,
                'created_at': timestamp,
                'updated_at': timestamp
            }
            imported_count += 1
        
        if imported_count > 0:
            self._save_secrets()
            logger.info(f"Imported {imported_count} secrets")
        
        return imported_count
    
    def export_secrets(self, keys: Optional[list] = None, include_metadata: bool = False) -> Dict[str, Any]:
        """
        Export secrets (decrypted).
        
        Args:
            keys: List of specific keys to export (None for all)
            include_metadata: Whether to include creation/update metadata
            
        Returns:
            Dictionary of exported secrets
        """
        if keys is None:
            keys = list(self._secrets_cache.keys())
        
        exported = {}
        for key in keys:
            if key not in self._secrets_cache:
                logger.warning(f"Secret '{key}' not found for export")
                continue
            
            secret_data = self._secrets_cache[key]
            if include_metadata:
                exported[key] = secret_data
            else:
                exported[key] = secret_data['value']
        
        logger.info(f"Exported {len(exported)} secrets")
        return exported
    
    def rotate_master_key(self, new_master_key: str) -> None:
        """
        Rotate the master encryption key.
        
        Args:
            new_master_key: New master key for encryption
        """
        # Backup current secrets
        current_secrets = self._secrets_cache.copy()
        
        # Re-initialize encryption with new key
        old_salt = self._salt
        try:
            self._initialize_encryption(new_master_key)
            
            # Re-encrypt with new key
            self._save_secrets()
            
            logger.info("Master key rotated successfully")
            
        except Exception as e:
            # Restore old salt and re-initialize with old key
            self._salt = old_salt
            logger.error(f"Failed to rotate master key: {e}")
            raise SecurityError(f"Master key rotation failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear in-memory secrets cache (forces reload from disk)."""
        self._secrets_cache.clear()
        self._load_secrets()
        logger.debug("Secrets cache cleared and reloaded")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_secrets_cache'):
            # Clear sensitive data from memory
            self._secrets_cache.clear()
        if hasattr(self, '_fernet'):
            self._fernet = None


class SecretConfigMixin:
    """
    Mixin for configuration classes to support secret management.
    
    Provides methods to integrate secret management into configuration classes.
    """
    
    def __init__(self, *args, secret_manager: Optional[SecretManager] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._secret_manager = secret_manager or SecretManager()
    
    def get_secret_value(self, key: str, default: Any = None) -> Any:
        """Get secret value from secret manager."""
        return self._secret_manager.get_secret(key, default)
    
    def set_secret_value(self, key: str, value: Any) -> None:
        """Set secret value in secret manager."""
        self._secret_manager.set_secret(key, value)
    
    def resolve_secret_placeholders(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve secret placeholders in configuration.
        
        Replaces values like "${secret:api_key}" with actual secret values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with resolved secrets
        """
        def _resolve_value(value):
            if isinstance(value, str) and value.startswith("${secret:") and value.endswith("}"):
                secret_key = value[9:-1]  # Remove "${secret:" and "}"
                try:
                    return self._secret_manager.get_secret(secret_key)
                except KeyError:
                    logger.warning(f"Secret '{secret_key}' not found, keeping placeholder")
                    return value
            elif isinstance(value, dict):
                return {k: _resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_resolve_value(item) for item in value]
            else:
                return value
        
        return _resolve_value(config)


# Convenience functions for global secret manager
_global_secret_manager: Optional[SecretManager] = None


def get_global_secret_manager() -> SecretManager:
    """Get or create global secret manager instance."""
    global _global_secret_manager
    if _global_secret_manager is None:
        _global_secret_manager = SecretManager()
    return _global_secret_manager


def set_secret(key: str, value: Union[str, dict, list]) -> None:
    """Set a secret using global secret manager."""
    get_global_secret_manager().set_secret(key, value)


def get_secret(key: str, default: Any = None) -> Any:
    """Get a secret using global secret manager."""
    return get_global_secret_manager().get_secret(key, default)


def delete_secret(key: str) -> bool:
    """Delete a secret using global secret manager."""
    return get_global_secret_manager().delete_secret(key)


def has_secret(key: str) -> bool:
    """Check if secret exists using global secret manager."""
    return get_global_secret_manager().has_secret(key)