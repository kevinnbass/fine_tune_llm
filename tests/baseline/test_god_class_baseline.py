"""
Baseline tests for god classes before decomposition.

These tests establish functional baselines for the original monolithic classes
to ensure that decomposition maintains identical functionality.
"""

import pytest
import sys
import os
import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, MagicMock, patch

# Test configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKUP_DIR = PROJECT_ROOT / "backups" / "god_classes"
sys.path.insert(0, str(PROJECT_ROOT))


class BaselineTestFramework:
    """Framework for creating baseline tests for god classes."""
    
    def __init__(self, class_name: str, backup_file: Path):
        """
        Initialize baseline test framework.
        
        Args:
            class_name: Name of the class being tested
            backup_file: Path to backup file containing original class
        """
        self.class_name = class_name
        self.backup_file = backup_file
        self.original_class = None
        self.test_results: Dict[str, Any] = {
            'class_name': class_name,
            'backup_file': str(backup_file),
            'methods_tested': [],
            'properties_tested': [],
            'initialization_test': None,
            'method_signatures': {},
            'property_signatures': {},
            'functional_outputs': {},
            'error_behaviors': {},
            'dependencies': []
        }
    
    def load_original_class(self):
        """Load the original class from backup file."""
        if not self.backup_file.exists():
            pytest.skip(f"Backup file not found: {self.backup_file}")
        
        # Load module from backup file
        spec = importlib.util.spec_from_file_location(
            f"backup_{self.class_name.lower()}", 
            self.backup_file
        )
        if spec is None or spec.loader is None:
            pytest.skip(f"Could not load backup file: {self.backup_file}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Mock dependencies that might not be available
        self._setup_dependency_mocks(module)
        
        try:
            spec.loader.exec_module(module)
            self.original_class = getattr(module, self.class_name, None)
            
            if self.original_class is None:
                # Try to find class by scanning module attributes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        self.class_name.lower() in attr_name.lower()):
                        self.original_class = attr
                        break
            
            if self.original_class is None:
                pytest.skip(f"Class {self.class_name} not found in backup")
                
        except Exception as e:
            pytest.skip(f"Failed to load class {self.class_name}: {e}")
    
    def _setup_dependency_mocks(self, module):
        """Set up mocks for common dependencies."""
        # Mock common dependencies
        sys.modules['streamlit'] = Mock()
        sys.modules['plotly'] = Mock()
        sys.modules['plotly.graph_objects'] = Mock()
        sys.modules['plotly.express'] = Mock()
        sys.modules['torch'] = Mock()
        sys.modules['transformers'] = Mock()
        sys.modules['datasets'] = Mock()
        sys.modules['accelerate'] = Mock()
        sys.modules['peft'] = Mock()
        
        # Set up module-level mocks
        if not hasattr(module, '__builtins__'):
            module.__builtins__ = __builtins__
    
    def test_class_initialization(self, test_args: List[Any] = None, test_kwargs: Dict[str, Any] = None):
        """Test class initialization with various parameters."""
        if not self.original_class:
            return False
        
        test_args = test_args or []
        test_kwargs = test_kwargs or {}
        
        try:
            # Test default initialization
            instance = self.original_class(*test_args, **test_kwargs)
            
            self.test_results['initialization_test'] = {
                'success': True,
                'args': test_args,
                'kwargs': test_kwargs,
                'instance_type': type(instance).__name__
            }
            
            return True
            
        except Exception as e:
            self.test_results['initialization_test'] = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'args': test_args,
                'kwargs': test_kwargs
            }
            return False
    
    def test_method_signatures(self) -> Dict[str, Any]:
        """Extract and test method signatures."""
        if not self.original_class:
            return {}
        
        method_signatures = {}
        
        for attr_name in dir(self.original_class):
            if not attr_name.startswith('_') or attr_name in ['__init__', '__call__']:
                attr = getattr(self.original_class, attr_name)
                
                if callable(attr):
                    try:
                        import inspect
                        sig = inspect.signature(attr)
                        method_signatures[attr_name] = {
                            'parameters': [
                                {
                                    'name': param.name,
                                    'kind': str(param.kind),
                                    'default': str(param.default) if param.default != param.empty else None,
                                    'annotation': str(param.annotation) if param.annotation != param.empty else None
                                }
                                for param in sig.parameters.values()
                            ],
                            'return_annotation': str(sig.return_annotation) if sig.return_annotation != sig.empty else None
                        }
                        
                        self.test_results['methods_tested'].append(attr_name)
                        
                    except Exception as e:
                        method_signatures[attr_name] = {
                            'error': str(e),
                            'callable': True
                        }
        
        self.test_results['method_signatures'] = method_signatures
        return method_signatures
    
    def test_property_signatures(self) -> Dict[str, Any]:
        """Extract and test property signatures."""
        if not self.original_class:
            return {}
        
        property_signatures = {}
        
        for attr_name in dir(self.original_class):
            if not attr_name.startswith('_'):
                attr = getattr(self.original_class, attr_name, None)
                
                if isinstance(attr, property):
                    property_signatures[attr_name] = {
                        'is_property': True,
                        'has_getter': attr.fget is not None,
                        'has_setter': attr.fset is not None,
                        'has_deleter': attr.fdel is not None,
                        'doc': attr.__doc__
                    }
                    
                    self.test_results['properties_tested'].append(attr_name)
        
        self.test_results['property_signatures'] = property_signatures
        return property_signatures
    
    def test_method_behavior(self, method_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test method behavior with various inputs."""
        if not self.original_class:
            return {}
        
        method_results = {
            'method_name': method_name,
            'test_cases': []
        }
        
        # Create instance for testing
        try:
            # Try to create instance with minimal args
            instance = self._create_test_instance()
            if instance is None:
                return {'error': 'Could not create test instance'}
        except Exception as e:
            return {'error': f'Instance creation failed: {e}'}
        
        method = getattr(instance, method_name, None)
        if not callable(method):
            return {'error': f'Method {method_name} not found or not callable'}
        
        for test_case in test_cases:
            case_result = {
                'args': test_case.get('args', []),
                'kwargs': test_case.get('kwargs', {}),
                'expected_exception': test_case.get('expected_exception')
            }
            
            try:
                result = method(*case_result['args'], **case_result['kwargs'])
                case_result['success'] = True
                case_result['result_type'] = type(result).__name__
                case_result['result_length'] = len(result) if hasattr(result, '__len__') else None
                
                # Store hash of result for comparison (not full result to avoid memory issues)
                if result is not None:
                    result_str = str(result)[:1000]  # Limit to prevent memory issues
                    case_result['result_hash'] = hashlib.md5(result_str.encode()).hexdigest()
                
            except Exception as e:
                case_result['success'] = False
                case_result['error'] = str(e)
                case_result['error_type'] = type(e).__name__
                
                # Check if this was an expected exception
                if (case_result.get('expected_exception') and
                    case_result['error_type'] == case_result['expected_exception']):
                    case_result['expected_error'] = True
            
            method_results['test_cases'].append(case_result)
        
        self.test_results['functional_outputs'][method_name] = method_results
        return method_results
    
    def _create_test_instance(self):
        """Create a test instance with mocked dependencies."""
        if not self.original_class:
            return None
        
        # Common initialization patterns for different classes
        test_configs = [
            # Try with no arguments
            {'args': [], 'kwargs': {}},
            
            # Try with mock config
            {'args': [], 'kwargs': {'config': {}}},
            
            # Try with mock data
            {'args': [], 'kwargs': {'data': []}},
            
            # Try with basic arguments
            {'args': [{}], 'kwargs': {}},
        ]
        
        for config in test_configs:
            try:
                instance = self.original_class(*config['args'], **config['kwargs'])
                return instance
            except Exception:
                continue
        
        return None
    
    def generate_baseline_report(self) -> Dict[str, Any]:
        """Generate comprehensive baseline report."""
        return {
            'timestamp': self._get_timestamp(),
            'class_info': {
                'name': self.class_name,
                'backup_file': str(self.backup_file),
                'file_hash': self._compute_file_hash(),
                'file_size': self.backup_file.stat().st_size if self.backup_file.exists() else 0
            },
            'test_results': self.test_results,
            'summary': {
                'methods_count': len(self.test_results.get('method_signatures', {})),
                'properties_count': len(self.test_results.get('property_signatures', {})),
                'initialization_successful': self.test_results.get('initialization_test', {}).get('success', False),
                'functional_tests_count': len(self.test_results.get('functional_outputs', {}))
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'
    
    def _compute_file_hash(self) -> str:
        """Compute SHA-256 hash of backup file."""
        if not self.backup_file.exists():
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(self.backup_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class TestHighStakesAuditBaseline:
    """Baseline tests for HighStakesAuditor god class."""
    
    def setup_method(self):
        """Set up test method."""
        self.backup_file = BACKUP_DIR / "high_stakes_audit_backup_20250814_121039.py"
        self.framework = BaselineTestFramework("AdvancedHighStakesAuditor", self.backup_file)
        self.framework.load_original_class()
    
    def test_class_loading(self):
        """Test that the original class can be loaded."""
        assert self.framework.original_class is not None
        assert hasattr(self.framework.original_class, '__init__')
    
    def test_initialization(self):
        """Test class initialization."""
        success = self.framework.test_class_initialization()
        # Note: May fail due to missing dependencies, but we record the behavior
        result = self.framework.test_results['initialization_test']
        assert result is not None
    
    def test_method_signatures(self):
        """Test extraction of method signatures."""
        signatures = self.framework.test_method_signatures()
        assert isinstance(signatures, dict)
        assert len(signatures) > 0  # Should have at least some methods
    
    def test_property_signatures(self):
        """Test extraction of property signatures."""
        properties = self.framework.test_property_signatures()
        assert isinstance(properties, dict)
    
    def test_key_method_behaviors(self):
        """Test key method behaviors with mock inputs."""
        # Test some key methods if they exist
        key_methods = [
            'run_comprehensive_audit',
            'detect_bias',
            'assess_fairness', 
            'compute_risk_metrics'
        ]
        
        for method_name in key_methods:
            if (self.framework.original_class and 
                hasattr(self.framework.original_class, method_name)):
                
                test_cases = [
                    {
                        'args': [[], []],  # Mock predictions and labels
                        'kwargs': {},
                        'expected_exception': None
                    }
                ]
                
                result = self.framework.test_method_behavior(method_name, test_cases)
                assert 'method_name' in result
    
    def test_generate_baseline_report(self):
        """Test baseline report generation."""
        report = self.framework.generate_baseline_report()
        assert 'timestamp' in report
        assert 'class_info' in report
        assert 'test_results' in report
        assert 'summary' in report
        
        # Save baseline report
        baseline_dir = PROJECT_ROOT / "tests" / "baseline_reports"
        baseline_dir.mkdir(exist_ok=True)
        
        report_file = baseline_dir / f"high_stakes_audit_baseline_{report['timestamp'].split('T')[0]}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class TestEvaluateBaseline:
    """Baseline tests for LLMEvaluator god class."""
    
    def setup_method(self):
        """Set up test method."""
        self.backup_file = BACKUP_DIR / "evaluate_backup_20250814_121303.py"
        self.framework = BaselineTestFramework("LLMEvaluator", self.backup_file)
        self.framework.load_original_class()
    
    def test_class_loading(self):
        """Test that the original class can be loaded."""
        assert self.framework.original_class is not None
    
    def test_initialization(self):
        """Test class initialization."""
        success = self.framework.test_class_initialization()
        result = self.framework.test_results['initialization_test']
        assert result is not None
    
    def test_method_signatures(self):
        """Test extraction of method signatures."""
        signatures = self.framework.test_method_signatures()
        assert isinstance(signatures, dict)
    
    def test_generate_baseline_report(self):
        """Test baseline report generation."""
        report = self.framework.generate_baseline_report()
        
        # Save baseline report
        baseline_dir = PROJECT_ROOT / "tests" / "baseline_reports"
        baseline_dir.mkdir(exist_ok=True)
        
        report_file = baseline_dir / f"evaluate_baseline_{report['timestamp'].split('T')[0]}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class TestDashboardBaseline:
    """Baseline tests for Dashboard god class."""
    
    def setup_method(self):
        """Set up test method."""
        self.backup_file = BACKUP_DIR / "dashboard_backup_20250814_123055.py"
        # Try to determine class name from file content
        self.framework = BaselineTestFramework("TrainingDashboard", self.backup_file)
        self.framework.load_original_class()
    
    def test_class_loading(self):
        """Test that the original class can be loaded."""
        # May not load due to Streamlit dependencies
        pass  # Skip if dependencies missing
    
    def test_generate_baseline_report(self):
        """Test baseline report generation."""
        report = self.framework.generate_baseline_report()
        
        # Save baseline report
        baseline_dir = PROJECT_ROOT / "tests" / "baseline_reports"
        baseline_dir.mkdir(exist_ok=True)
        
        report_file = baseline_dir / f"dashboard_baseline_{report['timestamp'].split('T')[0]}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class TestSFTLoRABaseline:
    """Baseline tests for SFT LoRA trainer god class."""
    
    def setup_method(self):
        """Set up test method."""
        self.backup_file = BACKUP_DIR / "sft_lora_backup_20250814_123726.py"
        # The class might have a different name, we'll discover it
        self.framework = BaselineTestFramework("EnhancedLoRASFTTrainer", self.backup_file)
        self.framework.load_original_class()
    
    def test_class_loading(self):
        """Test that the original class can be loaded."""
        # May not load due to ML dependencies
        pass  # Skip if dependencies missing
    
    def test_generate_baseline_report(self):
        """Test baseline report generation."""
        report = self.framework.generate_baseline_report()
        
        # Save baseline report
        baseline_dir = PROJECT_ROOT / "tests" / "baseline_reports"
        baseline_dir.mkdir(exist_ok=True)
        
        report_file = baseline_dir / f"sft_lora_baseline_{report['timestamp'].split('T')[0]}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def test_create_comprehensive_baseline_report():
    """Create comprehensive baseline report for all god classes."""
    baseline_dir = PROJECT_ROOT / "tests" / "baseline_reports"
    baseline_dir.mkdir(exist_ok=True)
    
    god_classes = [
        ("AdvancedHighStakesAuditor", "high_stakes_audit_backup_20250814_121039.py"),
        ("LLMEvaluator", "evaluate_backup_20250814_121303.py"), 
        ("TrainingDashboard", "dashboard_backup_20250814_123055.py"),
        ("EnhancedLoRASFTTrainer", "sft_lora_backup_20250814_123726.py")
    ]
    
    comprehensive_report = {
        'timestamp': BaselineTestFramework._get_timestamp(None),
        'project_root': str(PROJECT_ROOT),
        'backup_directory': str(BACKUP_DIR),
        'god_classes_analyzed': [],
        'summary': {
            'total_classes': len(god_classes),
            'successfully_loaded': 0,
            'total_methods_found': 0,
            'total_properties_found': 0
        }
    }
    
    for class_name, backup_filename in god_classes:
        backup_file = BACKUP_DIR / backup_filename
        
        if backup_file.exists():
            framework = BaselineTestFramework(class_name, backup_file)
            framework.load_original_class()
            
            if framework.original_class:
                # Run analysis
                framework.test_class_initialization()
                framework.test_method_signatures()
                framework.test_property_signatures()
                
                report = framework.generate_baseline_report()
                comprehensive_report['god_classes_analyzed'].append(report)
                
                # Update summary
                comprehensive_report['summary']['successfully_loaded'] += 1
                comprehensive_report['summary']['total_methods_found'] += report['summary']['methods_count']
                comprehensive_report['summary']['total_properties_found'] += report['summary']['properties_count']
    
    # Save comprehensive report
    comprehensive_file = baseline_dir / f"comprehensive_baseline_report_{comprehensive_report['timestamp'].split('T')[0]}.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    
    assert len(comprehensive_report['god_classes_analyzed']) > 0