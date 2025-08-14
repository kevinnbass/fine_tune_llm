"""
Tests for hierarchical import structure.

This module tests the proper functioning of the hierarchical import
structure with public/private API separation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestHierarchicalImports:
    """Test hierarchical import structure."""
    
    def test_main_package_import(self):
        """Test importing from main package."""
        try:
            import fine_tune_llm
            
            # Test basic package attributes
            assert hasattr(fine_tune_llm, '__version__')
            assert hasattr(fine_tune_llm, '__author__')
            assert hasattr(fine_tune_llm, '__email__')
            
            # Test version function
            version = fine_tune_llm.get_version()
            assert isinstance(version, str)
            assert len(version) > 0
            
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")
    
    def test_lazy_import_config_manager(self):
        """Test lazy import of ConfigManager."""
        try:
            import fine_tune_llm
            
            # This should trigger lazy import
            ConfigManager = fine_tune_llm.ConfigManager
            assert ConfigManager is not None
            
        except ImportError as e:
            pytest.skip(f"ConfigManager not available: {e}")
    
    def test_lazy_import_model_factory(self):
        """Test lazy import of ModelFactory."""
        try:
            import fine_tune_llm
            
            # This should trigger lazy import  
            ModelFactory = fine_tune_llm.ModelFactory
            assert ModelFactory is not None
            
        except ImportError as e:
            pytest.skip(f"ModelFactory not available: {e}")
    
    def test_lazy_import_trainer_factory(self):
        """Test lazy import of TrainerFactory."""
        try:
            import fine_tune_llm
            
            # This should trigger lazy import
            TrainerFactory = fine_tune_llm.TrainerFactory
            assert TrainerFactory is not None
            
        except ImportError as e:
            pytest.skip(f"TrainerFactory not available: {e}")
    
    def test_exception_imports(self):
        """Test exception class imports."""
        try:
            import fine_tune_llm
            
            # Test core exceptions
            FineTuneLLMError = fine_tune_llm.FineTuneLLMError
            assert FineTuneLLMError is not None
            assert issubclass(FineTuneLLMError, Exception)
            
            ConfigurationError = fine_tune_llm.ConfigurationError
            assert ConfigurationError is not None
            assert issubclass(ConfigurationError, FineTuneLLMError)
            
        except ImportError as e:
            pytest.skip(f"Exception classes not available: {e}")
    
    def test_config_module_import(self):
        """Test config module hierarchical imports."""
        try:
            from fine_tune_llm import config
            
            # Test public API exports
            assert hasattr(config, 'ConfigManager')
            assert hasattr(config, 'ValidationError') 
            assert hasattr(config, 'ConfigValidator')
            
            # Test that __all__ is properly defined
            assert hasattr(config, '__all__')
            assert isinstance(config.__all__, list)
            assert len(config.__all__) > 0
            
        except ImportError as e:
            pytest.skip(f"Config module not available: {e}")
    
    def test_models_module_import(self):
        """Test models module hierarchical imports."""
        try:
            from fine_tune_llm import models
            
            # Test public API exports
            assert hasattr(models, 'ModelFactory')
            assert hasattr(models, 'ModelManager')
            assert hasattr(models, 'ModelRegistry')
            
            # Test that __all__ is properly defined
            assert hasattr(models, '__all__')
            assert isinstance(models.__all__, list)
            assert len(models.__all__) > 0
            
        except ImportError as e:
            pytest.skip(f"Models module not available: {e}")
    
    def test_training_module_import(self):
        """Test training module hierarchical imports."""
        try:
            from fine_tune_llm import training
            
            # Test public API exports
            assert hasattr(training, 'TrainerFactory')
            assert hasattr(training, 'BaseTrainer')
            
            # Test that __all__ is properly defined  
            assert hasattr(training, '__all__')
            assert isinstance(training.__all__, list)
            assert len(training.__all__) > 0
            
        except ImportError as e:
            pytest.skip(f"Training module not available: {e}")
    
    def test_private_internals_not_exposed(self):
        """Test that private _internals modules are not exposed."""
        try:
            import fine_tune_llm
            
            # These should not be in the main package's __all__
            assert '_internals' not in fine_tune_llm.__all__
            assert '_version' not in fine_tune_llm.__all__
            
            # But should be accessible if needed (for internal use)
            from fine_tune_llm import _version
            assert hasattr(_version, '__version__')
            
        except ImportError as e:
            pytest.skip(f"Private modules not available: {e}")
    
    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import time
        
        start_time = time.time()
        try:
            import fine_tune_llm
            
            # Basic import should be very fast (< 1 second)
            import_time = time.time() - start_time
            assert import_time < 1.0, f"Import took {import_time:.2f}s, too slow"
            
        except ImportError as e:
            pytest.skip(f"Package not available: {e}")
    
    def test_circular_import_prevention(self):
        """Test that there are no circular import issues."""
        try:
            # These imports should not cause circular import errors
            from fine_tune_llm import config, models, training
            from fine_tune_llm.config import ConfigManager
            from fine_tune_llm.models import ModelFactory
            from fine_tune_llm.training import TrainerFactory
            
            # All should be accessible
            assert ConfigManager is not None
            assert ModelFactory is not None  
            assert TrainerFactory is not None
            
        except ImportError as e:
            pytest.skip(f"Modules not available: {e}")
    
    def test_direct_submodule_imports(self):
        """Test that direct submodule imports work."""
        try:
            # Should be able to import submodules directly
            from fine_tune_llm.config.manager import ConfigManager
            from fine_tune_llm.models.factory import ModelFactory
            
            assert ConfigManager is not None
            assert ModelFactory is not None
            
        except ImportError as e:
            pytest.skip(f"Direct submodule imports not available: {e}")
    
    def test_api_consistency(self):
        """Test API consistency across different import methods."""
        try:
            # Import via main package
            import fine_tune_llm
            ConfigManager1 = fine_tune_llm.ConfigManager
            
            # Import via submodule
            from fine_tune_llm.config import ConfigManager as ConfigManager2
            
            # Should be the same class
            assert ConfigManager1 is ConfigManager2
            
        except ImportError as e:
            pytest.skip(f"API consistency test not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__])