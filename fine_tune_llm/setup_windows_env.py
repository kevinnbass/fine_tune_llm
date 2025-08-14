"""
Setup script to fix Windows environment issues for testing
Run this to prepare your Windows environment for ML testing
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def setup_windows_environment():
    """Configure Windows environment for ML testing."""
    
    print("="*60)
    print("Windows Environment Setup for Fine-Tune LLM")
    print("="*60)
    print()
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("This script is for Windows only")
        return False
    
    # 1. Set environment variables
    print("1. Setting environment variables...")
    env_vars = {
        'PYTEST_TIMEOUT': '300',
        'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TensorFlow logging
        'TRANSFORMERS_VERBOSITY': 'error',  # Reduce transformers logging
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   Set {key}={value}")
    
    # 2. Install optimized packages
    print("\n2. Installing optimization packages...")
    packages = [
        'pytest-xdist',  # Parallel test execution
        'pytest-timeout',  # Timeout handling
        'pytest-mock',  # Better mocking
        'psutil',  # Process utilities
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package, '-q'], 
                         check=True, capture_output=True)
            print(f"   [OK] Installed {package}")
        except:
            print(f"   [FAIL] Failed to install {package}")
    
    # 3. Clear caches
    print("\n3. Clearing caches...")
    
    # Clear pytest cache
    pytest_cache = Path('.pytest_cache')
    if pytest_cache.exists():
        import shutil
        shutil.rmtree(pytest_cache)
        print("   [OK] Cleared pytest cache")
    
    # Clear Python caches
    for pycache in Path('.').rglob('__pycache__'):
        import shutil
        shutil.rmtree(pycache)
    print("   [OK] Cleared Python caches")
    
    # 4. Check GPU availability
    print("\n4. Checking hardware...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   [OK] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   [INFO] No CUDA GPU detected (CPU mode)")
    except ImportError:
        print("   [INFO] PyTorch not installed (install for GPU support)")
    
    # 5. Create optimized pytest.ini for Windows
    print("\n5. Creating optimized pytest configuration...")
    pytest_ini = """[pytest]
minversion = 6.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Windows optimizations
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --maxfail=5
    -p no:cacheprovider

# Timeout settings
timeout = 300
timeout_method = thread

# Markers
markers =
    quick: Quick tests without ML imports
    ml: Tests requiring ML libraries
    slow: Slow tests
    integration: Integration tests
    unit: Unit tests

# Ignore warnings
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::FutureWarning
    ignore::UserWarning:transformers
"""
    
    with open('pytest.ini', 'w') as f:
        f.write(pytest_ini)
    print("   [OK] Created pytest.ini")
    
    # 6. Test the setup
    print("\n6. Testing setup...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   [OK] Pytest is working: {result.stdout.strip()}")
        else:
            print("   [FAIL] Pytest not working properly")
    except:
        print("   [FAIL] Pytest not installed")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nRecommended test commands:")
    print("  Quick tests:  python -m pytest tests/ --quick")
    print("  Core tests:   test_windows.bat core")
    print("  All tests:    test_windows.bat all")
    print("\nOr use PowerShell:")
    print("  .\\test_windows.ps1 core")
    print("  .\\test_windows.ps1 fix  (to fix issues)")
    
    return True

if __name__ == "__main__":
    setup_windows_environment()