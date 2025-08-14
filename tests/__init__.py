"""
Test package for Fine-Tune LLM platform.

This package contains all tests for the platform, organized by component
and functionality with comprehensive coverage of all modules.
"""

# Test configuration
import os
import sys
from pathlib import Path

# Add source directory to Python path for testing
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Test constants
TEST_DATA_DIR = test_dir / "test_data"
TEST_FIXTURES_DIR = test_dir / "fixtures" 
TEST_OUTPUT_DIR = test_dir / "test_output"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_FIXTURES_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)