"""
Test registry for structured test hierarchy.

This module provides utilities to discover and run tests across
the organized test structure.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pytest


class TestRegistry:
    """Registry for managing structured test hierarchy."""
    
    def __init__(self, tests_root: Path):
        """Initialize test registry."""
        self.tests_root = tests_root
        self.test_categories = {
            'unit': 'Unit tests for individual components',
            'integration': 'Integration tests for component interactions',
            'end_to_end': 'End-to-end workflow tests',
            'specialized': 'Specialized tests (security, load, etc.)'
        }
        
        self.unit_modules = {
            'training': 'Training component tests',
            'inference': 'Inference engine tests', 
            'data': 'Data processing tests',
            'evaluation': 'Evaluation and metrics tests',
            'models': 'Model management tests',
            'config': 'Configuration system tests',
            'core': 'Core infrastructure tests',
            'monitoring': 'Monitoring and visualization tests',
            'services': 'Service layer tests',
            'utils': 'Utility function tests'
        }
        
        self.integration_modules = {
            'configuration': 'Configuration integration tests',
            'cross_module': 'Cross-module interaction tests',
            'database': 'Database integration tests',
            'external_services': 'External service integration tests',
            'ui_backend': 'UI-backend integration tests',
            'ui': 'UI integration tests',
            'pipeline': 'Pipeline integration tests'
        }
    
    def discover_tests(self) -> Dict[str, List[Path]]:
        """Discover all test files in the structured hierarchy."""
        discovered = {}
        
        for category in self.test_categories.keys():
            category_path = self.tests_root / category
            if category_path.exists():
                discovered[category] = self._find_test_files(category_path)
        
        # Add special directories
        for special_dir in ['baseline', 'equivalence', 'fixtures', 'mocks', 'utilities']:
            special_path = self.tests_root / special_dir
            if special_path.exists():
                discovered[special_dir] = self._find_test_files(special_path)
        
        return discovered
    
    def _find_test_files(self, directory: Path) -> List[Path]:
        """Find all test files in a directory."""
        test_files = []
        for path in directory.rglob('test_*.py'):
            test_files.append(path)
        return sorted(test_files)
    
    def run_category_tests(self, category: str, verbose: bool = False) -> int:
        """Run all tests in a specific category."""
        category_path = self.tests_root / category
        if not category_path.exists():
            print(f"Category '{category}' not found")
            return 1
        
        args = [str(category_path)]
        if verbose:
            args.append('-v')
        
        return pytest.main(args)
    
    def run_module_tests(self, category: str, module: str, verbose: bool = False) -> int:
        """Run tests for a specific module within a category."""
        module_path = self.tests_root / category / module
        if not module_path.exists():
            print(f"Module '{module}' in category '{category}' not found")
            return 1
        
        args = [str(module_path)]
        if verbose:
            args.append('-v')
        
        return pytest.main(args)
    
    def generate_test_report(self) -> str:
        """Generate a report of the test structure."""
        discovered = self.discover_tests()
        
        report = "# Test Structure Report\n\n"
        
        total_files = 0
        for category, files in discovered.items():
            report += f"## {category.title()} Tests\n"
            report += f"- **Description**: {self.test_categories.get(category, 'Special test directory')}\n"
            report += f"- **Test files**: {len(files)}\n"
            
            if files:
                report += "- **Files**:\n"
                for file_path in files:
                    relative_path = file_path.relative_to(self.tests_root)
                    report += f"  - {relative_path}\n"
            
            report += "\n"
            total_files += len(files)
        
        report += f"**Total test files**: {total_files}\n"
        return report
    
    def validate_structure(self) -> List[str]:
        """Validate the test structure and return any issues."""
        issues = []
        
        # Check for required __init__.py files
        for category in ['unit', 'integration']:
            category_path = self.tests_root / category
            if category_path.exists():
                init_file = category_path / '__init__.py'
                if not init_file.exists():
                    issues.append(f"Missing __init__.py in {category_path}")
                
                # Check subdirectories
                for subdir in category_path.iterdir():
                    if subdir.is_dir():
                        sub_init = subdir / '__init__.py'
                        if not sub_init.exists():
                            issues.append(f"Missing __init__.py in {subdir}")
        
        # Check for orphaned test files (not in proper structure)
        for file_path in self.tests_root.rglob('test_*.py'):
            relative_path = file_path.relative_to(self.tests_root)
            parts = relative_path.parts
            
            if len(parts) < 2:  # Test file directly in tests/ root
                if parts[0] not in ['test_registry.py']:  # Exclude known files
                    issues.append(f"Orphaned test file: {relative_path}")
        
        return issues


def main():
    """Main function for test registry operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test registry operations")
    parser.add_argument('--action', choices=['discover', 'report', 'validate', 'run'], 
                       default='report', help='Action to perform')
    parser.add_argument('--category', help='Test category to run')
    parser.add_argument('--module', help='Test module to run (requires category)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tests_root = Path(__file__).parent
    registry = TestRegistry(tests_root)
    
    if args.action == 'discover':
        discovered = registry.discover_tests()
        for category, files in discovered.items():
            print(f"{category}: {len(files)} test files")
    
    elif args.action == 'report':
        report = registry.generate_test_report()
        print(report)
    
    elif args.action == 'validate':
        issues = registry.validate_structure()
        if issues:
            print("Test structure issues found:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("Test structure is valid")
    
    elif args.action == 'run':
        if args.module and args.category:
            return registry.run_module_tests(args.category, args.module, args.verbose)
        elif args.category:
            return registry.run_category_tests(args.category, args.verbose)
        else:
            print("Please specify --category (and optionally --module) for running tests")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())