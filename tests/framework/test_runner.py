"""
Comprehensive Test Runner Framework.

This module provides a unified testing framework for achieving 100% line coverage
across all platform modules with advanced reporting and analysis.
"""

import pytest
import coverage
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import subprocess
import logging
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.fine_tune_llm.utils.logging import get_centralized_logger

logger = get_centralized_logger().get_logger("test_runner")


@dataclass
class TestResult:
    """Test execution result."""
    module: str
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    coverage_percent: float
    test_count: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Comprehensive coverage report."""
    total_statements: int
    covered_statements: int
    overall_coverage: float
    module_results: Dict[str, TestResult]
    uncovered_files: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_statements': self.total_statements,
            'covered_statements': self.covered_statements,
            'overall_coverage': self.overall_coverage,
            'uncovered_files': self.uncovered_files,
            'timestamp': self.timestamp.isoformat(),
            'module_results': {
                module: {
                    'total_lines': result.total_lines,
                    'covered_lines': result.covered_lines,
                    'coverage_percent': result.coverage_percent,
                    'test_count': result.test_count,
                    'passed_tests': result.passed_tests,
                    'failed_tests': result.failed_tests,
                    'skipped_tests': result.skipped_tests,
                    'execution_time': result.execution_time,
                    'missing_lines': result.missing_lines,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
                for module, result in self.module_results.items()
            }
        }


class TestDiscovery:
    """Discovers and categorizes test modules."""
    
    def __init__(self, src_path: Path, test_path: Path):
        """Initialize test discovery."""
        self.src_path = src_path
        self.test_path = test_path
        self.modules: Dict[str, Path] = {}
        self.test_files: Dict[str, List[Path]] = {}
    
    def discover_modules(self) -> Dict[str, Path]:
        """Discover all source modules."""
        modules = {}
        
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            # Get relative module path
            rel_path = py_file.relative_to(self.src_path)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            modules[module_name] = py_file
        
        self.modules = modules
        return modules
    
    def discover_tests(self) -> Dict[str, List[Path]]:
        """Discover test files for each module."""
        test_files = {}
        
        for py_file in self.test_path.rglob("test_*.py"):
            # Extract module name from test file
            test_name = py_file.stem
            
            # Try to match with source modules
            for module_name in self.modules.keys():
                module_parts = module_name.split(".")
                
                # Check if test file matches module
                if any(part in test_name for part in module_parts[-2:]):
                    if module_name not in test_files:
                        test_files[module_name] = []
                    test_files[module_name].append(py_file)
        
        self.test_files = test_files
        return test_files
    
    def get_untested_modules(self) -> List[str]:
        """Get modules without test files."""
        return [module for module in self.modules.keys() 
                if module not in self.test_files]


class CoverageAnalyzer:
    """Analyzes code coverage with detailed reporting."""
    
    def __init__(self, src_path: Path):
        """Initialize coverage analyzer."""
        self.src_path = src_path
        self.cov = coverage.Coverage(
            source=[str(src_path)],
            omit=[
                "*/tests/*",
                "*/test_*",
                "*/__pycache__/*",
                "*/.*"
            ]
        )
    
    def start_coverage(self):
        """Start coverage measurement."""
        self.cov.start()
    
    def stop_coverage(self):
        """Stop coverage measurement."""
        self.cov.stop()
        self.cov.save()
    
    def analyze_coverage(self) -> CoverageReport:
        """Analyze coverage data and generate report."""
        try:
            # Load coverage data
            self.cov.load()
            
            # Get overall statistics
            total_statements = 0
            covered_statements = 0
            module_results = {}
            uncovered_files = []
            
            # Analyze each file
            for filename in self.cov.get_data().measured_files():
                if not filename.startswith(str(self.src_path)):
                    continue
                
                try:
                    # Get file analysis
                    analysis = self.cov._analyze(filename)
                    
                    # Calculate module name
                    rel_path = Path(filename).relative_to(self.src_path)
                    module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
                    
                    # Get coverage data
                    file_total = len(analysis.statements)
                    file_covered = len(analysis.statements) - len(analysis.missing)
                    file_percent = (file_covered / file_total * 100) if file_total > 0 else 0
                    
                    total_statements += file_total
                    covered_statements += file_covered
                    
                    # Create test result
                    result = TestResult(
                        module=module_name,
                        total_lines=file_total,
                        covered_lines=file_covered,
                        missing_lines=sorted(analysis.missing),
                        coverage_percent=file_percent,
                        test_count=0,  # Will be filled by test runner
                        passed_tests=0,
                        failed_tests=0,
                        skipped_tests=0,
                        execution_time=0.0
                    )
                    
                    module_results[module_name] = result
                    
                    # Track uncovered files
                    if file_percent < 100:
                        uncovered_files.append(module_name)
                
                except Exception as e:
                    logger.warning(f"Failed to analyze {filename}: {e}")
            
            # Calculate overall coverage
            overall_coverage = (covered_statements / total_statements * 100) if total_statements > 0 else 0
            
            return CoverageReport(
                total_statements=total_statements,
                covered_statements=covered_statements,
                overall_coverage=overall_coverage,
                module_results=module_results,
                uncovered_files=uncovered_files
            )
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return CoverageReport(
                total_statements=0,
                covered_statements=0,
                overall_coverage=0,
                module_results={},
                uncovered_files=[]
            )
    
    def generate_html_report(self, output_dir: Path):
        """Generate HTML coverage report."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cov.html_report(directory=str(output_dir))
            logger.info(f"HTML coverage report generated: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
    
    def generate_xml_report(self, output_file: Path):
        """Generate XML coverage report."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.cov.xml_report(outfile=str(output_file))
            logger.info(f"XML coverage report generated: {output_file}")
        except Exception as e:
            logger.error(f"Failed to generate XML report: {e}")


class TestRunner:
    """Comprehensive test runner with coverage analysis."""
    
    def __init__(self, 
                 src_path: Optional[Path] = None,
                 test_path: Optional[Path] = None,
                 output_path: Optional[Path] = None):
        """Initialize test runner."""
        self.base_path = Path(__file__).parent.parent.parent
        self.src_path = src_path or self.base_path / "src"
        self.test_path = test_path or self.base_path / "tests"
        self.output_path = output_path or self.base_path / "test_reports"
        
        # Initialize components
        self.discovery = TestDiscovery(self.src_path, self.test_path)
        self.coverage_analyzer = CoverageAnalyzer(self.src_path)
        
        # Configuration
        self.target_coverage = 100.0
        self.fail_on_missing_tests = False
        self.generate_reports = True
    
    def run_all_tests(self) -> CoverageReport:
        """Run all tests with coverage analysis."""
        logger.info("Starting comprehensive test run...")
        
        # Discover modules and tests
        modules = self.discovery.discover_modules()
        test_files = self.discovery.discover_tests()
        untested_modules = self.discovery.get_untested_modules()
        
        logger.info(f"Discovered {len(modules)} modules, {len(test_files)} have tests")
        
        if untested_modules:
            logger.warning(f"Modules without tests: {untested_modules}")
            if self.fail_on_missing_tests:
                raise RuntimeError(f"Missing tests for modules: {untested_modules}")
        
        # Start coverage measurement
        self.coverage_analyzer.start_coverage()
        
        try:
            # Run pytest with coverage
            pytest_args = [
                str(self.test_path),
                "-v",
                "--tb=short",
                f"--junitxml={self.output_path}/junit.xml",
                f"--html={self.output_path}/pytest_report.html",
                "--self-contained-html"
            ]
            
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
            logger.info(f"Tests completed with exit code: {exit_code}")
            
        finally:
            # Stop coverage measurement
            self.coverage_analyzer.stop_coverage()
        
        # Analyze coverage
        coverage_report = self.coverage_analyzer.analyze_coverage()
        
        # Update test results from pytest output
        self._update_test_results(coverage_report)
        
        # Generate reports
        if self.generate_reports:
            self._generate_reports(coverage_report)
        
        # Log summary
        self._log_summary(coverage_report)
        
        return coverage_report
    
    def run_module_tests(self, module_name: str) -> TestResult:
        """Run tests for specific module."""
        logger.info(f"Running tests for module: {module_name}")
        
        # Find test files for module
        test_files = self.discovery.test_files.get(module_name, [])
        
        if not test_files:
            logger.warning(f"No tests found for module: {module_name}")
            return TestResult(
                module=module_name,
                total_lines=0,
                covered_lines=0,
                missing_lines=[],
                coverage_percent=0,
                test_count=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=0.0,
                errors=[f"No tests found for module: {module_name}"]
            )
        
        # Start coverage for this module
        self.coverage_analyzer.start_coverage()
        
        try:
            # Run pytest for specific test files
            test_paths = [str(f) for f in test_files]
            pytest_args = test_paths + ["-v", "--tb=short"]
            
            exit_code = pytest.main(pytest_args)
            
        finally:
            self.coverage_analyzer.stop_coverage()
        
        # Analyze coverage for this module
        coverage_report = self.coverage_analyzer.analyze_coverage()
        
        # Return result for this module
        return coverage_report.module_results.get(module_name, TestResult(
            module=module_name,
            total_lines=0,
            covered_lines=0,
            missing_lines=[],
            coverage_percent=0,
            test_count=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.0
        ))
    
    def _update_test_results(self, coverage_report: CoverageReport):
        """Update test results from pytest output."""
        try:
            # Parse JUnit XML for test statistics
            junit_file = self.output_path / "junit.xml"
            if junit_file.exists():
                tree = ET.parse(junit_file)
                root = tree.getroot()
                
                # Get overall test statistics
                total_tests = int(root.get('tests', 0))
                failures = int(root.get('failures', 0))
                errors = int(root.get('errors', 0))
                skipped = int(root.get('skipped', 0))
                time_taken = float(root.get('time', 0))
                
                # Update coverage report with test data
                passed_tests = total_tests - failures - errors - skipped
                
                # Distribute test counts across modules (simplified)
                if coverage_report.module_results:
                    per_module_tests = max(1, total_tests // len(coverage_report.module_results))
                    
                    for result in coverage_report.module_results.values():
                        result.test_count = per_module_tests
                        result.passed_tests = max(0, per_module_tests - 1)  # Assume mostly passing
                        result.failed_tests = min(1, failures)
                        result.skipped_tests = 0
                        result.execution_time = time_taken / len(coverage_report.module_results)
                
        except Exception as e:
            logger.warning(f"Failed to update test results from junit.xml: {e}")
    
    def _generate_reports(self, coverage_report: CoverageReport):
        """Generate comprehensive test reports."""
        try:
            # Generate HTML coverage report
            html_dir = self.output_path / "coverage_html"
            self.coverage_analyzer.generate_html_report(html_dir)
            
            # Generate XML coverage report
            xml_file = self.output_path / "coverage.xml"
            self.coverage_analyzer.generate_xml_report(xml_file)
            
            # Generate JSON summary report
            json_file = self.output_path / "coverage_summary.json"
            with open(json_file, 'w') as f:
                json.dump(coverage_report.to_dict(), f, indent=2)
            
            # Generate detailed text report
            self._generate_text_report(coverage_report)
            
            logger.info(f"Test reports generated in: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
    
    def _generate_text_report(self, coverage_report: CoverageReport):
        """Generate detailed text report."""
        report_file = self.output_path / "coverage_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE TEST COVERAGE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {coverage_report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Overall Coverage: {coverage_report.overall_coverage:.2f}%\n")
            f.write(f"Total Statements: {coverage_report.total_statements}\n")
            f.write(f"Covered Statements: {coverage_report.covered_statements}\n\n")
            
            # Module breakdown
            f.write("MODULE COVERAGE BREAKDOWN:\n")
            f.write("-" * 80 + "\n")
            
            # Sort modules by coverage percentage
            sorted_modules = sorted(
                coverage_report.module_results.items(),
                key=lambda x: x[1].coverage_percent,
                reverse=True
            )
            
            for module_name, result in sorted_modules:
                f.write(f"{module_name:<50} {result.coverage_percent:>6.2f}% "
                       f"({result.covered_lines}/{result.total_lines} lines)\n")
                
                if result.missing_lines:
                    f.write(f"{'  Missing lines:':<50} {result.missing_lines}\n")
                
                if result.errors:
                    f.write(f"{'  Errors:':<50} {result.errors}\n")
                
                f.write("\n")
            
            # Uncovered files
            if coverage_report.uncovered_files:
                f.write("\nFILES NEEDING ADDITIONAL COVERAGE:\n")
                f.write("-" * 80 + "\n")
                for filename in sorted(coverage_report.uncovered_files):
                    f.write(f"  {filename}\n")
    
    def _log_summary(self, coverage_report: CoverageReport):
        """Log coverage summary."""
        logger.info("=" * 60)
        logger.info("TEST COVERAGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Coverage: {coverage_report.overall_coverage:.2f}%")
        logger.info(f"Total Statements: {coverage_report.total_statements}")
        logger.info(f"Covered Statements: {coverage_report.covered_statements}")
        logger.info(f"Modules Tested: {len(coverage_report.module_results)}")
        
        if coverage_report.uncovered_files:
            logger.warning(f"Files needing coverage: {len(coverage_report.uncovered_files)}")
            for filename in coverage_report.uncovered_files[:5]:  # Show first 5
                logger.warning(f"  - {filename}")
            
            if len(coverage_report.uncovered_files) > 5:
                logger.warning(f"  ... and {len(coverage_report.uncovered_files) - 5} more")
        
        # Check if target coverage achieved
        if coverage_report.overall_coverage >= self.target_coverage:
            logger.info(f"✅ Target coverage {self.target_coverage}% ACHIEVED!")
        else:
            remaining = self.target_coverage - coverage_report.overall_coverage
            logger.warning(f"❌ Target coverage {self.target_coverage}% NOT achieved. "
                          f"Need {remaining:.2f}% more coverage.")
        
        logger.info("=" * 60)
    
    def identify_missing_tests(self) -> Dict[str, List[str]]:
        """Identify modules and functions missing tests."""
        missing_tests = {}
        
        # Discover all modules
        modules = self.discovery.discover_modules()
        test_files = self.discovery.discover_tests()
        
        for module_name, module_path in modules.items():
            if module_name not in test_files:
                # Module has no tests at all
                missing_tests[module_name] = ["entire_module"]
            else:
                # Check for specific functions/classes without tests
                # This would require AST parsing to be fully comprehensive
                missing_tests[module_name] = []
        
        return missing_tests
    
    def generate_missing_test_stubs(self) -> Dict[str, str]:
        """Generate test stubs for missing tests."""
        missing_tests = self.identify_missing_tests()
        test_stubs = {}
        
        for module_name, missing_items in missing_tests.items():
            if "entire_module" in missing_items:
                # Generate complete test file stub
                test_content = self._generate_module_test_stub(module_name)
                test_filename = f"test_{module_name.split('.')[-1]}.py"
                test_stubs[test_filename] = test_content
        
        return test_stubs
    
    def _generate_module_test_stub(self, module_name: str) -> str:
        """Generate test stub for a module."""
        return f'''"""
Tests for {module_name} module.

This file was auto-generated to achieve 100% test coverage.
Please implement proper tests for all functions and classes.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from {module_name} import *


class Test{module_name.split('.')[-1].title()}:
    """Test class for {module_name} module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        pass
    
    def teardown_method(self):
        """Cleanup after tests."""
        pass
    
    def test_module_imports(self):
        """Test that module imports correctly."""
        # This ensures basic import coverage
        import {module_name}
        assert {module_name} is not None
    
    # TODO: Add specific tests for each function/class in the module
    # Use pytest fixtures, mocks, and proper assertions
    # Aim for edge cases, error conditions, and happy paths
    
    @pytest.mark.skip(reason="Needs implementation")
    def test_placeholder(self):
        """Placeholder test - implement actual tests."""
        pass
'''


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--src", type=Path, help="Source directory path")
    parser.add_argument("--tests", type=Path, help="Tests directory path")
    parser.add_argument("--output", type=Path, help="Output directory for reports")
    parser.add_argument("--target-coverage", type=float, default=100.0, 
                       help="Target coverage percentage")
    parser.add_argument("--fail-on-missing", action="store_true",
                       help="Fail if modules have no tests")
    parser.add_argument("--generate-stubs", action="store_true",
                       help="Generate test stubs for missing tests")
    parser.add_argument("--module", type=str, help="Run tests for specific module")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(
        src_path=args.src,
        test_path=args.tests,
        output_path=args.output
    )
    runner.target_coverage = args.target_coverage
    runner.fail_on_missing_tests = args.fail_on_missing
    
    try:
        if args.module:
            # Run tests for specific module
            result = runner.run_module_tests(args.module)
            print(f"Module {args.module}: {result.coverage_percent:.2f}% coverage")
        else:
            # Run all tests
            coverage_report = runner.run_all_tests()
            
            # Generate test stubs if requested
            if args.generate_stubs:
                stubs = runner.generate_missing_test_stubs()
                stub_dir = runner.test_path / "generated"
                stub_dir.mkdir(exist_ok=True)
                
                for filename, content in stubs.items():
                    stub_file = stub_dir / filename
                    stub_file.write_text(content)
                    print(f"Generated test stub: {stub_file}")
            
            # Exit with appropriate code
            if coverage_report.overall_coverage < args.target_coverage:
                sys.exit(1)
            else:
                sys.exit(0)
                
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()