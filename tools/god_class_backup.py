#!/usr/bin/env python3
"""
God Class Backup System

Creates timestamped backups of god class files with checksum verification
before decomposition to ensure zero loss of functionality.
"""

import hashlib
import shutil
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import ast
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GodClassBackupSystem:
    """System for backing up and analyzing god class files before decomposition."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.backup_dir = self.project_root / "backups" / "god_classes"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # God class files to backup
        self.god_classes = {
            "high_stakes_audit": {
                "path": "fine_tune_llm/voters/llm/high_stakes_audit.py",
                "lines": 51000,
                "description": "High-stakes audit system"
            },
            "evaluate": {
                "path": "fine_tune_llm/voters/llm/evaluate.py",
                "lines": 45000,
                "description": "Evaluation system"
            },
            "dashboard": {
                "path": "fine_tune_llm/voters/llm/dashboard.py",
                "lines": 35000,
                "description": "Dashboard UI system"
            },
            "sft_lora": {
                "path": "fine_tune_llm/voters/llm/sft_lora.py",
                "lines": 36000,
                "description": "LoRA fine-tuning system"
            }
        }
        
        self.backup_metadata: Dict[str, Any] = {}
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file to extract metadata."""
        analysis = {
            "file_path": str(file_path),
            "file_size_bytes": file_path.stat().st_size,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "line_count": 0,
            "classes": [],
            "functions": [],
            "imports": [],
            "global_variables": [],
            "has_main": False
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                analysis["line_count"] = len(content.splitlines())
                
                # Parse AST to extract structure
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                        }
                        analysis["classes"].append(class_info)
                    
                    elif isinstance(node, ast.FunctionDef) and not any(
                        isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                        if hasattr(parent, 'body') and node in parent.body
                    ):
                        analysis["functions"].append({
                            "name": node.name,
                            "line_start": node.lineno
                        })
                    
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                analysis["imports"].append(alias.name)
                        else:
                            module = node.module or ""
                            for alias in node.names:
                                analysis["imports"].append(f"{module}.{alias.name}")
                
                # Check for if __name__ == "__main__"
                analysis["has_main"] = 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            
        return analysis
    
    def create_backup(self, class_name: str) -> Dict[str, Any]:
        """Create a timestamped backup of a god class file."""
        if class_name not in self.god_classes:
            raise ValueError(f"Unknown god class: {class_name}")
        
        god_class_info = self.god_classes[class_name]
        source_path = self.project_root / god_class_info["path"]
        
        if not source_path.exists():
            raise FileNotFoundError(f"God class file not found: {source_path}")
        
        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{class_name}_backup_{timestamp}.py"
        backup_path = self.backup_dir / backup_filename
        
        logger.info(f"Creating backup of {class_name} from {source_path}")
        
        # Calculate checksum of original
        original_checksum = self.calculate_checksum(source_path)
        
        # Analyze file structure
        file_analysis = self.analyze_file(source_path)
        
        # Copy file to backup location
        shutil.copy2(source_path, backup_path)
        
        # Verify backup
        backup_checksum = self.calculate_checksum(backup_path)
        
        if original_checksum != backup_checksum:
            raise ValueError("Backup verification failed: checksums don't match")
        
        # Create metadata
        metadata = {
            "class_name": class_name,
            "original_path": str(source_path),
            "backup_path": str(backup_path),
            "timestamp": timestamp,
            "checksum": original_checksum,
            "file_analysis": file_analysis,
            "description": god_class_info["description"],
            "expected_lines": god_class_info["lines"],
            "actual_lines": file_analysis["line_count"]
        }
        
        # Save metadata
        metadata_path = self.backup_dir / f"{class_name}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Backup created successfully:")
        logger.info(f"   - File: {backup_path}")
        logger.info(f"   - Size: {file_analysis['file_size_mb']:.2f} MB")
        logger.info(f"   - Lines: {file_analysis['line_count']:,}")
        logger.info(f"   - Classes: {len(file_analysis['classes'])}")
        logger.info(f"   - Functions: {len(file_analysis['functions'])}")
        logger.info(f"   - Checksum: {original_checksum}")
        
        self.backup_metadata[class_name] = metadata
        
        return metadata
    
    def verify_backup(self, class_name: str, metadata: Dict[str, Any]) -> bool:
        """Verify a backup file against its metadata."""
        backup_path = Path(metadata["backup_path"])
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        current_checksum = self.calculate_checksum(backup_path)
        
        if current_checksum != metadata["checksum"]:
            logger.error(f"Checksum mismatch for {class_name} backup")
            logger.error(f"  Expected: {metadata['checksum']}")
            logger.error(f"  Got: {current_checksum}")
            return False
        
        logger.info(f"✅ Backup verified for {class_name}")
        return True
    
    def run_tests_baseline(self, class_name: str) -> Dict[str, Any]:
        """Run tests for a god class and create baseline."""
        logger.info(f"Running tests for {class_name}...")
        
        test_results = {
            "class_name": class_name,
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "execution_time": 0,
            "test_output": ""
        }
        
        # Map class names to test files
        test_mapping = {
            "high_stakes_audit": ["test_high_stakes.py", "test_high_stakes_simple.py"],
            "evaluate": ["test_evaluate.py"],
            "dashboard": ["test_ui.py"],
            "sft_lora": ["test_sft_lora.py"]
        }
        
        test_files = test_mapping.get(class_name, [])
        
        for test_file in test_files:
            test_path = self.project_root / "fine_tune_llm" / "tests" / test_file
            
            if test_path.exists():
                logger.info(f"  Running {test_file}...")
                
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", str(test_path), "-v"],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root / "fine_tune_llm"
                    )
                    
                    test_results["test_output"] += f"\n--- {test_file} ---\n"
                    test_results["test_output"] += result.stdout
                    test_results["test_output"] += result.stderr
                    
                    # Parse pytest output for metrics
                    if "passed" in result.stdout:
                        import re
                        match = re.search(r'(\d+) passed', result.stdout)
                        if match:
                            test_results["tests_passed"] += int(match.group(1))
                    
                    if "failed" in result.stdout:
                        import re
                        match = re.search(r'(\d+) failed', result.stdout)
                        if match:
                            test_results["tests_failed"] += int(match.group(1))
                            
                except Exception as e:
                    logger.error(f"Error running test {test_file}: {e}")
        
        test_results["tests_run"] = test_results["tests_passed"] + test_results["tests_failed"]
        
        # Save test baseline
        baseline_path = self.backup_dir / f"{class_name}_test_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(baseline_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test baseline created: {baseline_path}")
        logger.info(f"  Tests run: {test_results['tests_run']}")
        logger.info(f"  Tests passed: {test_results['tests_passed']}")
        logger.info(f"  Tests failed: {test_results['tests_failed']}")
        
        return test_results
    
    def backup_all_god_classes(self) -> Dict[str, Dict[str, Any]]:
        """Backup all god class files."""
        results = {}
        
        logger.info("=" * 80)
        logger.info("Starting God Class Backup Process")
        logger.info("=" * 80)
        
        for class_name in self.god_classes.keys():
            try:
                logger.info(f"\nProcessing {class_name}...")
                
                # Create backup
                metadata = self.create_backup(class_name)
                
                # Verify backup
                is_valid = self.verify_backup(class_name, metadata)
                
                # Run tests (if available)
                test_results = self.run_tests_baseline(class_name)
                
                results[class_name] = {
                    "metadata": metadata,
                    "backup_valid": is_valid,
                    "test_baseline": test_results
                }
                
            except Exception as e:
                logger.error(f"Failed to backup {class_name}: {e}")
                results[class_name] = {
                    "error": str(e)
                }
        
        # Save overall backup summary
        summary_path = self.backup_dir / f"backup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("God Class Backup Process Complete")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("=" * 80)
        
        return results
    
    def restore_from_backup(self, class_name: str, backup_metadata: Dict[str, Any]) -> bool:
        """Restore a god class from backup."""
        logger.info(f"Restoring {class_name} from backup...")
        
        backup_path = Path(backup_metadata["backup_path"])
        original_path = Path(backup_metadata["original_path"])
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        # Verify backup integrity
        if not self.verify_backup(class_name, backup_metadata):
            logger.error("Backup verification failed, aborting restore")
            return False
        
        # Create safety backup of current file (if exists)
        if original_path.exists():
            safety_backup = original_path.with_suffix(f".safety_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
            shutil.copy2(original_path, safety_backup)
            logger.info(f"Created safety backup: {safety_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, original_path)
        
        # Verify restoration
        restored_checksum = self.calculate_checksum(original_path)
        if restored_checksum != backup_metadata["checksum"]:
            logger.error("Restoration verification failed")
            return False
        
        logger.info(f"✅ Successfully restored {class_name} from backup")
        return True

def main():
    """Main entry point for god class backup system."""
    backup_system = GodClassBackupSystem()
    
    # Backup all god classes
    results = backup_system.backup_all_god_classes()
    
    # Report results
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = len(results) - successful
    
    print(f"\n📊 Backup Results:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    
    for class_name, result in results.items():
        if "error" in result:
            print(f"  ⚠️  {class_name}: {result['error']}")
        else:
            print(f"  ✅ {class_name}: Backed up successfully")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())