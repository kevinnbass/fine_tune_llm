"""
Script to create baseline reports for god classes.

This script analyzes the original god class backup files and creates
comprehensive baseline reports for functional equivalence testing.
"""

import json
import hashlib
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file."""
    if not file_path.exists():
        return ""
    
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def analyze_python_file(file_path: Path) -> Dict[str, Any]:
    """Analyze Python file to extract class and function information."""
    if not file_path.exists():
        return {"error": "File not found"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic static analysis
        analysis = {
            'file_info': {
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'hash': compute_file_hash(file_path),
                'lines_of_code': len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            },
            'imports': [],
            'classes': {},
            'functions': [],
            'constants': []
        }
        
        lines = content.split('\n')
        current_class = None
        indentation_level = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                continue
            
            # Track imports
            if stripped.startswith('import ') or stripped.startswith('from '):
                analysis['imports'].append({
                    'line': line_num,
                    'statement': stripped
                })
            
            # Track class definitions
            elif stripped.startswith('class '):
                class_match = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
                current_class = class_match
                analysis['classes'][current_class] = {
                    'line_start': line_num,
                    'methods': [],
                    'properties': [],
                    'class_variables': []
                }
                indentation_level = len(line) - len(line.lstrip())
            
            # Track method definitions within classes
            elif current_class and stripped.startswith('def '):
                method_match = stripped.split('def ')[1].split('(')[0].strip()
                analysis['classes'][current_class]['methods'].append({
                    'name': method_match,
                    'line': line_num,
                    'is_private': method_match.startswith('_'),
                    'is_dunder': method_match.startswith('__') and method_match.endswith('__')
                })
            
            # Track standalone function definitions
            elif not current_class and stripped.startswith('def '):
                func_match = stripped.split('def ')[1].split('(')[0].strip()
                analysis['functions'].append({
                    'name': func_match,
                    'line': line_num
                })
            
            # Reset current class if we've outdented
            elif current_class and line and len(line) - len(line.lstrip()) <= indentation_level and not line.startswith(' '):
                if not stripped.startswith('def ') and not stripped.startswith('class '):
                    current_class = None
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}


def create_baseline_report(class_name: str, backup_file: Path) -> Dict[str, Any]:
    """Create baseline report for a god class."""
    print(f"Creating baseline report for {class_name}...")
    
    # Static analysis of the backup file
    static_analysis = analyze_python_file(backup_file)
    
    baseline_report = {
        'timestamp': get_timestamp(),
        'class_name': class_name,
        'backup_file': str(backup_file),
        'static_analysis': static_analysis,
        'metrics': {
            'total_classes': len(static_analysis.get('classes', {})),
            'total_methods': 0,
            'total_functions': len(static_analysis.get('functions', [])),
            'total_imports': len(static_analysis.get('imports', [])),
            'lines_of_code': static_analysis.get('file_info', {}).get('lines_of_code', 0),
            'file_size_bytes': static_analysis.get('file_info', {}).get('size', 0)
        },
        'complexity_indicators': {
            'method_count_per_class': {},
            'private_method_ratio': {},
            'class_depth_estimate': len(static_analysis.get('classes', {}))
        }
    }
    
    # Calculate metrics for each class
    for class_name_found, class_info in static_analysis.get('classes', {}).items():
        method_count = len(class_info.get('methods', []))
        private_methods = len([m for m in class_info.get('methods', []) if m.get('is_private', False)])
        
        baseline_report['metrics']['total_methods'] += method_count
        baseline_report['complexity_indicators']['method_count_per_class'][class_name_found] = method_count
        baseline_report['complexity_indicators']['private_method_ratio'][class_name_found] = (
            private_methods / method_count if method_count > 0 else 0
        )
    
    return baseline_report


def main():
    """Create baseline reports for all god classes."""
    project_root = Path(__file__).parent.parent.parent
    backup_dir = project_root / "backups" / "god_classes"
    baseline_dir = project_root / "tests" / "baseline_reports"
    
    # Ensure baseline reports directory exists
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # Map of expected class names to backup files
    god_classes = [
        ("AdvancedHighStakesAuditor", "high_stakes_audit_backup_20250814_121039.py"),
        ("LLMEvaluator", "evaluate_backup_20250814_121303.py"),
        ("TrainingDashboard", "dashboard_backup_20250814_123055.py"),
        ("EnhancedLoRASFTTrainer", "sft_lora_backup_20250814_123726.py")
    ]
    
    comprehensive_report = {
        'timestamp': get_timestamp(),
        'project_root': str(project_root),
        'backup_directory': str(backup_dir),
        'baseline_reports': [],
        'summary': {
            'total_classes_analyzed': 0,
            'total_backup_files': len(god_classes),
            'total_methods_found': 0,
            'total_lines_of_code': 0,
            'total_file_size_bytes': 0
        }
    }
    
    print("Creating baseline reports for god classes...")
    print(f"Backup directory: {backup_dir}")
    print(f"Baseline reports directory: {baseline_dir}")
    
    for class_name, backup_filename in god_classes:
        backup_file = backup_dir / backup_filename
        
        if backup_file.exists():
            print(f"\nProcessing: {class_name} ({backup_filename})")
            
            # Create individual baseline report
            baseline_report = create_baseline_report(class_name, backup_file)
            comprehensive_report['baseline_reports'].append(baseline_report)
            
            # Update summary statistics
            metrics = baseline_report.get('metrics', {})
            comprehensive_report['summary']['total_classes_analyzed'] += metrics.get('total_classes', 0)
            comprehensive_report['summary']['total_methods_found'] += metrics.get('total_methods', 0)
            comprehensive_report['summary']['total_lines_of_code'] += metrics.get('lines_of_code', 0)
            comprehensive_report['summary']['total_file_size_bytes'] += metrics.get('file_size_bytes', 0)
            
            # Save individual report
            individual_report_file = baseline_dir / f"{class_name.lower()}_baseline_report.json"
            with open(individual_report_file, 'w') as f:
                json.dump(baseline_report, f, indent=2, default=str)
            
            print(f"  - Lines of code: {metrics.get('lines_of_code', 0)}")
            print(f"  - Methods found: {metrics.get('total_methods', 0)}")
            print(f"  - Classes found: {metrics.get('total_classes', 0)}")
            print(f"  - Report saved: {individual_report_file}")
            
        else:
            print(f"\nWARNING: Backup file not found: {backup_file}")
    
    # Save comprehensive report
    comprehensive_file = baseline_dir / "comprehensive_baseline_report.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    
    print(f"\n=== BASELINE ANALYSIS COMPLETE ===")
    print(f"Total classes analyzed: {comprehensive_report['summary']['total_classes_analyzed']}")
    print(f"Total methods found: {comprehensive_report['summary']['total_methods_found']}")
    print(f"Total lines of code: {comprehensive_report['summary']['total_lines_of_code']}")
    print(f"Total file size: {comprehensive_report['summary']['total_file_size_bytes']} bytes")
    print(f"Comprehensive report saved: {comprehensive_file}")
    
    # Create a summary for the todo list
    summary_file = baseline_dir / "baseline_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# God Class Baseline Analysis Summary\n\n")
        f.write(f"**Analysis Date:** {comprehensive_report['timestamp']}\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total God Classes Analyzed:** {len(comprehensive_report['baseline_reports'])}\n")
        f.write(f"- **Total Methods Identified:** {comprehensive_report['summary']['total_methods_found']}\n")
        f.write(f"- **Total Lines of Code:** {comprehensive_report['summary']['total_lines_of_code']}\n")
        f.write(f"- **Combined File Size:** {comprehensive_report['summary']['total_file_size_bytes']:,} bytes\n\n")
        
        f.write("## Individual Class Analysis\n\n")
        for report in comprehensive_report['baseline_reports']:
            class_name = report['class_name']
            metrics = report.get('metrics', {})
            f.write(f"### {class_name}\n")
            f.write(f"- **File:** {Path(report['backup_file']).name}\n")
            f.write(f"- **Lines of Code:** {metrics.get('lines_of_code', 0)}\n")
            f.write(f"- **Methods:** {metrics.get('total_methods', 0)}\n")
            f.write(f"- **Functions:** {metrics.get('total_functions', 0)}\n")
            f.write(f"- **Imports:** {metrics.get('total_imports', 0)}\n")
            f.write(f"- **File Size:** {metrics.get('file_size_bytes', 0):,} bytes\n\n")
        
        f.write("## Files Created\n\n")
        f.write("- `comprehensive_baseline_report.json` - Complete analysis data\n")
        for report in comprehensive_report['baseline_reports']:
            class_name = report['class_name'].lower()
            f.write(f"- `{class_name}_baseline_report.json` - Individual {report['class_name']} analysis\n")
    
    print(f"Summary report saved: {summary_file}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)