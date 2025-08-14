"""
Script consolidation analysis tool.

This script analyzes existing scripts to identify overlapping functionality
and redundant code that can be consolidated or eliminated.
"""

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
import hashlib
import re


class ScriptAnalyzer:
    """Analyzes Python scripts to identify consolidation opportunities."""
    
    def __init__(self, project_root: Path):
        """Initialize script analyzer."""
        self.project_root = project_root
        self.scripts = {}
        self.analysis_results = {
            'duplicate_functions': [],
            'similar_imports': {},
            'redundant_scripts': [],
            'consolidation_opportunities': []
        }
    
    def analyze_all_scripts(self) -> Dict[str, Any]:
        """Analyze all Python scripts in the project."""
        print("Analyzing scripts for consolidation opportunities...")
        
        # Find all Python scripts
        script_paths = []
        
        # Check common script directories
        script_dirs = [
            self.project_root / "scripts",
            self.project_root / "fine_tune_llm" / "scripts",
            self.project_root / "fine_tune_llm" / "apps",
            self.project_root / "voters" / "llm"
        ]
        
        for script_dir in script_dirs:
            if script_dir.exists():
                script_paths.extend(list(script_dir.rglob("*.py")))
        
        print(f"Found {len(script_paths)} Python files to analyze")
        
        # Analyze each script
        for script_path in script_paths:
            if script_path.name != "__init__.py":  # Skip __init__ files
                try:
                    script_info = self._analyze_script(script_path)
                    self.scripts[str(script_path)] = script_info
                except Exception as e:
                    print(f"Error analyzing {script_path}: {e}")
        
        # Find consolidation opportunities
        self._find_duplicate_functions()
        self._find_similar_imports()
        self._find_redundant_scripts()
        self._suggest_consolidation_opportunities()
        
        return self.analysis_results
    
    def _analyze_script(self, script_path: Path) -> Dict[str, Any]:
        """Analyze a single script."""
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                'path': str(script_path),
                'error': f"Syntax error: {e}",
                'size': len(content),
                'lines': len(content.split('\n'))
            }
        
        # Extract information
        imports = []
        functions = []
        classes = []
        main_execution = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
            elif isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'signature_hash': self._hash_function_signature(node)
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
                })
        
        # Check for main execution pattern
        main_execution = 'if __name__ == "__main__"' in content
        
        return {
            'path': str(script_path),
            'name': script_path.name,
            'size': len(content),
            'lines': len(content.split('\n')),
            'imports': imports,
            'functions': functions,
            'classes': classes,
            'has_main_execution': main_execution,
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'purpose': self._infer_script_purpose(script_path, content)
        }
    
    def _hash_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Create a hash of function signature for comparison."""
        signature_parts = [
            func_node.name,
            [arg.arg for arg in func_node.args.args],
            [arg.arg for arg in func_node.args.kwonlyargs] if func_node.args.kwonlyargs else [],
            bool(func_node.args.vararg),
            bool(func_node.args.kwarg)
        ]
        signature_str = str(signature_parts)
        return hashlib.md5(signature_str.encode()).hexdigest()[:8]
    
    def _infer_script_purpose(self, script_path: Path, content: str) -> str:
        """Infer the purpose of a script from its path and content."""
        name = script_path.name.lower()
        content_lower = content.lower()
        
        purposes = []
        
        # Purpose keywords in filename
        if 'train' in name:
            purposes.append('training')
        if 'infer' in name or 'predict' in name:
            purposes.append('inference')
        if 'eval' in name or 'test' in name:
            purposes.append('evaluation')
        if 'dash' in name or 'ui' in name:
            purposes.append('ui')
        if 'data' in name or 'prepare' in name:
            purposes.append('data_processing')
        if 'merge' in name:
            purposes.append('model_merging')
        if 'tune' in name or 'hyperparam' in name:
            purposes.append('hyperparameter_tuning')
        
        # Purpose keywords in content
        if 'streamlit' in content_lower or 'st.' in content:
            purposes.append('streamlit_ui')
        if 'trainer' in content_lower and 'train(' in content_lower:
            purposes.append('training')
        if 'load_model' in content_lower and 'predict' in content_lower:
            purposes.append('inference')
        if 'dashboard' in content_lower:
            purposes.append('dashboard')
        
        return ', '.join(purposes) if purposes else 'utility'
    
    def _find_duplicate_functions(self):
        """Find functions with identical signatures across scripts."""
        function_signatures = defaultdict(list)
        
        for script_path, script_info in self.scripts.items():
            for func in script_info.get('functions', []):
                sig_hash = func['signature_hash']
                function_signatures[sig_hash].append({
                    'script': script_path,
                    'function': func['name'],
                    'args': func['args']
                })
        
        # Find duplicates (signature appears in multiple scripts)
        for sig_hash, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                self.analysis_results['duplicate_functions'].append({
                    'signature_hash': sig_hash,
                    'function_name': occurrences[0]['function'],
                    'args': occurrences[0]['args'],
                    'occurrences': occurrences,
                    'script_count': len(occurrences)
                })
    
    def _find_similar_imports(self):
        """Find scripts with very similar import patterns."""
        import_patterns = defaultdict(list)
        
        for script_path, script_info in self.scripts.items():
            imports = set(script_info.get('imports', []))
            
            # Create a pattern hash based on imports
            pattern_str = '|'.join(sorted(imports))
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:8]
            
            import_patterns[pattern_hash].append({
                'script': script_path,
                'imports': list(imports),
                'import_count': len(imports)
            })
        
        # Group scripts with very similar imports
        for pattern_hash, scripts in import_patterns.items():
            if len(scripts) > 1:
                self.analysis_results['similar_imports'][pattern_hash] = {
                    'scripts': scripts,
                    'common_imports': list(set.intersection(*[set(s['imports']) for s in scripts]))
                }
    
    def _find_redundant_scripts(self):
        """Find potentially redundant scripts."""
        # Group scripts by purpose
        purpose_groups = defaultdict(list)
        
        for script_path, script_info in self.scripts.items():
            purpose = script_info.get('purpose', 'utility')
            purpose_groups[purpose].append({
                'path': script_path,
                'name': script_info['name'],
                'size': script_info['size'],
                'functions': len(script_info.get('functions', [])),
                'classes': len(script_info.get('classes', []))
            })
        
        # Find groups with multiple scripts (potential redundancy)
        for purpose, scripts in purpose_groups.items():
            if len(scripts) > 1 and purpose != 'utility':
                self.analysis_results['redundant_scripts'].append({
                    'purpose': purpose,
                    'script_count': len(scripts),
                    'scripts': scripts
                })
    
    def _suggest_consolidation_opportunities(self):
        """Suggest specific consolidation opportunities."""
        opportunities = []
        
        # Opportunity 1: Scripts with duplicate functions
        if self.analysis_results['duplicate_functions']:
            opportunities.append({
                'type': 'extract_common_functions',
                'priority': 'high',
                'description': f"Extract {len(self.analysis_results['duplicate_functions'])} duplicate functions into shared utility modules",
                'affected_scripts': sum(len(dup['occurrences']) for dup in self.analysis_results['duplicate_functions']),
                'action': 'Create shared utility modules and refactor scripts to import common functions'
            })
        
        # Opportunity 2: Multiple training scripts
        training_scripts = [
            group for group in self.analysis_results['redundant_scripts']
            if 'training' in group['purpose']
        ]
        if training_scripts:
            opportunities.append({
                'type': 'consolidate_training_scripts',
                'priority': 'medium',
                'description': f"Consolidate {len(training_scripts)} training-related script groups",
                'action': 'Merge training scripts into a single unified training CLI'
            })
        
        # Opportunity 3: Multiple UI scripts
        ui_scripts = [
            group for group in self.analysis_results['redundant_scripts']
            if 'ui' in group['purpose'] or 'dashboard' in group['purpose']
        ]
        if ui_scripts:
            opportunities.append({
                'type': 'consolidate_ui_scripts',
                'priority': 'medium',
                'description': f"Consolidate {len(ui_scripts)} UI-related script groups",
                'action': 'Create unified UI launcher with multiple pages/modes'
            })
        
        # Opportunity 4: Small utility scripts
        small_scripts = [
            script_info for script_info in self.scripts.values()
            if script_info.get('size', 0) < 500 and len(script_info.get('functions', [])) <= 2
        ]
        if len(small_scripts) > 3:
            opportunities.append({
                'type': 'merge_small_utilities',
                'priority': 'low',
                'description': f"Merge {len(small_scripts)} small utility scripts",
                'action': 'Combine small utilities into larger, more cohesive modules'
            })
        
        self.analysis_results['consolidation_opportunities'] = opportunities
    
    def generate_consolidation_plan(self) -> Dict[str, Any]:
        """Generate a detailed consolidation plan."""
        plan = {
            'summary': {
                'total_scripts_analyzed': len(self.scripts),
                'duplicate_functions_found': len(self.analysis_results['duplicate_functions']),
                'redundant_script_groups': len(self.analysis_results['redundant_scripts']),
                'consolidation_opportunities': len(self.analysis_results['consolidation_opportunities'])
            },
            'actions': []
        }
        
        # Action 1: Create shared utilities
        if self.analysis_results['duplicate_functions']:
            plan['actions'].append({
                'action_type': 'create_shared_utilities',
                'priority': 1,
                'description': 'Extract duplicate functions into shared utility modules',
                'steps': [
                    'Create src/fine_tune_llm/scripts/utils.py',
                    'Move common functions to utils module',
                    'Update scripts to import from utils',
                    'Remove duplicate function definitions'
                ],
                'files_affected': list(set(
                    occ['script'] for dup in self.analysis_results['duplicate_functions']
                    for occ in dup['occurrences']
                ))
            })
        
        # Action 2: Consolidate training scripts
        training_scripts = []
        for group in self.analysis_results['redundant_scripts']:
            if 'training' in group['purpose']:
                training_scripts.extend([s['path'] for s in group['scripts']])
        
        if training_scripts:
            plan['actions'].append({
                'action_type': 'consolidate_training',
                'priority': 2,
                'description': 'Create unified training CLI',
                'steps': [
                    'Create scripts/train.py as unified training entry point',
                    'Add command-line arguments for different training modes',
                    'Migrate functionality from individual scripts',
                    'Deprecate old training scripts'
                ],
                'files_affected': training_scripts,
                'new_files': ['scripts/train.py']
            })
        
        # Action 3: Consolidate UI scripts
        ui_scripts = []
        for group in self.analysis_results['redundant_scripts']:
            if any(keyword in group['purpose'] for keyword in ['ui', 'dashboard', 'streamlit']):
                ui_scripts.extend([s['path'] for s in group['scripts']])
        
        if ui_scripts:
            plan['actions'].append({
                'action_type': 'consolidate_ui',
                'priority': 3,
                'description': 'Create unified UI launcher',
                'steps': [
                    'Create scripts/launch_ui.py as unified UI entry point',
                    'Add command-line arguments for different UI modes',
                    'Migrate UI components to shared modules',
                    'Update UI scripts to use shared components'
                ],
                'files_affected': ui_scripts,
                'new_files': ['scripts/launch_ui.py', 'src/fine_tune_llm/ui/shared.py']
            })
        
        return plan


def main():
    """Run script consolidation analysis."""
    project_root = Path(__file__).parent.parent
    analyzer = ScriptAnalyzer(project_root)
    
    # Run analysis
    results = analyzer.analyze_all_scripts()
    
    # Generate consolidation plan
    plan = analyzer.generate_consolidation_plan()
    
    # Save results
    output_dir = project_root / "tests" / "consolidation_reports"
    output_dir.mkdir(exist_ok=True)
    
    # Save analysis results
    with open(output_dir / "consolidation_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save consolidation plan
    with open(output_dir / "consolidation_plan.json", 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    
    # Create summary report
    with open(output_dir / "consolidation_summary.md", 'w') as f:
        f.write("# Script Consolidation Analysis\n\n")
        f.write(f"**Analysis Date:** {plan.get('timestamp', 'N/A')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Scripts Analyzed:** {plan['summary']['total_scripts_analyzed']}\n")
        f.write(f"- **Duplicate Functions Found:** {plan['summary']['duplicate_functions_found']}\n")
        f.write(f"- **Redundant Script Groups:** {plan['summary']['redundant_script_groups']}\n")
        f.write(f"- **Consolidation Opportunities:** {plan['summary']['consolidation_opportunities']}\n\n")
        
        f.write("## Consolidation Actions\n\n")
        for i, action in enumerate(plan['actions'], 1):
            f.write(f"### {i}. {action['description']}\n")
            f.write(f"**Priority:** {action['priority']}\n")
            f.write(f"**Type:** {action['action_type']}\n")
            f.write("**Steps:**\n")
            for step in action['steps']:
                f.write(f"- {step}\n")
            f.write(f"**Files Affected:** {len(action['files_affected'])}\n\n")
        
        f.write("## Duplicate Functions\n\n")
        for dup in results['duplicate_functions']:
            f.write(f"- **{dup['function_name']}** ({len(dup['occurrences'])} occurrences)\n")
            for occ in dup['occurrences']:
                f.write(f"  - {Path(occ['script']).name}\n")
        
        f.write("\n## Redundant Script Groups\n\n")
        for group in results['redundant_scripts']:
            f.write(f"- **{group['purpose']}** ({group['script_count']} scripts)\n")
            for script in group['scripts']:
                f.write(f"  - {script['name']} ({script['size']} bytes)\n")
    
    print(f"\n=== CONSOLIDATION ANALYSIS COMPLETE ===")
    print(f"Scripts analyzed: {plan['summary']['total_scripts_analyzed']}")
    print(f"Duplicate functions: {plan['summary']['duplicate_functions_found']}")
    print(f"Redundant groups: {plan['summary']['redundant_script_groups']}")
    print(f"Consolidation actions: {len(plan['actions'])}")
    print(f"Reports saved to: {output_dir}")
    
    return results, plan


if __name__ == "__main__":
    main()