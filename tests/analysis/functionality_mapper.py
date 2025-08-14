"""
Functionality and dependency mapper for god classes.

This script analyzes the god class backup files to create comprehensive
maps of functionality and dependencies for decomposition validation.
"""

import ast
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
import re


class FunctionalityMapper:
    """Maps functionality and dependencies from god class source code."""
    
    def __init__(self, file_path: Path):
        """
        Initialize functionality mapper.
        
        Args:
            file_path: Path to the source file to analyze
        """
        self.file_path = file_path
        self.source_code = ""
        self.ast_tree = None
        self.functionality_map = {
            'file_info': {},
            'imports': {},
            'classes': {},
            'functions': {},
            'dependencies': {},
            'call_graph': {},
            'data_flow': {}
        }
    
    def load_source_code(self) -> bool:
        """Load source code from file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
            
            # Parse AST
            self.ast_tree = ast.parse(self.source_code)
            
            self.functionality_map['file_info'] = {
                'path': str(self.file_path),
                'size': self.file_path.stat().st_size,
                'lines': len(self.source_code.split('\n')),
                'non_empty_lines': len([line for line in self.source_code.split('\n') if line.strip()]),
                'comment_lines': len([line for line in self.source_code.split('\n') if line.strip().startswith('#')])
            }
            
            return True
            
        except Exception as e:
            print(f"Error loading source code: {e}")
            return False
    
    def analyze_imports(self):
        """Analyze import statements and external dependencies."""
        imports = {
            'stdlib': [],
            'third_party': [],
            'local': [],
            'from_imports': {},
            'import_aliases': {}
        }
        
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    alias_name = alias.asname
                    
                    category = self._categorize_import(module_name)
                    imports[category].append(module_name)
                    
                    if alias_name:
                        imports['import_aliases'][alias_name] = module_name
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                imported_items = [alias.name for alias in node.names]
                
                category = self._categorize_import(module_name)
                if module_name not in imports['from_imports']:
                    imports['from_imports'][module_name] = {
                        'category': category,
                        'items': []
                    }
                imports['from_imports'][module_name]['items'].extend(imported_items)
        
        self.functionality_map['imports'] = imports
    
    def analyze_classes(self):
        """Analyze class definitions and their methods."""
        classes = {}
        
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': getattr(node, 'end_lineno', node.lineno),
                    'bases': [self._get_name(base) for base in node.bases],
                    'decorators': [self._get_name(dec) for dec in node.decorator_list],
                    'methods': {},
                    'properties': [],
                    'class_variables': [],
                    'instance_variables': set(),
                    'docstring': ast.get_docstring(node),
                    'complexity_metrics': {}
                }
                
                # Analyze methods and properties
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = self._analyze_method(item)
                        class_info['methods'][item.name] = method_info
                        
                        # Track instance variables from __init__
                        if item.name == '__init__':
                            instance_vars = self._extract_instance_variables(item)
                            class_info['instance_variables'].update(instance_vars)
                    
                    elif isinstance(item, ast.Assign):
                        # Class variables
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_info['class_variables'].append(target.id)
                    
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        # Annotated class variables
                        class_info['class_variables'].append(item.target.id)
                
                # Calculate complexity metrics
                class_info['complexity_metrics'] = self._calculate_class_complexity(node)
                
                classes[node.name] = class_info
        
        self.functionality_map['classes'] = classes
    
    def analyze_functions(self):
        """Analyze standalone function definitions."""
        functions = {}
        
        # Only get top-level functions (not methods)
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_method(node)  # Same analysis as methods
                functions[node.name] = func_info
        
        self.functionality_map['functions'] = functions
    
    def analyze_dependencies(self):
        """Analyze internal dependencies and call relationships."""
        dependencies = {
            'internal_calls': {},  # Function/method calls within the file
            'external_calls': {},  # Calls to imported modules
            'attribute_access': {},  # Attribute access patterns
            'inheritance_chain': {},  # Class inheritance relationships
            'composition_relationships': {}  # Object composition
        }
        
        # Track all function/method calls
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    caller_context = self._find_caller_context(node)
                    
                    if caller_context not in dependencies['internal_calls']:
                        dependencies['internal_calls'][caller_context] = []
                    dependencies['internal_calls'][caller_context].append(call_name)
            
            elif isinstance(node, ast.Attribute):
                attr_name = self._get_attribute_chain(node)
                if attr_name:
                    caller_context = self._find_caller_context(node)
                    
                    if caller_context not in dependencies['attribute_access']:
                        dependencies['attribute_access'][caller_context] = []
                    dependencies['attribute_access'][caller_context].append(attr_name)
        
        # Analyze inheritance relationships
        for class_name, class_info in self.functionality_map['classes'].items():
            if class_info['bases']:
                dependencies['inheritance_chain'][class_name] = class_info['bases']
        
        self.functionality_map['dependencies'] = dependencies
    
    def create_call_graph(self):
        """Create call graph showing function/method relationships."""
        call_graph = {}
        
        # For each class and function, track what it calls
        for class_name, class_info in self.functionality_map['classes'].items():
            for method_name, method_info in class_info['methods'].items():
                full_name = f"{class_name}.{method_name}"
                call_graph[full_name] = {
                    'type': 'method',
                    'calls': method_info.get('calls_made', []),
                    'called_by': [],  # Will be populated in second pass
                    'complexity': method_info.get('complexity', 0),
                    'line_count': method_info.get('line_count', 0)
                }
        
        for func_name, func_info in self.functionality_map['functions'].items():
            call_graph[func_name] = {
                'type': 'function',
                'calls': func_info.get('calls_made', []),
                'called_by': [],
                'complexity': func_info.get('complexity', 0),
                'line_count': func_info.get('line_count', 0)
            }
        
        # Second pass: populate called_by relationships
        for caller, caller_info in call_graph.items():
            for callee in caller_info['calls']:
                if callee in call_graph:
                    call_graph[callee]['called_by'].append(caller)
        
        self.functionality_map['call_graph'] = call_graph
    
    def analyze_data_flow(self):
        """Analyze data flow patterns and variable usage."""
        data_flow = {
            'global_variables': [],
            'shared_state': {},
            'parameter_flow': {},
            'return_patterns': {}
        }
        
        # Find global variables
        for node in self.ast_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        data_flow['global_variables'].append(target.id)
        
        # Analyze method parameter and return patterns
        for class_name, class_info in self.functionality_map['classes'].items():
            for method_name, method_info in class_info['methods'].items():
                full_name = f"{class_name}.{method_name}"
                
                data_flow['parameter_flow'][full_name] = {
                    'parameters': method_info.get('parameters', []),
                    'parameter_count': len(method_info.get('parameters', [])),
                    'has_self': 'self' in method_info.get('parameters', []),
                    'has_kwargs': method_info.get('has_kwargs', False),
                    'has_args': method_info.get('has_args', False)
                }
                
                data_flow['return_patterns'][full_name] = {
                    'return_statements': method_info.get('return_count', 0),
                    'returns_none': method_info.get('returns_none', False),
                    'return_complexity': 'multiple' if method_info.get('return_count', 0) > 1 else 'single'
                }
        
        self.functionality_map['data_flow'] = data_flow
    
    def _categorize_import(self, module_name: str) -> str:
        """Categorize import as stdlib, third-party, or local."""
        if not module_name:
            return 'local'
        
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 'logging', 'math',
            'random', 're', 'hashlib', 'urllib', 'http', 'subprocess',
            'threading', 'multiprocessing', 'asyncio', 'contextlib'
        }
        
        third_party_modules = {
            'numpy', 'pandas', 'torch', 'transformers', 'datasets',
            'streamlit', 'plotly', 'sklearn', 'matplotlib', 'seaborn',
            'requests', 'flask', 'django', 'pytest', 'click'
        }
        
        base_module = module_name.split('.')[0]
        
        if base_module in stdlib_modules:
            return 'stdlib'
        elif base_module in third_party_modules:
            return 'third_party'
        else:
            return 'local'
    
    def _analyze_method(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a method or function node."""
        method_info = {
            'name': node.name,
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'line_count': (getattr(node, 'end_lineno', node.lineno) - node.lineno + 1),
            'parameters': [arg.arg for arg in node.args.args],
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'has_args': node.args.vararg is not None,
            'has_kwargs': node.args.kwarg is not None,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'calls_made': [],
            'complexity': 0,
            'return_count': 0,
            'returns_none': False
        }
        
        # Analyze method body
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                call_name = self._get_call_name(item)
                if call_name:
                    method_info['calls_made'].append(call_name)
            
            elif isinstance(item, ast.Return):
                method_info['return_count'] += 1
                if item.value is None:
                    method_info['returns_none'] = True
            
            elif isinstance(item, (ast.If, ast.For, ast.While, ast.Try)):
                method_info['complexity'] += 1
        
        return method_info
    
    def _calculate_class_complexity(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate complexity metrics for a class."""
        metrics = {
            'method_count': 0,
            'line_count': getattr(node, 'end_lineno', node.lineno) - node.lineno + 1,
            'cyclomatic_complexity': 0,
            'public_methods': 0,
            'private_methods': 0,
            'property_count': 0,
            'inheritance_depth': len(node.bases)
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                metrics['method_count'] += 1
                
                if item.name.startswith('_'):
                    metrics['private_methods'] += 1
                else:
                    metrics['public_methods'] += 1
                
                # Count complexity within method
                for subitem in ast.walk(item):
                    if isinstance(subitem, (ast.If, ast.For, ast.While, ast.Try)):
                        metrics['cyclomatic_complexity'] += 1
        
        return metrics
    
    def _extract_instance_variables(self, init_node: ast.FunctionDef) -> Set[str]:
        """Extract instance variables from __init__ method."""
        instance_vars = set()
        
        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            instance_vars.add(target.attr)
        
        return instance_vars
    
    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_chain(node)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node.__class__.__name__)
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the name of a function/method call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_chain(node.func)
        return None
    
    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get the full attribute chain (e.g., 'self.attr.subattr')."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))
    
    def _find_caller_context(self, node) -> str:
        """Find the context (class.method or function) where a call is made."""
        # Walk up the AST to find the containing function/method
        parent = node
        while hasattr(parent, 'parent'):
            parent = parent.parent
            if isinstance(parent, ast.FunctionDef):
                # Find containing class if any
                class_parent = parent
                while hasattr(class_parent, 'parent'):
                    class_parent = class_parent.parent
                    if isinstance(class_parent, ast.ClassDef):
                        return f"{class_parent.name}.{parent.name}"
                return parent.name
        
        return "global"
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run complete analysis of the source file."""
        if not self.load_source_code():
            return {}
        
        print(f"Analyzing functionality map for {self.file_path.name}...")
        
        # Add parent references for context finding
        for node in ast.walk(self.ast_tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        
        self.analyze_imports()
        self.analyze_classes()
        self.analyze_functions()
        self.analyze_dependencies()
        self.create_call_graph()
        self.analyze_data_flow()
        
        # Add analysis metadata
        self.functionality_map['analysis_metadata'] = {
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'analyzer_version': '1.0.0',
            'file_analyzed': str(self.file_path),
            'analysis_complete': True
        }
        
        return self.functionality_map
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate high-level summary report."""
        if not self.functionality_map.get('analysis_metadata', {}).get('analysis_complete'):
            return {'error': 'Analysis not complete'}
        
        classes = self.functionality_map.get('classes', {})
        functions = self.functionality_map.get('functions', {})
        imports = self.functionality_map.get('imports', {})
        call_graph = self.functionality_map.get('call_graph', {})
        
        # Calculate summary statistics
        total_methods = sum(len(class_info['methods']) for class_info in classes.values())
        total_complexity = sum(
            method_info.get('complexity', 0)
            for class_info in classes.values()
            for method_info in class_info['methods'].values()
        )
        
        return {
            'file_summary': {
                'file_path': str(self.file_path),
                'total_classes': len(classes),
                'total_methods': total_methods,
                'total_functions': len(functions),
                'total_imports': sum(len(imports[key]) for key in ['stdlib', 'third_party', 'local']),
                'lines_of_code': self.functionality_map['file_info'].get('non_empty_lines', 0),
                'total_complexity': total_complexity
            },
            'complexity_analysis': {
                'average_methods_per_class': total_methods / len(classes) if classes else 0,
                'most_complex_class': self._find_most_complex_class(),
                'most_connected_method': self._find_most_connected_method(),
                'dependency_density': len(call_graph) / max(1, total_methods + len(functions))
            },
            'decomposition_candidates': self._identify_decomposition_candidates(),
            'external_dependencies': {
                'third_party_imports': imports.get('third_party', []),
                'high_coupling_indicators': self._find_coupling_indicators()
            }
        }
    
    def _find_most_complex_class(self) -> Dict[str, Any]:
        """Find the class with highest complexity."""
        max_complexity = 0
        most_complex = None
        
        for class_name, class_info in self.functionality_map.get('classes', {}).items():
            complexity = class_info.get('complexity_metrics', {}).get('cyclomatic_complexity', 0)
            if complexity > max_complexity:
                max_complexity = complexity
                most_complex = {
                    'name': class_name,
                    'complexity': complexity,
                    'method_count': class_info.get('complexity_metrics', {}).get('method_count', 0),
                    'line_count': class_info.get('complexity_metrics', {}).get('line_count', 0)
                }
        
        return most_complex or {}
    
    def _find_most_connected_method(self) -> Dict[str, Any]:
        """Find the method with most connections (calls + called_by)."""
        max_connections = 0
        most_connected = None
        
        for method_name, method_info in self.functionality_map.get('call_graph', {}).items():
            connections = len(method_info.get('calls', [])) + len(method_info.get('called_by', []))
            if connections > max_connections:
                max_connections = connections
                most_connected = {
                    'name': method_name,
                    'total_connections': connections,
                    'outgoing_calls': len(method_info.get('calls', [])),
                    'incoming_calls': len(method_info.get('called_by', []))
                }
        
        return most_connected or {}
    
    def _identify_decomposition_candidates(self) -> List[Dict[str, Any]]:
        """Identify classes/methods that are good candidates for decomposition."""
        candidates = []
        
        for class_name, class_info in self.functionality_map.get('classes', {}).items():
            metrics = class_info.get('complexity_metrics', {})
            method_count = metrics.get('method_count', 0)
            complexity = metrics.get('cyclomatic_complexity', 0)
            line_count = metrics.get('line_count', 0)
            
            # Criteria for decomposition candidate
            is_candidate = (
                method_count > 10 or  # Many methods
                complexity > 20 or   # High complexity
                line_count > 500     # Many lines
            )
            
            if is_candidate:
                candidates.append({
                    'class_name': class_name,
                    'reason': 'god_class',
                    'method_count': method_count,
                    'complexity': complexity,
                    'line_count': line_count,
                    'decomposition_suggestions': self._suggest_decomposition(class_info)
                })
        
        return candidates
    
    def _suggest_decomposition(self, class_info: Dict[str, Any]) -> List[str]:
        """Suggest decomposition strategies for a class."""
        suggestions = []
        
        methods = class_info.get('methods', {})
        method_names = list(methods.keys())
        
        # Group methods by common prefixes/patterns
        method_groups = {}
        for method_name in method_names:
            if '_' in method_name:
                prefix = method_name.split('_')[0]
                if prefix not in method_groups:
                    method_groups[prefix] = []
                method_groups[prefix].append(method_name)
        
        # Suggest extraction based on method groups
        for prefix, group_methods in method_groups.items():
            if len(group_methods) >= 3:
                suggestions.append(f"Extract {prefix}-related methods into separate class")
        
        # Check for utility methods
        utility_methods = [name for name in method_names if name.startswith('_') and not name.startswith('__')]
        if len(utility_methods) >= 3:
            suggestions.append("Extract utility methods into helper class")
        
        return suggestions
    
    def _find_coupling_indicators(self) -> List[str]:
        """Find indicators of tight coupling."""
        indicators = []
        
        # Check for excessive external dependencies
        imports = self.functionality_map.get('imports', {})
        third_party_count = len(imports.get('third_party', []))
        if third_party_count > 10:
            indicators.append(f"High third-party dependency count: {third_party_count}")
        
        # Check for complex call patterns
        call_graph = self.functionality_map.get('call_graph', {})
        highly_connected = [
            name for name, info in call_graph.items()
            if len(info.get('calls', [])) + len(info.get('called_by', [])) > 10
        ]
        
        if highly_connected:
            indicators.append(f"Highly connected methods: {len(highly_connected)}")
        
        return indicators


def main():
    """Create functionality maps for all god classes."""
    project_root = Path(__file__).parent.parent.parent
    backup_dir = project_root / "backups" / "god_classes"
    analysis_dir = project_root / "tests" / "analysis_reports"
    
    # Ensure analysis directory exists
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Map of class names to backup files
    god_classes = [
        ("AdvancedHighStakesAuditor", "high_stakes_audit_backup_20250814_121039.py"),
        ("LLMEvaluator", "evaluate_backup_20250814_121303.py"),
        ("TrainingDashboard", "dashboard_backup_20250814_123055.py"),
        ("EnhancedLoRASFTTrainer", "sft_lora_backup_20250814_123726.py")
    ]
    
    comprehensive_analysis = {
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'project_root': str(project_root),
        'backup_directory': str(backup_dir),
        'functionality_maps': [],
        'summary_statistics': {}
    }
    
    print("Creating functionality maps for god classes...")
    print(f"Backup directory: {backup_dir}")
    print(f"Analysis reports directory: {analysis_dir}")
    
    total_stats = {
        'total_classes': 0,
        'total_methods': 0,
        'total_functions': 0,
        'total_complexity': 0,
        'decomposition_candidates': 0
    }
    
    for class_name, backup_filename in god_classes:
        backup_file = backup_dir / backup_filename
        
        if backup_file.exists():
            print(f"\nProcessing: {class_name} ({backup_filename})")
            
            # Create functionality mapper
            mapper = FunctionalityMapper(backup_file)
            functionality_map = mapper.analyze_all()
            
            if functionality_map:
                # Generate summary report
                summary_report = mapper.generate_summary_report()
                
                combined_analysis = {
                    'class_name': class_name,
                    'functionality_map': functionality_map,
                    'summary_report': summary_report
                }
                
                comprehensive_analysis['functionality_maps'].append(combined_analysis)
                
                # Update total stats
                file_summary = summary_report.get('file_summary', {})
                total_stats['total_classes'] += file_summary.get('total_classes', 0)
                total_stats['total_methods'] += file_summary.get('total_methods', 0)
                total_stats['total_functions'] += file_summary.get('total_functions', 0)
                total_stats['total_complexity'] += file_summary.get('total_complexity', 0)
                
                decomposition_candidates = summary_report.get('decomposition_candidates', [])
                total_stats['decomposition_candidates'] += len(decomposition_candidates)
                
                # Save individual analysis
                individual_file = analysis_dir / f"{class_name.lower()}_functionality_map.json"
                with open(individual_file, 'w') as f:
                    json.dump(combined_analysis, f, indent=2, default=str)
                
                print(f"  - Classes: {file_summary.get('total_classes', 0)}")
                print(f"  - Methods: {file_summary.get('total_methods', 0)}")
                print(f"  - Functions: {file_summary.get('total_functions', 0)}")
                print(f"  - Complexity: {file_summary.get('total_complexity', 0)}")
                print(f"  - Decomposition candidates: {len(decomposition_candidates)}")
                print(f"  - Analysis saved: {individual_file}")
            
        else:
            print(f"\nWARNING: Backup file not found: {backup_file}")
    
    # Add summary statistics
    comprehensive_analysis['summary_statistics'] = total_stats
    
    # Save comprehensive analysis
    comprehensive_file = analysis_dir / "comprehensive_functionality_analysis.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_analysis, f, indent=2, default=str)
    
    # Create markdown summary
    summary_file = analysis_dir / "functionality_analysis_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# God Class Functionality Analysis Summary\n\n")
        f.write(f"**Analysis Date:** {comprehensive_analysis['timestamp']}\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Classes Analyzed:** {total_stats['total_classes']}\n")
        f.write(f"- **Total Methods Identified:** {total_stats['total_methods']}\n")
        f.write(f"- **Total Functions Identified:** {total_stats['total_functions']}\n")
        f.write(f"- **Total Complexity Score:** {total_stats['total_complexity']}\n")
        f.write(f"- **Decomposition Candidates:** {total_stats['decomposition_candidates']}\n\n")
        
        f.write("## Individual Class Analysis\n\n")
        for analysis in comprehensive_analysis['functionality_maps']:
            class_name = analysis['class_name']
            summary = analysis['summary_report']['file_summary']
            complexity = analysis['summary_report']['complexity_analysis']
            
            f.write(f"### {class_name}\n")
            f.write(f"- **Classes:** {summary.get('total_classes', 0)}\n")
            f.write(f"- **Methods:** {summary.get('total_methods', 0)}\n")
            f.write(f"- **Functions:** {summary.get('total_functions', 0)}\n")
            f.write(f"- **Lines of Code:** {summary.get('lines_of_code', 0)}\n")
            f.write(f"- **Complexity:** {summary.get('total_complexity', 0)}\n")
            f.write(f"- **Avg Methods/Class:** {complexity.get('average_methods_per_class', 0):.1f}\n")
            
            # Most complex class info
            most_complex = complexity.get('most_complex_class', {})
            if most_complex:
                f.write(f"- **Most Complex Class:** {most_complex.get('name', 'N/A')} (complexity: {most_complex.get('complexity', 0)})\n")
            
            f.write("\n")
        
        f.write("## Files Created\n\n")
        f.write("- `comprehensive_functionality_analysis.json` - Complete functionality analysis\n")
        for analysis in comprehensive_analysis['functionality_maps']:
            class_name = analysis['class_name'].lower()
            f.write(f"- `{class_name}_functionality_map.json` - Individual {analysis['class_name']} analysis\n")
    
    print(f"\n=== FUNCTIONALITY ANALYSIS COMPLETE ===")
    print(f"Total classes analyzed: {total_stats['total_classes']}")
    print(f"Total methods found: {total_stats['total_methods']}")
    print(f"Total complexity score: {total_stats['total_complexity']}")
    print(f"Decomposition candidates: {total_stats['decomposition_candidates']}")
    print(f"Comprehensive analysis saved: {comprehensive_file}")
    print(f"Summary report saved: {summary_file}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)