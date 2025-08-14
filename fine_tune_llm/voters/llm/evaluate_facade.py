"""
Facade pattern for backward compatibility with original evaluate.py.

This module provides a compatibility layer that maintains the original API
while delegating to the new decomposed components.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Import new decomposed components
from src.fine_tune_llm.evaluation import (
    LLMEvaluator,
    MetricsComputer,
    EvaluationPlotter,
    ReportGenerator
)

# Deprecation warning
warnings.warn(
    "evaluate.py has been decomposed into multiple modules. "
    "Please update your imports to use the new modular components from "
    "src.fine_tune_llm.evaluation/. This facade will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)


class OriginalLLMEvaluator:
    """
    Backward compatibility facade for the original LLMEvaluator.
    
    This class maintains the original API while delegating to new components.
    All original methods are preserved for backward compatibility.
    """
    
    def __init__(self, 
                 model=None,
                 tokenizer=None,
                 model_path: Optional[str] = None,
                 config_path: str = "configs/llm_lora.yaml"):
        """Initialize with backward compatibility."""
        # Create new evaluator
        self._evaluator = LLMEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            config_path=config_path
        )
        
        # Create supporting components
        self._metrics_computer = MetricsComputer(self._evaluator.config)
        self._plotter = EvaluationPlotter(self._evaluator.config)
        self._report_generator = ReportGenerator(self._evaluator.config)
        
        # Expose original attributes
        self.model = self._evaluator.model
        self.tokenizer = self._evaluator.tokenizer
        self.model_path = self._evaluator.model_path
        self.config_path = self._evaluator.config_path
        self.config = self._evaluator.config
        self.device = self._evaluator.device
        
        # Expose generation parameters
        self.max_new_tokens = self._evaluator.max_new_tokens
        self.temperature = self._evaluator.temperature
        self.top_p = self._evaluator.top_p
        self.do_sample = self._evaluator.do_sample
        
        # Expose evaluation parameters
        self.parse_json_output = self._evaluator.parse_json_output
        self.confidence_threshold = self._evaluator.confidence_threshold
        self.label_list = self._evaluator.label_list
        self.label_to_id = self._evaluator.label_to_id
    
    def load_config(self):
        """Load configuration (original API)."""
        return self._evaluator.load_config()
    
    def load_model(self):
        """Load model (original API)."""
        return self._evaluator.load_model()
    
    def evaluate_single(self, text: str, parse_json: bool = False) -> Dict:
        """Evaluate single text (original API)."""
        return self._evaluator.evaluate_single(text)
    
    def predict_single(self, text: str, metadata: Dict = None) -> Dict:
        """Predict single text (original API)."""
        return self._evaluator.predict_single(text, metadata)
    
    def evaluate_dataset(self, 
                        dataset, 
                        batch_size: int = 1, 
                        output_path: Optional[str] = None) -> List[Dict]:
        """Evaluate dataset (original API)."""
        output_path_obj = Path(output_path) if output_path else None
        return self._evaluator.evaluate_dataset(dataset, batch_size, output_path_obj)
    
    def compute_metrics(self,
                       predictions: Optional[List] = None,
                       labels: Optional[List] = None,
                       detailed_results: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, float]:
        """Compute metrics (original API)."""
        return self._evaluator.compute_metrics(predictions, labels, detailed_results, **kwargs)
    
    def get_label_distribution(self, labels: List[str]) -> Dict[str, float]:
        """Get label distribution (original API)."""
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        return {label: count / total for label, count in counts.items()}
    
    def create_visualizations(self, detailed_results: List[Dict], metrics: Dict, output_dir: Path):
        """Create visualizations (original API)."""
        return self._plotter.create_all_visualizations(metrics, detailed_results, output_dir)
    
    def create_advanced_visualizations(self, metrics: Dict, output_dir: Path):
        """Create advanced visualizations (original API)."""
        return self._plotter.create_all_visualizations(metrics, None, output_dir)
    
    def create_text_report(self, metrics: Dict, output_dir: Path):
        """Create text report (original API)."""
        return self._report_generator.generate_text_report(metrics, None, output_dir)
    
    def _label_to_index(self, label: str) -> int:
        """Convert label to index (original API)."""
        return self._evaluator._label_to_index(label)
    
    # Forward any other attributes to the new evaluator
    def __getattr__(self, name):
        """Forward undefined attributes to the new evaluator."""
        return getattr(self._evaluator, name)


# Module-level functions for backward compatibility
def main():
    """Main function for CLI usage (backward compatibility)."""
    import argparse
    import yaml
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Evaluate LLM model')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', default='configs/llm_lora.yaml', help='Config file path')
    parser.add_argument('--test-data', required=True, help='Test data file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    
    args = parser.parse_args()
    
    # Load test data
    test_data = load_test_data(args.test_data)
    
    # Create evaluator
    evaluator = LLMEvaluator(
        model_path=args.model_path,
        config_path=args.config
    )
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        test_data, 
        batch_size=args.batch_size,
        output_path=Path(args.output_dir) / "detailed_results.json"
    )
    
    # Compute metrics
    metrics = evaluator.compute_metrics(detailed_results=results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations and reports
    evaluator.create_visualizations(results, metrics, output_dir)
    evaluator.create_text_report(metrics, output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")


def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from file (backward compatibility)."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix in ['.jsonl', '.ndjson']:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return data


def compute_metrics(predictions: List,
                   labels: List,
                   probabilities: Optional[List] = None,
                   **kwargs) -> Dict[str, float]:
    """Compute metrics (backward compatibility function)."""
    metrics_computer = MetricsComputer()
    return metrics_computer.compute_all_metrics(
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        **kwargs
    )


def create_visualizations(metrics: Dict,
                         detailed_results: Optional[List[Dict]] = None,
                         output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Create visualizations (backward compatibility function)."""
    plotter = EvaluationPlotter()
    return plotter.create_all_visualizations(metrics, detailed_results, output_dir)


def generate_report(metrics: Dict,
                   detailed_results: Optional[List[Dict]] = None,
                   output_dir: Optional[Path] = None,
                   model_info: Optional[Dict] = None) -> Dict[str, Path]:
    """Generate report (backward compatibility function)."""
    generator = ReportGenerator()
    return generator.generate_all_reports(metrics, detailed_results, output_dir, model_info)


# Create alias for backward compatibility
LLMEvaluator = OriginalLLMEvaluator

# Export original class and function names for full backward compatibility
__all__ = [
    'LLMEvaluator',
    'OriginalLLMEvaluator',
    'main',
    'load_test_data',
    'compute_metrics',
    'create_visualizations',
    'generate_report'
]