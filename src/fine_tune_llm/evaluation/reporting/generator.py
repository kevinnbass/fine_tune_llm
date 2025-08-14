"""
Report generation module for evaluation results.

This module provides comprehensive report generation capabilities
including text reports, HTML reports, and JSON exports.
"""

import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import logging
from io import StringIO

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class ReportGenerator(BaseComponent):
    """Generate evaluation reports in various formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.report_config = self.config.get('reporting', {})
        
        # Report settings
        self.include_timestamp = self.report_config.get('include_timestamp', True)
        self.include_config = self.report_config.get('include_config', True)
        self.include_detailed_results = self.report_config.get('include_detailed_results', False)
        self.formats = self.report_config.get('formats', ['text', 'json'])
        
        # Templates
        self.templates = {
            'header': "=" * 80,
            'section': "-" * 40,
            'subsection': "." * 20
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.report_config = self.config.get('reporting', {})
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    def name(self) -> str:
        """Component name."""
        return "ReportGenerator"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def generate_all_reports(self,
                           metrics: Dict[str, Any],
                           detailed_results: Optional[List[Dict]] = None,
                           output_dir: Optional[Path] = None,
                           model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Generate all configured report formats.
        
        Args:
            metrics: Evaluation metrics
            detailed_results: Detailed evaluation results
            output_dir: Directory to save reports
            model_info: Model information
            
        Returns:
            Dictionary mapping format to file path
        """
        generated_reports = {}
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate each format
        for format_type in self.formats:
            try:
                if format_type == 'text':
                    path = self.generate_text_report(
                        metrics, detailed_results, output_dir, model_info
                    )
                    if path:
                        generated_reports['text'] = path
                        
                elif format_type == 'json':
                    path = self.generate_json_report(
                        metrics, detailed_results, output_dir, model_info
                    )
                    if path:
                        generated_reports['json'] = path
                        
                elif format_type == 'html':
                    path = self.generate_html_report(
                        metrics, detailed_results, output_dir, model_info
                    )
                    if path:
                        generated_reports['html'] = path
                        
            except Exception as e:
                logger.error(f"Error generating {format_type} report: {e}")
        
        logger.info(f"Generated {len(generated_reports)} reports")
        return generated_reports
    
    def generate_text_report(self,
                           metrics: Dict[str, Any],
                           detailed_results: Optional[List[Dict]] = None,
                           output_dir: Optional[Path] = None,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Generate text format report.
        
        Args:
            metrics: Evaluation metrics
            detailed_results: Detailed results
            output_dir: Output directory
            model_info: Model information
            
        Returns:
            Path to generated report
        """
        try:
            report = StringIO()
            
            # Header
            report.write(self.templates['header'] + "\n")
            report.write("EVALUATION REPORT\n")
            report.write(self.templates['header'] + "\n\n")
            
            # Timestamp
            if self.include_timestamp:
                report.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Model Information
            if model_info:
                report.write("MODEL INFORMATION\n")
                report.write(self.templates['section'] + "\n")
                for key, value in model_info.items():
                    report.write(f"  {key}: {value}\n")
                report.write("\n")
            
            # Overall Metrics
            report.write("OVERALL METRICS\n")
            report.write(self.templates['section'] + "\n")
            
            # Primary metrics
            primary_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in primary_metrics:
                if metric in metrics:
                    report.write(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n")
            report.write("\n")
            
            # Macro/Micro averages
            if any(k for k in metrics if 'macro' in k or 'micro' in k):
                report.write("AVERAGING STRATEGIES\n")
                report.write(self.templates['subsection'] + "\n")
                
                for avg_type in ['macro', 'micro']:
                    for metric in ['precision', 'recall', 'f1']:
                        key = f"{metric}_{avg_type}"
                        if key in metrics:
                            report.write(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}\n")
                report.write("\n")
            
            # Per-class metrics
            if 'per_class_metrics' in metrics:
                report.write("PER-CLASS METRICS\n")
                report.write(self.templates['section'] + "\n")
                
                for class_name, class_metrics in metrics['per_class_metrics'].items():
                    report.write(f"\n  {class_name}:\n")
                    for metric, value in class_metrics.items():
                        if isinstance(value, float):
                            report.write(f"    {metric}: {value:.4f}\n")
                        else:
                            report.write(f"    {metric}: {value}\n")
                report.write("\n")
            
            # Confidence metrics
            confidence_metrics = [k for k in metrics if 'confidence' in k]
            if confidence_metrics:
                report.write("CONFIDENCE ANALYSIS\n")
                report.write(self.templates['section'] + "\n")
                
                for metric in confidence_metrics:
                    report.write(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n")
                report.write("\n")
            
            # Calibration metrics
            if 'ece' in metrics or 'mce' in metrics:
                report.write("CALIBRATION METRICS\n")
                report.write(self.templates['section'] + "\n")
                
                if 'ece' in metrics:
                    report.write(f"  Expected Calibration Error (ECE): {metrics['ece']:.4f}\n")
                if 'mce' in metrics:
                    report.write(f"  Maximum Calibration Error (MCE): {metrics['mce']:.4f}\n")
                report.write("\n")
            
            # Abstention metrics
            abstention_metrics = [k for k in metrics if 'abstention' in k or 'coverage' in k]
            if abstention_metrics:
                report.write("ABSTENTION ANALYSIS\n")
                report.write(self.templates['section'] + "\n")
                
                for metric in abstention_metrics:
                    report.write(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n")
                report.write("\n")
            
            # Confusion Matrix
            if 'confusion_matrix' in metrics:
                report.write("CONFUSION MATRIX\n")
                report.write(self.templates['section'] + "\n")
                
                cm = metrics['confusion_matrix']
                if isinstance(cm, list):
                    for i, row in enumerate(cm):
                        report.write(f"  {row}\n")
                report.write("\n")
            
            # Summary statistics
            if detailed_results:
                report.write("SUMMARY STATISTICS\n")
                report.write(self.templates['section'] + "\n")
                
                total_samples = len(detailed_results)
                errors = sum(1 for r in detailed_results if 'error' in r)
                abstentions = sum(1 for r in detailed_results if r.get('abstain', False))
                
                report.write(f"  Total Samples: {total_samples}\n")
                report.write(f"  Errors: {errors} ({errors/total_samples*100:.1f}%)\n")
                report.write(f"  Abstentions: {abstentions} ({abstentions/total_samples*100:.1f}%)\n")
                report.write("\n")
            
            # Configuration
            if self.include_config and self.config:
                report.write("CONFIGURATION\n")
                report.write(self.templates['section'] + "\n")
                report.write(json.dumps(self.config, indent=2))
                report.write("\n")
            
            # Footer
            report.write(self.templates['header'] + "\n")
            report.write("END OF REPORT\n")
            report.write(self.templates['header'] + "\n")
            
            # Save to file
            if output_dir:
                output_path = output_dir / "evaluation_report.txt"
                with open(output_path, 'w') as f:
                    f.write(report.getvalue())
                logger.info(f"Saved text report to {output_path}")
                return output_path
            else:
                print(report.getvalue())
                return None
                
        except Exception as e:
            logger.error(f"Error generating text report: {e}")
            return None
    
    def generate_json_report(self,
                           metrics: Dict[str, Any],
                           detailed_results: Optional[List[Dict]] = None,
                           output_dir: Optional[Path] = None,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Generate JSON format report.
        
        Args:
            metrics: Evaluation metrics
            detailed_results: Detailed results
            output_dir: Output directory
            model_info: Model information
            
        Returns:
            Path to generated report
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat() if self.include_timestamp else None,
                'model_info': model_info,
                'metrics': metrics,
                'summary': {}
            }
            
            # Add summary statistics
            if detailed_results:
                report['summary'] = {
                    'total_samples': len(detailed_results),
                    'errors': sum(1 for r in detailed_results if 'error' in r),
                    'abstentions': sum(1 for r in detailed_results if r.get('abstain', False))
                }
                
                if self.include_detailed_results:
                    report['detailed_results'] = detailed_results
            
            # Add configuration
            if self.include_config:
                report['configuration'] = self.config
            
            # Save to file
            if output_dir:
                output_path = output_dir / "evaluation_report.json"
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Saved JSON report to {output_path}")
                return output_path
            else:
                print(json.dumps(report, indent=2, default=str))
                return None
                
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return None
    
    def generate_html_report(self,
                           metrics: Dict[str, Any],
                           detailed_results: Optional[List[Dict]] = None,
                           output_dir: Optional[Path] = None,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Generate HTML format report.
        
        Args:
            metrics: Evaluation metrics
            detailed_results: Detailed results
            output_dir: Output directory
            model_info: Model information
            
        Returns:
            Path to generated report
        """
        try:
            html = StringIO()
            
            # HTML header
            html.write("""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        h3 { color: #888; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric-value { font-weight: bold; }
        .good { color: green; }
        .warning { color: orange; }
        .poor { color: red; }
    </style>
</head>
<body>
""")
            
            # Title and timestamp
            html.write("<h1>Evaluation Report</h1>\n")
            if self.include_timestamp:
                html.write(f"<p>Generated: {datetime.now().isoformat()}</p>\n")
            
            # Model information
            if model_info:
                html.write("<h2>Model Information</h2>\n")
                html.write("<table>\n")
                for key, value in model_info.items():
                    html.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
                html.write("</table>\n")
            
            # Overall metrics
            html.write("<h2>Overall Metrics</h2>\n")
            html.write("<table>\n")
            html.write("<tr><th>Metric</th><th>Value</th></tr>\n")
            
            primary_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in primary_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    css_class = self._get_metric_class(value)
                    html.write(f"<tr><td>{metric.replace('_', ' ').title()}</td>")
                    html.write(f"<td class='metric-value {css_class}'>{value:.4f}</td></tr>\n")
            html.write("</table>\n")
            
            # Per-class metrics
            if 'per_class_metrics' in metrics:
                html.write("<h2>Per-Class Metrics</h2>\n")
                html.write("<table>\n")
                html.write("<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>\n")
                
                for class_name, class_metrics in metrics['per_class_metrics'].items():
                    html.write(f"<tr><td>{class_name}</td>")
                    for metric in ['precision', 'recall', 'f1_score', 'support']:
                        value = class_metrics.get(metric, 0)
                        if isinstance(value, float):
                            css_class = self._get_metric_class(value) if metric != 'support' else ''
                            html.write(f"<td class='{css_class}'>{value:.4f}</td>")
                        else:
                            html.write(f"<td>{value}</td>")
                    html.write("</tr>\n")
                html.write("</table>\n")
            
            # Close HTML
            html.write("</body>\n</html>")
            
            # Save to file
            if output_dir:
                output_path = output_dir / "evaluation_report.html"
                with open(output_path, 'w') as f:
                    f.write(html.getvalue())
                logger.info(f"Saved HTML report to {output_path}")
                return output_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return None
    
    def _get_metric_class(self, value: float) -> str:
        """Get CSS class based on metric value."""
        if value >= 0.8:
            return 'good'
        elif value >= 0.6:
            return 'warning'
        else:
            return 'poor'