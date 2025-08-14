"""
Consolidated Reporting Functionality.

This module provides a unified reporting system for all evaluation results,
metrics, and analyses across the platform with multiple output formats.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import threading
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import StringIO, BytesIO

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CSV = "csv"
    EXCEL = "excel"
    LATEX = "latex"
    INTERACTIVE = "interactive"


class ReportSection(Enum):
    """Report sections."""
    SUMMARY = "summary"
    METRICS = "metrics"
    VISUALIZATIONS = "visualizations"
    DETAILED_ANALYSIS = "detailed_analysis"
    RECOMMENDATIONS = "recommendations"
    METADATA = "metadata"
    APPENDIX = "appendix"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str
    format: ReportFormat = ReportFormat.HTML
    sections: List[ReportSection] = field(default_factory=lambda: list(ReportSection))
    include_visualizations: bool = True
    include_raw_data: bool = False
    output_dir: Optional[Path] = None
    template_path: Optional[Path] = None
    style_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    """Data for report generation."""
    metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    text_sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Any] = None


class BaseReportGenerator(ABC):
    """
    Base class for report generators.
    
    Provides interface for different report format generators.
    """
    
    def __init__(self, config: ReportConfig):
        """Initialize report generator."""
        self.config = config
    
    @abstractmethod
    def generate(self, data: ReportData) -> Union[str, bytes]:
        """Generate report from data."""
        pass
    
    @abstractmethod
    def save(self, content: Union[str, bytes], filename: str):
        """Save report to file."""
        pass
    
    def _prepare_output_path(self, filename: str) -> Path:
        """Prepare output file path."""
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            return self.config.output_dir / filename
        return Path(filename)


class JSONReportGenerator(BaseReportGenerator):
    """JSON report generator."""
    
    def generate(self, data: ReportData) -> str:
        """Generate JSON report."""
        report = {
            'title': self.config.title,
            'generated_at': datetime.now().isoformat(),
            'metadata': {**self.config.metadata, **data.metadata}
        }
        
        # Add sections
        if ReportSection.SUMMARY in self.config.sections:
            report['summary'] = self._generate_summary(data)
        
        if ReportSection.METRICS in self.config.sections:
            report['metrics'] = data.metrics
        
        if ReportSection.DETAILED_ANALYSIS in self.config.sections:
            report['analysis'] = data.text_sections
        
        if self.config.include_raw_data and data.raw_data is not None:
            report['raw_data'] = self._serialize_raw_data(data.raw_data)
        
        # Convert DataFrames to dict
        if data.tables:
            report['tables'] = {
                name: df.to_dict('records') 
                for name, df in data.tables.items()
            }
        
        return json.dumps(report, indent=2, default=str)
    
    def save(self, content: str, filename: str):
        """Save JSON report."""
        filepath = self._prepare_output_path(filename)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Saved JSON report to {filepath}")
    
    def _generate_summary(self, data: ReportData) -> Dict[str, Any]:
        """Generate summary section."""
        summary = {}
        
        if data.metrics:
            # Extract key metrics
            for metric_name, metric_value in data.metrics.items():
                if isinstance(metric_value, dict) and 'value' in metric_value:
                    summary[metric_name] = metric_value['value']
                else:
                    summary[metric_name] = metric_value
        
        return summary
    
    def _serialize_raw_data(self, raw_data: Any) -> Any:
        """Serialize raw data for JSON."""
        if isinstance(raw_data, np.ndarray):
            return raw_data.tolist()
        elif isinstance(raw_data, pd.DataFrame):
            return raw_data.to_dict('records')
        elif hasattr(raw_data, '__dict__'):
            return raw_data.__dict__
        return str(raw_data)


class HTMLReportGenerator(BaseReportGenerator):
    """HTML report generator with styling."""
    
    def generate(self, data: ReportData) -> str:
        """Generate HTML report."""
        html_parts = []
        
        # HTML header
        html_parts.append(self._generate_header())
        
        # Title and metadata
        html_parts.append(f"<h1>{self.config.title}</h1>")
        html_parts.append(f"<p class='metadata'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Generate sections
        if ReportSection.SUMMARY in self.config.sections:
            html_parts.append(self._generate_summary_section(data))
        
        if ReportSection.METRICS in self.config.sections:
            html_parts.append(self._generate_metrics_section(data))
        
        if ReportSection.VISUALIZATIONS in self.config.sections and self.config.include_visualizations:
            html_parts.append(self._generate_visualizations_section(data))
        
        if ReportSection.DETAILED_ANALYSIS in self.config.sections:
            html_parts.append(self._generate_analysis_section(data))
        
        # Tables
        if data.tables:
            html_parts.append(self._generate_tables_section(data))
        
        # HTML footer
        html_parts.append(self._generate_footer())
        
        return '\n'.join(html_parts)
    
    def save(self, content: str, filename: str):
        """Save HTML report."""
        filepath = self._prepare_output_path(filename)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Saved HTML report to {filepath}")
    
    def _generate_header(self) -> str:
        """Generate HTML header with styles."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.config.title}</title>
            <style>
                {self._get_default_styles()}
                {self.config.style_config.get('custom_css', '')}
            </style>
        </head>
        <body>
        <div class="container">
        """
    
    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
        </div>
        </body>
        </html>
        """
    
    def _get_default_styles(self) -> str:
        """Get default CSS styles."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }
        .metadata {
            color: #7f8c8d;
            font-style: italic;
        }
        .metric-card {
            display: inline-block;
            background: #ecf0f1;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            min-width: 150px;
        }
        .metric-name {
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-value {
            font-size: 24px;
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .visualization {
            margin: 20px 0;
            text-align: center;
        }
        .analysis-section {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }
        """
    
    def _generate_summary_section(self, data: ReportData) -> str:
        """Generate summary section."""
        html = ["<h2>Summary</h2>"]
        html.append("<div class='summary-grid'>")
        
        # Display key metrics as cards
        for metric_name, metric_value in data.metrics.items():
            if isinstance(metric_value, dict):
                value = metric_value.get('value', 'N/A')
            else:
                value = metric_value
            
            # Format value
            if isinstance(value, float):
                value = f"{value:.4f}"
            
            html.append(f"""
            <div class='metric-card'>
                <div class='metric-name'>{metric_name.replace('_', ' ').title()}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """)
        
        html.append("</div>")
        return '\n'.join(html)
    
    def _generate_metrics_section(self, data: ReportData) -> str:
        """Generate detailed metrics section."""
        html = ["<h2>Detailed Metrics</h2>"]
        
        for metric_name, metric_data in data.metrics.items():
            html.append(f"<h3>{metric_name.replace('_', ' ').title()}</h3>")
            
            if isinstance(metric_data, dict):
                html.append("<table>")
                html.append("<tr><th>Property</th><th>Value</th></tr>")
                
                for key, value in metric_data.items():
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    elif isinstance(value, list) and len(value) > 10:
                        value = f"[{len(value)} items]"
                    
                    html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                
                html.append("</table>")
            else:
                html.append(f"<p>{metric_data}</p>")
        
        return '\n'.join(html)
    
    def _generate_visualizations_section(self, data: ReportData) -> str:
        """Generate visualizations section."""
        html = ["<h2>Visualizations</h2>"]
        
        for viz_name, viz_data in data.visualizations.items():
            html.append(f"<div class='visualization'>")
            html.append(f"<h3>{viz_name.replace('_', ' ').title()}</h3>")
            
            # Embed visualization (assumes base64 encoded image)
            if isinstance(viz_data, str) and viz_data.startswith('data:image'):
                html.append(f"<img src='{viz_data}' alt='{viz_name}' />")
            elif isinstance(viz_data, bytes):
                import base64
                encoded = base64.b64encode(viz_data).decode('utf-8')
                html.append(f"<img src='data:image/png;base64,{encoded}' alt='{viz_name}' />")
            
            html.append("</div>")
        
        return '\n'.join(html)
    
    def _generate_analysis_section(self, data: ReportData) -> str:
        """Generate analysis section."""
        html = ["<h2>Detailed Analysis</h2>"]
        
        for section_name, section_text in data.text_sections.items():
            html.append(f"<div class='analysis-section'>")
            html.append(f"<h3>{section_name.replace('_', ' ').title()}</h3>")
            html.append(f"<p>{section_text}</p>")
            html.append("</div>")
        
        return '\n'.join(html)
    
    def _generate_tables_section(self, data: ReportData) -> str:
        """Generate tables section."""
        html = ["<h2>Data Tables</h2>"]
        
        for table_name, df in data.tables.items():
            html.append(f"<h3>{table_name.replace('_', ' ').title()}</h3>")
            html.append(df.to_html(classes='data-table', index=False))
        
        return '\n'.join(html)


class MarkdownReportGenerator(BaseReportGenerator):
    """Markdown report generator."""
    
    def generate(self, data: ReportData) -> str:
        """Generate Markdown report."""
        md_parts = []
        
        # Title and metadata
        md_parts.append(f"# {self.config.title}\n")
        md_parts.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Table of contents
        if len(self.config.sections) > 2:
            md_parts.append(self._generate_toc())
        
        # Generate sections
        if ReportSection.SUMMARY in self.config.sections:
            md_parts.append(self._generate_summary_section(data))
        
        if ReportSection.METRICS in self.config.sections:
            md_parts.append(self._generate_metrics_section(data))
        
        if ReportSection.DETAILED_ANALYSIS in self.config.sections:
            md_parts.append(self._generate_analysis_section(data))
        
        # Tables
        if data.tables:
            md_parts.append(self._generate_tables_section(data))
        
        # Metadata
        if ReportSection.METADATA in self.config.sections:
            md_parts.append(self._generate_metadata_section(data))
        
        return '\n'.join(md_parts)
    
    def save(self, content: str, filename: str):
        """Save Markdown report."""
        filepath = self._prepare_output_path(filename)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Saved Markdown report to {filepath}")
    
    def _generate_toc(self) -> str:
        """Generate table of contents."""
        toc = ["## Table of Contents\n"]
        
        for section in self.config.sections:
            section_name = section.value.replace('_', ' ').title()
            anchor = section.value.replace('_', '-')
            toc.append(f"- [{section_name}](#{anchor})")
        
        toc.append("")
        return '\n'.join(toc)
    
    def _generate_summary_section(self, data: ReportData) -> str:
        """Generate summary section."""
        md = ["## Summary\n"]
        
        # Create metrics table
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        
        for metric_name, metric_value in data.metrics.items():
            if isinstance(metric_value, dict):
                value = metric_value.get('value', 'N/A')
            else:
                value = metric_value
            
            if isinstance(value, float):
                value = f"{value:.4f}"
            
            md.append(f"| {metric_name.replace('_', ' ').title()} | {value} |")
        
        md.append("")
        return '\n'.join(md)
    
    def _generate_metrics_section(self, data: ReportData) -> str:
        """Generate metrics section."""
        md = ["## Detailed Metrics\n"]
        
        for metric_name, metric_data in data.metrics.items():
            md.append(f"### {metric_name.replace('_', ' ').title()}\n")
            
            if isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    elif isinstance(value, list) and len(value) > 10:
                        value = f"[{len(value)} items]"
                    
                    md.append(f"- **{key}**: {value}")
            else:
                md.append(f"{metric_data}")
            
            md.append("")
        
        return '\n'.join(md)
    
    def _generate_analysis_section(self, data: ReportData) -> str:
        """Generate analysis section."""
        md = ["## Detailed Analysis\n"]
        
        for section_name, section_text in data.text_sections.items():
            md.append(f"### {section_name.replace('_', ' ').title()}\n")
            md.append(section_text)
            md.append("")
        
        return '\n'.join(md)
    
    def _generate_tables_section(self, data: ReportData) -> str:
        """Generate tables section."""
        md = ["## Data Tables\n"]
        
        for table_name, df in data.tables.items():
            md.append(f"### {table_name.replace('_', ' ').title()}\n")
            md.append(df.to_markdown(index=False))
            md.append("")
        
        return '\n'.join(md)
    
    def _generate_metadata_section(self, data: ReportData) -> str:
        """Generate metadata section."""
        md = ["## Metadata\n"]
        
        all_metadata = {**self.config.metadata, **data.metadata}
        
        for key, value in all_metadata.items():
            md.append(f"- **{key}**: {value}")
        
        md.append("")
        return '\n'.join(md)


class UnifiedReporter:
    """
    Unified reporting system for all evaluation results.
    
    Provides centralized report generation with multiple formats,
    customizable templates, and comprehensive visualization support.
    """
    
    def __init__(self):
        """Initialize unified reporter."""
        # Report generators
        self.generators: Dict[ReportFormat, Type[BaseReportGenerator]] = {
            ReportFormat.JSON: JSONReportGenerator,
            ReportFormat.HTML: HTMLReportGenerator,
            ReportFormat.MARKDOWN: MarkdownReportGenerator
        }
        
        # Report history
        self.report_history: List[Dict[str, Any]] = []
        
        # Visualization generators
        self.viz_generators: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register default visualizations
        self._register_default_visualizations()
        
        logger.info("Initialized UnifiedReporter")
    
    def _register_default_visualizations(self):
        """Register default visualization generators."""
        self.register_visualization("metric_bars", self._generate_metric_bars)
        self.register_visualization("confusion_matrix", self._generate_confusion_matrix)
        self.register_visualization("calibration_plot", self._generate_calibration_plot)
        self.register_visualization("metric_history", self._generate_metric_history)
    
    def generate_report(self, 
                       config: ReportConfig,
                       data: ReportData,
                       auto_visualize: bool = True) -> Union[str, bytes]:
        """
        Generate report with specified configuration.
        
        Args:
            config: Report configuration
            data: Report data
            auto_visualize: Automatically generate visualizations
            
        Returns:
            Generated report content
        """
        try:
            with self._lock:
                # Auto-generate visualizations if requested
                if auto_visualize and config.include_visualizations:
                    self._auto_generate_visualizations(data)
                
                # Get appropriate generator
                generator_class = self.generators.get(config.format)
                if not generator_class:
                    raise ValueError(f"Unsupported report format: {config.format}")
                
                # Generate report
                generator = generator_class(config)
                content = generator.generate(data)
                
                # Save to history
                self.report_history.append({
                    'config': asdict(config),
                    'timestamp': datetime.now().isoformat(),
                    'format': config.format.value,
                    'sections': [s.value for s in config.sections]
                })
                
                logger.info(f"Generated {config.format.value} report: {config.title}")
                return content
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def save_report(self, 
                   content: Union[str, bytes],
                   filename: str,
                   config: ReportConfig):
        """
        Save report to file.
        
        Args:
            content: Report content
            filename: Output filename
            config: Report configuration
        """
        try:
            generator_class = self.generators.get(config.format)
            if not generator_class:
                raise ValueError(f"Unsupported report format: {config.format}")
            
            generator = generator_class(config)
            generator.save(content, filename)
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise
    
    def register_visualization(self, name: str, generator: Callable) -> bool:
        """
        Register visualization generator.
        
        Args:
            name: Visualization name
            generator: Generator function
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.viz_generators[name] = generator
            
            logger.info(f"Registered visualization: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register visualization {name}: {e}")
            return False
    
    def _auto_generate_visualizations(self, data: ReportData):
        """Automatically generate standard visualizations."""
        # Generate metric bars if metrics available
        if data.metrics:
            data.visualizations['metric_bars'] = self._generate_metric_bars(data.metrics)
        
        # Generate calibration plot if calibration data available
        if 'calibration' in data.metrics:
            data.visualizations['calibration_plot'] = self._generate_calibration_plot(
                data.metrics.get('calibration', {})
            )
    
    def _generate_metric_bars(self, metrics: Dict[str, Any]) -> bytes:
        """Generate bar chart of metrics."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract metric values
            names = []
            values = []
            
            for name, value in metrics.items():
                if isinstance(value, dict) and 'value' in value:
                    names.append(name.replace('_', ' ').title())
                    values.append(value['value'])
                elif isinstance(value, (int, float)):
                    names.append(name.replace('_', ' ').title())
                    values.append(value)
            
            # Create bar chart
            bars = ax.bar(names, values)
            
            # Color bars based on value
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' 
                     for v in values]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel('Value')
            ax.set_title('Evaluation Metrics')
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Convert to bytes
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate metric bars: {e}")
            return b''
    
    def _generate_confusion_matrix(self, cm_data: np.ndarray) -> bytes:
        """Generate confusion matrix heatmap."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {e}")
            return b''
    
    def _generate_calibration_plot(self, calibration_data: Dict[str, Any]) -> bytes:
        """Generate calibration plot."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Extract bin data
            bin_metrics = calibration_data.get('bin_metrics', [])
            
            if bin_metrics:
                confidences = [b['confidence'] for b in bin_metrics]
                accuracies = [b['accuracy'] for b in bin_metrics]
                
                # Plot calibration curve
                ax.plot(confidences, accuracies, 'o-', label='Calibration')
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Accuracy')
                ax.set_title('Calibration Plot')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate calibration plot: {e}")
            return b''
    
    def _generate_metric_history(self, history_data: List[Dict[str, Any]]) -> bytes:
        """Generate metric history plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(history_data)
            
            for column in df.select_dtypes(include=[np.number]).columns:
                ax.plot(df.index, df[column], label=column, marker='o')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.set_title('Metric History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate metric history: {e}")
            return b''
    
    def create_comparison_report(self, 
                                results: List[Dict[str, Any]],
                                config: ReportConfig) -> Union[str, bytes]:
        """
        Create comparison report for multiple results.
        
        Args:
            results: List of evaluation results to compare
            config: Report configuration
            
        Returns:
            Comparison report content
        """
        # Prepare comparison data
        comparison_df = pd.DataFrame(results)
        
        data = ReportData(
            tables={'comparison': comparison_df},
            text_sections={
                'overview': f"Comparison of {len(results)} evaluation runs",
                'best_performer': self._identify_best_performer(results)
            }
        )
        
        # Add comparison visualizations
        if config.include_visualizations:
            data.visualizations['comparison_chart'] = self._generate_comparison_chart(comparison_df)
        
        return self.generate_report(config, data)
    
    def _identify_best_performer(self, results: List[Dict[str, Any]]) -> str:
        """Identify best performing configuration."""
        if not results:
            return "No results to compare"
        
        # Simple heuristic: highest average metric value
        best_idx = 0
        best_score = 0
        
        for i, result in enumerate(results):
            metrics = result.get('metrics', {})
            avg_score = np.mean([
                v['value'] if isinstance(v, dict) else v 
                for v in metrics.values() 
                if isinstance(v, (dict, float, int))
            ])
            
            if avg_score > best_score:
                best_score = avg_score
                best_idx = i
        
        return f"Configuration {best_idx + 1} with average score: {best_score:.4f}"
    
    def _generate_comparison_chart(self, df: pd.DataFrame) -> bytes:
        """Generate comparison chart."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot metrics for each configuration
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            x = np.arange(len(df))
            width = 0.8 / len(numeric_cols)
            
            for i, col in enumerate(numeric_cols):
                offset = (i - len(numeric_cols)/2) * width
                ax.bar(x + offset, df[col], width, label=col)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Value')
            ax.set_title('Metric Comparison')
            ax.legend()
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate comparison chart: {e}")
            return b''


# Global reporter instance
_reporter = None

def get_reporter() -> UnifiedReporter:
    """Get global reporter instance."""
    global _reporter
    if _reporter is None:
        _reporter = UnifiedReporter()
    return _reporter


# Convenience functions

def generate_evaluation_report(
    metrics: Dict[str, Any],
    title: str = "Evaluation Report",
    format: ReportFormat = ReportFormat.HTML,
    output_dir: Optional[Path] = None,
    save: bool = True
) -> Union[str, bytes]:
    """
    Generate evaluation report from metrics.
    
    Args:
        metrics: Evaluation metrics
        title: Report title
        format: Output format
        output_dir: Output directory
        save: Whether to save report
        
    Returns:
        Report content
    """
    reporter = get_reporter()
    
    config = ReportConfig(
        title=title,
        format=format,
        output_dir=output_dir,
        sections=[
            ReportSection.SUMMARY,
            ReportSection.METRICS,
            ReportSection.VISUALIZATIONS
        ]
    )
    
    data = ReportData(metrics=metrics)
    
    content = reporter.generate_report(config, data)
    
    if save and output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{title.lower().replace(' ', '_')}_{timestamp}.{format.value}"
        reporter.save_report(content, filename, config)
    
    return content