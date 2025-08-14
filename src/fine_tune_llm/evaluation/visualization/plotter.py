"""
Visualization module for evaluation results.

This module provides comprehensive visualization capabilities for
model evaluation metrics and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class EvaluationPlotter(BaseComponent):
    """Create visualizations for evaluation results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation plotter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.viz_config = self.config.get('visualization', {})
        
        # Plotting parameters
        self.figure_size = self.viz_config.get('figure_size', (12, 8))
        self.dpi = self.viz_config.get('dpi', 100)
        self.style = self.viz_config.get('style', 'seaborn')
        self.color_palette = self.viz_config.get('palette', 'Set2')
        
        # Set style
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
        
        # Output directory
        self.output_dir = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.viz_config = self.config.get('visualization', {})
    
    def cleanup(self) -> None:
        """Clean up resources."""
        plt.close('all')
    
    @property
    def name(self) -> str:
        """Component name."""
        return "EvaluationPlotter"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def create_all_visualizations(self,
                                 metrics: Dict[str, Any],
                                 detailed_results: Optional[List[Dict]] = None,
                                 output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Create all standard visualizations.
        
        Args:
            metrics: Computed metrics
            detailed_results: Detailed evaluation results
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        try:
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                cm_path = self.plot_confusion_matrix(
                    np.array(metrics['confusion_matrix']),
                    save_path=self.output_dir / 'confusion_matrix.png' if self.output_dir else None
                )
                if cm_path:
                    created_files['confusion_matrix'] = cm_path
            
            # Per-class metrics
            if 'per_class_metrics' in metrics:
                pc_path = self.plot_per_class_metrics(
                    metrics['per_class_metrics'],
                    save_path=self.output_dir / 'per_class_metrics.png' if self.output_dir else None
                )
                if pc_path:
                    created_files['per_class_metrics'] = pc_path
            
            # Confidence distribution
            if detailed_results:
                conf_path = self.plot_confidence_distribution(
                    detailed_results,
                    save_path=self.output_dir / 'confidence_dist.png' if self.output_dir else None
                )
                if conf_path:
                    created_files['confidence_distribution'] = conf_path
            
            # Metrics summary
            summary_path = self.plot_metrics_summary(
                metrics,
                save_path=self.output_dir / 'metrics_summary.png' if self.output_dir else None
            )
            if summary_path:
                created_files['metrics_summary'] = summary_path
            
            # Calibration plot
            if 'ece' in metrics or 'mce' in metrics:
                cal_path = self.plot_calibration(
                    metrics,
                    save_path=self.output_dir / 'calibration.png' if self.output_dir else None
                )
                if cal_path:
                    created_files['calibration'] = cal_path
            
            logger.info(f"Created {len(created_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return created_files
    
    def plot_confusion_matrix(self,
                             confusion_matrix: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Optional class names
            save_path: Path to save figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Normalize for display
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Create heatmap
            sns.heatmap(
                cm_normalized,
                annot=confusion_matrix,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names if class_names else range(confusion_matrix.shape[0]),
                yticklabels=class_names if class_names else range(confusion_matrix.shape[0]),
                ax=ax,
                cbar_kws={'label': 'Normalized Count'}
            )
            
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved confusion matrix to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            return None
    
    def plot_per_class_metrics(self,
                              per_class_metrics: Dict[str, Dict[str, float]],
                              save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot per-class metrics comparison.
        
        Args:
            per_class_metrics: Per-class metrics dictionary
            save_path: Path to save figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Prepare data
            classes = list(per_class_metrics.keys())
            metrics_names = ['precision', 'recall', 'f1_score']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, metric_name in enumerate(metrics_names):
                values = [per_class_metrics[cls].get(metric_name, 0) for cls in classes]
                
                ax = axes[idx]
                bars = ax.bar(range(len(classes)), values)
                
                # Color bars based on value
                for bar, val in zip(bars, values):
                    if val < 0.5:
                        bar.set_color('red')
                    elif val < 0.7:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.set_ylim([0, 1])
                ax.set_ylabel('Score')
                ax.set_title(metric_name.capitalize().replace('_', ' '), fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.2f}', ha='center', va='bottom')
            
            plt.suptitle('Per-Class Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved per-class metrics to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error plotting per-class metrics: {e}")
            return None
    
    def plot_confidence_distribution(self,
                                    detailed_results: List[Dict],
                                    save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot confidence score distribution.
        
        Args:
            detailed_results: List of evaluation results with confidence scores
            save_path: Path to save figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Extract confidence scores
            confidences = [r.get('confidence', 0.5) for r in detailed_results]
            correct = []
            incorrect = []
            
            for result in detailed_results:
                conf = result.get('confidence', 0.5)
                if 'ground_truth' in result and 'prediction' in result:
                    if result['ground_truth'] == result['prediction']:
                        correct.append(conf)
                    else:
                        incorrect.append(conf)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Overall distribution
            ax1 = axes[0]
            ax1.hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.2f}')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Count')
            ax1.set_title('Overall Confidence Distribution', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Correct vs Incorrect
            ax2 = axes[1]
            if correct and incorrect:
                ax2.hist(correct, bins=20, alpha=0.5, color='green', 
                        label=f'Correct (n={len(correct)})', edgecolor='black')
                ax2.hist(incorrect, bins=20, alpha=0.5, color='red',
                        label=f'Incorrect (n={len(incorrect)})', edgecolor='black')
                ax2.set_xlabel('Confidence Score')
                ax2.set_ylabel('Count')
                ax2.set_title('Confidence by Prediction Correctness', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved confidence distribution to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error plotting confidence distribution: {e}")
            return None
    
    def plot_metrics_summary(self,
                           metrics: Dict[str, Any],
                           save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot summary of key metrics.
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Select key metrics to display
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            macro_metrics = ['precision_macro', 'recall_macro', 'f1_macro']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            metric_names = []
            metric_values = []
            
            for metric in key_metrics:
                if metric in metrics:
                    metric_names.append(metric.replace('_', ' ').title())
                    metric_values.append(metrics[metric])
            
            # Add macro metrics if available
            for metric in macro_metrics:
                if metric in metrics:
                    metric_names.append(metric.replace('_', ' ').title())
                    metric_values.append(metrics[metric])
            
            if metric_names:
                # Create bar plot
                bars = ax.bar(range(len(metric_names)), metric_values)
                
                # Color code bars
                for bar, val in zip(bars, metric_values):
                    if val < 0.5:
                        bar.set_color('red')
                    elif val < 0.7:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                ax.set_xticks(range(len(metric_names)))
                ax.set_xticklabels(metric_names, rotation=45, ha='right')
                ax.set_ylim([0, 1])
                ax.set_ylabel('Score')
                ax.set_title('Metrics Summary', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')
                
                # Add horizontal line at 0.5 and 0.7
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
                ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
                ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved metrics summary to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error plotting metrics summary: {e}")
            return None
    
    def plot_calibration(self,
                        metrics: Dict[str, Any],
                        save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot calibration metrics.
        
        Args:
            metrics: Metrics dictionary with calibration info
            save_path: Path to save figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Extract calibration metrics
            ece = metrics.get('ece', 0)
            mce = metrics.get('mce', 0)
            
            # Create bar plot
            cal_metrics = ['ECE', 'MCE']
            cal_values = [ece, mce]
            
            bars = ax.bar(cal_metrics, cal_values, color=['blue', 'red'])
            
            ax.set_ylabel('Calibration Error')
            ax.set_title('Model Calibration Metrics', fontsize=14, fontweight='bold')
            ax.set_ylim([0, max(cal_values) * 1.2 if cal_values else 1])
            
            # Add value labels
            for bar, val in zip(bars, cal_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{val:.4f}', ha='center', va='bottom')
            
            # Add interpretation text
            interpretation = "Lower values indicate better calibration"
            ax.text(0.5, 0.95, interpretation, transform=ax.transAxes,
                   ha='center', va='top', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved calibration plot to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error plotting calibration: {e}")
            return None