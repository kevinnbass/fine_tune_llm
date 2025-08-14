"""
Facade pattern for backward compatibility with original dashboard.py.

This module provides a compatibility layer that maintains the original API
while delegating to the new decomposed components.
"""

import warnings
from typing import Dict, List, Optional, Any
import threading
import time
import pandas as pd
from pathlib import Path

# Import new decomposed components
from src.fine_tune_llm.monitoring import (
    MetricsCollector,
    TrainingMetrics as NewTrainingMetrics,
    DashboardRenderer
)

# Deprecation warning
warnings.warn(
    "dashboard.py has been decomposed into multiple modules. "
    "Please update your imports to use the new modular components from "
    "src.fine_tune_llm.monitoring/. This facade will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)


class TrainingMetrics:
    """
    Backward compatibility facade for TrainingMetrics.
    
    This class maintains the original API while delegating to new components.
    """
    
    def __init__(self):
        """Initialize with backward compatibility."""
        self._metrics = NewTrainingMetrics()
        
        # Expose original attributes for compatibility
        self.metrics = self._metrics.metrics
        self.timestamps = self._metrics.timestamps
        self.lock = self._metrics.lock
        self.start_time = self._metrics.start_time
        self.max_history = self._metrics.max_history
        self.update_interval = self._metrics.update_interval
    
    def add_metrics(self, metrics_dict: Dict[str, Any], timestamp=None):
        """Add metrics (original API)."""
        return self._metrics.add_metrics(metrics_dict, timestamp)
    
    def get_latest(self, metric_name: str):
        """Get latest value (original API).""" 
        return self._metrics.get_latest(metric_name)
    
    def get_history(self, metric_name: str, hours=None):
        """Get history (original API)."""
        return self._metrics.get_history(metric_name, hours)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get DataFrame (original API)."""
        return self._metrics.get_metrics_dataframe()
    
    def get_summary_stats(self):
        """Get summary stats (original API)."""
        return self._metrics.get_summary_stats()
    
    def save_to_file(self, filepath):
        """Save to file (original API)."""
        return self._metrics.save_to_file(filepath)
    
    def load_from_file(self, filepath):
        """Load from file (original API).""" 
        return self._metrics.load_from_file(filepath)
    
    # Forward any other attributes
    def __getattr__(self, name):
        """Forward undefined attributes to new metrics."""
        return getattr(self._metrics, name)


class TrainingDashboard:
    """
    Backward compatibility facade for TrainingDashboard.
    
    This class maintains the original API while delegating to new components.
    """
    
    def __init__(self, 
                 metrics_file: str = "artifacts/models/llm_lora/training_metrics.json",
                 demo_mode: bool = False,
                 update_interval: int = 5):
        """Initialize with backward compatibility."""
        # Create new components
        config = {
            'collector': {
                'update_interval': update_interval,
                'auto_save': True,
                'save_path': Path(metrics_file).parent
            },
            'dashboard': {
                'update_interval': update_interval,
                'theme': 'dark',
                'show_advanced_plots': True
            }
        }
        
        self._metrics_collector = MetricsCollector(config)
        self._renderer = DashboardRenderer(self._metrics_collector, config)
        
        # Original attributes for compatibility
        self.metrics_file = Path(metrics_file)
        self.demo_mode = demo_mode
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Create TrainingMetrics facade for backward compatibility
        self.training_metrics = TrainingMetrics()
        
        # Load existing metrics if available
        if self.metrics_file.exists() and not demo_mode:
            try:
                self.training_metrics.load_from_file(self.metrics_file)
            except Exception as e:
                print(f"Warning: Could not load metrics file: {e}")
        
        # Setup demo data if needed
        if demo_mode:
            self._setup_demo_data()
    
    def start_monitoring(self):
        """Start monitoring (original API)."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._metrics_collector.start_monitoring()
        
        # Start monitoring thread for file updates
        if not self.demo_mode:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring (original API)."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self._metrics_collector.stop_monitoring()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def add_metrics(self, metrics_dict: Dict[str, Any]):
        """Add metrics (original API)."""
        self.training_metrics.add_metrics(metrics_dict)
        self._metrics_collector.add_metrics(metrics_dict)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics DataFrame (original API)."""
        return self.training_metrics.get_metrics_dataframe()
    
    def create_dashboard(self):
        """Create dashboard (original API)."""
        # Transfer data from training_metrics to metrics_collector
        df = self.training_metrics.get_metrics_dataframe()
        if not df.empty:
            for timestamp, row in df.iterrows():
                metrics_dict = row.dropna().to_dict()
                self._metrics_collector.collect_metrics(metrics_dict)
        
        # Render dashboard
        return self._renderer.create_dashboard()
    
    def _monitor_loop(self):
        """Monitor loop for file updates (original API compatibility)."""
        last_modified = 0
        
        while self.monitoring:
            try:
                if self.metrics_file.exists():
                    current_modified = self.metrics_file.stat().st_mtime
                    if current_modified > last_modified:
                        self._update_metrics()
                        last_modified = current_modified
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(5)
    
    def _update_metrics(self):
        """Update metrics from file (original API compatibility)."""
        try:
            if self.metrics_file.exists():
                self.training_metrics.load_from_file(self.metrics_file)
                
                # Get latest metrics and add to collector
                latest_stats = self.training_metrics.get_summary_stats()
                if latest_stats:
                    latest_values = {
                        name: stats.get('latest', 0.0) 
                        for name, stats in latest_stats.items()
                    }
                    self._metrics_collector.collect_metrics(latest_values)
                    
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _setup_demo_data(self):
        """Setup demo data for testing (original API compatibility)."""
        import random
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample training metrics
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(100):
            timestamp = base_time + timedelta(minutes=i)
            
            # Generate realistic training metrics
            epoch_progress = i / 100.0
            train_loss = 2.5 * np.exp(-2 * epoch_progress) + 0.1 + random.normal(0, 0.05)
            eval_loss = train_loss + 0.2 + random.normal(0, 0.03)
            
            metrics = {
                'train_loss': max(0.05, train_loss),
                'eval_loss': max(0.05, eval_loss), 
                'learning_rate': 2e-4 * (1 - epoch_progress),
                'epoch': epoch_progress * 3,
                'accuracy': 0.5 + 0.4 * epoch_progress + random.normal(0, 0.02),
                'f1_score': 0.4 + 0.5 * epoch_progress + random.normal(0, 0.02),
                'gpu_memory_used': 8.5 + random.normal(0, 0.5),
                'ece': 0.15 * (1 - epoch_progress) + random.normal(0, 0.01),
                'grad_norm': 1.0 + random.normal(0, 0.2)
            }
            
            self.training_metrics.add_metrics(metrics, timestamp)
    
    # Delegate render methods for backward compatibility
    def _render_current_status(self):
        """Render current status (original API)."""
        return self._renderer._render_current_status()
    
    def _render_training_plots(self):
        """Render training plots (original API)."""
        return self._renderer._render_training_plots()
    
    def _render_advanced_plots(self):
        """Render advanced plots (original API)."""
        return self._renderer._render_advanced_plots()
    
    def _render_metrics_table(self):
        """Render metrics table (original API)."""
        return self._renderer._render_metrics_table()
    
    def _render_risk_prediction_interface(self):
        """Render risk prediction interface (original API)."""
        # This was a complex method in the original - simplified for compatibility
        import streamlit as st
        st.subheader("🎯 Risk-Controlled Prediction Interface")
        st.info("Risk prediction interface has been moved to a separate component.")
        st.write("Please use the new RiskPredictionInterface from src.fine_tune_llm.inference.risk_control")
    
    def _get_metric_delta(self, metric_name: str):
        """Get metric delta (original API)."""
        return self._renderer._get_metric_delta(metric_name)
    
    def _make_risk_controlled_prediction(self, text: str, confidence_level: float, cost_matrix=None):
        """Make risk-controlled prediction (original API compatibility)."""
        # Simplified implementation for backward compatibility
        return {
            'prediction': 'positive',  # placeholder
            'confidence': 0.8,
            'risk_score': 0.2,
            'abstain': False
        }
    
    # Forward any other attributes to renderer or collector
    def __getattr__(self, name):
        """Forward undefined attributes."""
        # Try renderer first, then collector
        if hasattr(self._renderer, name):
            return getattr(self._renderer, name)
        elif hasattr(self._metrics_collector, name):
            return getattr(self._metrics_collector, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Export original class names for full backward compatibility
__all__ = [
    'TrainingMetrics',
    'TrainingDashboard'
]