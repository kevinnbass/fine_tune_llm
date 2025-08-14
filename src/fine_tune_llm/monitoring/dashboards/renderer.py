"""
Dashboard rendering module for training monitoring.

This module provides comprehensive dashboard rendering capabilities
using Streamlit for interactive training monitoring.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ...core.interfaces import BaseComponent
from ..collectors import MetricsCollector

logger = logging.getLogger(__name__)


class DashboardRenderer(BaseComponent):
    """Render interactive training dashboards using Streamlit."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize dashboard renderer.
        
        Args:
            metrics_collector: Metrics collector instance
            config: Configuration dictionary
        """
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.dashboard_config = self.config.get('dashboard', {})
        
        # UI settings
        self.theme = self.dashboard_config.get('theme', 'dark')
        self.update_interval = self.dashboard_config.get('update_interval', 5)
        self.show_advanced_plots = self.dashboard_config.get('show_advanced_plots', True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        
        # Cache for metric deltas
        self._last_values = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.dashboard_config = self.config.get('dashboard', {})
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._last_values.clear()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "DashboardRenderer"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def create_dashboard(self):
        """Create the main Streamlit dashboard."""
        st.set_page_config(
            page_title="Training Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Dashboard header
        st.title("🚀 LLM Training Dashboard")
        st.markdown("Real-time monitoring of model training progress")
        
        # Sidebar controls
        with st.sidebar:
            self._render_sidebar_controls()
        
        # Main dashboard sections
        self._render_current_status()
        self._render_training_plots()
        
        if self.show_advanced_plots:
            self._render_advanced_plots()
        
        self._render_metrics_table()
        
        # Auto-refresh
        time.sleep(self.update_interval)
        st.experimental_rerun()
    
    def _render_sidebar_controls(self):
        """Render sidebar controls."""
        st.header("⚙️ Controls")
        
        # Time range selector
        st.subheader("Time Range")
        time_options = {
            "Last 1 Hour": 1,
            "Last 6 Hours": 6, 
            "Last 24 Hours": 24,
            "All Time": None
        }
        selected_time = st.selectbox(
            "Select time range:",
            options=list(time_options.keys()),
            index=1
        )
        self.time_range = time_options[selected_time]
        
        # Refresh controls
        st.subheader("Refresh")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Now"):
                st.experimental_rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        # Display settings
        st.subheader("Display")
        self.show_advanced_plots = st.checkbox("Show Advanced Plots", value=True)
        self.show_raw_data = st.checkbox("Show Raw Data", value=False)
        
        # Export options
        st.subheader("Export")
        if st.button("📊 Export Metrics"):
            self._export_metrics()
    
    def _render_current_status(self):
        """Render current training status section."""
        st.header("📈 Current Status")
        
        # Get latest metrics
        latest_stats = self.metrics_collector.get_summary_stats()
        
        if not latest_stats:
            st.warning("No metrics available yet. Training may not have started.")
            return
        
        # Key metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        # Training Loss
        with col1:
            loss_val = latest_stats.get('train_loss', {}).get('latest', 0.0)
            loss_delta = self._get_metric_delta('train_loss')
            st.metric(
                label="Training Loss",
                value=f"{loss_val:.4f}",
                delta=f"{loss_delta:.4f}" if loss_delta is not None else None
            )
        
        # Validation Loss
        with col2:
            val_loss = latest_stats.get('eval_loss', {}).get('latest', 0.0)
            val_delta = self._get_metric_delta('eval_loss')
            st.metric(
                label="Validation Loss",
                value=f"{val_loss:.4f}",
                delta=f"{val_delta:.4f}" if val_delta is not None else None
            )
        
        # Learning Rate
        with col3:
            lr_val = latest_stats.get('learning_rate', {}).get('latest', 0.0)
            st.metric(
                label="Learning Rate",
                value=f"{lr_val:.2e}"
            )
        
        # GPU Memory
        with col4:
            gpu_mem = latest_stats.get('gpu_memory_used', {}).get('latest', 0.0)
            st.metric(
                label="GPU Memory (GB)",
                value=f"{gpu_mem:.1f}"
            )
        
        # Progress information
        if 'epoch' in latest_stats:
            current_epoch = latest_stats['epoch']['latest']
            total_epochs = self.config.get('training', {}).get('epochs', 3)
            progress = current_epoch / total_epochs if total_epochs > 0 else 0.0
            
            st.progress(progress)
            st.write(f"Epoch {current_epoch:.0f} of {total_epochs}")
    
    def _render_training_plots(self):
        """Render main training plots."""
        st.header("📊 Training Progress")
        
        df = self.metrics_collector.get_metrics_dataframe()
        
        if df.empty:
            st.info("No training data available yet.")
            return
        
        # Filter by time range if specified
        if self.time_range:
            cutoff_time = datetime.now() - timedelta(hours=self.time_range)
            df = df[df.index >= cutoff_time]
        
        # Loss plot
        if 'train_loss' in df.columns or 'eval_loss' in df.columns:
            st.subheader("Loss Curves")
            
            fig = go.Figure()
            
            if 'train_loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['train_loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color=self.colors['primary'])
                ))
            
            if 'eval_loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['eval_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color=self.colors['warning'])
                ))
            
            fig.update_layout(
                title="Training and Validation Loss",
                xaxis_title="Time",
                yaxis_title="Loss",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Learning rate plot
        if 'learning_rate' in df.columns:
            st.subheader("Learning Rate Schedule")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['learning_rate'],
                mode='lines',
                name='Learning Rate',
                line=dict(color=self.colors['success'])
            ))
            
            fig.update_layout(
                title="Learning Rate Over Time",
                xaxis_title="Time",
                yaxis_title="Learning Rate",
                yaxis_type="log"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        perf_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        available_perf = [m for m in perf_metrics if m in df.columns]
        
        if available_perf:
            st.subheader("Performance Metrics")
            
            fig = go.Figure()
            colors_cycle = [self.colors['primary'], self.colors['secondary'], 
                          self.colors['success'], self.colors['info']]
            
            for i, metric in enumerate(available_perf):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors_cycle[i % len(colors_cycle)])
                ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Time", 
                yaxis_title="Score",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_advanced_plots(self):
        """Render advanced monitoring plots."""
        st.header("🔬 Advanced Analytics")
        
        df = self.metrics_collector.get_metrics_dataframe()
        
        if df.empty:
            return
        
        col1, col2 = st.columns(2)
        
        # GPU utilization
        with col1:
            gpu_metrics = [col for col in df.columns if 'gpu' in col.lower()]
            if gpu_metrics:
                st.subheader("GPU Utilization")
                
                fig = go.Figure()
                
                for metric in gpu_metrics:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[metric],
                        mode='lines',
                        name=metric.replace('_', ' ').title()
                    ))
                
                fig.update_layout(
                    title="GPU Resources",
                    xaxis_title="Time",
                    yaxis_title="Usage"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Calibration metrics
        with col2:
            cal_metrics = [col for col in df.columns if any(x in col.lower() for x in ['ece', 'mce', 'brier'])]
            if cal_metrics:
                st.subheader("Model Calibration")
                
                fig = go.Figure()
                
                for metric in cal_metrics:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[metric],
                        mode='lines',
                        name=metric.upper()
                    ))
                
                fig.update_layout(
                    title="Calibration Metrics",
                    xaxis_title="Time",
                    yaxis_title="Error"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Gradient norms
        grad_metrics = [col for col in df.columns if 'grad' in col.lower() and 'norm' in col.lower()]
        if grad_metrics:
            st.subheader("Gradient Analysis")
            
            fig = go.Figure()
            
            for metric in grad_metrics:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title()
                ))
            
            fig.update_layout(
                title="Gradient Norms Over Time",
                xaxis_title="Time",
                yaxis_title="Gradient Norm",
                yaxis_type="log"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_metrics_table(self):
        """Render detailed metrics table."""
        st.header("📋 Detailed Metrics")
        
        # Get summary statistics
        stats = self.metrics_collector.get_summary_stats()
        
        if not stats:
            st.info("No metrics data available.")
            return
        
        # Convert to display format
        display_data = []
        for metric_name, metric_stats in stats.items():
            display_data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Current': f"{metric_stats['latest']:.4f}",
                'Mean': f"{metric_stats['mean']:.4f}",
                'Min': f"{metric_stats['min']:.4f}",
                'Max': f"{metric_stats['max']:.4f}",
                'Count': metric_stats['count']
            })
        
        # Display as DataFrame
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True)
        
        # Show raw data if requested
        if hasattr(self, 'show_raw_data') and self.show_raw_data:
            st.subheader("Raw Data")
            raw_df = self.metrics_collector.get_metrics_dataframe()
            st.dataframe(raw_df)
    
    def _get_metric_delta(self, metric_name: str) -> Optional[float]:
        """Calculate metric delta from last update."""
        current_value = self.metrics_collector.get_latest(metric_name)
        
        if current_value is None:
            return None
        
        if metric_name in self._last_values:
            delta = current_value - self._last_values[metric_name]
            self._last_values[metric_name] = current_value
            return delta
        else:
            self._last_values[metric_name] = current_value
            return None
    
    def _export_metrics(self):
        """Export metrics to file."""
        try:
            df = self.metrics_collector.get_metrics_dataframe()
            
            if df.empty:
                st.error("No data to export")
                return
            
            # Convert to CSV
            csv = df.to_csv()
            
            # Offer download
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Export failed: {e}")