"""
Real-time training dashboard with comprehensive metrics visualization.

This module provides a web-based dashboard for monitoring LLM fine-tuning progress,
including advanced calibration metrics, conformal prediction statistics, and 
abstention-aware training progress.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import numpy as np

# Dashboard dependencies
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Import advanced metrics if available
try:
    from .metrics import MetricsAggregator
    from .conformal import ConformalPredictor, RiskControlledPredictor
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Structure for training metrics."""
    epoch: int
    step: int
    timestamp: float
    
    # Core training metrics
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    
    # Classification metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    # Advanced calibration metrics
    ece: Optional[float] = None
    mce: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Confidence metrics
    mean_confidence: Optional[float] = None
    confidence_accuracy_correlation: Optional[float] = None
    
    # Abstention metrics
    abstention_rate: Optional[float] = None
    effective_accuracy: Optional[float] = None
    abstention_cost: Optional[float] = None
    
    # Conformal prediction metrics
    conformal_coverage: Optional[float] = None
    conformal_avg_set_size: Optional[float] = None
    
    # Risk-aware metrics
    average_risk: Optional[float] = None
    risk_reduction: Optional[float] = None
    
    # System metrics
    gpu_memory_used: Optional[float] = None
    training_speed: Optional[float] = None  # samples/sec


class TrainingDashboard:
    """Real-time training dashboard with advanced metrics."""
    
    def __init__(self, 
                 metrics_path: Optional[str] = None,
                 update_interval: int = 5,
                 max_history: int = 1000):
        """
        Initialize training dashboard.
        
        Args:
            metrics_path: Path to metrics file for monitoring
            update_interval: Update interval in seconds
            max_history: Maximum number of metrics to keep in history
        """
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.update_interval = update_interval
        self.max_history = max_history
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.current_metrics: Optional[TrainingMetrics] = None
        
        # Dashboard state
        self.is_running = False
        self.last_update = 0
        self.monitor_thread = None
        
        # Initialize components
        if ADVANCED_METRICS_AVAILABLE and self.metrics_path:
            self.metrics_aggregator = MetricsAggregator(save_path=self.metrics_path)
            self.conformal_predictor = ConformalPredictor()
            self.risk_controlled_predictor = RiskControlledPredictor()
        else:
            self.metrics_aggregator = None
            self.conformal_predictor = None
            self.risk_controlled_predictor = None
        
        logger.info("Training dashboard initialized")
    
    def start_monitoring(self):
        """Start monitoring training metrics."""
        if not DASHBOARD_AVAILABLE:
            logger.warning("Dashboard dependencies not available")
            return False
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Started training monitoring")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring training metrics."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped training monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._update_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update metrics from file."""
        if not self.metrics_path or not self.metrics_path.exists():
            return
            
        try:
            # Check if file was modified
            file_mtime = self.metrics_path.stat().st_mtime
            if file_mtime <= self.last_update:
                return
                
            self.last_update = file_mtime
            
            # Load metrics
            with open(self.metrics_path, 'r') as f:
                data = json.load(f)
            
            # Convert to TrainingMetrics objects
            if isinstance(data, list) and data:
                latest_data = data[-1]  # Get most recent metrics
                
                # Extract core metrics with safe defaults
                metrics = TrainingMetrics(
                    epoch=latest_data.get('epoch', 0),
                    step=latest_data.get('step', 0),
                    timestamp=latest_data.get('timestamp', time.time()),
                    train_loss=latest_data.get('train_loss', latest_data.get('loss', 0.0)),
                    eval_loss=latest_data.get('eval_loss', None),
                    learning_rate=latest_data.get('learning_rate', 0.0),
                    
                    # Classification metrics
                    accuracy=latest_data.get('accuracy', None),
                    f1_score=latest_data.get('f1_macro', latest_data.get('f1_score', None)),
                    precision=latest_data.get('precision_macro', latest_data.get('precision', None)),
                    recall=latest_data.get('recall_macro', latest_data.get('recall', None)),
                    
                    # Advanced metrics
                    ece=latest_data.get('ece', None),
                    mce=latest_data.get('mce', None),
                    brier_score=latest_data.get('brier_score', None),
                    mean_confidence=latest_data.get('confidence_mean', latest_data.get('mean_confidence', None)),
                    confidence_accuracy_correlation=latest_data.get('confidence_accuracy_correlation', None),
                    
                    # Abstention metrics
                    abstention_rate=latest_data.get('abstention_rate', None),
                    effective_accuracy=latest_data.get('abstention_effective_accuracy', None),
                    abstention_cost=latest_data.get('abstention_avg_cost_per_sample', None),
                    
                    # Conformal metrics
                    conformal_coverage=latest_data.get('conformal_coverage', None),
                    conformal_avg_set_size=latest_data.get('conformal_avg_set_size', None),
                    
                    # Risk metrics
                    average_risk=latest_data.get('risk_average_risk', None),
                    risk_reduction=latest_data.get('risk_reduction', None),
                )
                
                # Update current metrics
                self.current_metrics = metrics
                
                # Add to history
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                    
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")
    
    def add_metrics(self, metrics_dict: Dict[str, Any]):
        """Add metrics manually (for testing or direct integration)."""
        try:
            metrics = TrainingMetrics(
                epoch=metrics_dict.get('epoch', len(self.metrics_history)),
                step=metrics_dict.get('step', 0),
                timestamp=time.time(),
                **{k: v for k, v in metrics_dict.items() 
                   if k not in ['epoch', 'step'] and hasattr(TrainingMetrics, k)}
            )
            
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Limit history size
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
                
        except Exception as e:
            logger.warning(f"Failed to add metrics: {e}")
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        data = [asdict(m) for m in self.metrics_history]
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def create_dashboard(self):
        """Create Streamlit dashboard."""
        if not DASHBOARD_AVAILABLE:
            return "Dashboard dependencies not available. Install streamlit and plotly."
        
        st.set_page_config(
            page_title="LLM Training Dashboard", 
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 LLM Fine-Tuning Dashboard")
        st.markdown("Real-time monitoring with advanced calibration and conformal prediction metrics")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, self.update_interval)
        
        if st.sidebar.button("Manual Refresh") or auto_refresh:
            self._update_metrics()
        
        # Risk prediction interface link
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Prediction Interface")
        st.sidebar.markdown("""
        **Risk-Controlled Predictions**: Launch the dedicated interface for making 
        predictions with statistical guarantees and risk control.
        """)
        
        if st.sidebar.button("🚀 Launch Risk Prediction UI"):
            st.sidebar.info("""
            To launch the risk prediction interface, run:
            ```bash
            python scripts/launch_risk_ui.py
            ```
            Or directly:
            ```bash
            streamlit run scripts/risk_prediction_ui.py
            ```
            """)
        
        st.sidebar.markdown("---")
        
        # Main dashboard
        if not self.metrics_history:
            st.warning("No training metrics available. Start training to see live updates.")
            return
        
        # Current status
        self._render_current_status()
        
        # Metrics plots
        self._render_training_plots()
        
        # Advanced metrics
        if ADVANCED_METRICS_AVAILABLE:
            self._render_advanced_plots()
            
        # Risk-controlled prediction interface
        if ADVANCED_METRICS_AVAILABLE:
            self._render_risk_prediction_interface()
        
        # Metrics table
        self._render_metrics_table()
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
    
    def _render_current_status(self):
        """Render current training status."""
        if not self.current_metrics:
            return
            
        st.header("📊 Current Training Status")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Epoch", 
                f"{self.current_metrics.epoch}",
                delta=None
            )
            if self.current_metrics.train_loss:
                st.metric(
                    "Training Loss", 
                    f"{self.current_metrics.train_loss:.4f}",
                    delta=self._get_metric_delta('train_loss')
                )
        
        with col2:
            if self.current_metrics.accuracy:
                st.metric(
                    "Accuracy", 
                    f"{self.current_metrics.accuracy:.3f}",
                    delta=self._get_metric_delta('accuracy')
                )
            if self.current_metrics.learning_rate:
                st.metric(
                    "Learning Rate", 
                    f"{self.current_metrics.learning_rate:.2e}",
                    delta=None
                )
        
        with col3:
            if self.current_metrics.ece is not None:
                st.metric(
                    "ECE (Calibration)", 
                    f"{self.current_metrics.ece:.4f}",
                    delta=self._get_metric_delta('ece'),
                    delta_color="inverse"  # Lower is better
                )
            if self.current_metrics.abstention_rate is not None:
                st.metric(
                    "Abstention Rate", 
                    f"{self.current_metrics.abstention_rate:.3f}",
                    delta=self._get_metric_delta('abstention_rate')
                )
        
        with col4:
            if self.current_metrics.conformal_coverage is not None:
                st.metric(
                    "Conformal Coverage", 
                    f"{self.current_metrics.conformal_coverage:.3f}",
                    delta=self._get_metric_delta('conformal_coverage')
                )
            if self.current_metrics.average_risk is not None:
                st.metric(
                    "Average Risk", 
                    f"{self.current_metrics.average_risk:.4f}",
                    delta=self._get_metric_delta('average_risk'),
                    delta_color="inverse"  # Lower is better
                )
    
    def _get_metric_delta(self, metric_name: str) -> Optional[float]:
        """Get delta for a metric compared to previous value."""
        if len(self.metrics_history) < 2:
            return None
            
        current_val = getattr(self.current_metrics, metric_name, None)
        previous_val = getattr(self.metrics_history[-2], metric_name, None)
        
        if current_val is not None and previous_val is not None:
            return current_val - previous_val
        return None
    
    def _render_training_plots(self):
        """Render core training plots."""
        st.header("📈 Training Progress")
        
        df = self.get_metrics_dataframe()
        
        if df.empty:
            st.warning("No metrics data available")
            return
        
        # Training loss plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Loss")
            if 'train_loss' in df.columns and df['train_loss'].notna().any():
                fig_loss = px.line(df, x='epoch', y='train_loss', title='Training Loss Over Time')
                fig_loss.update_layout(height=400)
                st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.subheader("Learning Rate")
            if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
                fig_lr = px.line(df, x='epoch', y='learning_rate', title='Learning Rate Schedule')
                fig_lr.update_layout(height=400, yaxis_type="log")
                st.plotly_chart(fig_lr, use_container_width=True)
        
        # Classification metrics
        st.subheader("Classification Metrics")
        
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in metrics_to_plot if m in df.columns and df[m].notna().any()]
        
        if available_metrics:
            fig_metrics = go.Figure()
            
            for metric in available_metrics:
                fig_metrics.add_trace(go.Scatter(
                    x=df['epoch'], 
                    y=df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
            
            fig_metrics.update_layout(
                title='Classification Performance Over Time',
                xaxis_title='Epoch',
                yaxis_title='Score',
                height=400,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
    
    def _render_advanced_plots(self):
        """Render advanced metrics plots."""
        st.header("🔬 Advanced Metrics")
        
        df = self.get_metrics_dataframe()
        
        if df.empty:
            return
        
        # Calibration metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Calibration Quality")
            calibration_metrics = ['ece', 'mce', 'brier_score']
            available_cal_metrics = [m for m in calibration_metrics if m in df.columns and df[m].notna().any()]
            
            if available_cal_metrics:
                fig_cal = go.Figure()
                
                for metric in available_cal_metrics:
                    fig_cal.add_trace(go.Scatter(
                        x=df['epoch'],
                        y=df[metric], 
                        mode='lines+markers',
                        name=metric.upper().replace('_', ' '),
                        line=dict(width=2)
                    ))
                
                fig_cal.update_layout(
                    title='Model Calibration Over Time',
                    xaxis_title='Epoch',
                    yaxis_title='Error',
                    height=400
                )
                
                st.plotly_chart(fig_cal, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Analysis")
            confidence_metrics = ['mean_confidence', 'confidence_accuracy_correlation']
            available_conf_metrics = [m for m in confidence_metrics if m in df.columns and df[m].notna().any()]
            
            if available_conf_metrics:
                fig_conf = make_subplots(specs=[[{"secondary_y": True}]])
                
                if 'mean_confidence' in available_conf_metrics:
                    fig_conf.add_trace(
                        go.Scatter(x=df['epoch'], y=df['mean_confidence'],
                                 name='Mean Confidence', line=dict(color='blue')),
                        secondary_y=False
                    )
                
                if 'confidence_accuracy_correlation' in available_conf_metrics:
                    fig_conf.add_trace(
                        go.Scatter(x=df['epoch'], y=df['confidence_accuracy_correlation'],
                                 name='Confidence-Accuracy Correlation', line=dict(color='red')),
                        secondary_y=True
                    )
                
                fig_conf.update_xaxes(title_text="Epoch")
                fig_conf.update_yaxes(title_text="Confidence", secondary_y=False)
                fig_conf.update_yaxes(title_text="Correlation", secondary_y=True)
                fig_conf.update_layout(height=400, title="Confidence Metrics")
                
                st.plotly_chart(fig_conf, use_container_width=True)
        
        # Abstention and risk metrics
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Abstention Analysis")
            abstention_metrics = ['abstention_rate', 'effective_accuracy']
            available_abs_metrics = [m for m in abstention_metrics if m in df.columns and df[m].notna().any()]
            
            if available_abs_metrics:
                fig_abs = go.Figure()
                
                for metric in available_abs_metrics:
                    fig_abs.add_trace(go.Scatter(
                        x=df['epoch'],
                        y=df[metric],
                        mode='lines+markers', 
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
                
                fig_abs.update_layout(
                    title='Abstention Behavior Over Time',
                    xaxis_title='Epoch',
                    yaxis_title='Rate/Score',
                    height=400,
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig_abs, use_container_width=True)
        
        with col4:
            st.subheader("Risk-Aware Metrics")
            risk_metrics = ['average_risk', 'risk_reduction']
            available_risk_metrics = [m for m in risk_metrics if m in df.columns and df[m].notna().any()]
            
            if available_risk_metrics:
                fig_risk = go.Figure()
                
                for metric in available_risk_metrics:
                    fig_risk.add_trace(go.Scatter(
                        x=df['epoch'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
                
                fig_risk.update_layout(
                    title='Risk Metrics Over Time',
                    xaxis_title='Epoch', 
                    yaxis_title='Risk Score',
                    height=400
                )
                
                st.plotly_chart(fig_risk, use_container_width=True)
    
    def _render_risk_prediction_interface(self):
        """Render risk-controlled prediction interface."""
        st.header("⚖️ Risk-Controlled Prediction Interface")
        
        # Description
        st.markdown("""
        This interface allows you to make predictions with statistical guarantees and risk control.
        Enter text below to get predictions with conformal prediction sets and risk assessments.
        """)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Text")
            user_text = st.text_area(
                "Enter text for prediction:",
                height=100,
                placeholder="Enter the text you want to classify..."
            )
        
        with col2:
            st.subheader("Prediction Settings")
            
            # Confidence level for conformal prediction
            confidence_level = st.slider(
                "Confidence Level", 
                min_value=0.80, 
                max_value=0.99, 
                value=0.90, 
                step=0.01,
                help="Statistical confidence level for prediction sets"
            )
            
            # Risk tolerance
            risk_tolerance = st.slider(
                "Risk Tolerance",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                help="Maximum acceptable risk for high-stakes decisions"
            )
            
            # Cost matrix settings
            st.subheader("Cost Matrix")
            false_positive_cost = st.number_input(
                "False Positive Cost", 
                min_value=0.1, 
                max_value=10.0, 
                value=1.0,
                step=0.1
            )
            false_negative_cost = st.number_input(
                "False Negative Cost", 
                min_value=0.1, 
                max_value=10.0, 
                value=5.0,
                step=0.1
            )
        
        # Prediction button
        if st.button("🎯 Make Risk-Controlled Prediction", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text for prediction.")
            else:
                self._make_risk_controlled_prediction(
                    user_text, 
                    confidence_level, 
                    risk_tolerance,
                    false_positive_cost,
                    false_negative_cost
                )
        
        # Demo section
        st.subheader("📊 Risk Control Visualization")
        
        # Show current model calibration status if available
        if self.current_metrics and self.current_metrics.ece is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Calibration status
                ece_color = "🟢" if self.current_metrics.ece < 0.05 else "🟡" if self.current_metrics.ece < 0.10 else "🔴"
                st.metric(
                    "Model Calibration",
                    f"{ece_color} ECE: {self.current_metrics.ece:.4f}",
                    help="Expected Calibration Error - measures how well-calibrated the model's confidence is"
                )
            
            with col2:
                # Coverage status
                if self.current_metrics.conformal_coverage is not None:
                    coverage_color = "🟢" if self.current_metrics.conformal_coverage > 0.90 else "🟡" if self.current_metrics.conformal_coverage > 0.85 else "🔴"
                    st.metric(
                        "Conformal Coverage",
                        f"{coverage_color} {self.current_metrics.conformal_coverage:.3f}",
                        help="Current conformal prediction coverage rate"
                    )
            
            with col3:
                # Risk level
                if self.current_metrics.average_risk is not None:
                    risk_color = "🟢" if self.current_metrics.average_risk < 0.10 else "🟡" if self.current_metrics.average_risk < 0.25 else "🔴"
                    st.metric(
                        "Average Risk",
                        f"{risk_color} {self.current_metrics.average_risk:.4f}",
                        help="Current average risk level across predictions"
                    )
    
    def _make_risk_controlled_prediction(self, text: str, confidence_level: float, 
                                       risk_tolerance: float, fp_cost: float, fn_cost: float):
        """Make a risk-controlled prediction with the given parameters."""
        try:
            st.subheader("🎯 Prediction Results")
            
            # Since we don't have actual model inference here, we'll create a demo prediction
            # In a real implementation, this would call the actual model
            
            # Demo prediction probabilities (normally from model inference)
            demo_probs = np.array([0.15, 0.25, 0.45, 0.15])  # [HIGH_RISK, MEDIUM_RISK, LOW_RISK, NO_RISK]
            labels = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
            
            st.info("📝 **Note**: This is a demo interface. In production, this would connect to your trained model.")
            
            # Display base prediction
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Base Prediction")
                max_prob_idx = np.argmax(demo_probs)
                max_prob = demo_probs[max_prob_idx]
                predicted_label = labels[max_prob_idx]
                
                st.write(f"**Predicted Class**: {predicted_label}")
                st.write(f"**Confidence**: {max_prob:.3f}")
                
                # Show all probabilities
                prob_df = pd.DataFrame({
                    'Class': labels,
                    'Probability': demo_probs
                })
                st.bar_chart(prob_df.set_index('Class'))
            
            with col2:
                st.subheader("Risk Analysis")
                
                # Create cost matrix
                cost_matrix = np.array([
                    [0.0, fp_cost, fp_cost, fp_cost],  # HIGH_RISK predictions
                    [fn_cost, 0.0, fp_cost, fp_cost],  # MEDIUM_RISK predictions  
                    [fn_cost, fn_cost, 0.0, fp_cost],  # LOW_RISK predictions
                    [fn_cost, fn_cost, fn_cost, 0.0]   # NO_RISK predictions
                ])
                
                # Compute expected costs
                expected_costs = demo_probs @ cost_matrix
                min_cost_idx = np.argmin(expected_costs)
                min_cost_label = labels[min_cost_idx]
                min_cost_value = expected_costs[min_cost_idx]
                
                st.write(f"**Risk-Optimal Decision**: {min_cost_label}")
                st.write(f"**Expected Cost**: {min_cost_value:.3f}")
                
                # Risk assessment
                high_risk_prob = demo_probs[0] + demo_probs[1]  # HIGH + MEDIUM risk
                risk_status = "🔴 HIGH" if high_risk_prob > 0.7 else "🟡 MEDIUM" if high_risk_prob > 0.3 else "🟢 LOW"
                st.write(f"**Risk Level**: {risk_status}")
            
            # Conformal prediction set
            st.subheader("🎯 Conformal Prediction Set")
            alpha = 1 - confidence_level
            
            # Demo conformal prediction (normally computed by ConformalPredictor)
            # Sort probabilities and include labels until we exceed confidence threshold
            sorted_indices = np.argsort(demo_probs)[::-1]
            cumulative_prob = 0.0
            prediction_set = []
            
            for idx in sorted_indices:
                cumulative_prob += demo_probs[idx]
                prediction_set.append(labels[idx])
                if cumulative_prob >= confidence_level:
                    break
            
            st.write(f"**Prediction Set** (at {confidence_level:.1%} confidence):")
            for i, label in enumerate(prediction_set):
                prob = demo_probs[labels.index(label)]
                st.write(f"  {i+1}. {label} (p={prob:.3f})")
            
            st.write(f"**Set Size**: {len(prediction_set)} classes")
            
            # Risk-controlled decision
            st.subheader("⚖️ Risk-Controlled Decision")
            
            if high_risk_prob > risk_tolerance:
                st.error(f"🚫 **ABSTAIN** - Risk level ({high_risk_prob:.3f}) exceeds tolerance ({risk_tolerance:.3f})")
                st.write("**Recommendation**: Human review required for this prediction.")
            else:
                if min_cost_label == predicted_label:
                    st.success(f"✅ **PROCEED** with prediction: {predicted_label}")
                else:
                    st.warning(f"⚠️ **ADJUST** - Risk analysis suggests: {min_cost_label} instead of {predicted_label}")
                
                st.write(f"Risk level ({high_risk_prob:.3f}) is within acceptable tolerance.")
            
            # Additional metrics
            st.subheader("📊 Prediction Metrics")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                entropy = -np.sum(demo_probs * np.log(demo_probs + 1e-8))
                st.metric("Entropy (Uncertainty)", f"{entropy:.3f}")
            
            with metrics_col2:
                confidence_gap = max_prob - np.partition(demo_probs, -2)[-2]  # Gap between top 2
                st.metric("Confidence Gap", f"{confidence_gap:.3f}")
            
            with metrics_col3:
                effective_sample_size = 1 / np.sum(demo_probs ** 2)
                st.metric("Effective Sample Size", f"{effective_sample_size:.1f}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    def _render_metrics_table(self):
        """Render detailed metrics table."""
        st.header("📋 Detailed Metrics")
        
        df = self.get_metrics_dataframe()
        
        if df.empty:
            st.warning("No metrics data available")
            return
        
        # Show last 10 entries
        display_df = df.tail(10).copy()
        
        # Format datetime
        display_df['Time'] = display_df['datetime'].dt.strftime('%H:%M:%S')
        
        # Select relevant columns
        columns_to_show = ['Time', 'epoch', 'step', 'train_loss', 'accuracy', 'ece', 'abstention_rate']
        available_columns = [col for col in columns_to_show if col in display_df.columns]
        
        if available_columns:
            # Format numeric columns
            numeric_columns = display_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(4)
            
            st.dataframe(
                display_df[available_columns].sort_values('epoch', ascending=False),
                use_container_width=True
            )
        
        # Download button
        if st.button("📥 Download Full Metrics"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def run_dashboard(metrics_path: Optional[str] = None, port: int = 8501):
    """
    Run the training dashboard.
    
    Args:
        metrics_path: Path to metrics file
        port: Port to run Streamlit on
    """
    if not DASHBOARD_AVAILABLE:
        logger.error("Dashboard dependencies not available. Please install: pip install streamlit plotly")
        return False
    
    dashboard = TrainingDashboard(metrics_path=metrics_path)
    dashboard.start_monitoring()
    
    try:
        # This would typically be run via: streamlit run dashboard.py
        dashboard.create_dashboard()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    finally:
        dashboard.stop_monitoring()
    
    return True


if __name__ == "__main__":
    import sys
    
    metrics_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/llm_lora/training_metrics.json"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8501
    
    run_dashboard(metrics_path, port)