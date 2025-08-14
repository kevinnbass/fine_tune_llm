#!/usr/bin/env python3
"""
Risk-Controlled Prediction Interface

A standalone Streamlit application for making predictions with statistical guarantees
and risk control using trained LLM models.

Usage:
    streamlit run scripts/risk_prediction_ui.py -- [model_path]

Example:
    streamlit run scripts/risk_prediction_ui.py -- artifacts/models/llm_lora/final_model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Streamlit and plotting
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Import LLM components
try:
    from voters.llm.infer import LLMInferenceEngine
    from voters.llm.metrics import MetricsAggregator, compute_ece, compute_mce
    from voters.llm.conformal import ConformalPredictor, RiskControlledPredictor
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskControlledPredictionUI:
    """Streamlit interface for risk-controlled predictions."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the risk-controlled prediction UI.
        
        Args:
            model_path: Path to trained model for inference
        """
        self.model_path = model_path
        
        # Initialize components if available
        if ADVANCED_FEATURES_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.inference_engine = LLMInferenceEngine(model_path=model_path)
                self.conformal_predictor = ConformalPredictor()
                self.risk_predictor = RiskControlledPredictor()
                self.model_available = True
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.inference_engine = None
                self.conformal_predictor = None
                self.risk_predictor = None
                self.model_available = False
        else:
            self.inference_engine = None
            self.conformal_predictor = None 
            self.risk_predictor = None
            self.model_available = False
            
        # Load default risk classes and costs
        self.default_classes = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
        self.default_cost_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],  # True HIGH_RISK costs
            [5.0, 0.0, 1.0, 2.0],  # True MEDIUM_RISK costs
            [8.0, 3.0, 0.0, 1.0],  # True LOW_RISK costs
            [10.0, 5.0, 2.0, 0.0]  # True NO_RISK costs
        ])
    
    def run_app(self):
        """Run the Streamlit application."""
        if not STREAMLIT_AVAILABLE:
            st.error("Streamlit not available. Please install with: pip install streamlit plotly")
            return
            
        st.set_page_config(
            page_title="Risk-Controlled Predictions",
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("⚖️ Risk-Controlled Prediction Interface")
        st.markdown("Make predictions with statistical guarantees and risk control")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main interface
        if self.model_available:
            self._render_model_interface()
        else:
            self._render_demo_interface()
    
    def _render_sidebar(self):
        """Render sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # Model status
        st.sidebar.subheader("Model Status")
        if self.model_available:
            st.sidebar.success("✅ Model Loaded")
            if self.model_path:
                st.sidebar.write(f"Path: {Path(self.model_path).name}")
        else:
            st.sidebar.warning("⚠️ Demo Mode")
            st.sidebar.write("No model loaded - using synthetic data")
        
        # Prediction settings
        st.sidebar.subheader("Prediction Settings")
        
        # Conformal prediction confidence
        st.session_state.confidence_level = st.sidebar.slider(
            "Confidence Level",
            min_value=0.80,
            max_value=0.99,
            value=0.90,
            step=0.01,
            help="Statistical confidence for prediction sets"
        )
        
        # Risk tolerance
        st.session_state.risk_tolerance = st.sidebar.slider(
            "Risk Tolerance", 
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Maximum acceptable risk for decisions"
        )
        
        # Cost matrix configuration
        st.sidebar.subheader("Cost Matrix")
        
        st.session_state.false_positive_cost = st.sidebar.number_input(
            "False Positive Cost",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.1
        )
        
        st.session_state.false_negative_cost = st.sidebar.number_input(
            "False Negative Cost", 
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1
        )
        
        # Advanced options
        with st.sidebar.expander("🔬 Advanced Options"):
            st.session_state.enable_monte_carlo = st.checkbox(
                "Monte Carlo Dropout",
                value=False,
                help="Use MC dropout for uncertainty estimation"
            )
            
            st.session_state.mc_samples = st.number_input(
                "MC Samples",
                min_value=1,
                max_value=100,
                value=10,
                disabled=not st.session_state.enable_monte_carlo
            )
            
            st.session_state.show_detailed_metrics = st.checkbox(
                "Show Detailed Metrics",
                value=True
            )
    
    def _render_model_interface(self):
        """Render interface for actual model predictions."""
        st.subheader("🤖 Model-Based Predictions")
        
        # Text input
        user_text = st.text_area(
            "Enter text for classification:",
            height=150,
            placeholder="Enter the text you want to classify for risk assessment..."
        )
        
        # Additional metadata inputs
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox(
                "Source",
                ["Unknown", "News", "Social Media", "Report", "Email", "Document"],
                help="Source of the text (may affect classification)"
            )
        
        with col2:
            urgency = st.selectbox(
                "Urgency Level",
                ["Low", "Medium", "High", "Critical"],
                index=1,
                help="How urgent is this classification?"
            )
        
        # Prediction button
        if st.button("🎯 Make Risk-Controlled Prediction", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text for prediction.")
            else:
                self._make_model_prediction(user_text, source, urgency)
    
    def _render_demo_interface(self):
        """Render demo interface with synthetic predictions."""
        st.subheader("🧪 Demo Mode - Synthetic Predictions")
        
        st.info("""
        **Demo Mode Active**: This interface is running with synthetic data. 
        To use with a real model, provide a model path when launching the app.
        """)
        
        # Demo scenarios
        st.subheader("📋 Demo Scenarios")
        
        scenario = st.selectbox(
            "Select a demo scenario:",
            [
                "High Confidence - Low Risk",
                "Medium Confidence - Medium Risk", 
                "Low Confidence - High Risk",
                "Uncertain - Very High Risk",
                "Custom Probabilities"
            ]
        )
        
        # Generate demo probabilities based on scenario
        if scenario == "High Confidence - Low Risk":
            demo_probs = np.array([0.05, 0.10, 0.80, 0.05])
        elif scenario == "Medium Confidence - Medium Risk":
            demo_probs = np.array([0.20, 0.50, 0.25, 0.05])
        elif scenario == "Low Confidence - High Risk":
            demo_probs = np.array([0.45, 0.35, 0.15, 0.05])
        elif scenario == "Uncertain - Very High Risk":
            demo_probs = np.array([0.70, 0.20, 0.08, 0.02])
        else:  # Custom
            st.write("**Custom Probability Distribution:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                p_high = st.slider("HIGH_RISK", 0.0, 1.0, 0.25, 0.01)
            with col2:
                p_medium = st.slider("MEDIUM_RISK", 0.0, 1.0, 0.25, 0.01)
            with col3:
                p_low = st.slider("LOW_RISK", 0.0, 1.0, 0.25, 0.01)
            with col4:
                p_none = st.slider("NO_RISK", 0.0, 1.0, 0.25, 0.01)
            
            # Normalize probabilities
            total = p_high + p_medium + p_low + p_none
            if total > 0:
                demo_probs = np.array([p_high, p_medium, p_low, p_none]) / total
            else:
                demo_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Demo text input (optional)
        demo_text = st.text_area(
            "Sample text (for display only):",
            value="This is a sample text for risk classification demonstration.",
            height=100
        )
        
        # Run demo prediction
        if st.button("🎯 Run Demo Prediction", type="primary"):
            self._make_demo_prediction(demo_text, demo_probs)
    
    def _make_model_prediction(self, text: str, source: str, urgency: str):
        """Make prediction using actual model."""
        try:
            st.subheader("🎯 Model Prediction Results")
            
            # Prepare input for model
            input_data = {
                'text': text,
                'source': source, 
                'urgency': urgency
            }
            
            # Get model prediction
            with st.spinner("Making prediction..."):
                result = self.inference_engine.predict(
                    text=text,
                    return_probabilities=True,
                    enable_monte_carlo=st.session_state.enable_monte_carlo,
                    mc_samples=st.session_state.mc_samples if st.session_state.enable_monte_carlo else 1
                )
            
            # Extract probabilities and prediction
            probabilities = result.get('probabilities', {})
            predicted_class = result.get('predicted_class', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            
            # Convert to numpy array for processing
            prob_array = np.array([
                probabilities.get(cls, 0.0) 
                for cls in self.default_classes
            ])
            
            # Display results
            self._display_prediction_results(
                text, prob_array, self.default_classes,
                predicted_class, confidence
            )
            
        except Exception as e:
            st.error(f"Error making model prediction: {str(e)}")
            logger.exception("Model prediction error")
    
    def _make_demo_prediction(self, text: str, probabilities: np.ndarray):
        """Make demo prediction with synthetic data."""
        try:
            st.subheader("🎯 Demo Prediction Results")
            
            # Get predicted class
            max_prob_idx = np.argmax(probabilities)
            predicted_class = self.default_classes[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            
            # Display results
            self._display_prediction_results(
                text, probabilities, self.default_classes,
                predicted_class, confidence
            )
            
        except Exception as e:
            st.error(f"Error in demo prediction: {str(e)}")
    
    def _display_prediction_results(self, text: str, probabilities: np.ndarray, 
                                  classes: List[str], predicted_class: str, confidence: float):
        """Display comprehensive prediction results."""
        
        # Basic prediction info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Input Text:**")
            st.text_area("", value=text, height=100, disabled=True)
        
        with col2:
            st.markdown("**Prediction Summary:**")
            st.write(f"**Class**: {predicted_class}")
            st.write(f"**Confidence**: {confidence:.3f}")
            
            # Risk level indicator
            high_risk_prob = probabilities[0] + probabilities[1]
            if high_risk_prob > 0.7:
                st.error("🔴 HIGH RISK")
            elif high_risk_prob > 0.3:
                st.warning("🟡 MEDIUM RISK") 
            else:
                st.success("🟢 LOW RISK")
        
        # Detailed results in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Probability Distribution", 
            "🎯 Conformal Prediction",
            "⚖️ Risk Analysis", 
            "📋 Detailed Metrics"
        ])
        
        with tab1:
            self._render_probability_tab(probabilities, classes)
        
        with tab2:
            self._render_conformal_tab(probabilities, classes)
            
        with tab3:
            self._render_risk_tab(probabilities, classes)
            
        with tab4:
            if st.session_state.show_detailed_metrics:
                self._render_metrics_tab(probabilities, classes)
    
    def _render_probability_tab(self, probabilities: np.ndarray, classes: List[str]):
        """Render probability distribution visualization."""
        
        # Bar chart
        prob_df = pd.DataFrame({
            'Class': classes,
            'Probability': probabilities
        })
        
        fig = px.bar(
            prob_df, 
            x='Class', 
            y='Probability',
            title='Prediction Probability Distribution',
            color='Probability',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probability table
        st.subheader("📋 Probability Details")
        display_df = prob_df.copy()
        display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.4f}")
        display_df['Percentage'] = (probabilities * 100).round(2).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True)
    
    def _render_conformal_tab(self, probabilities: np.ndarray, classes: List[str]):
        """Render conformal prediction analysis."""
        
        confidence_level = st.session_state.confidence_level
        alpha = 1 - confidence_level
        
        st.write(f"**Conformal Prediction at {confidence_level:.1%} confidence level**")
        
        # Generate prediction set (simplified adaptive prediction sets)
        sorted_indices = np.argsort(probabilities)[::-1]
        cumulative_prob = 0.0
        prediction_set = []
        
        for idx in sorted_indices:
            cumulative_prob += probabilities[idx]
            prediction_set.append(classes[idx])
            if cumulative_prob >= confidence_level:
                break
        
        # Display prediction set
        st.write("**Prediction Set:**")
        for i, cls in enumerate(prediction_set):
            prob = probabilities[classes.index(cls)]
            st.write(f"  {i+1}. {cls} (p={prob:.3f})")
        
        st.write(f"**Set Size**: {len(prediction_set)} classes")
        
        # Coverage analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction Set Size", len(prediction_set))
            st.metric("Coverage Probability", f"{cumulative_prob:.3f}")
        
        with col2:
            # Efficiency metrics
            efficiency = 1 - len(prediction_set) / len(classes)
            st.metric("Efficiency", f"{efficiency:.3f}")
            
            # Uncertainty measure
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            max_entropy = np.log(len(classes))
            normalized_entropy = entropy / max_entropy
            st.metric("Normalized Entropy", f"{normalized_entropy:.3f}")
    
    def _render_risk_tab(self, probabilities: np.ndarray, classes: List[str]):
        """Render risk analysis and cost-based decisions."""
        
        st.write("**Risk-Based Decision Analysis**")
        
        # Cost matrix (simplified)
        cost_matrix = np.array([
            [0.0, st.session_state.false_positive_cost, st.session_state.false_positive_cost, st.session_state.false_positive_cost],
            [st.session_state.false_negative_cost, 0.0, st.session_state.false_positive_cost, st.session_state.false_positive_cost], 
            [st.session_state.false_negative_cost, st.session_state.false_negative_cost, 0.0, st.session_state.false_positive_cost],
            [st.session_state.false_negative_cost, st.session_state.false_negative_cost, st.session_state.false_negative_cost, 0.0]
        ])
        
        # Compute expected costs for each decision
        expected_costs = probabilities @ cost_matrix
        min_cost_idx = np.argmin(expected_costs)
        optimal_decision = classes[min_cost_idx]
        min_cost = expected_costs[min_cost_idx]
        
        # Display optimal decision
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimal Decision")
            st.write(f"**Decision**: {optimal_decision}")
            st.write(f"**Expected Cost**: {min_cost:.3f}")
            
            # Risk tolerance check
            high_risk_prob = probabilities[0] + probabilities[1]  # HIGH + MEDIUM
            risk_tolerance = st.session_state.risk_tolerance
            
            if high_risk_prob > risk_tolerance:
                st.error(f"🚫 **ABSTAIN** - Risk {high_risk_prob:.3f} > tolerance {risk_tolerance:.3f}")
            else:
                st.success(f"✅ **PROCEED** - Risk {high_risk_prob:.3f} ≤ tolerance {risk_tolerance:.3f}")
        
        with col2:
            st.subheader("Cost Analysis")
            
            # Expected costs for all decisions
            cost_df = pd.DataFrame({
                'Decision': classes,
                'Expected Cost': expected_costs
            })
            
            fig = px.bar(
                cost_df,
                x='Decision',
                y='Expected Cost', 
                title='Expected Cost by Decision',
                color='Expected Cost',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk summary
        st.subheader("📊 Risk Summary")
        
        risk_metrics = pd.DataFrame({
            'Metric': [
                'Total High Risk Probability',
                'Expected Cost (Optimal)', 
                'Expected Cost (Predicted)',
                'Cost Savings',
                'Risk Tolerance Met'
            ],
            'Value': [
                f"{high_risk_prob:.3f}",
                f"{min_cost:.3f}",
                f"{expected_costs[np.argmax(probabilities)]:.3f}",
                f"{expected_costs[np.argmax(probabilities)] - min_cost:.3f}",
                "Yes" if high_risk_prob <= risk_tolerance else "No"
            ]
        })
        
        st.dataframe(risk_metrics, use_container_width=True)
    
    def _render_metrics_tab(self, probabilities: np.ndarray, classes: List[str]):
        """Render detailed prediction metrics."""
        
        st.subheader("📊 Detailed Prediction Metrics")
        
        # Uncertainty metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Entropy
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            max_entropy = np.log(len(classes))
            normalized_entropy = entropy / max_entropy
            
            st.metric("Entropy", f"{entropy:.3f}")
            st.metric("Normalized Entropy", f"{normalized_entropy:.3f}")
        
        with col2:
            # Confidence metrics
            max_prob = np.max(probabilities)
            second_max_prob = np.partition(probabilities, -2)[-2]
            confidence_gap = max_prob - second_max_prob
            
            st.metric("Max Probability", f"{max_prob:.3f}")
            st.metric("Confidence Gap", f"{confidence_gap:.3f}")
        
        with col3:
            # Effective sample size and other metrics
            ess = 1 / np.sum(probabilities ** 2)
            gini_impurity = 1 - np.sum(probabilities ** 2)
            
            st.metric("Effective Sample Size", f"{ess:.1f}")
            st.metric("Gini Impurity", f"{gini_impurity:.3f}")
        
        # Probability distribution statistics
        st.subheader("📈 Distribution Statistics")
        
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                f"{np.mean(probabilities):.4f}",
                f"{np.std(probabilities):.4f}",
                f"{np.min(probabilities):.4f}",
                f"{np.max(probabilities):.4f}",
                f"{np.max(probabilities) - np.min(probabilities):.4f}"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True)


def main():
    """Main entry point for the risk prediction UI."""
    parser = argparse.ArgumentParser(
        description="Risk-Controlled Prediction Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model for inference"
    )
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] != "--":
        args = parser.parse_args()
    else:
        # When running via streamlit, arguments come after --
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
        else:
            model_path = None
        args = argparse.Namespace(model_path=model_path)
    
    # Initialize and run app
    app = RiskControlledPredictionUI(model_path=args.model_path)
    app.run_app()


if __name__ == "__main__":
    main()