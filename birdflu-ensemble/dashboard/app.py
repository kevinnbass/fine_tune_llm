"""Real-time monitoring dashboard using Streamlit."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.models import DatabaseManager, DatabaseOperations
from monitoring.metrics import MonitoringSystem
from cache.prediction_cache import PredictionCache

# Page config
st.set_page_config(
    page_title="BirdFlu Ensemble Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize dashboard components."""
    # Database
    db_url = st.secrets.get("database_url", "postgresql://user:pass@localhost/birdflu")
    db_manager = DatabaseManager(db_url)
    db_ops = DatabaseOperations(db_manager)
    
    # Monitoring
    monitoring = MonitoringSystem()
    
    # Cache
    cache = PredictionCache()
    
    return db_ops, monitoring, cache

db_ops, monitoring, cache = init_components()


def main():
    """Main dashboard application."""
    st.title("🦅 BirdFlu Ensemble Monitoring Dashboard")
    st.markdown("Real-time monitoring for the bird flu classification system")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["System Overview", "Model Performance", "Data Quality", "Error Analysis", 
         "Alerts & Incidents", "A/B Testing", "System Health"]
    )
    
    # Time range selector
    st.sidebar.subheader("Time Range")
    time_range = st.sidebar.selectbox(
        "Select time range",
        ["Last Hour", "Last 4 Hours", "Last 24 Hours", "Last 7 Days"],
        index=2
    )
    
    hours_map = {
        "Last Hour": 1,
        "Last 4 Hours": 4,
        "Last 24 Hours": 24,
        "Last 7 Days": 168
    }
    hours = hours_map[time_range]
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Page routing
    if page == "System Overview":
        show_system_overview(hours)
    elif page == "Model Performance":
        show_model_performance(hours)
    elif page == "Data Quality":
        show_data_quality(hours)
    elif page == "Error Analysis":
        show_error_analysis(hours)
    elif page == "Alerts & Incidents":
        show_alerts(hours)
    elif page == "A/B Testing":
        show_ab_testing(hours)
    elif page == "System Health":
        show_system_health(hours)


def show_system_overview(hours: int):
    """Show system overview page."""
    st.header("System Overview")
    
    # Get metrics
    metrics = db_ops.get_performance_metrics(hours)
    monitoring_stats = monitoring.get_metrics_summary(window_size=1000)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Predictions",
            metrics.get('total_predictions', 0),
            delta=None
        )
    
    with col2:
        abstention_rate = metrics.get('abstention_rate', 0)
        st.metric(
            "Abstention Rate",
            f"{abstention_rate:.1%}",
            delta=f"{abstention_rate - 0.15:.1%}" if abstention_rate else None,
            delta_color="inverse"
        )
    
    with col3:
        accuracy = metrics.get('accuracy')
        st.metric(
            "Accuracy",
            f"{accuracy:.1%}" if accuracy else "N/A",
            delta=None
        )
    
    with col4:
        avg_latency = metrics.get('avg_latency_ms', 0)
        st.metric(
            "Avg Latency",
            f"{avg_latency:.0f}ms",
            delta=f"{avg_latency - 200:.0f}ms" if avg_latency else None,
            delta_color="inverse"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Volume Over Time")
        
        # Generate sample time series data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        time_points = pd.date_range(start_time, end_time, freq='H')
        
        # Simulate prediction counts
        np.random.seed(42)
        pred_counts = np.random.poisson(50, len(time_points))
        
        df_volume = pd.DataFrame({
            'timestamp': time_points,
            'predictions': pred_counts
        })
        
        fig = px.line(df_volume, x='timestamp', y='predictions',
                     title='Predictions per Hour')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Decision Distribution")
        
        # Sample decision distribution
        decisions = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK', 'ABSTAIN']
        counts = [120, 300, 450, 800, 150]  # Sample data
        
        fig = px.pie(values=counts, names=decisions,
                    title='Decision Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance breakdown
    st.subheader("Voter Performance Breakdown")
    
    voter_data = {
        'Voter': ['Regex DSL', 'TF-IDF LR', 'TF-IDF SVM', 'Label Model', 'LLM LoRA'],
        'Accuracy': [0.85, 0.82, 0.81, 0.79, 0.88],
        'Avg Latency (ms)': [5, 15, 12, 25, 450],
        'Cost (cents)': [0.0001, 0.001, 0.001, 0.002, 0.05],
        'Usage Rate': [0.95, 0.85, 0.82, 0.78, 0.15]
    }
    
    df_voters = pd.DataFrame(voter_data)
    st.dataframe(df_voters, use_container_width=True)


def show_model_performance(hours: int):
    """Show model performance page."""
    st.header("Model Performance Analysis")
    
    # Model selector
    model_options = ["All Models", "Regex DSL", "TF-IDF LR", "TF-IDF SVM", "Label Model", "LLM LoRA", "Ensemble"]
    selected_model = st.selectbox("Select Model", model_options)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Classification Metrics")
        
        # Sample metrics
        metrics_data = {
            'Metric': ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'Precision', 'Recall'],
            'Value': [0.876, 0.871, 0.845, 0.882, 0.860],
            'Target': [0.85, 0.85, 0.80, 0.85, 0.80]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics['Status'] = df_metrics.apply(
            lambda row: '✅' if row['Value'] >= row['Target'] else '❌', axis=1
        )
        
        st.dataframe(df_metrics, use_container_width=True)
    
    with col2:
        st.subheader("Operational Metrics")
        
        op_metrics = {
            'Metric': ['Abstention Rate', 'Cache Hit Rate', 'Error Rate', 'P95 Latency'],
            'Value': ['12.3%', '67.8%', '2.1%', '185ms'],
            'Target': ['<15%', '>60%', '<5%', '<200ms'],
            'Status': ['✅', '✅', '✅', '✅']
        }
        
        st.dataframe(pd.DataFrame(op_metrics), use_container_width=True)
    
    with col3:
        st.subheader("Cost Analysis")
        
        cost_data = {
            'Metric': ['Avg Cost per Prediction', 'LLM Call Rate', 'Daily Cost', 'Monthly Projection'],
            'Value': ['$0.0089', '8.2%', '$125.67', '$3,770'],
            'Target': ['<$0.01', '<10%', '<$200', '<$6,000'],
            'Status': ['✅', '✅', '✅', '✅']
        }
        
        st.dataframe(pd.DataFrame(cost_data), use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    # Sample confusion matrix
    labels = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
    conf_matrix = np.array([
        [450, 20, 5, 2],
        [15, 380, 25, 8],
        [3, 18, 420, 12],
        [1, 5, 15, 485]
    ])
    
    fig = px.imshow(conf_matrix, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels, y=labels,
                    color_continuous_scale='Blues',
                    text_auto=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance over time
    st.subheader("Performance Trends")
    
    # Generate sample time series
    dates = pd.date_range(datetime.now() - timedelta(days=7), datetime.now(), freq='H')
    np.random.seed(42)
    accuracy = 0.85 + 0.05 * np.random.randn(len(dates)).cumsum() * 0.01
    f1_score = 0.82 + 0.05 * np.random.randn(len(dates)).cumsum() * 0.01
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines', name='Accuracy'))
    fig.add_trace(go.Scatter(x=dates, y=f1_score, mode='lines', name='F1 Score'))
    fig.update_layout(title="Performance Over Time", yaxis_title="Score")
    
    st.plotly_chart(fig, use_container_width=True)


def show_data_quality(hours: int):
    """Show data quality page."""
    st.header("Data Quality & Drift Detection")
    
    # Drift alerts
    drift_alerts = db_ops.get_drift_alerts(hours)
    
    if drift_alerts:
        st.subheader("🚨 Active Drift Alerts")
        
        for alert in drift_alerts:
            severity_color = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }
            
            st.warning(
                f"{severity_color.get(alert['severity'], '⚪')} "
                f"**{alert['type'].title()} Drift** in {alert['feature']} "
                f"(Score: {alert['score']:.3f}, {alert['detected_at'].strftime('%H:%M')})"
            )
    else:
        st.success("✅ No drift alerts in the selected time range")
    
    # Drift monitoring charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Drift Scores")
        
        # Sample drift data
        features = ['text_length', 'word_count', 'caps_ratio', 'special_chars', 'embeddings']
        drift_scores = [0.02, 0.15, 0.08, 0.34, 0.12]
        threshold = 0.05
        
        colors = ['red' if score > threshold else 'green' for score in drift_scores]
        
        fig = go.Figure(data=[
            go.Bar(x=features, y=drift_scores, marker_color=colors)
        ])
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                     annotation_text="Drift Threshold")
        fig.update_layout(title="Current Drift Scores", yaxis_title="Drift Score")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Data Distribution Comparison")
        
        # Sample distribution comparison
        np.random.seed(42)
        reference_data = np.random.normal(100, 15, 1000)
        current_data = np.random.normal(105, 18, 300)  # Slightly drifted
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=reference_data, name='Reference', opacity=0.7))
        fig.add_trace(go.Histogram(x=current_data, name='Current', opacity=0.7))
        fig.update_layout(title="Text Length Distribution", barmode='overlay')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Input validation stats
    st.subheader("Input Validation Statistics")
    
    validation_data = {
        'Validation Check': [
            'Text Length', 'Character Encoding', 'Special Characters',
            'Binary Content', 'Suspicious Patterns'
        ],
        'Pass Rate': [98.7, 99.2, 96.8, 99.9, 94.5],
        'Failure Count': [23, 15, 58, 2, 102]
    }
    
    df_validation = pd.DataFrame(validation_data)
    
    fig = px.bar(df_validation, x='Validation Check', y='Pass Rate',
                title='Input Validation Pass Rates (%)')
    fig.update_yaxis(range=[90, 100])
    
    st.plotly_chart(fig, use_container_width=True)


def show_error_analysis(hours: int):
    """Show error analysis page."""
    st.header("Error Analysis & Debugging")
    
    # Error summary
    error_analysis = db_ops.get_error_analysis(hours)
    
    if error_analysis:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", error_analysis['total_feedback'])
        
        with col2:
            st.metric("Error Count", error_analysis['error_count'])
        
        with col3:
            error_rate = error_analysis['error_rate']
            st.metric("Error Rate", f"{error_rate:.1%}")
    
    # Error breakdown
    st.subheader("Error Types")
    
    if error_analysis and error_analysis.get('errors_by_type'):
        errors = error_analysis['errors_by_type']
        
        fig = px.pie(
            values=list(errors.values()),
            names=list(errors.keys()),
            title='Error Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No error data available for the selected time range")
    
    # Recent errors table
    st.subheader("Recent Errors")
    
    # Sample error data
    error_data = {
        'Timestamp': [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=5),
            datetime.now() - timedelta(hours=8)
        ],
        'Error Type': ['Misclassification', 'False Positive', 'Abstention'],
        'Predicted': ['MEDIUM_RISK', 'HIGH_RISK', 'ABSTAIN'],
        'Actual': ['HIGH_RISK', 'NO_RISK', 'LOW_RISK'],
        'Confidence': [0.72, 0.85, 0.45],
        'Text Preview': [
            'Avian flu detected in local farms...',
            'Weather forecast for tomorrow shows...',
            'Unclear text with mixed content...'
        ]
    }
    
    df_errors = pd.DataFrame(error_data)
    st.dataframe(df_errors, use_container_width=True)


def show_alerts(hours: int):
    """Show alerts and incidents page."""
    st.header("Alerts & Incidents")
    
    # Alert status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Critical Alerts", 0, delta=None)
    
    with col2:
        st.metric("Warning Alerts", 2, delta="+1")
    
    with col3:
        st.metric("System Status", "Healthy", delta=None)
    
    # Active alerts
    st.subheader("Active Alerts")
    
    alerts = [
        {
            'Severity': 'Warning',
            'Type': 'Performance',
            'Message': 'LLM latency above threshold (520ms)',
            'Time': datetime.now() - timedelta(minutes=15),
            'Status': 'Active'
        },
        {
            'Severity': 'Warning', 
            'Type': 'Data Quality',
            'Message': 'Text length distribution drift detected',
            'Time': datetime.now() - timedelta(hours=2),
            'Status': 'Investigating'
        }
    ]
    
    for alert in alerts:
        severity_color = {'Critical': '🔴', 'Warning': '🟠', 'Info': '🔵'}
        
        with st.expander(f"{severity_color.get(alert['Severity'], '⚪')} {alert['Message']}"):
            st.write(f"**Type:** {alert['Type']}")
            st.write(f"**Severity:** {alert['Severity']}")
            st.write(f"**Time:** {alert['Time'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Status:** {alert['Status']}")
    
    # Alert history
    st.subheader("Alert History")
    
    # Sample alert history chart
    dates = pd.date_range(datetime.now() - timedelta(days=7), datetime.now(), freq='D')
    critical = np.random.poisson(0.5, len(dates))
    warning = np.random.poisson(2, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=critical, name='Critical', marker_color='red'))
    fig.add_trace(go.Bar(x=dates, y=warning, name='Warning', marker_color='orange'))
    
    fig.update_layout(title="Alerts Over Time", barmode='stack')
    st.plotly_chart(fig, use_container_width=True)


def show_ab_testing(hours: int):
    """Show A/B testing page."""
    st.header("A/B Testing & Experiments")
    
    st.info("🧪 A/B testing framework coming soon!")
    
    # Placeholder for A/B testing metrics
    st.subheader("Active Experiments")
    st.write("No active experiments")
    
    st.subheader("Experiment History")
    st.write("No completed experiments")


def show_system_health(hours: int):
    """Show system health page."""
    st.header("System Health & Resources")
    
    # Resource metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "45%", delta="-5%")
    
    with col2:
        st.metric("Memory Usage", "6.2GB", delta="+0.3GB")
    
    with col3:
        st.metric("Disk Usage", "67%", delta="+2%")
    
    with col4:
        st.metric("Active Connections", 24, delta="+3")
    
    # Service status
    st.subheader("Service Status")
    
    services = {
        'API Server': '✅ Healthy',
        'Redis Cache': '✅ Healthy', 
        'PostgreSQL': '✅ Healthy',
        'MLflow': '✅ Healthy',
        'Prometheus': '🟡 Warning',
        'Grafana': '✅ Healthy'
    }
    
    cols = st.columns(3)
    for i, (service, status) in enumerate(services.items()):
        with cols[i % 3]:
            st.write(f"**{service}:** {status}")
    
    # Resource usage over time
    st.subheader("Resource Usage Trends")
    
    times = pd.date_range(datetime.now() - timedelta(hours=24), datetime.now(), freq='H')
    np.random.seed(42)
    cpu_usage = 40 + 10 * np.sin(np.arange(len(times)) * 0.2) + np.random.normal(0, 5, len(times))
    memory_usage = 5.5 + 0.5 * np.sin(np.arange(len(times)) * 0.15) + np.random.normal(0, 0.2, len(times))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=times, y=cpu_usage, name="CPU %"),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=times, y=memory_usage, name="Memory GB"),
        secondary_y=True
    )
    
    fig.update_layout(title="System Resources Over Time")
    fig.update_yaxis(title_text="CPU Usage (%)", secondary_y=False)
    fig.update_yaxis(title_text="Memory Usage (GB)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()