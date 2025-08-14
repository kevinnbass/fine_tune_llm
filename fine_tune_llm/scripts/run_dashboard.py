#!/usr/bin/env python3
"""
Launch the LLM training dashboard.

This script starts a Streamlit-based web dashboard for monitoring LLM fine-tuning progress.
The dashboard displays real-time metrics including advanced calibration metrics,
conformal prediction statistics, and abstention-aware training progress.

Usage:
    python scripts/run_dashboard.py [metrics_path] [port]
    
Example:
    python scripts/run_dashboard.py artifacts/models/llm_lora/training_metrics.json 8501
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voters.llm.dashboard import TrainingDashboard, DASHBOARD_AVAILABLE


def main():
    parser = argparse.ArgumentParser(
        description="Launch LLM Training Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch with default settings
    python scripts/run_dashboard.py
    
    # Specify metrics file
    python scripts/run_dashboard.py --metrics artifacts/training_metrics.json
    
    # Custom port
    python scripts/run_dashboard.py --port 8502
    
    # Auto-start browser
    python scripts/run_dashboard.py --auto-open
"""
    )
    
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        default="artifacts/models/llm_lora/training_metrics.json",
        help="Path to training metrics JSON file"
    )
    
    parser.add_argument(
        "--port", "-p", 
        type=int,
        default=8501,
        help="Port to run dashboard on (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run dashboard on (default: localhost)"
    )
    
    parser.add_argument(
        "--auto-open", "-o",
        action="store_true",
        help="Automatically open browser"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true", 
        help="Run with demo data (for testing)"
    )
    
    args = parser.parse_args()
    
    if not DASHBOARD_AVAILABLE:
        print("❌ Dashboard dependencies not available!")
        print("Please install required packages:")
        print("  pip install streamlit plotly pandas")
        sys.exit(1)
    
    # Verify metrics file exists or create demo data
    metrics_path = Path(args.metrics)
    
    if args.demo:
        print("🧪 Creating demo data...")
        create_demo_data(metrics_path)
    elif not metrics_path.exists():
        print(f"⚠️  Metrics file not found: {metrics_path}")
        print("   Either start training or use --demo flag")
        response = input("Create demo data? (y/n): ")
        if response.lower().startswith('y'):
            create_demo_data(metrics_path)
        else:
            print("Dashboard will show empty state until training starts")
    
    print(f"🚀 Starting LLM Training Dashboard...")
    print(f"   📊 Metrics file: {metrics_path}")
    print(f"   🌐 URL: http://{args.host}:{args.port}")
    print(f"   🔄 Auto-refresh: Enabled")
    print()
    print("Dashboard features:")
    print("   ✅ Real-time training progress")
    print("   ✅ Advanced calibration metrics (ECE, MCE, Brier Score)")
    print("   ✅ Conformal prediction statistics") 
    print("   ✅ Abstention analysis")
    print("   ✅ Risk-aware metrics")
    print("   ✅ Risk-controlled prediction interface")
    print("   ✅ Interactive plots and data export")
    print()
    
    # Set up streamlit command
    streamlit_cmd = [
        "streamlit", "run",
        str(Path(__file__).parent.parent / "voters" / "llm" / "dashboard.py"),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--", str(metrics_path)
    ]
    
    if args.auto_open:
        streamlit_cmd.extend(["--server.headless", "false"])
    
    # Launch dashboard
    try:
        import subprocess
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)


def create_demo_data(metrics_path: Path):
    """Create demo training metrics for testing the dashboard."""
    import json
    import time
    import numpy as np
    
    # Ensure parent directory exists
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate realistic training progression
    demo_metrics = []
    
    base_time = time.time() - 3600  # Start 1 hour ago
    
    for epoch in range(5):
        for step in range(20):
            # Simulate realistic training progression
            progress = (epoch * 20 + step) / 100.0
            
            # Core metrics with realistic progression
            train_loss = 2.5 * np.exp(-progress * 2) + 0.3 + np.random.normal(0, 0.05)
            accuracy = 0.6 + 0.3 * (1 - np.exp(-progress * 3)) + np.random.normal(0, 0.02)
            accuracy = np.clip(accuracy, 0, 1)
            
            # Advanced metrics
            ece = 0.2 * np.exp(-progress * 1.5) + 0.02 + abs(np.random.normal(0, 0.01))
            mce = ece * 1.5 + abs(np.random.normal(0, 0.02))
            brier_score = 0.25 * np.exp(-progress * 1.2) + 0.05 + abs(np.random.normal(0, 0.01))
            
            # Confidence metrics
            mean_confidence = 0.5 + 0.4 * (1 - np.exp(-progress * 2)) + np.random.normal(0, 0.03)
            mean_confidence = np.clip(mean_confidence, 0, 1)
            
            # Abstention metrics
            abstention_rate = 0.3 * np.exp(-progress * 1.8) + 0.05 + abs(np.random.normal(0, 0.02))
            effective_accuracy = accuracy * (1 - abstention_rate)
            
            # Conformal prediction metrics
            conformal_coverage = 0.85 + 0.1 * (1 - np.exp(-progress * 2)) + np.random.normal(0, 0.02)
            conformal_coverage = np.clip(conformal_coverage, 0, 1)
            conformal_avg_set_size = 2.5 * np.exp(-progress * 1.5) + 1.1 + abs(np.random.normal(0, 0.1))
            
            # Risk metrics
            average_risk = 1.0 * np.exp(-progress * 1.8) + 0.1 + abs(np.random.normal(0, 0.05))
            risk_reduction = 1 - np.exp(-progress * 2) + np.random.normal(0, 0.03)
            risk_reduction = np.clip(risk_reduction, 0, 1)
            
            metrics = {
                "timestamp": base_time + epoch * 1800 + step * 90,  # 30min per epoch, 1.5min per step
                "epoch": epoch,
                "step": step,
                "train_loss": float(train_loss),
                "accuracy": float(accuracy),
                "f1_macro": float(accuracy * 0.95),  # F1 typically slightly lower
                "precision_macro": float(accuracy * 0.98),
                "recall_macro": float(accuracy * 0.93),
                "learning_rate": 2e-4 * (0.95 ** epoch),  # Decaying LR
                
                # Advanced metrics
                "ece": float(ece),
                "mce": float(mce),
                "brier_score": float(brier_score),
                "confidence_mean": float(mean_confidence),
                "confidence_accuracy_correlation": float(0.3 + 0.4 * progress),
                
                # Abstention metrics
                "abstention_rate": float(abstention_rate),
                "abstention_effective_accuracy": float(effective_accuracy),
                "abstention_avg_cost_per_sample": float(abstention_rate * 0.3 + (1-accuracy) * 1.0),
                
                # Conformal metrics
                "conformal_coverage": float(conformal_coverage),
                "conformal_avg_set_size": float(conformal_avg_set_size),
                
                # Risk metrics
                "risk_average_risk": float(average_risk),
                "risk_reduction": float(risk_reduction),
            }
            
            demo_metrics.append(metrics)
    
    # Save demo metrics
    with open(metrics_path, 'w') as f:
        json.dump(demo_metrics, f, indent=2)
    
    print(f"✅ Created demo data with {len(demo_metrics)} metrics entries")


if __name__ == "__main__":
    main()