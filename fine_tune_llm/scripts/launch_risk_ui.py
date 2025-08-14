#!/usr/bin/env python3
"""
Launch the Risk-Controlled Prediction Interface

A simple launcher for the risk-controlled prediction Streamlit app.

Usage:
    python scripts/launch_risk_ui.py [model_path] [--port PORT]

Example:
    python scripts/launch_risk_ui.py artifacts/models/llm_lora/final_model --port 8502
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Launch Risk-Controlled Prediction Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch in demo mode
    python scripts/launch_risk_ui.py
    
    # Launch with a trained model
    python scripts/launch_risk_ui.py artifacts/models/llm_lora/final_model
    
    # Custom port
    python scripts/launch_risk_ui.py --port 8502
    
    # Model with custom port
    python scripts/launch_risk_ui.py artifacts/models/llm_lora/final_model --port 8502
"""
    )
    
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to trained model (optional - runs in demo mode without)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8502,
        help="Port to run the interface on (default: 8502)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost", 
        help="Host to run on (default: localhost)"
    )
    
    parser.add_argument(
        "--auto-open", "-o",
        action="store_true",
        help="Automatically open browser"
    )
    
    args = parser.parse_args()
    
    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Please install with:")
        print("   pip install streamlit plotly pandas")
        sys.exit(1)
    
    # Validate model path if provided
    if args.model_path and not Path(args.model_path).exists():
        print(f"⚠️  Model path not found: {args.model_path}")
        response = input("Continue in demo mode? (y/n): ")
        if not response.lower().startswith('y'):
            sys.exit(1)
        args.model_path = None
    
    # Print startup info
    print("🚀 Launching Risk-Controlled Prediction Interface...")
    if args.model_path:
        print(f"   🤖 Model: {args.model_path}")
    else:
        print("   🧪 Mode: Demo (synthetic data)")
    print(f"   🌐 URL: http://{args.host}:{args.port}")
    print()
    
    print("Interface features:")
    print("   ✅ Interactive risk-controlled predictions")
    print("   ✅ Conformal prediction with statistical guarantees")
    print("   ✅ Cost-based decision analysis")
    print("   ✅ Real-time risk assessment")
    print("   ✅ Customizable risk tolerance and cost matrices")
    print("   ✅ Comprehensive uncertainty quantification")
    print()
    
    # Build streamlit command
    script_dir = Path(__file__).parent
    ui_script = script_dir / "risk_prediction_ui.py"
    
    cmd = [
        "streamlit", "run", str(ui_script),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]
    
    if not args.auto_open:
        cmd.extend(["--server.headless", "true"])
    
    # Add model path if provided
    if args.model_path:
        cmd.extend(["--", str(args.model_path)])
    
    # Launch the interface
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Risk prediction interface stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()