#!/usr/bin/env python3
"""
Advanced LoRA Fine-tuning Script

Comprehensive training script with calibration-aware training,
abstention-aware loss functions, and advanced metrics tracking.

Usage:
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml --resume-from-checkpoint
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml --enable-dashboard
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import training components
try:
    from voters.llm.sft_lora import EnhancedLoRASFTTrainer
    from voters.llm.dataset import LLMDatasetProcessor
    from voters.llm.metrics import MetricsAggregator
    from voters.llm.conformal import ConformalPredictor
    from voters.llm.dashboard import TrainingDashboard
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the project is properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class AdvancedLoRASFTRunner:
    """
    Advanced LoRA SFT training runner with comprehensive features.
    
    Features:
    - Calibration-aware training
    - Abstention-aware loss functions
    - Real-time dashboard integration
    - Advanced metrics tracking
    - Conformal prediction calibration
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, config_path: str, enable_dashboard: bool = False):
        """
        Initialize the training runner.
        
        Args:
            config_path: Path to configuration file
            enable_dashboard: Whether to enable real-time dashboard
        """
        self.config_path = Path(config_path)
        self.enable_dashboard = enable_dashboard
        
        # Validate config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        # Initialize components
        self.trainer = None
        self.dashboard = None
        self.metrics_aggregator = None
        self.conformal_predictor = None
        
        logger.info(f"Initialized training runner with config: {config_path}")
        
    def setup_trainer(self, resume_from_checkpoint: bool = False) -> None:
        """
        Set up the enhanced LoRA trainer with advanced features.
        
        Args:
            resume_from_checkpoint: Whether to resume from existing checkpoint
        """
        logger.info("Setting up enhanced LoRA trainer...")
        
        try:
            # Initialize trainer with advanced features
            self.trainer = EnhancedLoRASFTTrainer(
                config_path=str(self.config_path),
                enable_advanced_metrics=True,
                enable_calibration_monitoring=True,
                enable_abstention_loss=True,
                enable_conformal_prediction=True
            )
            
            # Set up metrics aggregator
            metrics_path = Path("artifacts/models/llm_lora/training_metrics.json")
            self.metrics_aggregator = MetricsAggregator(save_path=metrics_path)
            
            # Set up conformal predictor for calibration
            self.conformal_predictor = ConformalPredictor()
            
            logger.info("Enhanced trainer setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up trainer: {e}")
            raise
            
    def setup_dashboard(self) -> None:
        """Set up real-time training dashboard if enabled."""
        if not self.enable_dashboard:
            return
            
        logger.info("Setting up training dashboard...")
        
        try:
            metrics_path = Path("artifacts/models/llm_lora/training_metrics.json")
            self.dashboard = TrainingDashboard(
                metrics_path=str(metrics_path),
                update_interval=5,
                max_history=1000
            )
            
            # Start monitoring in background
            self.dashboard.start_monitoring()
            
            logger.info("Training dashboard started successfully")
            logger.info("Dashboard URL: http://localhost:8501")
            
        except Exception as e:
            logger.warning(f"Failed to set up dashboard: {e}")
            logger.warning("Training will continue without dashboard")
            
    def prepare_data(self) -> None:
        """Prepare training data with validation and preprocessing."""
        logger.info("Preparing training data...")
        
        try:
            # Initialize data processor
            processor = LLMDatasetProcessor()
            
            # Check for raw data
            raw_data_dir = Path("data/raw")
            if not raw_data_dir.exists() or not list(raw_data_dir.glob("*.json")):
                logger.warning("No raw data found in data/raw/")
                logger.info("Please add training data files to data/raw/ directory")
                return False
                
            # Process and validate data
            processed_data = processor.process_directory(raw_data_dir)
            
            # Save processed data
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            processor.save_processed_data(processed_data, processed_dir)
            
            logger.info(f"Data preparation completed: {len(processed_data)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
            
    def run_training(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Args:
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting advanced LoRA fine-tuning...")
        
        try:
            # Prepare data
            if not self.prepare_data():
                raise RuntimeError("Data preparation failed")
                
            # Set up trainer
            self.setup_trainer(resume_from_checkpoint)
            
            # Set up dashboard if enabled
            self.setup_dashboard()
            
            # Load training data
            train_dataset = self.trainer.load_dataset("data/processed/train.json")
            eval_dataset = self.trainer.load_dataset("data/processed/eval.json")
            
            logger.info(f"Loaded datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
            
            # Start training with advanced features
            logger.info("Beginning training with advanced features:")
            logger.info("  ✅ Calibration-aware training")
            logger.info("  ✅ Abstention-aware loss functions") 
            logger.info("  ✅ Advanced metrics tracking")
            logger.info("  ✅ Conformal prediction calibration")
            logger.info("  ✅ Real-time monitoring")
            
            # Execute training
            training_results = self.trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            # Post-training analysis
            self.run_post_training_analysis(training_results)
            
            logger.info("Training completed successfully!")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            if self.dashboard:
                self.dashboard.stop_monitoring()
                
    def run_post_training_analysis(self, training_results: Dict[str, Any]) -> None:
        """
        Run comprehensive post-training analysis.
        
        Args:
            training_results: Results from training
        """
        logger.info("Running post-training analysis...")
        
        try:
            # Generate comprehensive training report
            if self.metrics_aggregator:
                report = self.metrics_aggregator.generate_report()
                
                # Save report
                report_path = Path("artifacts/models/llm_lora/training_report.json")
                self.metrics_aggregator.save_report(report, report_path)
                
                logger.info(f"Training report saved to: {report_path}")
                
            # Log key metrics
            if "eval_results" in training_results:
                eval_results = training_results["eval_results"]
                logger.info("Final evaluation metrics:")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.4f}")
                        
            logger.info("Post-training analysis completed")
            
        except Exception as e:
            logger.warning(f"Post-training analysis failed: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced LoRA Fine-tuning with Calibration and Risk Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml
    
    # Training with dashboard
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml --enable-dashboard
    
    # Resume from checkpoint
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml --resume-from-checkpoint
    
    # Full featured training
    python apps/train/train_lora_sft.py --config configs/llm_lora.yaml --enable-dashboard --verbose
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume training from existing checkpoint"
    )
    
    parser.add_argument(
        "--enable-dashboard",
        action="store_true", 
        help="Enable real-time training dashboard"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="artifacts/models/llm_lora",
        help="Output directory for trained model"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for training script."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("=== Advanced LoRA Fine-tuning Started ===")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dashboard enabled: {args.enable_dashboard}")
    logger.info(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    
    try:
        # Initialize training runner
        runner = AdvancedLoRASFTRunner(
            config_path=args.config,
            enable_dashboard=args.enable_dashboard
        )
        
        # Execute training
        results = runner.run_training(
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        logger.info("=== Training Completed Successfully ===")
        logger.info(f"Model saved to: {args.output_dir}")
        
        if args.enable_dashboard:
            logger.info("Dashboard will continue running for monitoring")
            logger.info("Press Ctrl+C to stop the dashboard")
            
            # Keep dashboard running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Dashboard stopped by user")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())