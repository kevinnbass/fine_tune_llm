"""
Unified training script for all LLM fine-tuning operations.

This script consolidates various training modes into a single entry point
with command-line arguments for different training configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fine_tune_llm.services import TrainingService, ModelService
from fine_tune_llm.config import ConfigManager
from fine_tune_llm.core.events import EventBus
from fine_tune_llm.core.dependency_injection import DIContainer
from fine_tune_llm.utils.logging import setup_logging, get_logger
from fine_tune_llm.core.exceptions import TrainingError, ConfigurationError

logger = get_logger(__name__)


class UnifiedTrainingCLI:
    """Unified command-line interface for all training operations."""
    
    def __init__(self):
        """Initialize the unified training CLI."""
        self.config_manager = None
        self.training_service = None
        self.model_service = None
        self.event_bus = None
        self.di_container = DIContainer()
    
    def setup_services(self, config_dir: Optional[Path] = None):
        """Set up services and dependencies."""
        # Set up logging
        setup_logging(log_level="INFO", enable_file_logging=True)
        
        # Initialize configuration manager
        config_dir = config_dir or Path("configs")
        self.config_manager = ConfigManager(config_dir=config_dir)
        
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Register services in DI container
        self.di_container.register_singleton(ConfigManager, self.config_manager)
        self.di_container.register_singleton(EventBus, self.event_bus)
        
        # Initialize services
        self.model_service = ModelService(
            config_manager=self.config_manager,
            event_bus=self.event_bus,
            di_container=self.di_container
        )
        
        self.training_service = TrainingService(
            config_manager=self.config_manager,
            event_bus=self.event_bus,
            di_container=self.di_container
        )
        
        # Initialize services
        self.model_service.initialize({})
        self.training_service.initialize({})
        
        # Start services
        self.model_service.start()
        self.training_service.start()
        
        logger.info("Services initialized successfully")
    
    def train_lora(self, args: argparse.Namespace) -> bool:
        """Run LoRA fine-tuning."""
        logger.info("Starting LoRA fine-tuning")
        
        try:
            # Prepare training configuration
            training_config = {
                'trainer_type': 'calibrated',
                'training_args': {
                    'output_dir': args.output_dir,
                    'num_train_epochs': args.epochs,
                    'per_device_train_batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'logging_steps': args.logging_steps,
                    'save_steps': args.save_steps,
                    'evaluation_strategy': 'steps' if args.eval_steps else 'no',
                    'eval_steps': args.eval_steps,
                    'save_total_limit': args.save_total_limit,
                    'load_best_model_at_end': True,
                    'metric_for_best_model': 'eval_loss',
                },
                'lora_config': {
                    'r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'target_modules': args.target_modules,
                    'lora_dropout': args.lora_dropout,
                },
                'enable_calibration': args.enable_calibration,
                'enable_conformal_prediction': args.enable_conformal_prediction
            }
            
            # Model configuration
            model_config = {
                'model_type': 'llm_lora',
                'model_name_or_path': args.model_name,
                'tokenizer_name': args.tokenizer_name or args.model_name,
                'use_auth_token': args.use_auth_token,
                'trust_remote_code': args.trust_remote_code
            }
            
            # Data configuration
            data_config = {
                'train_path': args.train_data,
                'eval_path': args.eval_data,
                'data_format': args.data_format,
                'max_length': args.max_length,
                'loader': {
                    'batch_size': args.batch_size,
                    'num_workers': args.num_workers
                }
            }
            
            # Start training
            job_id = self.training_service.start_training(
                training_config=training_config,
                model_config=model_config,
                data_config=data_config
            )
            
            logger.info(f"Training job started with ID: {job_id}")
            
            # Monitor training progress
            return self._monitor_training(job_id)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def train_high_stakes(self, args: argparse.Namespace) -> bool:
        """Run high-stakes training with enhanced monitoring."""
        logger.info("Starting high-stakes training with enhanced monitoring")
        
        try:
            # Enhanced configuration for high-stakes training
            training_config = {
                'trainer_type': 'calibrated',
                'training_args': {
                    'output_dir': args.output_dir,
                    'num_train_epochs': args.epochs,
                    'per_device_train_batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'logging_steps': max(10, args.logging_steps // 2),  # More frequent logging
                    'save_steps': max(50, args.save_steps // 2),  # More frequent saves
                    'evaluation_strategy': 'steps',
                    'eval_steps': max(100, args.eval_steps // 2),  # More frequent evaluation
                    'save_total_limit': args.save_total_limit * 2,  # Keep more checkpoints
                    'load_best_model_at_end': True,
                    'metric_for_best_model': 'eval_calibration_ece',  # Use calibration metric
                },
                'lora_config': {
                    'r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'target_modules': args.target_modules,
                    'lora_dropout': args.lora_dropout,
                },
                # Enhanced monitoring for high-stakes
                'enable_calibration': True,
                'enable_conformal_prediction': True,
                'enable_bias_detection': True,
                'enable_fairness_analysis': True,
                'risk_monitoring': True,
                'calibration_config': {
                    'enable_ece_monitoring': True,
                    'enable_mce_monitoring': True,
                    'calibration_adjustment': True,
                    'temperature_scaling': True
                }
            }
            
            # Model and data config same as LoRA
            model_config = {
                'model_type': 'llm_lora',
                'model_name_or_path': args.model_name,
                'tokenizer_name': args.tokenizer_name or args.model_name,
                'use_auth_token': args.use_auth_token,
                'trust_remote_code': args.trust_remote_code
            }
            
            data_config = {
                'train_path': args.train_data,
                'eval_path': args.eval_data,
                'data_format': args.data_format,
                'max_length': args.max_length,
                'loader': {
                    'batch_size': args.batch_size,
                    'num_workers': args.num_workers
                }
            }
            
            # Start high-stakes training
            job_id = self.training_service.start_training(
                training_config=training_config,
                model_config=model_config,
                data_config=data_config
            )
            
            logger.info(f"High-stakes training job started with ID: {job_id}")
            
            return self._monitor_training(job_id, high_stakes=True)
            
        except Exception as e:
            logger.error(f"High-stakes training failed: {e}")
            return False
    
    def tune_hyperparameters(self, args: argparse.Namespace) -> bool:
        """Run hyperparameter tuning."""
        logger.info("Starting hyperparameter tuning")
        
        try:
            # Define hyperparameter search space
            search_space = {
                'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                'lora_r': [8, 16, 32, 64],
                'lora_alpha': [16, 32, 64, 128],
                'batch_size': [4, 8, 16],
                'epochs': [2, 3, 4, 5]
            }
            
            best_config = None
            best_score = float('inf')
            
            # Simple grid search (could be enhanced with more sophisticated methods)
            import itertools
            
            # Sample subset of combinations for efficiency
            param_combinations = list(itertools.product(
                search_space['learning_rate'][:3],
                search_space['lora_r'][:3],  
                search_space['lora_alpha'][:3],
                search_space['batch_size'][:2],
                search_space['epochs'][:3]
            ))
            
            # Limit to reasonable number of trials
            max_trials = min(args.max_trials or 10, len(param_combinations))
            selected_combinations = param_combinations[:max_trials]
            
            logger.info(f"Running hyperparameter tuning with {max_trials} trials")
            
            for i, (lr, r, alpha, bs, epochs) in enumerate(selected_combinations, 1):
                logger.info(f"Trial {i}/{max_trials}: lr={lr}, r={r}, alpha={alpha}, bs={bs}, epochs={epochs}")
                
                # Create temporary args for this trial
                trial_args = argparse.Namespace(**vars(args))
                trial_args.learning_rate = lr
                trial_args.lora_r = r
                trial_args.lora_alpha = alpha
                trial_args.batch_size = bs
                trial_args.epochs = epochs
                trial_args.output_dir = f"{args.output_dir}/trial_{i}"
                
                # Run training for this trial
                success = self.train_lora(trial_args)
                
                if success:
                    # Get trial results (simplified - would need actual metric extraction)
                    trial_score = self._get_trial_score(trial_args.output_dir)
                    
                    if trial_score < best_score:
                        best_score = trial_score
                        best_config = {
                            'learning_rate': lr,
                            'lora_r': r, 
                            'lora_alpha': alpha,
                            'batch_size': bs,
                            'epochs': epochs,
                            'score': trial_score
                        }
                        logger.info(f"New best configuration found: {best_config}")
            
            # Save best configuration
            if best_config:
                best_config_path = Path(args.output_dir) / "best_hyperparameters.json"
                with open(best_config_path, 'w') as f:
                    json.dump(best_config, f, indent=2)
                
                logger.info(f"Hyperparameter tuning completed. Best config saved to {best_config_path}")
                return True
            else:
                logger.error("No successful trials in hyperparameter tuning")
                return False
                
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            return False
    
    def _monitor_training(self, job_id: str, high_stakes: bool = False) -> bool:
        """Monitor training progress."""
        import time
        
        logger.info(f"Monitoring training job: {job_id}")
        
        try:
            while self.training_service.is_training():
                status = self.training_service.get_training_status(job_id)
                
                if status.get('status') == 'completed':
                    logger.info("Training completed successfully")
                    
                    # Print final metrics
                    final_metrics = status.get('final_metrics', {})
                    if final_metrics:
                        logger.info("Final metrics:")
                        for metric, value in final_metrics.items():
                            logger.info(f"  {metric}: {value}")
                    
                    return True
                
                elif status.get('status') == 'failed':
                    error = status.get('error', 'Unknown error')
                    logger.error(f"Training failed: {error}")
                    return False
                
                elif status.get('status') in ['training', 'preparing']:
                    # Log progress
                    elapsed_time = status.get('elapsed_time', 0)
                    current_metrics = status.get('current_metrics', {})
                    
                    logger.info(f"Training in progress... Elapsed: {elapsed_time:.1f}s")
                    if current_metrics:
                        logger.info(f"Current metrics: {current_metrics}")
                    
                    # Enhanced monitoring for high-stakes training
                    if high_stakes:
                        self._check_high_stakes_conditions(status)
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
            
            logger.warning("Training monitoring ended unexpectedly")
            return False
            
        except KeyboardInterrupt:
            logger.info("Training monitoring interrupted by user")
            self.training_service.stop_training()
            return False
    
    def _check_high_stakes_conditions(self, status: Dict[str, Any]):
        """Check high-stakes training conditions."""
        current_metrics = status.get('current_metrics', {})
        
        # Check for concerning patterns
        if 'eval_loss' in current_metrics and current_metrics['eval_loss'] > 2.0:
            logger.warning("High evaluation loss detected - consider early stopping")
        
        if 'calibration_ece' in current_metrics and current_metrics['calibration_ece'] > 0.2:
            logger.warning("Poor calibration detected - model may be overconfident")
        
        if 'gradient_norm' in current_metrics and current_metrics['gradient_norm'] > 10.0:
            logger.warning("High gradient norm detected - potential training instability")
    
    def _get_trial_score(self, output_dir: str) -> float:
        """Get score for hyperparameter tuning trial."""
        # Simplified scoring - in practice would read actual metrics
        try:
            # Look for training results
            results_file = Path(output_dir) / "training_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results.get('eval_loss', 1.0)
            else:
                # Default score if no results found
                return 1.0
        except Exception:
            return 1.0
    
    def cleanup(self):
        """Clean up services and resources."""
        if self.training_service:
            self.training_service.stop()
            self.training_service.cleanup()
        
        if self.model_service:
            self.model_service.stop()
            self.model_service.cleanup()
        
        logger.info("Services cleaned up")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified training script for LLM fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Training mode')
    
    # Common arguments
    def add_common_args(subparser):
        subparser.add_argument('--model-name', required=True, help='Model name or path')
        subparser.add_argument('--tokenizer-name', help='Tokenizer name (defaults to model name)')
        subparser.add_argument('--train-data', required=True, help='Training data path')
        subparser.add_argument('--eval-data', help='Evaluation data path')
        subparser.add_argument('--output-dir', required=True, help='Output directory')
        subparser.add_argument('--config-dir', help='Configuration directory', type=Path)
        
        # Training hyperparameters
        subparser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
        subparser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
        subparser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
        subparser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
        
        # LoRA parameters
        subparser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
        subparser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
        subparser.add_argument('--lora-dropout', type=float, default=0.1, help='LoRA dropout')
        subparser.add_argument('--target-modules', nargs='+', help='LoRA target modules')
        
        # Training configuration
        subparser.add_argument('--logging-steps', type=int, default=100, help='Logging interval')
        subparser.add_argument('--save-steps', type=int, default=500, help='Save interval')
        subparser.add_argument('--eval-steps', type=int, default=500, help='Evaluation interval')
        subparser.add_argument('--save-total-limit', type=int, default=3, help='Maximum saved checkpoints')
        subparser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
        
        # Data format
        subparser.add_argument('--data-format', default='jsonl', choices=['jsonl', 'json', 'csv'], help='Data format')
        
        # Advanced options
        subparser.add_argument('--use-auth-token', action='store_true', help='Use HuggingFace auth token')
        subparser.add_argument('--trust-remote-code', action='store_true', help='Trust remote code')
        subparser.add_argument('--enable-calibration', action='store_true', help='Enable calibration monitoring')
        subparser.add_argument('--enable-conformal-prediction', action='store_true', help='Enable conformal prediction')
    
    # LoRA training
    lora_parser = subparsers.add_parser('lora', help='LoRA fine-tuning')
    add_common_args(lora_parser)
    
    # High-stakes training  
    high_stakes_parser = subparsers.add_parser('high-stakes', help='High-stakes training with enhanced monitoring')
    add_common_args(high_stakes_parser)
    
    # Hyperparameter tuning
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    add_common_args(tune_parser)
    tune_parser.add_argument('--max-trials', type=int, default=10, help='Maximum tuning trials')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    # Initialize CLI
    cli = UnifiedTrainingCLI()
    
    try:
        # Set up services
        cli.setup_services(args.config_dir)
        
        # Run training based on mode
        success = False
        
        if args.mode == 'lora':
            success = cli.train_lora(args)
        elif args.mode == 'high-stakes':
            success = cli.train_high_stakes(args)
        elif args.mode == 'tune':
            success = cli.tune_hyperparameters(args)
        else:
            logger.error(f"Unknown training mode: {args.mode}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1
    finally:
        cli.cleanup()


if __name__ == "__main__":
    exit(main())