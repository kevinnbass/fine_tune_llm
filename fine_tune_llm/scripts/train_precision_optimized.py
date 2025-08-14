#!/usr/bin/env python
"""Precision-optimized training script with all advanced features."""

import argparse
import yaml
import json
import logging
from pathlib import Path
import sys
import torch
from datasets import Dataset, load_dataset

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from voters.llm.sft_lora import EnhancedLoRASFTTrainer
from voters.llm.moe import MoEWrapper
from voters.llm.moa import MoAOrchestrator
from voters.llm.precision_optimizers import (
    ORPOTrainer, PrecisionPruner, EfficientAttentionSkipper,
    MemoryEfficientOptimizers
)
from voters.llm.data_efficiency import (
    DEFTSelector, ContinuousLearner, DataPurifier, HighStakesPreprocessor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Precision-optimized LLM fine-tuning")
    
    # Basic arguments
    parser.add_argument("--config", type=str, default="configs/llm_lora.yaml",
                        help="Path to configuration file")
    parser.add_argument("--train-data", type=str, default="data/processed/train.json",
                        help="Path to training data")
    parser.add_argument("--val-data", type=str, default="data/processed/val.json",
                        help="Path to validation data")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/precision_optimized",
                        help="Output directory for model")
    
    # PEFT methods
    parser.add_argument("--peft-method", type=str, 
                        choices=["lora", "qlora", "dora", "adalora", "dylora", "lorafa"],
                        default="lora", help="PEFT method to use")
    
    # Advanced features
    parser.add_argument("--use-moe", action="store_true",
                        help="Use Mixture of Experts")
    parser.add_argument("--use-moa", action="store_true",
                        help="Use Mixture of Agents")
    parser.add_argument("--use-orpo", action="store_true",
                        help="Use ORPO for preference optimization")
    parser.add_argument("--enable-pruning", action="store_true",
                        help="Enable model pruning")
    parser.add_argument("--enable-eas", action="store_true",
                        help="Enable Efficient Attention Skipping")
    parser.add_argument("--use-lomo", action="store_true",
                        help="Use LOMO optimizer")
    parser.add_argument("--use-mezo", action="store_true",
                        help="Use MeZO optimizer")
    parser.add_argument("--use-deft", action="store_true",
                        help="Use DEFT for data-efficient training")
    parser.add_argument("--enable-continuous-learning", action="store_true",
                        help="Enable continuous learning")
    parser.add_argument("--purify-data", action="store_true",
                        help="Apply data purification")
    parser.add_argument("--high-stakes-preprocessing", action="store_true",
                        help="Apply high-stakes preprocessing")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    return parser.parse_args()


def load_configuration(config_path: str, args) -> dict:
    """Load and update configuration based on arguments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config based on command line arguments
    if args.peft_method == "dylora":
        config['peft']['dylora']['enabled'] = True
    elif args.peft_method == "lorafa":
        config['peft']['lorafa']['enabled'] = True
    elif args.peft_method == "qlora":
        config['lora']['quantization']['enabled'] = True
    else:
        config['lora']['method'] = args.peft_method
    
    if args.use_moe:
        config['moe']['enabled'] = True
    if args.use_moa:
        config['moa']['enabled'] = True
    if args.use_orpo:
        config['alignment']['orpo']['enabled'] = True
    if args.enable_pruning:
        config['pruning']['enabled'] = True
    if args.enable_eas:
        config['eas']['enabled'] = True
    if args.use_lomo:
        config['lomo']['enabled'] = True
    if args.use_mezo:
        config['mezo']['enabled'] = True
    if args.use_deft:
        config['deft']['enabled'] = True
    if args.enable_continuous_learning:
        config['continuous_learning']['enabled'] = True
    if args.purify_data:
        config['data_quality']['purification']['enabled'] = True
    if args.high_stakes_preprocessing:
        config['data_quality']['high_stakes_preprocessing']['enabled'] = True
    
    # Update training parameters
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    
    return config


def load_data(train_path: str, val_path: str) -> tuple:
    """Load training and validation data."""
    logger.info(f"Loading training data from {train_path}")
    
    if train_path.endswith('.json'):
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        train_dataset = Dataset.from_list(train_data)
    else:
        train_dataset = load_dataset('json', data_files=train_path)['train']
    
    if val_path and Path(val_path).exists():
        logger.info(f"Loading validation data from {val_path}")
        if val_path.endswith('.json'):
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            val_dataset = Dataset.from_list(val_data)
        else:
            val_dataset = load_dataset('json', data_files=val_path)['train']
    else:
        val_dataset = None
    
    return train_dataset, val_dataset


def apply_data_processing(dataset: Dataset, config: dict) -> Dataset:
    """Apply data processing based on configuration."""
    
    # Data purification
    if config.get('data_quality', {}).get('purification', {}).get('enabled', False):
        logger.info("Applying data purification...")
        purifier = DataPurifier(config)
        dataset = purifier.purify_dataset(dataset)
    
    # High-stakes preprocessing
    if config.get('data_quality', {}).get('high_stakes_preprocessing', {}).get('enabled', False):
        logger.info("Applying high-stakes preprocessing...")
        preprocessor = HighStakesPreprocessor(config)
        dataset = preprocessor.preprocess(dataset)
    
    return dataset


def setup_trainer(config: dict, train_dataset: Dataset, val_dataset: Dataset):
    """Setup trainer with all configured features."""
    
    # Initialize base trainer
    trainer = EnhancedLoRASFTTrainer(config_path=None)
    trainer.config = config  # Use our modified config
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Apply Half Fine-Tuning if enabled
    if config.get('training', {}).get('hft', {}).get('enabled', False):
        logger.info("Applying Half Fine-Tuning layer selection...")
        trainer.apply_hft_layer_selection(trainer.model)
    
    # Setup MoE if enabled
    if config.get('moe', {}).get('enabled', False):
        logger.info("Setting up Mixture of Experts...")
        moe_wrapper = MoEWrapper(config_path=None)
        moe_wrapper.config = config
        moe_wrapper.initialize_experts()
        trainer.model = moe_wrapper  # Replace model with MoE wrapper
    
    # Setup MoA if enabled
    if config.get('moa', {}).get('enabled', False):
        logger.info("Setting up Mixture of Agents...")
        moa_orchestrator = MoAOrchestrator(config_path=None)
        moa_orchestrator.config = config
        moa_orchestrator.initialize_agents()
        # MoA typically used for inference, but can augment training
    
    # Apply pruning if enabled
    if config.get('pruning', {}).get('enabled', False):
        logger.info("Applying model pruning...")
        pruner = PrecisionPruner(trainer.model, config)
        pruner.prune_model(val_dataset)
    
    # Apply EAS if enabled
    if config.get('eas', {}).get('enabled', False):
        logger.info("Applying Efficient Attention Skipping...")
        eas = EfficientAttentionSkipper(trainer.model, config)
        eas.apply_skipping(val_dataset)
    
    # Setup optimizer
    if config.get('lomo', {}).get('enabled', False):
        logger.info("Using LOMO optimizer...")
        optimizer = MemoryEfficientOptimizers.get_lomo_optimizer(trainer.model, config)
    elif config.get('mezo', {}).get('enabled', False):
        logger.info("Using MeZO optimizer...")
        optimizer = MemoryEfficientOptimizers.get_mezo_optimizer(trainer.model, config)
    else:
        optimizer = None  # Use default
    
    # Apply DEFT if enabled
    if config.get('deft', {}).get('enabled', False):
        logger.info("Applying DEFT data selection...")
        deft_selector = DEFTSelector(trainer.model, trainer.tokenizer, config)
        train_dataset = deft_selector.select_data(train_dataset)
    
    # Setup continuous learning if enabled
    continuous_learner = None
    if config.get('continuous_learning', {}).get('enabled', False):
        logger.info("Setting up continuous learning...")
        continuous_learner = ContinuousLearner(trainer.model, config)
    
    # Setup ORPO if enabled
    orpo_trainer = None
    if config.get('alignment', {}).get('orpo', {}).get('enabled', False):
        logger.info("Setting up ORPO trainer...")
        orpo_trainer = ORPOTrainer(trainer.model, trainer.tokenizer, config)
    
    return trainer, optimizer, continuous_learner, orpo_trainer


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_configuration(args.config, args)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    train_dataset, val_dataset = load_data(args.train_data, args.val_data)
    
    # Apply data processing
    train_dataset = apply_data_processing(train_dataset, config)
    if val_dataset:
        val_dataset = apply_data_processing(val_dataset, config)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Setup trainer and components
    trainer, optimizer, continuous_learner, orpo_trainer = setup_trainer(
        config, train_dataset, val_dataset
    )
    
    # Train model
    logger.info("Starting precision-optimized training...")
    
    if config.get('training', {}).get('forward_only', {}).get('enabled', False):
        # Forward-only training
        logger.info("Using forward-only fine-tuning...")
        # Implement forward-only training logic
        # This would involve in-context learning without backprop
    else:
        # Standard training with all enhancements
        trained_model = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            resume_from_checkpoint=args.resume
        )
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir / "final")
    trainer.tokenizer.save_pretrained(output_dir / "final")
    
    # Save training metrics
    if hasattr(trainer, 'trainer') and hasattr(trainer.trainer, 'state'):
        metrics = {
            'final_loss': trainer.trainer.state.log_history[-1].get('loss', None),
            'final_eval_loss': trainer.trainer.state.log_history[-1].get('eval_loss', None),
            'total_steps': trainer.trainer.state.global_step,
        }
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()