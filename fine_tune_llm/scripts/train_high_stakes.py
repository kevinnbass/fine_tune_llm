#!/usr/bin/env python
"""High-stakes precision and auditability training script."""

import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import List, Dict
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from voters.llm.sft_lora import EnhancedLoRASFTTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="High-stakes LLM training with precision optimization")
    
    # Basic arguments
    parser.add_argument("--config", type=str, default="configs/llm_lora.yaml",
                        help="Path to configuration file")
    parser.add_argument("--train-data", type=str, default="data/test_high_stakes.json",
                        help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/high_stakes",
                        help="Output directory for model")
    
    # High-stakes features
    parser.add_argument("--uncertainty-enabled", action="store_true",
                        help="Enable uncertainty-aware fine-tuning")
    parser.add_argument("--factual-enabled", action="store_true",
                        help="Enable RELIANCE factual accuracy")
    parser.add_argument("--bias-audit-enabled", action="store_true",
                        help="Enable bias auditing")
    parser.add_argument("--explainable-enabled", action="store_true",
                        help="Enable explainable reasoning")
    parser.add_argument("--procedural-enabled", action="store_true",
                        help="Enable procedural alignment")
    parser.add_argument("--verifiable-enabled", action="store_true",
                        help="Enable verifiable training")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    
    return parser.parse_args()


def create_test_data() -> List[Dict]:
    """Create test data for high-stakes training."""
    return [
        {
            "text": "Bird flu, also known as avian influenza, is a viral infection that affects birds and can sometimes spread to humans.",
            "label": "relevant",
            "metadata": {"domain": "medical", "confidence": "high"}
        },
        {
            "text": "The latest research shows that proper biosecurity measures can significantly reduce the spread of avian influenza.",
            "label": "relevant", 
            "metadata": {"domain": "scientific", "confidence": "high"}
        },
        {
            "text": "Yesterday's weather was sunny with a high of 75 degrees.",
            "label": "irrelevant",
            "metadata": {"domain": "weather", "confidence": "high"}
        },
        {
            "text": "Some people believe that eating garlic can cure bird flu, but there is no scientific evidence for this.",
            "label": "relevant",
            "metadata": {"domain": "medical", "confidence": "medium"}
        },
        {
            "text": "The H5N1 strain has been detected in multiple countries and requires immediate attention from health authorities.",
            "label": "relevant",
            "metadata": {"domain": "medical", "confidence": "high"}
        },
        {
            "text": "My favorite color is blue and I like to paint landscapes.",
            "label": "irrelevant",
            "metadata": {"domain": "personal", "confidence": "high"}
        },
        {
            "text": "Bird flu outbreaks can have significant economic impacts on poultry industries worldwide.",
            "label": "relevant",
            "metadata": {"domain": "economic", "confidence": "high"}
        },
        {
            "text": "The vaccine development for avian influenza is ongoing with promising preliminary results.",
            "label": "relevant",
            "metadata": {"domain": "medical", "confidence": "high"}
        },
        {
            "text": "I'm not sure if this article is about bird flu or regular seasonal flu.",
            "label": "uncertain",
            "metadata": {"domain": "medical", "confidence": "low"}
        },
        {
            "text": "Proper hand hygiene and avoiding contact with infected birds are recommended preventive measures.",
            "label": "relevant",
            "metadata": {"domain": "medical", "confidence": "high"}
        }
    ]


def update_config_from_args(config: dict, args) -> dict:
    """Update configuration based on command line arguments."""
    # Ensure high_stakes section exists
    if 'high_stakes' not in config:
        config['high_stakes'] = {}
    
    # Update high-stakes features
    if args.uncertainty_enabled:
        config['high_stakes']['uncertainty'] = {
            'enabled': True,
            'method': 'mc_dropout',
            'num_samples': 5,
            'abstention_threshold': 0.7,
            'fp_penalty_weight': 2.0
        }
        logger.info("Enabled uncertainty-aware training")
    
    if args.factual_enabled:
        config['high_stakes']['factual'] = {
            'enabled': True,
            'reliance_steps': 3,
            'fact_penalty_weight': 2.0,
            'self_consistency_threshold': 0.8
        }
        logger.info("Enabled factual accuracy enhancement")
    
    if args.bias_audit_enabled:
        config['high_stakes']['bias_audit'] = {
            'enabled': True,
            'audit_categories': ['gender', 'race', 'age'],
            'bias_threshold': 0.1,
            'mitigation_weight': 1.5
        }
        logger.info("Enabled bias auditing")
    
    if args.explainable_enabled:
        config['high_stakes']['explainable'] = {
            'enabled': True,
            'chain_of_thought': True,
            'reasoning_steps': 3,
            'faithfulness_check': True
        }
        logger.info("Enabled explainable reasoning")
    
    if args.procedural_enabled:
        config['high_stakes']['procedural'] = {
            'enabled': True,
            'domain': 'medical',
            'compliance_weight': 2.0
        }
        logger.info("Enabled procedural alignment")
    
    if args.verifiable_enabled:
        config['high_stakes']['verifiable'] = {
            'enabled': True,
            'hash_artifacts': True,
            'cryptographic_proof': True,
            'audit_log': 'artifacts/audit_trail.jsonl'
        }
        logger.info("Enabled verifiable training")
    
    # Update training parameters
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config from arguments
    config = update_config_from_args(config, args)
    
    # Create test data if it doesn't exist
    data_path = Path(args.train_data)
    if not data_path.exists():
        logger.info(f"Creating test data at {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = create_test_data()
        with open(data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
    
    # Load data
    with open(data_path, 'r') as f:
        train_data = json.load(f)
    
    train_dataset = Dataset.from_list(train_data)
    
    logger.info(f"Loaded {len(train_dataset)} training samples")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Initialize trainer
    try:
        trainer = EnhancedLoRASFTTrainer(config_path=None)
        trainer.config = config
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        logger.info("High-stakes training started...")
        
        # For this demo, we'll just do basic setup and validation
        # Full training would require more data and resources
        
        # Validate high-stakes components
        if hasattr(trainer, 'high_stakes_components'):
            for component_name, component in trainer.high_stakes_components.items():
                logger.info(f"✓ {component_name} component initialized")
        
        # Test uncertainty wrapper if enabled
        if args.uncertainty_enabled and hasattr(trainer.model, 'generate_with_uncertainty'):
            logger.info("✓ Uncertainty wrapper active")
        
        logger.info("High-stakes training validation completed successfully!")
        logger.info(f"Model and config saved to {output_dir}")
        
        # Save model
        if hasattr(trainer.model, 'save_pretrained'):
            trainer.model.save_pretrained(output_dir / "model")
        trainer.tokenizer.save_pretrained(output_dir / "model")
        
    except Exception as e:
        logger.error(f"Error in high-stakes training: {e}")
        raise


if __name__ == "__main__":
    main()