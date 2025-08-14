"""
Enhanced LoRA Supervised Fine-Tuning trainer.

This module provides a comprehensive LoRA SFT trainer with advanced features
including calibration awareness, high-stakes auditing, and conformal prediction.
"""

import os
import yaml
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging

from .base import BaseTrainer
from .calibrated import CalibratedTrainer
from ...core.exceptions import TrainingError, ModelError, ConfigurationError
from ...models import ModelManager

logger = logging.getLogger(__name__)


class EnhancedLoRASFTTrainer(BaseTrainer):
    """
    Enhanced LoRA Supervised Fine-Tuning trainer with advanced capabilities.
    
    Features:
    - LoRA/QLoRA parameter-efficient fine-tuning
    - Calibration-aware training with ECE/MCE monitoring
    - High-stakes auditing integration
    - Conformal prediction support
    - Advanced metrics tracking
    - Multi-GPU support with accelerate
    """
    
    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        """
        Initialize Enhanced LoRA SFT Trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize base trainer
        super().__init__(self.config)
        
        # Model configuration
        self.model_config = self.config.get('model', {})
        self.model_id = self.model_config.get('model_id', 'ZHIPU-AI/glm-4-9b-chat')
        self.tokenizer_id = self.model_config.get('tokenizer_id', self.model_id)
        
        # LoRA configuration
        self.lora_config = self.config.get('lora', {})
        
        # Training configuration  
        self.training_config = self.config.get('training', {})
        
        # Advanced features configuration
        self.calibration_config = self.config.get('calibration', {})
        self.high_stakes_config = self.config.get('high_stakes', {})
        self.conformal_config = self.config.get('conformal_prediction', {})
        
        # Initialize components
        self.model_manager = ModelManager(self.config)
        
        # High-stakes components (optional)
        self.high_stakes_auditor = None
        self.metrics_aggregator = None
        self.conformal_predictor = None
        
        # Initialize advanced components if enabled
        if self.high_stakes_config.get('enabled', False):
            self._initialize_high_stakes_components()
        
        logger.info(f"Initialized EnhancedLoRASFTTrainer:")
        logger.info(f"  - Model: {self.model_id}")
        logger.info(f"  - LoRA rank: {self.lora_config.get('r', 16)}")
        logger.info(f"  - High-stakes auditing: {self.high_stakes_config.get('enabled', False)}")
        logger.info(f"  - Calibration monitoring: {self.calibration_config.get('enabled', False)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise ConfigurationError(f"Failed to load config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'model_id': 'ZHIPU-AI/glm-4-9b-chat',
                'tokenizer_id': 'ZHIPU-AI/glm-4-9b-chat',
                'trust_remote_code': True
            },
            'lora': {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            },
            'training': {
                'output_dir': 'artifacts/models/llm_lora',
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'gradient_accumulation_steps': 1,
                'learning_rate': 2e-4,
                'weight_decay': 0.01,
                'warmup_ratio': 0.03,
                'bf16': True,
                'logging_steps': 10,
                'evaluation_strategy': 'steps',
                'eval_steps': 100,
                'save_steps': 500,
                'load_best_model_at_end': True
            },
            'quantization': {
                'use_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_compute_dtype': 'bfloat16'
            }
        }
    
    def _initialize_high_stakes_components(self):
        """Initialize high-stakes auditing components."""
        try:
            from ...evaluation.auditing import AdvancedHighStakesAuditor
            from ...evaluation.metrics import MetricsComputer
            from ...inference.conformal import ConformalPredictor
            
            # High-stakes auditor
            if self.high_stakes_config.get('auditor_enabled', True):
                self.high_stakes_auditor = AdvancedHighStakesAuditor(self.high_stakes_config)
                logger.info("Initialized high-stakes auditor")
            
            # Metrics aggregator
            if self.high_stakes_config.get('metrics_enabled', True):
                self.metrics_aggregator = MetricsComputer(self.config)
                logger.info("Initialized advanced metrics aggregator")
            
            # Conformal predictor
            if self.conformal_config.get('enabled', False):
                method = self.conformal_config.get('method', 'lac')
                self.conformal_predictor = ConformalPredictor(method=method)
                logger.info(f"Initialized conformal predictor: {method}")
                
        except ImportError as e:
            logger.warning(f"Some high-stakes components not available: {e}")
        except Exception as e:
            logger.error(f"Error initializing high-stakes components: {e}")
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for QLoRA."""
        quant_config = self.config.get('quantization', {})
        
        if not quant_config.get('use_4bit', False):
            return None
        
        # Convert string dtype to torch dtype
        compute_dtype_str = quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
        compute_dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }
        compute_dtype = compute_dtype_map.get(compute_dtype_str, torch.bfloat16)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        return quantization_config
    
    def get_peft_config(self):
        """Get PEFT (LoRA) configuration."""
        from peft import TaskType
        
        # Model-specific target modules
        model_type = self.model_id.split('/')[-1].lower()
        
        if 'glm' in model_type:
            default_targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif 'qwen' in model_type:
            default_targets = ["q_proj", "v_proj", "k_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"]
        else:
            default_targets = ["q_proj", "v_proj"]
        
        target_modules = self.lora_config.get('target_modules', default_targets)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('lora_alpha', 32),
            lora_dropout=self.lora_config.get('lora_dropout', 0.1),
            target_modules=target_modules,
            bias="none",
            inference_mode=False
        )
        
        return peft_config
    
    def setup_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Setup model and tokenizer with LoRA configuration."""
        try:
            # Get quantization config
            quantization_config = self.get_quantization_config()
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=self.model_config.get('trust_remote_code', True),
                torch_dtype=torch.bfloat16 if quantization_config is None else None
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare model for training
            if quantization_config:
                model = prepare_model_for_kbit_training(model)
            
            # Add LoRA adapter
            peft_config = self.get_peft_config()
            model = get_peft_model(model, peft_config)
            
            # Print model info
            model.print_trainable_parameters()
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("Successfully setup model and tokenizer with LoRA")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise ModelError(f"Failed to setup model: {e}")
    
    def get_model_specific_prompt(self, text: str, metadata: Dict = None) -> str:
        """
        Format input text with model-specific prompt template.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Formatted prompt string
        """
        # Default system prompt
        system_prompt = (
            "You are a helpful assistant for classification tasks. "
            "Analyze the given text and provide a classification with reasoning."
        )
        
        # Model-specific formatting
        model_type = self.model_id.split('/')[-1].lower()
        
        if 'glm' in model_type:
            # GLM-4 chat format
            formatted = f"[gMASK]sop<|system|>\n{system_prompt}<|user|>\n{text}<|assistant|>\n"
        elif 'qwen' in model_type:
            # Qwen chat format
            formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Generic format
            formatted = f"System: {system_prompt}\nUser: {text}\nAssistant: "
        
        return formatted
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for LoRA SFT training.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Processed dataset ready for training
        """
        def format_sample(sample):
            """Format individual sample for training."""
            input_text = sample.get('input', sample.get('text', ''))
            output_text = sample.get('output', sample.get('response', ''))
            
            # Format with model-specific prompt
            prompt = self.get_model_specific_prompt(input_text)
            full_text = prompt + output_text
            
            # Tokenize
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.training_config.get('max_length', 2048),
                padding=False
            )
            
            # Create labels (copy of input_ids)
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Apply formatting to dataset
        formatted_dataset = dataset.map(
            format_sample,
            remove_columns=dataset.column_names,
            desc="Formatting dataset"
        )
        
        logger.info(f"Prepared dataset with {len(formatted_dataset)} samples")
        return formatted_dataset
    
    def train(self) -> Dict[str, Any]:
        """
        Execute LoRA SFT training with advanced features.
        
        Returns:
            Training results and metrics
        """
        try:
            # Setup model and tokenizer
            if self.model is None or self.tokenizer is None:
                self.setup_model_and_tokenizer()
            
            # Validate configuration
            self.validate_configuration()
            
            # Get training arguments
            training_args = self.get_training_arguments()
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM
            )
            
            # Create trainer with calibration awareness
            if self.calibration_config.get('enabled', False):
                trainer = CalibratedTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    data_collator=data_collator,
                    metrics_aggregator=self.metrics_aggregator,
                    conformal_predictor=self.conformal_predictor,
                    calibration_config=self.calibration_config
                )
            else:
                from transformers import Trainer
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    data_collator=data_collator
                )
            
            # Add callbacks
            self._add_training_callbacks(trainer)
            
            # Start training
            logger.info("Starting LoRA SFT training...")
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model()
            trainer.save_state()
            
            # Conduct high-stakes audit if enabled
            if self.high_stakes_auditor and self.eval_dataset:
                logger.info("Conducting high-stakes audit...")
                audit_results = self.high_stakes_auditor.conduct_comprehensive_audit(
                    model=trainer.model,
                    test_data=self.eval_dataset
                )
                train_result.audit_results = audit_results
            
            logger.info("Training completed successfully")
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"LoRA SFT training failed: {e}")
    
    def _add_training_callbacks(self, trainer):
        """Add training callbacks to trainer."""
        from ..callbacks import create_callback
        
        try:
            # Progress callback
            progress_callback = create_callback(
                'progress',
                report_interval=self.training_config.get('progress_interval', 100)
            )
            trainer.add_callback(progress_callback)
            
            # Resource monitor callback
            resource_callback = create_callback(
                'resource_monitor',
                monitor_interval=self.training_config.get('resource_interval', 50)
            )
            trainer.add_callback(resource_callback)
            
            # Early stopping if configured
            if self.training_config.get('early_stopping_patience'):
                early_stop_callback = create_callback(
                    'early_stopping',
                    early_stopping_patience=self.training_config['early_stopping_patience']
                )
                trainer.add_callback(early_stop_callback)
            
            # Calibration monitoring if enabled
            if self.calibration_config.get('enabled', False):
                calibration_callback = create_callback(
                    'calibration_monitor',
                    adjustment_threshold=self.calibration_config.get('adjustment_threshold', 0.05),
                    lr_reduction_factor=self.calibration_config.get('lr_reduction_factor', 0.8)
                )
                trainer.add_callback(calibration_callback)
            
            # Metrics aggregation if available
            if self.metrics_aggregator:
                metrics_callback = create_callback(
                    'metrics_aggregator',
                    metrics_aggregator=self.metrics_aggregator
                )
                trainer.add_callback(metrics_callback)
            
        except Exception as e:
            logger.warning(f"Error adding callbacks: {e}")
    
    def set_datasets(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Set training and evaluation datasets."""
        self.train_dataset = self.prepare_dataset(train_dataset)
        
        if eval_dataset:
            self.eval_dataset = self.prepare_dataset(eval_dataset)
        
        logger.info(f"Set datasets: train={len(self.train_dataset)}, "
                   f"eval={len(self.eval_dataset) if self.eval_dataset else 0}")
    
    def load_model(self, model_path: Union[str, Path]):
        """Load trained LoRA model."""
        try:
            from peft import PeftModel
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                trust_remote_code=self.model_config.get('trust_remote_code', True),
                torch_dtype=torch.bfloat16
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            logger.info(f"Loaded LoRA model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        info = super().get_model_info()
        
        # Add LoRA-specific info
        if self.model and hasattr(self.model, 'peft_config'):
            peft_config = self.model.peft_config
            if peft_config:
                config_dict = peft_config.get('default', {})
                info.update({
                    'lora_rank': getattr(config_dict, 'r', None),
                    'lora_alpha': getattr(config_dict, 'lora_alpha', None),
                    'lora_dropout': getattr(config_dict, 'lora_dropout', None),
                    'target_modules': getattr(config_dict, 'target_modules', None)
                })
        
        # Add configuration info
        info.update({
            'model_id': self.model_id,
            'config_path': str(self.config_path),
            'high_stakes_enabled': self.high_stakes_config.get('enabled', False),
            'calibration_enabled': self.calibration_config.get('enabled', False),
            'conformal_enabled': self.conformal_config.get('enabled', False)
        })
        
        return info