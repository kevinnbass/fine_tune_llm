"""LoRA SFT training script for LLM voter."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_from_disk
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class LoRASFTTrainer:
    """LoRA Supervised Fine-Tuning for LLM voter."""
    
    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        """
        Initialize LoRA SFT trainer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lora_config = self.config['lora']
        self.instruction_format = self.config['instruction_format']
        
        # Setup paths
        self.output_dir = Path("artifacts/models/llm_lora")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with LoRA."""
        logger.info(f"Loading model: {self.lora_config['model_id']}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lora_config['tokenizer_id'],
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for efficient training
        bnb_config = None
        if self.lora_config.get('load_in_8bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        elif self.lora_config.get('load_in_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.lora_config['model_id'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.lora_config['bf16'] else torch.float16
        )
        
        # Prepare for k-bit training if quantized
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            target_modules=self.lora_config['target_modules'],
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
        
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset: Raw dataset with instruction/input/output
            
        Returns:
            Tokenized dataset
        """
        def format_example(example):
            """Format single example."""
            text_parts = []
            
            if example.get('instruction'):
                text_parts.append(f"### Instruction:\n{example['instruction']}")
            
            if example.get('input'):
                text_parts.append(f"### Input:\n{example['input']}")
            
            if example.get('output'):
                text_parts.append(f"### Output:\n{example['output']}")
            
            return "\n\n".join(text_parts)
        
        def tokenize_function(examples):
            """Tokenize examples."""
            # Format texts
            texts = [format_example(ex) for ex in examples]
            
            # Tokenize
            model_inputs = self.tokenizer(
                texts,
                max_length=self.lora_config['max_length'],
                padding=self.lora_config['padding'],
                truncation=self.lora_config['truncation'],
                return_tensors=None
            )
            
            # Set labels (same as input_ids for causal LM)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.lora_config.get('num_proc', 4),
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: EvalPrediction object
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        # Compute perplexity
        loss = predictions.mean()
        perplexity = torch.exp(torch.tensor(loss)).item()
        
        return {
            "perplexity": perplexity,
            "loss": loss
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.lora_config['num_epochs'],
            per_device_train_batch_size=self.lora_config['batch_size'],
            per_device_eval_batch_size=self.lora_config['batch_size'],
            gradient_accumulation_steps=self.lora_config['gradient_accumulation_steps'],
            warmup_ratio=self.lora_config['warmup_ratio'],
            learning_rate=self.lora_config['learning_rate'],
            weight_decay=self.lora_config['weight_decay'],
            max_grad_norm=self.lora_config['max_grad_norm'],
            
            # Optimization
            optim=self.lora_config['optim'],
            fp16=self.lora_config.get('fp16', False),
            bf16=self.lora_config.get('bf16', True),
            gradient_checkpointing=self.lora_config['gradient_checkpointing'],
            
            # Evaluation
            evaluation_strategy=self.lora_config['eval_strategy'],
            eval_steps=self.lora_config['eval_steps'],
            save_strategy="steps",
            save_steps=self.lora_config['save_steps'],
            logging_steps=self.lora_config['logging_steps'],
            
            # Best model
            load_best_model_at_end=self.lora_config['load_best_model_at_end'],
            metric_for_best_model="loss",
            greater_is_better=False,
            save_total_limit=self.lora_config['save_total_limit'],
            
            # Other
            report_to=["tensorboard"],
            push_to_hub=False,
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Callbacks
        callbacks = []
        if self.lora_config['early_stopping']:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.lora_config['early_stopping_patience']
                )
            )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        logger.info("Saving model...")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self.lora_config, f, indent=2)
        
        logger.info(f"Training complete. Model saved to {self.output_dir}")
        
        return trainer
    
    def evaluate_json_compliance(self, model_path: str, test_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate model's JSON schema compliance.
        
        Args:
            model_path: Path to saved model
            test_dataset: Test dataset
            
        Returns:
            Dictionary of compliance metrics
        """
        from .infer import LLMVoterInference
        from .dataset import SFTDatasetBuilder
        
        voter = LLMVoterInference(model_path)
        builder = SFTDatasetBuilder()
        
        valid_count = 0
        abstain_count = 0
        total = len(test_dataset)
        
        for example in tqdm(test_dataset, desc="Evaluating JSON compliance"):
            # Generate prediction
            result = voter.predict(example['input'])
            
            if result['abstain']:
                abstain_count += 1
            else:
                # Validate output
                is_valid, _ = builder.validate_output(json.dumps(result.get('raw_output', {})))
                if is_valid:
                    valid_count += 1
        
        metrics = {
            'json_compliance_rate': valid_count / total,
            'abstention_rate': abstain_count / total,
            'valid_prediction_rate': (valid_count + abstain_count) / total
        }
        
        logger.info(f"JSON Compliance Metrics: {metrics}")
        return metrics


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA model for LLM voter")
    parser.add_argument("--config", default="configs/llm_lora.yaml", help="Config file path")
    parser.add_argument("--train_data", required=True, help="Training data path")
    parser.add_argument("--eval_data", help="Evaluation data path")
    parser.add_argument("--resume", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load datasets
    if args.train_data.endswith('.json'):
        with open(args.train_data) as f:
            train_data = json.load(f)
        train_dataset = Dataset.from_list(train_data)
    else:
        train_dataset = load_from_disk(args.train_data)
    
    eval_dataset = None
    if args.eval_data:
        if args.eval_data.endswith('.json'):
            with open(args.eval_data) as f:
                eval_data = json.load(f)
            eval_dataset = Dataset.from_list(eval_data)
        else:
            eval_dataset = load_from_disk(args.eval_data)
    
    # Initialize trainer
    trainer = LoRASFTTrainer(args.config)
    
    # Train
    trainer.train(
        train_dataset,
        eval_dataset,
        resume_from_checkpoint=args.resume
    )
    
    # Evaluate JSON compliance if eval data available
    if eval_dataset:
        trainer.evaluate_json_compliance(
            str(trainer.output_dir / "final"),
            eval_dataset
        )


if __name__ == "__main__":
    main()