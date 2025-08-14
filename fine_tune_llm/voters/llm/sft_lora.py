"""Enhanced LoRA SFT training script with QLoRA, DoRA, and multi-model support."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import Dataset, load_from_disk
import evaluate
import logging
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# Optional W&B import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not available. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class EnhancedLoRASFTTrainer:
    """Enhanced LoRA Supervised Fine-Tuning with QLoRA, DoRA, and multi-model support."""

    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        """Initialize enhanced LoRA SFT trainer."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Get selected model configuration
        selected_model = self.config["selected_model"]
        self.model_config = self.config["model_options"][selected_model]

        # Setup paths
        self.output_dir = Path("artifacts/models/llm_lora")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluation metrics if enabled
        self.metrics = {}
        if self.config["evaluation"]["enabled"]:
            for metric_name in self.config["evaluation"]["metrics"]:
                self.metrics[metric_name] = evaluate.load(metric_name)

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for QLoRA."""
        quant_config = self.config["lora"]["quantization"]

        if not quant_config["enabled"]:
            return None

        compute_dtype = (
            torch.bfloat16 if quant_config["compute_dtype"] == "bfloat16" else torch.float16
        )

        if quant_config["bits"] == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quant_config["double_quant"],
                bnb_4bit_quant_type=quant_config["quant_type"],
            )
        elif quant_config["bits"] == 8:
            return BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=compute_dtype)
        else:
            raise ValueError(f"Unsupported quantization bits: {quant_config['bits']}")

    def get_peft_config(self):
        """Get PEFT configuration based on method."""
        method = self.config["lora"]["method"]

        if method == "adalora":
            # AdaLoRA configuration
            adalora_config = self.config["lora"]["adalora"]
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["lora"]["r"],
                lora_alpha=self.config["lora"]["lora_alpha"],
                lora_dropout=self.config["lora"]["lora_dropout"],
                target_modules=self.model_config["target_modules"],
                bias="none",
                target_r=adalora_config["target_r"],
                init_r=adalora_config.get("init_r", self.config["lora"]["r"]),
                tinit=adalora_config.get("tinit", 0),
                tfinal=adalora_config.get("tfinal", 0),
                deltaT=adalora_config.get("deltaT", 1),
            )
        else:
            # Standard LoRA or DoRA configuration
            use_dora = method == "dora"
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["lora"]["r"],
                lora_alpha=self.config["lora"]["lora_alpha"],
                lora_dropout=self.config["lora"]["lora_dropout"],
                target_modules=self.model_config["target_modules"],
                bias="none",
                use_dora=use_dora,
            )

        return peft_config

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with LoRA/QLoRA/DoRA."""
        logger.info(f"Loading model: {self.model_config['model_id']}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["tokenizer_id"],
            trust_remote_code=True,
            padding_side="right",  # Important for training
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quantization_config = self.get_quantization_config()

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model_id"],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
        )

        # Prepare for k-bit training if quantized
        if quantization_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            )

        # Configure LoRA/DoRA/AdaLoRA
        peft_config = self.get_peft_config()

        # Apply LoRA/DoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        method_name = {"lora": "LoRA", "dora": "DoRA", "adalora": "AdaLoRA"}.get(
            self.config["lora"]["method"], "LoRA"
        )
        logger.info(f"Model setup complete with {method_name}")

    def get_model_specific_prompt(self, text: str, metadata: Dict = None) -> str:
        """Format prompt according to model-specific chat template."""
        chat_template = self.model_config.get("chat_template", "generic")

        system_prompt = self.config["instruction_format"]["system_prompt"].strip()

        if chat_template == "llama":
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        elif chat_template == "mistral":
            return f"<s>[INST] {system_prompt}\n\n{text} [/INST] "
        elif chat_template == "qwen":
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        else:  # glm or generic
            return f"{system_prompt}\n\nText to classify:\n{text}\n\nResponse:"

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training with model-specific formatting."""

        def format_example(example):
            """Format single example with model-specific prompt."""
            # Get input text and metadata
            text = example.get("text", example.get("input", ""))
            metadata = example.get("metadata", {})

            # Format prompt
            prompt = self.get_model_specific_prompt(text, metadata)

            # Add response
            output = example.get("output", example.get("response", ""))
            full_text = prompt + output

            return full_text

        def tokenize_function(examples):
            """Tokenize examples."""
            # Format texts
            texts = [format_example(ex) for ex in examples]

            # Tokenize
            model_inputs = self.tokenizer(
                texts,
                max_length=self.config["training"]["max_length"],
                padding=self.config["training"]["padding"],
                truncation=self.config["training"]["truncation"],
                return_tensors=None,
            )

            # Set labels (same as input_ids for causal LM)
            model_inputs["labels"] = model_inputs["input_ids"].copy()

            return model_inputs

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config["training"].get("num_proc", 4),
            remove_columns=dataset.column_names,
        )

        return tokenized_dataset

    def split_dataset(self, dataset: Dataset) -> tuple:
        """Split dataset into train/validation sets."""
        if not self.config["evaluation"]["enabled"]:
            return dataset, None

        val_split = self.config["evaluation"]["val_split"]

        # Convert to list for splitting
        data_list = list(dataset)

        # Split
        train_data, val_data = train_test_split(
            data_list,
            test_size=val_split,
            random_state=42,
            stratify=[ex.get("label", 0) for ex in data_list] if "label" in data_list[0] else None,
        )

        return Dataset.from_list(train_data), Dataset.from_list(val_data)

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

        # Basic loss-based metrics
        loss = predictions.mean()
        perplexity = torch.exp(torch.tensor(loss)).item()

        metrics = {"perplexity": perplexity, "loss": loss}

        return metrics

    def get_scheduler(self, optimizer, num_training_steps: int):
        """Get learning rate scheduler."""
        scheduler_config = self.config["training"]["scheduler"]

        if scheduler_config["type"] == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_config["warmup_steps"],
                num_training_steps=num_training_steps,
            )
        elif scheduler_config["type"] == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_config["warmup_steps"],
                num_training_steps=num_training_steps,
            )
        else:
            return None

    def init_wandb(self):
        """Initialize Weights & Biases if enabled."""
        wandb_config = self.config["training"]["logging"]

        if not wandb_config.get("wandb", False):
            return

        if not WANDB_AVAILABLE:
            logger.warning("W&B logging enabled but wandb not installed. Skipping...")
            return

        # Prepare W&B config
        wandb_init_kwargs = {
            "project": wandb_config["project_name"],
            "config": {
                "model": self.config["selected_model"],
                "lora": self.config["lora"],
                "training": self.config["training"],
                "model_config": self.model_config,
            },
            "tags": wandb_config.get("tags", []),
            "notes": wandb_config.get("notes", ""),
        }

        # Add optional parameters
        if wandb_config.get("entity"):
            wandb_init_kwargs["entity"] = wandb_config["entity"]
        if wandb_config.get("run_name"):
            wandb_init_kwargs["name"] = wandb_config["run_name"]

        # Initialize W&B
        wandb.init(**wandb_init_kwargs)
        logger.info("W&B logging initialized")

    def get_report_to(self) -> list:
        """Get list of reporting tools."""
        report_to = ["tensorboard"]

        wandb_config = self.config["training"]["logging"]
        if wandb_config.get("wandb", False) and WANDB_AVAILABLE:
            report_to.append("wandb")

        return report_to

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Trainer:
        """Train the model with all enhancements."""
        # Initialize W&B if enabled
        self.init_wandb()

        # Setup model and tokenizer
        self.setup_model_and_tokenizer()

        # Split dataset if needed
        if eval_dataset is None and self.config["evaluation"]["enabled"]:
            train_dataset, eval_dataset = self.split_dataset(train_dataset)

        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.prepare_dataset(eval_dataset)

        # Calculate total training steps for scheduler
        num_training_steps = (
            len(train_dataset)
            // (
                self.config["training"]["batch_size"]
                * self.config["training"]["gradient_accumulation_steps"]
            )
        ) * self.config["training"]["num_epochs"]

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            # Optimization
            optim=self.config["training"]["optim"],
            fp16=self.config["training"].get("fp16", False),
            bf16=self.config["training"].get("bf16", True),
            gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            # Evaluation
            evaluation_strategy=self.config["training"]["eval_strategy"] if eval_dataset else "no",
            eval_steps=self.config["training"]["eval_steps"],
            save_strategy="steps",
            save_steps=self.config["training"]["save_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            # Best model
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"]
            and eval_dataset is not None,
            metric_for_best_model=self.config["training"]["metric_for_best_model"],
            greater_is_better=self.config["training"]["greater_is_better"],
            save_total_limit=self.config["training"]["save_total_limit"],
            # Logging
            report_to=self.get_report_to(),
            logging_dir=str(self.output_dir / "logs"),
            push_to_hub=False,
            remove_unused_columns=False,
            # Memory optimization
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Callbacks
        callbacks = []
        if self.config["training"]["early_stopping"] and eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["training"]["early_stopping_patience"]
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
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )

        # Setup custom scheduler if specified
        if self.config["training"]["scheduler"]["type"] != "none":
            # Get optimizer from trainer
            optimizer = trainer.create_optimizer()
            scheduler = self.get_scheduler(optimizer, num_training_steps)
            trainer.optimizer = optimizer
            trainer.lr_scheduler = scheduler

        # Log training info
        method_name = "DoRA" if self.config["lora"]["method"] == "dora" else "LoRA"
        quant_info = ""
        if self.config["lora"]["quantization"]["enabled"]:
            quant_info = f" + QLoRA {self.config['lora']['quantization']['bits']}-bit"

        logger.info(f"Starting training with {method_name}{quant_info}")
        logger.info(f"Model: {self.model_config['model_id']}")
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation samples: {len(eval_dataset)}")

        # Train
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        logger.info("Saving model...")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))

        # Save config
        config_to_save = {
            "selected_model": self.config["selected_model"],
            "model_config": self.model_config,
            "lora_config": self.config["lora"],
            "training_config": self.config["training"],
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)

        logger.info(f"Training complete. Model saved to {self.output_dir}")

        return trainer


def train_and_eval(config: Dict) -> float:
    """Wrapper function for hyperparameter tuning."""
    # This is a simplified version for Optuna integration
    # In practice, you'd load your dataset and run a quick training
    trainer = EnhancedLoRASFTTrainer()
    trainer.config = config

    # Mock training for demonstration
    # In real implementation, you'd load actual data and train
    return 0.5  # Return validation loss


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced LoRA training with QLoRA, DoRA, and multi-model support"
    )
    parser.add_argument("--config", default="configs/llm_lora.yaml", help="Config file path")
    parser.add_argument("--train_data", required=True, help="Training data path")
    parser.add_argument("--eval_data", help="Evaluation data path")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--model", help="Override selected model")
    parser.add_argument(
        "--quantization", type=str, choices=["4bit", "8bit", "none"], default="none"
    )
    parser.add_argument("--lora-method", type=str, choices=["lora", "dora"], default="lora")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.model:
        config["selected_model"] = args.model

    if args.quantization != "none":
        config["lora"]["quantization"]["enabled"] = True
        config["lora"]["quantization"]["bits"] = int(args.quantization[0])

    config["lora"]["method"] = args.lora_method

    # Save updated config
    with open(args.config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load datasets
    if args.train_data.endswith(".json"):
        with open(args.train_data) as f:
            train_data = json.load(f)
        train_dataset = Dataset.from_list(train_data)
    else:
        train_dataset = load_from_disk(args.train_data)

    eval_dataset = None
    if args.eval_data:
        if args.eval_data.endswith(".json"):
            with open(args.eval_data) as f:
                eval_data = json.load(f)
            eval_dataset = Dataset.from_list(eval_data)
        else:
            eval_dataset = load_from_disk(args.eval_data)

    # Initialize trainer
    trainer = EnhancedLoRASFTTrainer(args.config)

    # Train
    trainer.train(train_dataset, eval_dataset, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
