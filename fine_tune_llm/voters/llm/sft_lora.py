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

# Import high-stakes modules if available
try:
    from .uncertainty import MCDropoutWrapper, compute_uncertainty_aware_loss, should_abstain
    from .fact_check import RELIANCEFactChecker, FactualDataFilter
    from .high_stakes_audit import BiasAuditor, ExplainableReasoning, ProceduralAlignment, VerifiableTraining
    from .metrics import (
        compute_ece, compute_mce, compute_brier_score,
        compute_abstention_metrics, compute_risk_aware_metrics,
        compute_confidence_metrics, MetricsAggregator
    )
    from .conformal import ConformalPredictor, RiskControlledPredictor
    from .utils import MetricsTracker, ErrorHandler
    HIGH_STAKES_AVAILABLE = True
except ImportError:
    HIGH_STAKES_AVAILABLE = False

logger = logging.getLogger(__name__)


class CalibratedTrainer(Trainer):
    """Enhanced Trainer with calibration monitoring and advanced metrics."""
    
    def __init__(self, metrics_aggregator=None, conformal_predictor=None, 
                 abstention_loss_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_aggregator = metrics_aggregator
        self.conformal_predictor = conformal_predictor
        self.calibration_history = []
        self.best_ece = float('inf')
        
        # Abstention-aware loss configuration
        self.abstention_loss_config = abstention_loss_config or {}
        self.use_abstention_loss = self.abstention_loss_config.get('enabled', False)
        self.abstention_threshold = self.abstention_loss_config.get('confidence_threshold', 0.7)
        self.abstention_penalty = self.abstention_loss_config.get('abstention_penalty', 0.3)
        self.uncertainty_weight = self.abstention_loss_config.get('uncertainty_weight', 0.1)
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with calibration and advanced metrics."""
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if not HIGH_STAKES_AVAILABLE or eval_dataset is None:
            return eval_results
            
        try:
            # Get predictions and labels for advanced metrics
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            predictions, labels, probabilities = self._get_predictions_and_labels(eval_dataloader)
            
            # Compute calibration metrics
            if len(probabilities) > 0 and len(labels) > 0:
                # Convert to binary for ECE if multiclass
                if probabilities.shape[1] > 2:
                    # Use max probability and correctness for ECE
                    max_probs = np.max(probabilities, axis=1)
                    predicted_classes = np.argmax(probabilities, axis=1)
                    correct = (predicted_classes == labels).astype(float)
                    ece = compute_ece(correct, max_probs)
                    mce = compute_mce(correct, max_probs)
                else:
                    # Binary classification
                    ece = compute_ece(labels, probabilities[:, 1])
                    mce = compute_mce(labels, probabilities[:, 1])
                
                eval_results[f"{metric_key_prefix}_ece"] = ece
                eval_results[f"{metric_key_prefix}_mce"] = mce
                
                # Brier score
                brier = compute_brier_score(labels, max_probs if probabilities.shape[1] > 2 else probabilities[:, 1])
                eval_results[f"{metric_key_prefix}_brier_score"] = brier
                
                # Confidence metrics
                confidence_metrics = compute_confidence_metrics(
                    max_probs if probabilities.shape[1] > 2 else probabilities[:, 1],
                    labels,
                    predicted_classes if probabilities.shape[1] > 2 else (probabilities[:, 1] > 0.5).astype(int)
                )
                
                for key, value in confidence_metrics.items():
                    eval_results[f"{metric_key_prefix}_confidence_{key}"] = value
                
                # Abstention metrics (using confidence threshold)
                confidence_threshold = 0.7
                abstentions = max_probs < confidence_threshold
                if abstentions.sum() > 0:
                    abstention_metrics = compute_abstention_metrics(
                        labels, predicted_classes, abstentions
                    )
                    for key, value in abstention_metrics.items():
                        eval_results[f"{metric_key_prefix}_abstention_{key}"] = value
                
                # Track calibration for learning rate adjustment
                self.calibration_history.append(ece)
                if ece < self.best_ece:
                    self.best_ece = ece
                
                # Conformal prediction calibration
                if self.conformal_predictor is not None:
                    self.conformal_predictor.calibrate(probabilities, labels)
                    coverage_metrics = self.conformal_predictor.evaluate_coverage(probabilities, labels)
                    for key, value in coverage_metrics.items():
                        eval_results[f"{metric_key_prefix}_conformal_{key}"] = value
                
                # Add to metrics aggregator
                if self.metrics_aggregator is not None:
                    epoch = int(self.state.epoch) if hasattr(self.state, 'epoch') else 0
                    metrics_to_add = {k.replace(f"{metric_key_prefix}_", ""): v for k, v in eval_results.items() if k.startswith(metric_key_prefix)}
                    self.metrics_aggregator.add_metrics(metrics_to_add, epoch=epoch)
                
        except Exception as e:
            logger.warning(f"Advanced metrics computation failed: {e}")
            
        return eval_results
    
    def _get_predictions_and_labels(self, dataloader):
        """Extract predictions, labels, and probabilities from dataloader."""
        predictions = []
        labels = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model outputs
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                probabilities.append(probs.cpu().numpy())
                
                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                predictions.append(preds.cpu().numpy())
                
                # Get labels (assuming they're in the batch)
                if 'labels' in batch:
                    labels.append(batch['labels'].cpu().numpy())
        
        # Concatenate all batches
        if predictions and labels and probabilities:
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            probabilities = np.concatenate(probabilities)
            
            # Handle sequence-to-sequence case (take last token)
            if len(predictions.shape) > 1:
                predictions = predictions[:, -1]
                labels = labels[:, -1]
                probabilities = probabilities[:, -1, :]
            
            return predictions, labels, probabilities
        else:
            return np.array([]), np.array([]), np.array([]).reshape(0, 2)
    
    def should_adjust_learning_rate(self):
        """Check if learning rate should be adjusted based on calibration."""
        if len(self.calibration_history) < 3:
            return False
            
        # Check if ECE is consistently increasing
        recent_ece = self.calibration_history[-3:]
        return all(recent_ece[i] > recent_ece[i-1] for i in range(1, len(recent_ece)))
    
    def adjust_learning_rate(self, factor=0.9):
        """Adjust learning rate based on calibration drift."""
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= factor
                logger.info(f"Adjusted learning rate from {old_lr:.2e} to {param_group['lr']:.2e} due to calibration drift")
    
    def compute_abstention_aware_loss(self, model, inputs, return_outputs=False):
        """
        Compute abstention-aware loss that encourages confident predictions
        and penalizes uncertain predictions appropriately.
        """
        if not self.use_abstention_loss:
            # Fall back to standard loss computation
            return model(**inputs) if return_outputs else model(**inputs).loss
            
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get('labels')
        
        if labels is None:
            return outputs if return_outputs else outputs.loss
            
        # Standard cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Reshape for sequence modeling if needed
        if logits.dim() > 2:
            # Causal LM case - take last token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Filter out -100 labels (padding)
            active_mask = shift_labels != -100
            active_logits = shift_logits[active_mask]
            active_labels = shift_labels[active_mask]
        else:
            active_logits = logits
            active_labels = labels
            
        if len(active_logits) == 0:
            return outputs if return_outputs else torch.tensor(0.0, device=logits.device)
            
        # Compute base loss
        base_losses = loss_fct(active_logits, active_labels)
        
        # Compute confidence (max probability)
        probs = torch.softmax(active_logits, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        
        # Abstention-aware loss components
        # 1. Standard loss weighted by confidence
        confidence_weighted_loss = base_losses * confidences
        
        # 2. Uncertainty penalty - penalize low-confidence predictions
        uncertainty_penalty = torch.relu(self.abstention_threshold - confidences) * self.abstention_penalty
        
        # 3. Entropy regularization to encourage confident predictions
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        entropy_penalty = entropy * self.uncertainty_weight
        
        # Combine loss components
        total_loss = confidence_weighted_loss + uncertainty_penalty + entropy_penalty
        final_loss = total_loss.mean()
        
        if return_outputs:
            outputs.loss = final_loss
            return outputs
        return final_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use abstention-aware loss."""
        return self.compute_abstention_aware_loss(model, inputs, return_outputs)


class CalibrationMonitorCallback:
    """Callback to monitor calibration and adjust learning rate."""
    
    def on_evaluate(self, args, state, control, trainer=None, **kwargs):
        """Called after evaluation."""
        if isinstance(trainer, CalibratedTrainer):
            # Check if learning rate should be adjusted
            if trainer.should_adjust_learning_rate():
                trainer.adjust_learning_rate()
                logger.info("Learning rate adjusted due to calibration drift")


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
        
        # Initialize advanced metrics tracking
        self.metrics_aggregator = None
        self.conformal_predictor = None
        self.risk_controlled_predictor = None
        
        if HIGH_STAKES_AVAILABLE:
            metrics_path = self.output_dir / "training_metrics.json"
            self.metrics_aggregator = MetricsAggregator(save_path=metrics_path)
            
            # Initialize conformal prediction components
            advanced_config = self.config.get('advanced_metrics', {})
            if advanced_config.get('conformal_prediction', {}).get('enabled', False):
                alpha = advanced_config['conformal_prediction'].get('alpha', 0.1)
                self.conformal_predictor = ConformalPredictor(alpha=alpha)
                logger.info(f"Initialized conformal predictor with alpha={alpha}")
                
            if advanced_config.get('risk_control', {}).get('enabled', False):
                self.risk_controlled_predictor = RiskControlledPredictor(alpha=alpha)
                logger.info("Initialized risk-controlled predictor")
        
        # Initialize high-stakes components if available
        self.high_stakes_components = {}
        if HIGH_STAKES_AVAILABLE:
            self._initialize_high_stakes_components()
    
    def _initialize_high_stakes_components(self):
        """Initialize high-stakes precision and auditing components."""
        high_stakes_config = self.config.get('high_stakes', {})
        
        # Bias auditor
        if high_stakes_config.get('bias_audit', {}).get('enabled', False):
            self.high_stakes_components['bias_auditor'] = BiasAuditor(self.config)
            logger.info("Initialized bias auditor")
        
        # Procedural alignment
        if high_stakes_config.get('procedural', {}).get('enabled', False):
            self.high_stakes_components['procedural'] = ProceduralAlignment(self.config)
            logger.info("Initialized procedural alignment")
        
        # Verifiable training
        if high_stakes_config.get('verifiable', {}).get('enabled', False):
            self.high_stakes_components['verifiable'] = VerifiableTraining(self.config)
            logger.info("Initialized verifiable training")

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

        # Apply uncertainty wrapper if enabled
        if HIGH_STAKES_AVAILABLE:
            uncertainty_config = self.config.get('high_stakes', {}).get('uncertainty', {})
            if uncertainty_config.get('enabled', False):
                if uncertainty_config.get('method') == 'mc_dropout':
                    self.model = MCDropoutWrapper(
                        self.model, 
                        num_samples=uncertainty_config.get('num_samples', 5),
                        dropout_rate=0.1
                    )
                    logger.info("Applied MC Dropout uncertainty wrapper")

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
        
        # Add calibration monitoring callback
        if HIGH_STAKES_AVAILABLE and self.metrics_aggregator is not None:
            callbacks.append(CalibrationMonitorCallback())

        # Initialize enhanced trainer with calibration monitoring
        if HIGH_STAKES_AVAILABLE and self.metrics_aggregator is not None:
            # Get abstention loss configuration
            abstention_loss_config = self.config.get('advanced_metrics', {}).get('abstention_loss', {})
            
            trainer = CalibratedTrainer(
                metrics_aggregator=self.metrics_aggregator,
                conformal_predictor=self.conformal_predictor,
                abstention_loss_config=abstention_loss_config,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                compute_metrics=self.compute_metrics if eval_dataset else None,
            )
        else:
            # Fallback to standard trainer
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
