"""Hyperparameter tuning using Optuna for LoRA fine-tuning."""

import optuna
import yaml
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import logging
from datasets import Dataset
from voters.llm.sft_lora import EnhancedLoRASFTTrainer
from voters.llm.evaluate import LLMEvaluator

logger = logging.getLogger(__name__)


class OptunaHyperparameterTuner:
    """Hyperparameter tuning using Optuna."""

    def __init__(
        self,
        config_path: str = "configs/llm_lora.yaml",
        train_dataset: Dataset = None,
        val_dataset: Dataset = None,
        n_trials: int = 20,
    ):
        """
        Initialize tuner.

        Args:
            config_path: Path to base configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            n_trials: Number of optimization trials
        """
        self.config_path = config_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_trials = n_trials

        # Load base config
        with open(config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        # Setup results directory
        self.results_dir = Path("artifacts/hyperparameter_tuning")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Metric to minimize (validation loss or 1-f1_score)
        """
        # Sample hyperparameters
        params = self.sample_hyperparameters(trial)

        # Create trial-specific config
        trial_config = self.create_trial_config(params)

        # Create temporary config file for this trial
        trial_dir = self.results_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)

        trial_config_path = trial_dir / "config.yaml"
        with open(trial_config_path, "w") as f:
            yaml.dump(trial_config, f, default_flow_style=False)

        try:
            # Train model with trial config
            logger.info(f"Starting trial {trial.number} with params: {params}")

            trainer = EnhancedLoRASFTTrainer(str(trial_config_path))

            # Reduce epochs for faster tuning
            trial_config["training"]["num_epochs"] = min(trial_config["training"]["num_epochs"], 2)
            trial_config["training"]["save_steps"] = 50
            trial_config["training"]["eval_steps"] = 50
            trial_config["training"]["logging_steps"] = 10

            # Set output directory for this trial
            trainer.output_dir = trial_dir / "model"
            trainer.output_dir.mkdir(exist_ok=True)

            # Train model
            result_trainer = trainer.train(self.train_dataset, self.val_dataset)

            # Evaluate model
            if self.val_dataset:
                evaluator = LLMEvaluator(str(trainer.output_dir / "final"))
                metrics = evaluator.evaluate_dataset(self.val_dataset)

                # Save trial results
                trial_results = {
                    "trial_number": trial.number,
                    "hyperparameters": params,
                    "metrics": metrics,
                }

                with open(trial_dir / "results.json", "w") as f:
                    json.dump(trial_results, f, indent=2)

                # Return metric to minimize
                if "f1_macro" in metrics:
                    objective_value = 1.0 - metrics["f1_macro"]  # Minimize 1-F1
                else:
                    objective_value = metrics.get("loss", 1.0)  # Minimize loss

                logger.info(f"Trial {trial.number} completed. Objective: {objective_value:.4f}")

                return objective_value
            else:
                # No validation set, use training loss
                return 0.5  # Placeholder

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return high penalty for failed trials
            return 1.0

        finally:
            # Clean up model files to save space (keep only best)
            if trial_dir.exists() and (trial_dir / "model").exists():
                # Keep only the config and results, remove model files
                model_dir = trial_dir / "model"
                if model_dir.exists():
                    shutil.rmtree(model_dir)

    def sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for trial."""
        params = {
            # LoRA parameters
            "lora_rank": trial.suggest_int("lora_rank", 8, 64, step=8),
            "lora_alpha": trial.suggest_int("lora_alpha", 16, 128, step=16),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.05, 0.3),
            # Training parameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.1),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
            # Scheduler
            "scheduler_type": trial.suggest_categorical("scheduler_type", ["cosine", "linear"]),
            # Method
            "lora_method": trial.suggest_categorical("lora_method", ["lora", "dora"]),
            # Quantization (optional)
            "use_quantization": trial.suggest_categorical("use_quantization", [False, True]),
            "quantization_bits": (
                trial.suggest_categorical("quantization_bits", [4, 8])
                if trial.params.get("use_quantization", False)
                else 4
            ),
        }

        return params

    def create_trial_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for trial."""
        trial_config = self.base_config.copy()

        # Update LoRA settings
        trial_config["lora"]["r"] = params["lora_rank"]
        trial_config["lora"]["lora_alpha"] = params["lora_alpha"]
        trial_config["lora"]["lora_dropout"] = params["lora_dropout"]
        trial_config["lora"]["method"] = params["lora_method"]

        # Update quantization
        trial_config["lora"]["quantization"]["enabled"] = params["use_quantization"]
        if params["use_quantization"]:
            trial_config["lora"]["quantization"]["bits"] = params["quantization_bits"]

        # Update training settings
        trial_config["training"]["learning_rate"] = params["learning_rate"]
        trial_config["training"]["batch_size"] = params["batch_size"]
        trial_config["training"]["warmup_ratio"] = params["warmup_ratio"]
        trial_config["training"]["weight_decay"] = params["weight_decay"]
        trial_config["training"]["scheduler"]["type"] = params["scheduler_type"]

        # Adjust gradient accumulation based on batch size
        target_effective_batch = 16
        trial_config["training"]["gradient_accumulation_steps"] = max(
            1, target_effective_batch // params["batch_size"]
        )

        return trial_config

    def optimize(self, study_name: str = "lora_hyperparameter_tuning") -> optuna.Study:
        """Run hyperparameter optimization."""

        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Run optimization
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")

        study.optimize(
            self.objective, n_trials=self.n_trials, timeout=None, callbacks=[self.trial_callback]
        )

        # Save study results
        self.save_study_results(study)

        return study

    def trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function called after each trial."""
        logger.info(f"Trial {trial.number} finished with value: {trial.value}")

        if trial.value is not None:
            logger.info(f"Current best value: {study.best_value}")
            logger.info(f"Current best params: {study.best_params}")

    def save_study_results(self, study: optuna.Study):
        """Save optimization results."""

        # Best parameters
        best_params = study.best_params
        best_value = study.best_value

        # Create summary
        summary = {
            "best_value": best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "study_name": study.study_name,
        }

        # Save summary
        with open(self.results_dir / "optimization_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save all trials
        trials_data = []
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
            trials_data.append(trial_data)

        with open(self.results_dir / "all_trials.json", "w") as f:
            json.dump(trials_data, f, indent=2)

        # Create best configuration
        best_config = self.create_trial_config(best_params)
        with open(self.results_dir / "best_config.yaml", "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)

        # Print summary
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Best Value: {best_value:.4f}")
        print("Best Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print("=" * 60)
        print(f"Results saved to: {self.results_dir}")
        print(f"Best config saved to: {self.results_dir / 'best_config.yaml'}")
        print("=" * 60)

        logger.info("Optimization completed successfully")


def main():
    """Main hyperparameter tuning function."""
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for LoRA fine-tuning")
    parser.add_argument("--config", default="configs/llm_lora.yaml", help="Base config file")
    parser.add_argument("--train-data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--study-name", default="lora_tuning", help="Study name")

    args = parser.parse_args()

    # Load datasets
    if args.train_data.endswith(".json"):
        with open(args.train_data) as f:
            train_data = json.load(f)
        train_dataset = Dataset.from_list(train_data)
    else:
        train_dataset = Dataset.load_from_disk(args.train_data)

    val_dataset = None
    if args.val_data:
        if args.val_data.endswith(".json"):
            with open(args.val_data) as f:
                val_data = json.load(f)
            val_dataset = Dataset.from_list(val_data)
        else:
            val_dataset = Dataset.load_from_disk(args.val_data)

    # Subsample for faster tuning if datasets are large
    if len(train_dataset) > 1000:
        train_dataset = train_dataset.select(range(1000))
        logger.info("Subsampled training dataset to 1000 examples for faster tuning")

    if val_dataset and len(val_dataset) > 200:
        val_dataset = val_dataset.select(range(200))
        logger.info("Subsampled validation dataset to 200 examples for faster tuning")

    # Initialize tuner
    tuner = OptunaHyperparameterTuner(
        config_path=args.config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=args.n_trials,
    )

    # Run optimization
    study = tuner.optimize(args.study_name)

    return study


if __name__ == "__main__":
    main()
