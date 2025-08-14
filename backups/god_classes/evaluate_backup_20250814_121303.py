"""Evaluation pipeline for fine-tuned LLM models."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import advanced metrics if available
try:
    from .metrics import (
        compute_ece, compute_mce, compute_brier_score,
        compute_abstention_metrics, compute_risk_aware_metrics,
        compute_confidence_metrics, MetricsAggregator,
        compute_reliability_diagram_data
    )
    from .conformal import ConformalPredictor, RiskControlledPredictor
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Comprehensive evaluation for fine-tuned LLM models."""

    def __init__(self, model, tokenizer, model_path: Optional[str] = None, config_path: str = "configs/llm_lora.yaml"):
        """
        Initialize evaluator.

        Args:
            model: Model instance for evaluation
            tokenizer: Tokenizer instance  
            model_path: Optional path to fine-tuned model
            config_path: Path to configuration file
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = Path(model_path) if model_path else None
        self.config_path = config_path
        
        # Initialize tracking
        self.predictions = []
        self.results = {}
        
        # Try to load config if available
        try:
            self.load_config()
        except FileNotFoundError:
            # Use default config if not found
            self.config = {
                'instruction_format': {
                    'system_prompt': 'You are a helpful assistant.'
                }
            }

        # Load evaluation metrics if evaluate library is available
        try:
            self.metrics = {
                "accuracy": evaluate.load("accuracy"),
                "f1": evaluate.load("f1"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
            }
        except:
            self.metrics = {}
        
        # Initialize advanced metrics components
        self.metrics_aggregator = None
        self.conformal_predictor = None
        self.risk_controlled_predictor = None
        
        if ADVANCED_METRICS_AVAILABLE:
            # Initialize metrics tracking
            if model_path:
                metrics_path = Path(model_path).parent / "evaluation_metrics.json"
            else:
                metrics_path = Path("artifacts/evaluation_metrics.json")
            self.metrics_aggregator = MetricsAggregator(save_path=metrics_path)
            
            # Initialize conformal prediction components
            advanced_config = self.config.get('advanced_metrics', {})
            if advanced_config.get('conformal_prediction', {}).get('enabled', False):
                alpha = advanced_config['conformal_prediction'].get('alpha', 0.1)
                method = advanced_config['conformal_prediction'].get('method', 'lac')
                self.conformal_predictor = ConformalPredictor(alpha=alpha, method=method)
                logger.info(f"Initialized conformal predictor with alpha={alpha}, method={method}")
                
            if advanced_config.get('risk_control', {}).get('enabled', False):
                self.risk_controlled_predictor = RiskControlledPredictor(alpha=alpha)
                logger.info("Initialized risk-controlled predictor")
            
            logger.info("Advanced metrics components initialized")

    def load_config(self):
        """Load configuration."""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def load_model(self):
        """Load the fine-tuned model."""
        logger.info(f"Loading model from {self.model_path}")

        # Get model configuration
        selected_model = self.config["selected_model"]
        model_config = self.config["model_options"][selected_model]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        logger.info("Model loaded successfully")
    
    def evaluate_single(self, text: str, parse_json: bool = False) -> Dict:
        """Evaluate a single text input."""
        return self.predict_single(text, {})

    def predict_single(self, text: str, metadata: Dict = None) -> Dict:
        """
        Generate prediction for a single text.

        Args:
            text: Input text
            metadata: Additional metadata

        Returns:
            Prediction dictionary
        """
        # Format prompt according to model type
        system_prompt = self.config["instruction_format"]["system_prompt"].strip()
        prompt = f"{system_prompt}\n\nText to classify:\n{text}\n\nResponse:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part
        prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_text = full_response[prompt_length:].strip()

        # Parse JSON response
        try:
            # Look for JSON-like content
            start_idx = generated_text.find("{")
            end_idx = generated_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = generated_text[start_idx:end_idx]
                result = json.loads(json_str)

                return {
                    "decision": result.get("decision", "unknown"),
                    "rationale": result.get("rationale", ""),
                    "abstain": result.get("abstain", False),
                    "confidence": result.get("confidence", 0.5),
                    "raw_output": generated_text,
                    "valid_json": True,
                }
            else:
                # No valid JSON found
                return {
                    "decision": "parse_error",
                    "rationale": "Could not parse JSON response",
                    "abstain": True,
                    "confidence": 0.0,
                    "raw_output": generated_text,
                    "valid_json": False,
                }

        except json.JSONDecodeError:
            return {
                "decision": "parse_error",
                "rationale": "Invalid JSON format",
                "abstain": True,
                "confidence": 0.0,
                "raw_output": generated_text,
                "valid_json": False,
            }

    def evaluate_dataset(self, dataset, batch_size: int = 1, output_path: Optional[str] = None) -> List[Dict]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Evaluation dataset (list of dicts or Dataset)
            batch_size: Batch size for processing
            output_path: Optional path to save detailed results

        Returns:
            List of evaluation results
        """
        # Handle different dataset types
        if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
            # Dataset-like object
            data_list = [dataset[i] for i in range(len(dataset))]
        else:
            # Assume it's already a list
            data_list = dataset
            
        if not data_list:
            return []
            
        logger.info(f"Evaluating on {len(data_list)} examples")

        detailed_results = []

        # Process in batches (simplified - just iterate for now)
        for i, example in enumerate(tqdm(data_list, desc="Evaluating")):
            try:
                # Get prediction
                pred = self.predict_single(
                    example.get("text", example.get("input", "")), example.get("metadata", {})
                )

                # Get ground truth
                true_label = example.get("label", example.get("output", "unknown"))

                # Store detailed result
                result = {
                    "text": example.get("text", example.get("input", "")),
                    "label": true_label,
                    "prediction": pred["decision"],
                    "abstain": pred["abstain"],
                    "confidence": pred["confidence"],
                    "rationale": pred["rationale"],
                    "valid_json": pred["valid_json"],
                    "raw_output": pred["raw_output"],
                }
                
                # Add parsed response if available
                if pred.get("valid_json"):
                    try:
                        parsed = json.loads(pred["raw_output"])
                        result["parsed_response"] = parsed
                    except:
                        result["parsed_response"] = None
                        
                detailed_results.append(result)
                
            except Exception as e:
                # Handle errors gracefully
                result = {
                    "text": example.get("text", example.get("input", "")),
                    "label": example.get("label", example.get("output", "unknown")),
                    "prediction": "error",
                    "error": str(e),
                    "abstain": False,
                    "confidence": 0.0,
                    "rationale": "",
                    "valid_json": False,
                    "raw_output": ""
                }
                detailed_results.append(result)
                logger.warning(f"Error processing example {i}: {e}")

        # Save detailed results if requested
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            with open(output_dir / "detailed_results.json", "w") as f:
                json.dump(detailed_results, f, indent=2)

            logger.info(f"Results saved to {output_dir}")

        return detailed_results

    def compute_metrics(
        self, predictions: List[str], ground_truth: List[str], detailed_results: List[Dict]
    ) -> Dict:
        """Compute comprehensive evaluation metrics including advanced metrics."""

        # Filter out abstentions for core metrics
        non_abstain_indices = [
            i
            for i, result in enumerate(detailed_results)
            if not result["abstain"] and result.get("prediction", result.get("predicted_label")) != "parse_error"
        ]

        if len(non_abstain_indices) == 0:
            logger.warning("No valid predictions found!")
            return {
                "error": "No valid predictions",
                "total_examples": len(predictions),
                "abstention_rate": 1.0,
                "json_compliance_rate": 0.0,
            }

        filtered_preds = [predictions[i] for i in non_abstain_indices]
        filtered_truth = [ground_truth[i] for i in non_abstain_indices]

        # Core classification metrics
        accuracy = accuracy_score(filtered_truth, filtered_preds)
        f1_macro = f1_score(filtered_truth, filtered_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(filtered_truth, filtered_preds, average="weighted", zero_division=0)
        precision_macro = precision_score(
            filtered_truth, filtered_preds, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            filtered_truth, filtered_preds, average="macro", zero_division=0
        )

        # Model behavior metrics
        abstention_rate = sum(result["abstain"] for result in detailed_results) / len(
            detailed_results
        )
        json_compliance_rate = sum(result["valid_json"] for result in detailed_results) / len(
            detailed_results
        )

        # Confidence statistics
        confidences = [
            result["confidence"] for result in detailed_results if result["confidence"] > 0
        ]
        confidence_stats = {
            "mean": np.mean(confidences) if confidences else 0,
            "std": np.std(confidences) if confidences else 0,
            "min": np.min(confidences) if confidences else 0,
            "max": np.max(confidences) if confidences else 0,
        }

        # Error analysis
        parse_errors = sum(
            1 for result in detailed_results if result.get("prediction", result.get("predicted_label")) == "parse_error"
        )
        parse_error_rate = parse_errors / len(detailed_results)

        metrics = {
            # Core metrics
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            # Model behavior
            "abstention_rate": abstention_rate,
            "json_compliance_rate": json_compliance_rate,
            "parse_error_rate": parse_error_rate,
            # Coverage
            "total_examples": len(predictions),
            "valid_predictions": len(non_abstain_indices),
            "coverage": len(non_abstain_indices) / len(predictions),
            # Confidence
            "confidence_stats": confidence_stats,
            # Label distribution
            "label_distribution": self.get_label_distribution(ground_truth),
            "prediction_distribution": self.get_label_distribution(filtered_preds),
        }

        # Add advanced metrics if available
        if ADVANCED_METRICS_AVAILABLE and len(confidences) > 0 and len(filtered_preds) > 0:
            try:
                # Convert predictions to binary for ECE/MCE
                y_true_binary = np.array([pred == truth for pred, truth in zip(filtered_preds, filtered_truth)], dtype=float)
                y_confidences = np.array([detailed_results[i]["confidence"] for i in non_abstain_indices])
                
                # Calibration metrics
                if len(y_true_binary) > 0 and len(y_confidences) > 0:
                    metrics["ece"] = compute_ece(y_true_binary, y_confidences)
                    metrics["mce"] = compute_mce(y_true_binary, y_confidences)
                    metrics["brier_score"] = compute_brier_score(y_true_binary, y_confidences)
                    
                    # Reliability diagram data
                    reliability_data = compute_reliability_diagram_data(y_true_binary, y_confidences)
                    metrics["reliability_data"] = reliability_data

                # Confidence metrics
                confidence_metrics = compute_confidence_metrics(
                    y_confidences, 
                    np.array([self._label_to_index(truth) for truth in filtered_truth]), 
                    np.array([self._label_to_index(pred) for pred in filtered_preds])
                )
                metrics.update({f"confidence_{k}": v for k, v in confidence_metrics.items()})
                
                # Abstention metrics
                if abstention_rate > 0:
                    abstentions = np.array([result["abstain"] for result in detailed_results])
                    all_preds = np.array([self._label_to_index(result.get("prediction", "unknown")) for result in detailed_results])
                    all_truth = np.array([self._label_to_index(ground_truth[i]) for i in range(len(detailed_results))])
                    
                    abstention_metrics = compute_abstention_metrics(all_truth, all_preds, abstentions)
                    metrics.update({f"abstention_{k}": v for k, v in abstention_metrics.items()})
                
                # Risk-aware metrics
                risk_metrics = compute_risk_aware_metrics(
                    np.array([self._label_to_index(truth) for truth in filtered_truth]),
                    np.array([self._label_to_index(pred) for pred in filtered_preds])
                )
                metrics.update({f"risk_{k}": v for k, v in risk_metrics.items()})
                
                # Conformal prediction metrics if enabled
                if self.conformal_predictor is not None:
                    # For conformal prediction, we need probability distributions
                    # For now, we'll create mock probabilities based on confidence
                    n_samples = len(y_confidences)
                    n_classes = len(set(filtered_truth + filtered_preds))
                    
                    # Create simple probability distributions
                    probs = np.zeros((n_samples, max(4, n_classes)))  # Ensure at least 4 classes
                    for i, conf in enumerate(y_confidences):
                        pred_idx = self._label_to_index(filtered_preds[i])
                        probs[i, pred_idx] = conf
                        # Distribute remaining probability
                        remaining_prob = 1.0 - conf
                        other_prob = remaining_prob / (probs.shape[1] - 1)
                        for j in range(probs.shape[1]):
                            if j != pred_idx:
                                probs[i, j] = other_prob
                    
                    # Calibrate conformal predictor
                    try:
                        labels_idx = np.array([self._label_to_index(truth) for truth in filtered_truth])
                        self.conformal_predictor.calibrate(probs, labels_idx)
                        
                        # Evaluate coverage
                        coverage_metrics = self.conformal_predictor.evaluate_coverage(probs, labels_idx)
                        metrics.update({f"conformal_{k}": v for k, v in coverage_metrics.items()})
                        
                    except Exception as e:
                        logger.warning(f"Conformal prediction evaluation failed: {e}")
                
                logger.info("Advanced metrics computed successfully")
                
            except Exception as e:
                logger.warning(f"Advanced metrics computation failed: {e}")
        
        # Add to metrics aggregator if available
        if self.metrics_aggregator is not None:
            self.metrics_aggregator.add_metrics(metrics)

        return metrics

    def _label_to_index(self, label: str) -> int:
        """Convert label string to index for advanced metrics."""
        label_map = {
            "HIGH_RISK": 0,
            "MEDIUM_RISK": 1, 
            "LOW_RISK": 2,
            "NO_RISK": 3,
            "unknown": 3,
            "parse_error": 3
        }
        return label_map.get(label, 3)
    
    def get_label_distribution(self, labels: List[str]) -> Dict[str, float]:
        """Get label distribution."""
        from collections import Counter

        counts = Counter(labels)
        total = len(labels)
        return {label: count / total for label, count in counts.items()}

    def create_visualizations(self, detailed_results: List[Dict], metrics: Dict, output_dir: Path):
        """Create evaluation visualizations."""

        # Create results DataFrame
        df = pd.DataFrame(detailed_results)

        # 1. Confusion Matrix
        if len(df[~df["abstain"]]) > 0:
            valid_df = df[~df["abstain"] & (df["predicted_label"] != "parse_error")]
            if len(valid_df) > 0:
                cm = confusion_matrix(valid_df["true_label"], valid_df["predicted_label"])
                labels = sorted(
                    list(
                        set(valid_df["true_label"].unique())
                        | set(valid_df["predicted_label"].unique())
                    )
                )

                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
                plt.title("Confusion Matrix")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.tight_layout()
                plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
                plt.close()

        # 2. Metrics Summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Core metrics
        core_metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        core_values = [metrics[m] for m in core_metrics]
        axes[0, 0].bar(core_metrics, core_values)
        axes[0, 0].set_title("Core Classification Metrics")
        axes[0, 0].set_ylim(0, 1)

        # Model behavior
        behavior_metrics = ["abstention_rate", "json_compliance_rate", "parse_error_rate"]
        behavior_values = [metrics[m] for m in behavior_metrics]
        axes[0, 1].bar(behavior_metrics, behavior_values)
        axes[0, 1].set_title("Model Behavior Metrics")
        axes[0, 1].set_ylim(0, 1)

        # Confidence distribution
        if df["confidence"].sum() > 0:
            axes[1, 0].hist(df[df["confidence"] > 0]["confidence"], bins=20, alpha=0.7)
            axes[1, 0].set_title("Confidence Distribution")
            axes[1, 0].set_xlabel("Confidence")
            axes[1, 0].set_ylabel("Frequency")

        # Label distribution
        label_dist = metrics["label_distribution"]
        axes[1, 1].bar(label_dist.keys(), label_dist.values())
        axes[1, 1].set_title("True Label Distribution")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "metrics_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Advanced metrics visualizations
        if ADVANCED_METRICS_AVAILABLE:
            self.create_advanced_visualizations(metrics, output_dir)
        
        # 4. Results summary report
        self.create_text_report(metrics, output_dir)

        logger.info("Visualizations created successfully")
    
    def create_advanced_visualizations(self, metrics: Dict, output_dir: Path):
        """Create advanced metrics visualizations."""
        try:
            # Reliability diagram
            if "reliability_data" in metrics:
                reliability_data = metrics["reliability_data"]
                if reliability_data and "bin_accuracies" in reliability_data:
                    plt.figure(figsize=(10, 8))
                    
                    bin_centers = reliability_data["bin_centers"]
                    bin_accuracies = reliability_data["bin_accuracies"] 
                    bin_confidences = reliability_data["bin_confidences"]
                    bin_counts = reliability_data["bin_counts"]
                    
                    # Plot reliability diagram
                    plt.subplot(2, 2, 1)
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                    plt.scatter(bin_confidences, bin_accuracies, s=bin_counts*10, alpha=0.7, label='Model')
                    plt.xlabel('Confidence')
                    plt.ylabel('Accuracy')
                    plt.title('Reliability Diagram')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # ECE and MCE bars
                    plt.subplot(2, 2, 2)
                    calibration_metrics = []
                    calibration_values = []
                    if "ece" in metrics:
                        calibration_metrics.append("ECE")
                        calibration_values.append(metrics["ece"])
                    if "mce" in metrics:
                        calibration_metrics.append("MCE")
                        calibration_values.append(metrics["mce"])
                    if "brier_score" in metrics:
                        calibration_metrics.append("Brier")
                        calibration_values.append(metrics["brier_score"])
                    
                    if calibration_metrics:
                        plt.bar(calibration_metrics, calibration_values)
                        plt.title('Calibration Metrics')
                        plt.ylabel('Score')
                    
                    # Confidence distribution by correctness
                    plt.subplot(2, 2, 3)
                    if "confidence_mean_confidence_correct" in metrics and "confidence_mean_confidence_incorrect" in metrics:
                        conf_correct = metrics["confidence_mean_confidence_correct"]
                        conf_incorrect = metrics["confidence_mean_confidence_incorrect"]
                        plt.bar(['Correct', 'Incorrect'], [conf_correct, conf_incorrect], alpha=0.7)
                        plt.title('Confidence by Correctness')
                        plt.ylabel('Mean Confidence')
                    
                    # Risk metrics
                    plt.subplot(2, 2, 4)
                    risk_metrics = {}
                    for key, value in metrics.items():
                        if key.startswith("risk_") and isinstance(value, (int, float)):
                            risk_metrics[key.replace("risk_", "")] = value
                    
                    if risk_metrics:
                        plt.bar(risk_metrics.keys(), risk_metrics.values())
                        plt.title('Risk-Aware Metrics')
                        plt.xticks(rotation=45)
                        plt.ylabel('Score')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / "advanced_metrics.png", dpi=300, bbox_inches="tight")
                    plt.close()
            
            # Conformal prediction visualization
            conformal_metrics = {k.replace("conformal_", ""): v for k, v in metrics.items() if k.startswith("conformal_")}
            if conformal_metrics:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                # Coverage vs target
                if "coverage" in conformal_metrics and "target_coverage" in conformal_metrics:
                    coverage = conformal_metrics["coverage"]
                    target = conformal_metrics["target_coverage"] 
                    plt.bar(['Actual', 'Target'], [coverage, target], alpha=0.7)
                    plt.title('Conformal Prediction Coverage')
                    plt.ylabel('Coverage')
                    plt.ylim(0, 1)
                
                plt.subplot(1, 2, 2)
                # Set size distribution
                size_metrics = {k: v for k, v in conformal_metrics.items() if k.startswith("size_")}
                if size_metrics:
                    sizes = [int(k.split("_")[1]) for k in size_metrics.keys()]
                    counts = list(size_metrics.values())
                    plt.bar(sizes, counts, alpha=0.7)
                    plt.title('Prediction Set Size Distribution')
                    plt.xlabel('Set Size')
                    plt.ylabel('Count')
                
                plt.tight_layout()
                plt.savefig(output_dir / "conformal_metrics.png", dpi=300, bbox_inches="tight")
                plt.close()
                
            logger.info("Advanced visualizations created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create advanced visualizations: {e}")

    def create_text_report(self, metrics: Dict, output_dir: Path):
        """Create a text summary report."""

        report = f"""
# Model Evaluation Report

## Overview
- Total Examples: {metrics['total_examples']}
- Valid Predictions: {metrics['valid_predictions']}
- Coverage: {metrics['coverage']:.3f}

## Core Classification Metrics
- Accuracy: {metrics['accuracy']:.3f}
- F1 Score (Macro): {metrics['f1_macro']:.3f}
- F1 Score (Weighted): {metrics['f1_weighted']:.3f}
- Precision (Macro): {metrics['precision_macro']:.3f}
- Recall (Macro): {metrics['recall_macro']:.3f}

## Model Behavior
- Abstention Rate: {metrics['abstention_rate']:.3f}
- JSON Compliance Rate: {metrics['json_compliance_rate']:.3f}
- Parse Error Rate: {metrics['parse_error_rate']:.3f}

## Confidence Statistics
- Mean Confidence: {metrics['confidence_stats']['mean']:.3f}
- Std Confidence: {metrics['confidence_stats']['std']:.3f}
- Min Confidence: {metrics['confidence_stats']['min']:.3f}
- Max Confidence: {metrics['confidence_stats']['max']:.3f}

## Label Distribution
"""

        for label, proportion in metrics["label_distribution"].items():
            report += f"- {label}: {proportion:.3f}\n"

        report += "\n## Prediction Distribution\n"
        for label, proportion in metrics["prediction_distribution"].items():
            report += f"- {label}: {proportion:.3f}\n"

        # Add advanced metrics to report
        if ADVANCED_METRICS_AVAILABLE:
            # Calibration metrics
            calibration_section = "\n## Advanced Calibration Metrics\n"
            if "ece" in metrics:
                calibration_section += f"- Expected Calibration Error (ECE): {metrics['ece']:.4f}\n"
            if "mce" in metrics:
                calibration_section += f"- Maximum Calibration Error (MCE): {metrics['mce']:.4f}\n"
            if "brier_score" in metrics:
                calibration_section += f"- Brier Score: {metrics['brier_score']:.4f}\n"
            
            # Advanced confidence metrics
            confidence_section = "\n## Advanced Confidence Analysis\n"
            for key, value in metrics.items():
                if key.startswith("confidence_") and isinstance(value, (int, float)):
                    clean_name = key.replace("confidence_", "").replace("_", " ").title()
                    confidence_section += f"- {clean_name}: {value:.4f}\n"
            
            # Abstention metrics
            abstention_section = "\n## Abstention Analysis\n"
            for key, value in metrics.items():
                if key.startswith("abstention_") and isinstance(value, (int, float)):
                    clean_name = key.replace("abstention_", "").replace("_", " ").title()
                    abstention_section += f"- {clean_name}: {value:.4f}\n"
            
            # Risk metrics
            risk_section = "\n## Risk-Aware Analysis\n"
            for key, value in metrics.items():
                if key.startswith("risk_") and isinstance(value, (int, float)):
                    clean_name = key.replace("risk_", "").replace("_", " ").title()
                    risk_section += f"- {clean_name}: {value:.4f}\n"
            
            # Conformal prediction metrics
            conformal_section = "\n## Conformal Prediction Analysis\n"
            for key, value in metrics.items():
                if key.startswith("conformal_") and isinstance(value, (int, float)):
                    clean_name = key.replace("conformal_", "").replace("_", " ").title()
                    conformal_section += f"- {clean_name}: {value:.4f}\n"
            
            report += calibration_section + confidence_section + abstention_section + risk_section + conformal_section

        with open(output_dir / "evaluation_report.txt", "w") as f:
            f.write(report)


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM model")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--test-data", required=True, help="Test dataset path")
    parser.add_argument("--config", default="configs/llm_lora.yaml", help="Config file path")
    parser.add_argument("--output", default="artifacts/evaluation", help="Output directory")

    args = parser.parse_args()

    # Load test dataset
    if args.test_data.endswith(".json"):
        with open(args.test_data) as f:
            test_data = json.load(f)
        test_dataset = Dataset.from_list(test_data)
    else:
        test_dataset = Dataset.load_from_disk(args.test_data)

    # Initialize evaluator
    evaluator = LLMEvaluator(args.model_path, args.config)

    # Run evaluation
    metrics = evaluator.evaluate_dataset(test_dataset, args.output)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.3f}")
    print(f"Abstention Rate: {metrics['abstention_rate']:.3f}")
    print(f"JSON Compliance: {metrics['json_compliance_rate']:.3f}")
    print(f"Coverage: {metrics['coverage']:.3f}")
    print("=" * 50)


# Additional helper functions expected by tests

def load_test_data(file_path: str) -> List[Dict]:
    """
    Load test data from file.
    
    Args:
        file_path: Path to test data file
        
    Returns:
        List of test data samples
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return data


def compute_metrics(
    predictions: List[str], 
    labels: List[str], 
    confidences: Optional[List[float]] = None,
    metrics: Optional[List[str]] = None
) -> Dict:
    """
    Compute evaluation metrics for predictions.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        confidences: Optional list of confidence scores
        metrics: Optional list of specific metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    
    if len(predictions) == 0:
        return {}
    
    # Handle abstentions
    abstention_mask = [pred != "abstain" for pred in predictions]
    non_abstain_preds = [pred for pred, mask in zip(predictions, abstention_mask) if mask]
    non_abstain_labels = [label for label, mask in zip(labels, abstention_mask) if mask]
    
    metrics_dict = {}
    
    # Basic metrics
    if len(non_abstain_preds) > 0:
        metrics_dict['accuracy'] = accuracy_score(non_abstain_labels, non_abstain_preds)
        
        # Get unique labels for multi-class handling
        unique_labels = list(set(non_abstain_labels + non_abstain_preds))
        
        if len(unique_labels) == 2:
            # Binary classification
            metrics_dict['precision'] = precision_score(non_abstain_labels, non_abstain_preds, average='binary', zero_division=0)
            metrics_dict['recall'] = recall_score(non_abstain_labels, non_abstain_preds, average='binary', zero_division=0)
            metrics_dict['f1_score'] = f1_score(non_abstain_labels, non_abstain_preds, average='binary', zero_division=0)
        else:
            # Multi-class
            metrics_dict['macro_precision'] = precision_score(non_abstain_labels, non_abstain_preds, average='macro', zero_division=0)
            metrics_dict['macro_recall'] = recall_score(non_abstain_labels, non_abstain_preds, average='macro', zero_division=0)
            metrics_dict['macro_f1'] = f1_score(non_abstain_labels, non_abstain_preds, average='macro', zero_division=0)
            metrics_dict['precision'] = metrics_dict['macro_precision']
            metrics_dict['recall'] = metrics_dict['macro_recall']
            metrics_dict['f1_score'] = metrics_dict['macro_f1']
    
    # Abstention rate
    abstention_count = sum(1 for pred in predictions if pred == "abstain")
    metrics_dict['abstention_rate'] = abstention_count / len(predictions)
    
    # Confidence metrics
    if confidences:
        valid_confidences = [conf for conf in confidences if conf is not None and conf > 0]
        if valid_confidences:
            metrics_dict['avg_confidence'] = np.mean(valid_confidences)
            metrics_dict['confidence_std'] = np.std(valid_confidences)
            
            # Calibration error (simplified)
            if len(non_abstain_preds) > 0:
                correct_preds = [p == l for p, l in zip(non_abstain_preds, non_abstain_labels)]
                conf_subset = [confidences[i] for i, mask in enumerate(abstention_mask) if mask]
                if len(conf_subset) == len(correct_preds):
                    calibration_error = abs(np.mean(conf_subset) - np.mean(correct_preds))
                    metrics_dict['calibration_error'] = calibration_error
    
    return metrics_dict


def create_visualizations(
    predictions: List[str], 
    labels: List[str], 
    confidences: Optional[List[float]] = None,
    save_dir: Optional[str] = None,
    plots: Optional[List[str]] = None
):
    """
    Create visualizations for evaluation results.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        confidences: Optional list of confidence scores
        save_dir: Optional directory to save plots
        plots: Optional list of specific plots to create
    """
    if len(predictions) == 0 or len(labels) == 0:
        return
    
    # Filter out abstentions for confusion matrix
    abstention_mask = [pred != "abstain" for pred in predictions]
    non_abstain_preds = [pred for pred, mask in zip(predictions, abstention_mask) if mask]
    non_abstain_labels = [label for label, mask in zip(labels, abstention_mask) if mask]
    
    # Create confusion matrix if we have non-abstain predictions
    if len(non_abstain_preds) > 0 and (plots is None or 'confusion_matrix' in plots):
        try:
            # Get unique labels
            unique_labels = sorted(list(set(non_abstain_labels + non_abstain_preds)))
            cm = confusion_matrix(non_abstain_labels, non_abstain_preds, labels=unique_labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_dir:
                plt.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create confusion matrix: {e}")
    
    # Create confidence histogram if available
    if confidences and (plots is None or 'confidence_histogram' in plots):
        try:
            valid_confidences = [c for c in confidences if c is not None and c > 0]
            if valid_confidences:
                plt.figure(figsize=(10, 6))
                plt.hist(valid_confidences, bins=20, alpha=0.7, edgecolor='black')
                plt.title('Confidence Score Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Frequency')
                
                if save_dir:
                    plt.savefig(Path(save_dir) / 'confidence_histogram.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Could not create confidence histogram: {e}")
    
    # Create calibration plot if available
    if confidences and len(non_abstain_preds) > 0 and (plots is None or 'calibration_plot' in plots):
        try:
            correct_preds = [p == l for p, l in zip(non_abstain_preds, non_abstain_labels)]
            conf_subset = [confidences[i] for i, mask in enumerate(abstention_mask) if mask]
            
            if len(conf_subset) == len(correct_preds):
                # Bin by confidence
                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                bin_confidences = []
                bin_accuracies = []
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in conf_subset]
                    if any(in_bin):
                        bin_confidence = np.mean([conf_subset[i] for i, in_b in enumerate(in_bin) if in_b])
                        bin_accuracy = np.mean([correct_preds[i] for i, in_b in enumerate(in_bin) if in_b])
                        bin_confidences.append(bin_confidence)
                        bin_accuracies.append(bin_accuracy)
                
                if bin_confidences and bin_accuracies:
                    plt.figure(figsize=(8, 8))
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                    plt.plot(bin_confidences, bin_accuracies, 'ro-', label='Model Calibration')
                    plt.xlabel('Confidence')
                    plt.ylabel('Accuracy')
                    plt.title('Calibration Plot')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    if save_dir:
                        plt.savefig(Path(save_dir) / 'calibration_plot.png', dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Could not create calibration plot: {e}")


def generate_report(
    predictions: List[str], 
    labels: List[str], 
    confidences: Optional[List[float]] = None,
    metadata: Optional[Dict] = None,
    errors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    include_sections: Optional[List[str]] = None
) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        confidences: Optional list of confidence scores
        metadata: Optional metadata about the evaluation
        errors: Optional list of errors encountered
        save_path: Optional path to save the report
        include_sections: Optional list of sections to include
        
    Returns:
        Report dictionary
    """
    # Compute metrics
    metrics = compute_metrics(predictions, labels, confidences)
    
    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_predictions': len(predictions),
            'accuracy': metrics.get('accuracy', 0.0),
            'abstention_rate': metrics.get('abstention_rate', 0.0)
        },
        'metrics': metrics
    }
    
    # Add confidence statistics if available
    if confidences:
        valid_confidences = [c for c in confidences if c is not None and c > 0]
        if valid_confidences:
            report['confidence_stats'] = {
                'avg_confidence': np.mean(valid_confidences),
                'std_confidence': np.std(valid_confidences),
                'min_confidence': np.min(valid_confidences),
                'max_confidence': np.max(valid_confidences)
            }
    
    # Add metadata if provided
    if metadata:
        report['metadata'] = metadata
    
    # Add error statistics if provided
    if errors:
        error_count = sum(1 for error in errors if error is not None)
        report['error_stats'] = {
            'error_count': error_count,
            'error_rate': error_count / len(predictions) if predictions else 0.0,
            'error_types': list(set(error for error in errors if error is not None))
        }
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {save_path}")
    
    return report


if __name__ == "__main__":
    main()
