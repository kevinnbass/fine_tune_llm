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

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Comprehensive evaluation for fine-tuned LLM models."""

    def __init__(self, model_path: str, config_path: str = "configs/llm_lora.yaml"):
        """
        Initialize evaluator.

        Args:
            model_path: Path to fine-tuned model
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.config_path = config_path
        self.load_config()
        self.load_model()

        # Load evaluation metrics
        self.metrics = {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }

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

    def evaluate_dataset(self, dataset: Dataset, output_path: Optional[str] = None) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Evaluation dataset
            output_path: Optional path to save detailed results

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on {len(dataset)} examples")

        predictions = []
        ground_truth = []
        detailed_results = []

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            # Get prediction
            pred = self.predict_single(
                example.get("text", example.get("input", "")), example.get("metadata", {})
            )

            # Get ground truth
            true_label = example.get("label", example.get("output", "unknown"))

            # Store results
            predictions.append(pred["decision"])
            ground_truth.append(true_label)

            # Store detailed result
            detailed_results.append(
                {
                    "example_id": i,
                    "input_text": example.get("text", example.get("input", "")),
                    "true_label": true_label,
                    "predicted_label": pred["decision"],
                    "abstain": pred["abstain"],
                    "confidence": pred["confidence"],
                    "rationale": pred["rationale"],
                    "valid_json": pred["valid_json"],
                    "raw_output": pred["raw_output"],
                }
            )

        # Compute metrics
        metrics = self.compute_metrics(predictions, ground_truth, detailed_results)

        # Save detailed results if requested
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            with open(output_dir / "detailed_results.json", "w") as f:
                json.dump(detailed_results, f, indent=2)

            # Save metrics
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Create visualizations
            self.create_visualizations(detailed_results, metrics, output_dir)

            logger.info(f"Results saved to {output_dir}")

        return metrics

    def compute_metrics(
        self, predictions: List[str], ground_truth: List[str], detailed_results: List[Dict]
    ) -> Dict:
        """Compute comprehensive evaluation metrics."""

        # Filter out abstentions for core metrics
        non_abstain_indices = [
            i
            for i, result in enumerate(detailed_results)
            if not result["abstain"] and result["predicted_label"] != "parse_error"
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
            1 for result in detailed_results if result["predicted_label"] == "parse_error"
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

        return metrics

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

        # 3. Results summary report
        self.create_text_report(metrics, output_dir)

        logger.info("Visualizations created successfully")

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


if __name__ == "__main__":
    main()
