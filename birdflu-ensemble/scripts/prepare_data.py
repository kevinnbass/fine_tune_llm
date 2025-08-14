#!/usr/bin/env python3
"""
Enhanced data preparation script with augmentation support.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import random
import logging
from sklearn.model_selection import train_test_split

# Data augmentation imports
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logging.warning("nlpaug not available. Data augmentation disabled.")

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Data augmentation for text classification."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize augmenter with configuration."""
        self.config = config
        self.augmenters = []

        if not AUGMENTATION_AVAILABLE:
            logger.warning("Data augmentation not available. Install nlpaug package.")
            return

        # Initialize augmenters based on config
        methods = config.get("methods", [])
        aug_p = config.get("aug_p", 0.1)

        if "synonym_replacement" in methods:
            self.augmenters.append(naw.SynonymAug(aug_p=aug_p, aug_min=1, aug_max=3))

        if "random_insertion" in methods:
            self.augmenters.append(naw.RandomWordAug(action="insert", aug_p=aug_p))

        if "random_swap" in methods:
            self.augmenters.append(naw.RandomWordAug(action="swap", aug_p=aug_p))

        if "random_deletion" in methods:
            self.augmenters.append(naw.RandomWordAug(action="delete", aug_p=aug_p))

        if "contextual_embedding" in methods:
            # Requires model download, use carefully
            try:
                self.augmenters.append(
                    naw.ContextualWordEmbsAug(
                        model_path="distilbert-base-uncased", action="substitute", aug_p=aug_p
                    )
                )
            except Exception as e:
                logger.warning(f"Could not initialize contextual augmenter: {e}")

        logger.info(f"Initialized {len(self.augmenters)} augmenters")

    def augment_text(self, text: str) -> List[str]:
        """
        Augment a single text.

        Args:
            text: Input text

        Returns:
            List of augmented texts (including original)
        """
        if not self.augmenters or not AUGMENTATION_AVAILABLE:
            return [text]

        augmented_texts = [text]  # Always include original

        for augmenter in self.augmenters:
            try:
                aug_text = augmenter.augment(text)
                if isinstance(aug_text, list):
                    augmented_texts.extend(aug_text)
                else:
                    augmented_texts.append(aug_text)
            except Exception as e:
                logger.warning(f"Augmentation failed for text: {e}")
                continue

        # Remove duplicates and empty strings
        unique_texts = []
        for aug_text in augmented_texts:
            if aug_text and aug_text.strip() and aug_text not in unique_texts:
                unique_texts.append(aug_text)

        return unique_texts

    def augment_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Augment entire dataset.

        Args:
            dataset: List of examples

        Returns:
            Augmented dataset
        """
        if not self.config.get("enabled", False):
            return dataset

        logger.info(f"Augmenting dataset of {len(dataset)} examples")

        augmented_dataset = []

        for example in dataset:
            original_text = example.get("text", example.get("input", ""))

            # Get augmented versions
            augmented_texts = self.augment_text(original_text)

            # Create examples for each augmented text
            for aug_text in augmented_texts:
                aug_example = example.copy()

                # Update text field
                if "text" in example:
                    aug_example["text"] = aug_text
                elif "input" in example:
                    aug_example["input"] = aug_text

                # Mark as augmented (except for original)
                if aug_text != original_text:
                    aug_example["augmented"] = True

                augmented_dataset.append(aug_example)

        logger.info(f"Created {len(augmented_dataset)} examples after augmentation")

        return augmented_dataset


class EnhancedDataPreparator:
    """Enhanced data preparation with augmentation and validation."""

    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        """Initialize preparator."""
        self.config_path = config_path
        self.load_config()

        # Initialize augmenter
        aug_config = self.config.get("data", {}).get("augmentation", {})
        self.augmenter = DataAugmenter(aug_config)

    def load_config(self):
        """Load configuration."""
        if Path(self.config_path).exists():
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            self.config = {
                "data": {
                    "augmentation": {
                        "enabled": False,
                        "methods": ["synonym_replacement"],
                        "aug_p": 0.1,
                    }
                },
                "evaluation": {"enabled": True, "val_split": 0.1},
            }

    def prepare_bird_flu_data(
        self, input_path: str, output_dir: str, val_split: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Prepare bird flu classification data with augmentation and splitting.

        Args:
            input_path: Path to raw data file
            output_dir: Directory to save processed data
            val_split: Validation split ratio (if None, uses config)

        Returns:
            Dictionary with paths to created files
        """
        logger.info(f"Preparing data from {input_path}")

        # Load raw data
        with open(input_path, "r") as f:
            raw_data = json.load(f)

        # Validate input format
        if not isinstance(raw_data, list):
            raise ValueError("Input data must be a list of examples")

        if len(raw_data) == 0:
            raise ValueError("Input data is empty")

        # Check required fields
        first_example = raw_data[0]
        if "text" not in first_example and "input" not in first_example:
            raise ValueError("Examples must have 'text' or 'input' field")

        # Process examples
        processed_data = []

        for i, example in enumerate(raw_data):
            # Format for instruction tuning
            text_field = example.get("text", example.get("input", ""))
            label = example.get("label", example.get("output", "unknown"))

            # Create structured output
            if isinstance(label, str):
                # Simple label to JSON
                output_json = {
                    "decision": label,
                    "rationale": f"This text appears to be classified as '{label}' based on the content analysis.",
                    "abstain": False,
                }
            else:
                # Already structured
                output_json = label

            processed_example = {
                "text": text_field,
                "label": label if isinstance(label, str) else label.get("decision", "unknown"),
                "output": json.dumps(output_json),
                "metadata": example.get("metadata", {}),
                "example_id": i,
            }

            processed_data.append(processed_example)

        logger.info(f"Processed {len(processed_data)} examples")

        # Apply data augmentation
        if self.config.get("data", {}).get("augmentation", {}).get("enabled", False):
            logger.info("Applying data augmentation...")
            processed_data = self.augmenter.augment_dataset(processed_data)

        # Split data if requested
        output_paths = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if val_split is None:
            val_split = self.config.get("evaluation", {}).get("val_split", 0.1)

        if val_split > 0 and self.config.get("evaluation", {}).get("enabled", True):
            # Split into train/val
            train_data, val_data = train_test_split(
                processed_data,
                test_size=val_split,
                random_state=42,
                stratify=[ex["label"] for ex in processed_data],
            )

            # Save train data
            train_path = output_dir / "train.json"
            with open(train_path, "w") as f:
                json.dump(train_data, f, indent=2)
            output_paths["train"] = str(train_path)

            # Save val data
            val_path = output_dir / "val.json"
            with open(val_path, "w") as f:
                json.dump(val_data, f, indent=2)
            output_paths["val"] = str(val_path)

            logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val")

        else:
            # Save all as train data
            train_path = output_dir / "train.json"
            with open(train_path, "w") as f:
                json.dump(processed_data, f, indent=2)
            output_paths["train"] = str(train_path)

        # Save full processed data
        full_path = output_dir / "full_processed.json"
        with open(full_path, "w") as f:
            json.dump(processed_data, f, indent=2)
        output_paths["full"] = str(full_path)

        # Save data statistics
        stats = self.compute_dataset_stats(processed_data)
        stats_path = output_dir / "data_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        output_paths["stats"] = str(stats_path)

        logger.info(f"Data preparation completed. Files saved to {output_dir}")

        return output_paths

    def compute_dataset_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute dataset statistics."""

        # Basic stats
        total_examples = len(dataset)

        # Label distribution
        labels = [ex["label"] for ex in dataset]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Text length stats
        text_lengths = [len(ex["text"].split()) for ex in dataset]

        # Augmentation stats
        augmented_count = sum(1 for ex in dataset if ex.get("augmented", False))

        stats = {
            "total_examples": total_examples,
            "augmented_examples": augmented_count,
            "original_examples": total_examples - augmented_count,
            "label_distribution": label_counts,
            "text_length_stats": {
                "mean": sum(text_lengths) / len(text_lengths),
                "min": min(text_lengths),
                "max": max(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths) // 2],
            },
        }

        return stats

    def validate_processed_data(self, data_path: str) -> bool:
        """Validate processed data format."""

        with open(data_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("Data must be a list")
            return False

        if len(data) == 0:
            logger.error("Data is empty")
            return False

        # Check required fields
        required_fields = ["text", "label", "output"]
        for i, example in enumerate(data[:10]):  # Check first 10
            for field in required_fields:
                if field not in example:
                    logger.error(f"Example {i} missing field: {field}")
                    return False

            # Validate JSON output
            try:
                output = json.loads(example["output"])
                if "decision" not in output:
                    logger.error(f"Example {i} output missing 'decision' field")
                    return False
            except json.JSONDecodeError:
                logger.error(f"Example {i} has invalid JSON output")
                return False

        logger.info("Data validation passed")
        return True


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Enhanced data preparation with augmentation")
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--config", default="configs/llm_lora.yaml", help="Config file")
    parser.add_argument("--val-split", type=float, help="Validation split ratio")
    parser.add_argument("--validate", action="store_true", help="Validate processed data")

    args = parser.parse_args()

    # Initialize preparator
    preparator = EnhancedDataPreparator(args.config)

    # Prepare data
    output_paths = preparator.prepare_bird_flu_data(args.input, args.output, args.val_split)

    # Print results
    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETED")
    print("=" * 50)
    for file_type, path in output_paths.items():
        print(f"{file_type.upper()}: {path}")
    print("=" * 50)

    # Validate if requested
    if args.validate:
        print("\nValidating processed data...")
        for file_type, path in output_paths.items():
            if file_type in ["train", "val", "full"]:
                is_valid = preparator.validate_processed_data(path)
                print(f"{file_type.upper()}: {'✅ Valid' if is_valid else '❌ Invalid'}")


if __name__ == "__main__":
    main()
