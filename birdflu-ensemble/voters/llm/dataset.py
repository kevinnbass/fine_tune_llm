"""Dataset builder for LLM SFT with JSON schema and abstention examples."""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class SFTDatasetBuilder:
    """Build instruction-following dataset for LLM fine-tuning."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        include_abstain_examples: bool = True,
        abstain_ratio: float = 0.1,
    ):
        """
        Initialize dataset builder.

        Args:
            system_prompt: System instruction for the model
            include_abstain_examples: Whether to include abstention examples
            abstain_ratio: Fraction of examples that should abstain
        """
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.include_abstain_examples = include_abstain_examples
        self.abstain_ratio = abstain_ratio

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a specialized classifier for bird flu content.
Analyze the text and return a JSON response with your classification.
You MUST return valid JSON with "decision", "rationale", and "abstain" fields.
If you are uncertain or the text is ambiguous, set "abstain": true.

Valid labels: HIGH_RISK, MEDIUM_RISK, LOW_RISK, NO_RISK"""

    def build_example(
        self,
        text: str,
        label: str,
        metadata: Optional[Dict] = None,
        force_abstain: bool = False,
        abstain_reason: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Build a single training example.

        Args:
            text: Input text to classify
            label: Ground truth label
            metadata: Optional metadata
            force_abstain: Whether this should be an abstention example
            abstain_reason: Reason for abstention

        Returns:
            Dictionary with instruction, input, and output
        """
        # Build input
        input_parts = [f"Text to classify:\n{text}"]

        if metadata:
            meta_str = json.dumps(metadata, indent=2)
            input_parts.append(f"\nMetadata:\n{meta_str}")

        input_text = "\n".join(input_parts)

        # Build output
        if force_abstain:
            output = {
                "decision": None,
                "rationale": abstain_reason or "Content is ambiguous or uncertain",
                "abstain": True,
            }
        else:
            # Generate appropriate rationale based on label
            rationale = self._generate_rationale(text, label)
            output = {"decision": label, "rationale": rationale, "abstain": False}

        return {
            "instruction": self.system_prompt,
            "input": input_text,
            "output": json.dumps(output, indent=2),
        }

    def _generate_rationale(self, text: str, label: str) -> str:
        """Generate rationale for classification."""
        rationales = {
            "HIGH_RISK": [
                "Text contains explicit H5N1 or avian flu outbreak information",
                "Clear evidence of bird flu transmission or pandemic risk",
                "Direct mention of H5N1 virus with concerning context",
            ],
            "MEDIUM_RISK": [
                "Text discusses bird health issues with potential flu indicators",
                "Mentions biosecurity measures related to poultry",
                "Contains indirect references to avian disease concerns",
            ],
            "LOW_RISK": [
                "General flu discussion without specific bird flu indicators",
                "Bird health mentioned without direct flu connection",
                "Minimal risk indicators present",
            ],
            "NO_RISK": [
                "No bird flu or related content detected",
                "Text is unrelated to avian influenza",
                "Content does not indicate any flu-related risk",
            ],
        }

        return random.choice(rationales.get(label, ["Classification based on content analysis"]))

    def create_abstention_examples(self, n_examples: int = 100) -> List[Dict[str, str]]:
        """Create synthetic abstention examples."""
        abstention_templates = [
            {
                "text": "The document appears corrupted with mixed characters: @#$%^&*()_+ [DATA ERROR]",
                "reason": "Text appears corrupted or unreadable",
            },
            {
                "text": "这是一段中文文本，讨论了一些医学内容但不清楚具体主题。",
                "reason": "Non-English text requiring specialized analysis",
            },
            {
                "text": "Something about birds... or maybe flu? The connection is unclear and the context is missing.",
                "reason": "Ambiguous content without clear classification indicators",
            },
            {
                "text": "[REDACTED] mentioned [CLASSIFIED] in relation to [REMOVED]",
                "reason": "Insufficient information due to redacted content",
            },
            {"text": "Very short.", "reason": "Text too brief for reliable classification"},
            {
                "text": "The symptoms might indicate various conditions including but not limited to respiratory issues that could be related to multiple pathogens.",
                "reason": "Medical content too vague for specific classification",
            },
        ]

        examples = []
        for _ in range(n_examples):
            template = random.choice(abstention_templates)
            example = self.build_example(
                text=template["text"],
                label=None,
                force_abstain=True,
                abstain_reason=template["reason"],
            )
            examples.append(example)

        return examples

    def build_dataset(
        self,
        data_path: str,
        weak_labels_path: Optional[str] = None,
        gold_labels_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """
        Build full SFT dataset from various sources.

        Args:
            data_path: Path to main data file
            weak_labels_path: Path to weak supervision labels
            gold_labels_path: Path to gold standard labels
            max_samples: Maximum number of samples

        Returns:
            HuggingFace Dataset object
        """
        examples = []

        # Load main data
        df = pd.read_csv(data_path)
        if max_samples:
            df = df.head(max_samples)

        # Load labels
        if gold_labels_path and Path(gold_labels_path).exists():
            gold_df = pd.read_csv(gold_labels_path)
            df = df.merge(gold_df[["id", "label"]], on="id", how="left", suffixes=("", "_gold"))
            label_col = "label_gold"
        elif weak_labels_path and Path(weak_labels_path).exists():
            weak_df = pd.read_csv(weak_labels_path)
            df = df.merge(weak_df[["id", "label", "confidence"]], on="id", how="left")
            # Filter high-confidence weak labels
            df = df[df["confidence"] > 0.8]
            label_col = "label"
        else:
            raise ValueError("Need either gold or weak labels")

        # Build examples
        for _, row in df.iterrows():
            # Skip if no label
            if pd.isna(row.get(label_col)):
                continue

            # Randomly add abstention examples
            if self.include_abstain_examples and random.random() < self.abstain_ratio:
                # Corrupt the text slightly to create abstention case
                text = row["text"][:50] + " [UNCLEAR CONTINUATION...]"
                example = self.build_example(
                    text=text,
                    label=row[label_col],
                    metadata=row.get("metadata"),
                    force_abstain=True,
                    abstain_reason="Incomplete or unclear text",
                )
            else:
                example = self.build_example(
                    text=row["text"],
                    label=row[label_col],
                    metadata=row.get("metadata"),
                    force_abstain=False,
                )

            examples.append(example)

        # Add synthetic abstention examples
        if self.include_abstain_examples:
            n_abstain = int(len(examples) * self.abstain_ratio)
            abstain_examples = self.create_abstention_examples(n_abstain)
            examples.extend(abstain_examples)

        # Shuffle
        random.shuffle(examples)

        # Convert to HF Dataset
        dataset = Dataset.from_list(examples)

        logger.info(f"Created dataset with {len(dataset)} examples")
        logger.info(f"Abstention examples: {sum(1 for e in examples if 'true' in e['output'])}")

        return dataset

    def format_for_training(self, example: Dict) -> str:
        """
        Format example for training (concatenate all parts).

        Args:
            example: Dataset example

        Returns:
            Formatted string for training
        """
        parts = []

        if example.get("instruction"):
            parts.append(f"### Instruction:\n{example['instruction']}")

        if example.get("input"):
            parts.append(f"### Input:\n{example['input']}")

        if example.get("output"):
            parts.append(f"### Output:\n{example['output']}")

        return "\n\n".join(parts)

    def validate_output(self, output_text: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate model output against schema.

        Args:
            output_text: Raw model output

        Returns:
            Tuple of (is_valid, parsed_output)
        """
        try:
            # Try to parse JSON
            output = json.loads(output_text)

            # Check required fields
            required_fields = {"decision", "rationale", "abstain"}
            if not all(field in output for field in required_fields):
                return False, None

            # Check decision validity
            if not output["abstain"]:
                valid_labels = {"HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"}
                if output["decision"] not in valid_labels:
                    return False, None
            else:
                # For abstention, decision should be None or null
                if output["decision"] is not None:
                    return False, None

            # Check types
            if not isinstance(output["abstain"], bool):
                return False, None

            if not isinstance(output["rationale"], str):
                return False, None

            return True, output

        except (json.JSONDecodeError, KeyError, TypeError):
            return False, None


def create_balanced_dataset(
    data_sources: List[str], output_path: str, samples_per_class: int = 1000
) -> Dataset:
    """
    Create a balanced dataset from multiple sources.

    Args:
        data_sources: List of data file paths
        output_path: Where to save the dataset
        samples_per_class: Number of samples per class

    Returns:
        Balanced dataset
    """
    builder = SFTDatasetBuilder()
    all_examples = []

    class_examples = {"HIGH_RISK": [], "MEDIUM_RISK": [], "LOW_RISK": [], "NO_RISK": []}

    # Collect examples by class
    for source in data_sources:
        df = pd.read_csv(source)

        for _, row in df.iterrows():
            label = row.get("label")
            if label in class_examples:
                example = builder.build_example(
                    text=row["text"], label=label, metadata=row.get("metadata")
                )
                class_examples[label].append(example)

    # Balance classes
    min_samples = min(len(examples) for examples in class_examples.values())
    n_samples = min(min_samples, samples_per_class)

    for label, examples in class_examples.items():
        sampled = random.sample(examples, min(len(examples), n_samples))
        all_examples.extend(sampled)

    # Add abstention examples
    n_abstain = int(len(all_examples) * 0.1)
    abstain_examples = builder.create_abstention_examples(n_abstain)
    all_examples.extend(abstain_examples)

    # Shuffle and create dataset
    random.shuffle(all_examples)
    dataset = Dataset.from_list(all_examples)

    # Save
    dataset.save_to_disk(output_path)
    logger.info(f"Saved balanced dataset to {output_path}")

    return dataset
