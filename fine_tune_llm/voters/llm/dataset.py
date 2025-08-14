"""Dataset builder for LLM SFT with JSON schema and abstention examples."""

import json
import random
import yaml
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from datasets import Dataset
import logging
from collections import Counter
import copy

logger = logging.getLogger(__name__)


def apply_augmentation(examples: List[Dict], augmentation_config: Dict = None) -> List[Dict]:
    """Apply data augmentation to examples."""
    if not augmentation_config or not augmentation_config.get('enabled', False):
        return examples
    
    # Simple augmentation - could be enhanced with nlpaug or other libraries
    augmented = []
    for example in examples:
        augmented.append(example)  # Original
        
        # Simple paraphrasing augmentation
        if random.random() < 0.3:  # 30% chance
            aug_example = copy.deepcopy(example)
            # Simple word substitutions
            text = aug_example.get('text', '')
            if 'bird flu' in text.lower():
                text = text.replace('bird flu', 'avian influenza')
            elif 'avian influenza' in text.lower():
                text = text.replace('avian influenza', 'bird flu')
            aug_example['text'] = text
            augmented.append(aug_example)
    
    return augmented


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


# Additional helper functions expected by tests

def load_labels(labels_path: str) -> Dict[str, str]:
    """
    Load label definitions from YAML file.
    
    Args:
        labels_path: Path to labels YAML file
        
    Returns:
        Dictionary mapping label names to descriptions
    """
    with open(labels_path, 'r') as f:
        labels = yaml.safe_load(f)
    return labels


def build_examples(
    raw_data: List[Dict], 
    labels: Dict[str, str], 
    config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Build training examples from raw data.
    
    Args:
        raw_data: List of raw data samples
        labels: Label definitions
        config: Configuration dictionary
        
    Returns:
        List of formatted examples
    """
    examples = []
    
    for sample in raw_data:
        text = sample['text']
        label = sample['label']
        metadata = sample.get('metadata', {})
        
        # Check if label is valid
        if label not in labels:
            raise KeyError(f"Unknown label: {label}")
        
        # Build input according to template
        input_template = config['instruction_format']['input_template']
        input_text = input_template.format(text=text, metadata=json.dumps(metadata) if metadata else '{}')
        
        # Build output
        output_template = config['instruction_format']['output_template']
        
        output_data = {
            'decision': label,
            'abstain': False
        }
        
        # Add rationale if requested
        if config['instruction_format'].get('add_rationale', False):
            output_data['rationale'] = add_rationale(sample, labels)
        else:
            output_data['rationale'] = f"This content is classified as {label} based on the analysis."
        
        # Add confidence if specified in template
        if 'confidence' in output_template:
            output_data['confidence'] = 0.85  # Default confidence
        
        example = {
            'input': input_text,
            'output': json.dumps(output_data)
        }
        
        examples.append(example)
    
    return examples


def create_abstention_examples(labels: Dict[str, str], config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Create abstention examples for training.
    
    Args:
        labels: Label definitions
        config: Configuration dictionary
        
    Returns:
        List of abstention examples
    """
    if not config['abstention']['enabled']:
        return []
    
    examples = []
    examples_per_label = config['abstention']['examples_per_label']
    uncertainty_phrases = config['abstention']['uncertainty_phrases']
    
    for label in labels.keys():
        for _ in range(examples_per_label):
            # Create uncertain text with uncertainty phrases
            uncertainty_phrase = random.choice(uncertainty_phrases)
            text = f"{uncertainty_phrase} about the content. The information seems ambiguous."
            
            # Build input
            input_template = config['instruction_format']['input_template']
            input_text = input_template.format(text=text, metadata='{}')
            
            # Build abstention output
            output_data = {
                'decision': None,
                'rationale': f"Unable to confidently classify due to ambiguous content. {uncertainty_phrase}.",
                'abstain': True
            }
            
            if 'confidence' in config['instruction_format']['output_template']:
                output_data['confidence'] = 0.0
            
            example = {
                'input': input_text,
                'output': json.dumps(output_data)
            }
            
            examples.append(example)
    
    return examples


def add_rationale(example: Dict[str, Any], labels: Dict[str, str]) -> str:
    """
    Generate rationale for a classification decision.
    
    Args:
        example: Data example with text and label
        labels: Label definitions
        
    Returns:
        Generated rationale string
    """
    text = example['text']
    label = example['label']
    
    # Check if label exists in definitions
    if label not in labels:
        return f"Classification decision based on content analysis."
    
    # Generate rationale based on label and content
    rationale_templates = {
        'relevant': [
            f"This content is relevant because it directly relates to the topic of interest.",
            f"The text contains key indicators that make it relevant to the classification task.",
            f"Based on the content analysis, this text contains relevant information."
        ],
        'irrelevant': [
            f"This content is irrelevant as it does not relate to the topic of interest.",
            f"The text lacks key indicators that would make it relevant to the classification task.", 
            f"Based on content analysis, this text does not contain relevant information."
        ],
        'uncertain': [
            f"The content presents ambiguous information that makes classification uncertain.",
            f"There are conflicting indicators in the text that make definitive classification difficult.",
            f"The content lacks clear indicators for confident classification."
        ]
    }
    
    # Use label-specific templates if available, otherwise generic
    templates = rationale_templates.get(label, [
        f"Content classified as {label} based on analysis of key features.",
        f"The text exhibits characteristics consistent with the {label} category."
    ])
    
    base_rationale = random.choice(templates)
    
    # Add metadata considerations if present
    if 'metadata' in example and example['metadata']:
        metadata = example['metadata']
        if 'confidence' in metadata:
            confidence_level = metadata['confidence']
            base_rationale += f" The source confidence level is {confidence_level}."
        if 'source' in metadata:
            source = metadata['source']
            base_rationale += f" This information comes from a {source} source."
    
    return base_rationale


def validate_output_format(output_text: str) -> Tuple[bool, Optional[Dict]]:
    """
    Validate that output text matches expected JSON format.
    
    Args:
        output_text: Raw output text to validate
        
    Returns:
        Tuple of (is_valid, parsed_output)
    """
    if not output_text or output_text.strip() == "":
        return False, None
        
    if output_text is None:
        return False, None
    
    try:
        # Parse JSON
        parsed = json.loads(output_text)
        
        # Check for required fields
        required_fields = {'decision', 'rationale', 'abstain'}
        if not all(field in parsed for field in required_fields):
            return False, None
        
        # Validate field types
        if not isinstance(parsed['abstain'], bool):
            return False, None
        
        if not isinstance(parsed['rationale'], str):
            return False, None
        
        # If not abstaining, decision should be present and not None
        if not parsed['abstain'] and parsed['decision'] is None:
            return False, None
        
        # If abstaining, decision should be None
        if parsed['abstain'] and parsed['decision'] is not None:
            return False, None
            
        return True, parsed
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, None


def create_balanced_dataset(
    raw_data: List[Dict[str, Any]], 
    target_samples_per_class: int
) -> List[Dict[str, Any]]:
    """
    Create a balanced dataset with equal samples per class.
    
    Args:
        raw_data: List of raw data samples
        target_samples_per_class: Target number of samples per class
        
    Returns:
        Balanced dataset
    """
    if not raw_data:
        return []
    
    # Group by label - handle both raw data and processed examples
    label_groups = {}
    for sample in raw_data:
        # Extract label from different formats
        if 'label' in sample:
            label = sample['label']
        elif 'output' in sample:
            # For processed examples, extract from output JSON
            try:
                output_data = json.loads(sample['output'])
                label = output_data.get('decision', 'unknown')
            except (json.JSONDecodeError, KeyError):
                label = 'unknown'
        else:
            label = 'unknown'
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(sample)
    
    balanced_data = []
    
    for label, samples in label_groups.items():
        if len(samples) >= target_samples_per_class:
            # Undersample
            selected_samples = random.sample(samples, target_samples_per_class)
        else:
            # Oversample with repetition
            selected_samples = []
            for _ in range(target_samples_per_class):
                selected_samples.append(random.choice(samples))
        
        balanced_data.extend(selected_samples)
    
    # Shuffle the final dataset
    random.shuffle(balanced_data)
    
    return balanced_data
