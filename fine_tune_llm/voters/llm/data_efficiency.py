"""Data-efficient fine-tuning methods including DEFT, continuous learning, and data quality."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datasets import Dataset
import json
import yaml
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import hashlib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


class DEFTSelector:
    """Data Efficient Fine-Tuning with influence-based sample selection."""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.deft_config = config['deft']
        self.influence_method = self.deft_config['influence_score_method']
        self.data_fraction = self.deft_config['data_fraction']
        self.influence_scores = {}
        
    def compute_influence_scores(self, dataset: Dataset) -> Dict[int, float]:
        """
        Compute influence scores for each data point.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Dictionary mapping data indices to influence scores
        """
        scores = {}
        
        if self.influence_method == 'precision':
            scores = self.compute_precision_influence(dataset)
        elif self.influence_method == 'gradient':
            scores = self.compute_gradient_influence(dataset)
        else:
            scores = self.compute_loss_influence(dataset)
            
        return scores
    
    def compute_precision_influence(self, dataset: Dataset) -> Dict[int, float]:
        """
        Compute influence based on precision impact.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Influence scores based on precision
        """
        scores = {}
        
        # Get baseline precision on validation set
        baseline_precision = self.evaluate_precision(dataset)
        
        for idx in range(len(dataset)):
            # Create dataset without this sample
            indices = list(range(len(dataset)))
            indices.remove(idx)
            subset = dataset.select(indices)
            
            # Measure precision without this sample
            precision_without = self.evaluate_precision(subset)
            
            # Influence is the drop in precision when sample is removed
            influence = baseline_precision - precision_without
            scores[idx] = influence
            
        return scores
    
    def compute_gradient_influence(self, dataset: Dataset) -> Dict[int, float]:
        """
        Compute influence using gradient similarity.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Gradient-based influence scores
        """
        scores = {}
        gradients = []
        
        self.model.train()
        for idx, sample in enumerate(dataset):
            # Compute gradient for this sample
            inputs = self.tokenizer(
                sample['text'],
                return_tensors='pt',
                truncation=True,
                max_length=2048
            )
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            
            # Collect gradients
            sample_grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    sample_grad.append(param.grad.flatten().cpu().numpy())
            
            if sample_grad:
                sample_grad = np.concatenate(sample_grad)
                gradients.append(sample_grad)
                
            # Clear gradients
            self.model.zero_grad()
        
        # Compute influence as gradient magnitude
        for idx, grad in enumerate(gradients):
            scores[idx] = np.linalg.norm(grad)
            
        return scores
    
    def compute_loss_influence(self, dataset: Dataset) -> Dict[int, float]:
        """
        Compute influence based on loss values.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Loss-based influence scores
        """
        scores = {}
        
        self.model.eval()
        with torch.no_grad():
            for idx, sample in enumerate(dataset):
                inputs = self.tokenizer(
                    sample['text'],
                    return_tensors='pt',
                    truncation=True,
                    max_length=2048
                )
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                # Higher loss = more influential (harder examples)
                scores[idx] = outputs.loss.item()
                
        return scores
    
    def evaluate_precision(self, dataset: Dataset) -> float:
        """Evaluate model precision on dataset."""
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for sample in dataset:
                inputs = self.tokenizer(
                    sample['text'],
                    return_tensors='pt',
                    truncation=True,
                    max_length=2048
                )
                
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                if 'label' in sample:
                    label = self.tokenizer(
                        sample['label'],
                        return_tensors='pt'
                    )['input_ids']
                    correct += (preds == label).all().item()
                    total += 1
                    
        return correct / total if total > 0 else 0
    
    def select_data(self, dataset: Dataset) -> Dataset:
        """
        Select most influential data samples.
        
        Args:
            dataset: Full training dataset
            
        Returns:
            Subset of most influential samples
        """
        # Compute influence scores
        scores = self.compute_influence_scores(dataset)
        
        # Sort by influence
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Select top fraction
        num_samples = int(len(dataset) * self.data_fraction)
        selected_indices = sorted_indices[:num_samples]
        
        logger.info(f"Selected {num_samples}/{len(dataset)} most influential samples")
        
        return dataset.select(selected_indices)


class ContinuousLearner:
    """Continuous learning with replay buffer for high-precision examples."""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.cl_config = config['continuous_learning']
        self.buffer_size = self.cl_config['replay_buffer_size']
        self.high_precision_only = self.cl_config['high_precision_examples_only']
        self.update_interval = self.cl_config['incremental_update_interval']
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.seen_hashes = set()  # Avoid duplicates
        
    def add_to_buffer(self, sample: Dict, precision_score: float):
        """
        Add sample to replay buffer if it meets criteria.
        
        Args:
            sample: Data sample
            precision_score: Precision score for this sample
        """
        # Check precision threshold
        if self.high_precision_only and precision_score < 0.95:
            return
            
        # Compute hash to avoid duplicates
        sample_hash = hashlib.md5(
            json.dumps(sample, sort_keys=True).encode()
        ).hexdigest()
        
        if sample_hash not in self.seen_hashes:
            self.replay_buffer.append({
                'data': sample,
                'precision': precision_score,
                'timestamp': len(self.seen_hashes)  # Track order
            })
            self.seen_hashes.add(sample_hash)
            
    def get_replay_batch(self, batch_size: int) -> List[Dict]:
        """
        Get batch from replay buffer for continuous learning.
        
        Args:
            batch_size: Size of replay batch
            
        Returns:
            List of samples from buffer
        """
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
            
        # Prioritize high-precision recent examples
        sorted_buffer = sorted(
            self.replay_buffer,
            key=lambda x: x['precision'] * (1 + x['timestamp'] / 1000),
            reverse=True
        )
        
        return sorted_buffer[:batch_size]
    
    def incremental_update(self, new_data: Dataset):
        """
        Perform incremental update with new data.
        
        Args:
            new_data: New data for incremental learning
        """
        # Combine new data with replay buffer
        replay_samples = self.get_replay_batch(len(new_data) // 2)
        
        combined_data = list(new_data) + [s['data'] for s in replay_samples]
        
        logger.info(f"Incremental update with {len(new_data)} new + {len(replay_samples)} replay samples")
        
        return Dataset.from_list(combined_data)


class DataPurifier:
    """Data purification for safety in high-stakes fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.purification_config = config['data_quality']['purification']
        self.safety_filter = self.purification_config['safety_filter']
        self.quality_threshold = self.purification_config['quality_threshold']
        
        # Initialize safety keywords for filtering
        self.unsafe_keywords = [
            'harmful', 'dangerous', 'illegal', 'unethical',
            'violence', 'hate', 'discrimination'
        ]
        
    def purify_dataset(self, dataset: Dataset) -> Dataset:
        """
        Purify dataset by removing harmful or low-quality samples.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Purified dataset
        """
        purified_samples = []
        removed_count = 0
        
        for sample in dataset:
            if self.is_safe(sample) and self.is_high_quality(sample):
                purified_samples.append(sample)
            else:
                removed_count += 1
                
        logger.info(f"Removed {removed_count}/{len(dataset)} samples during purification")
        
        return Dataset.from_list(purified_samples)
    
    def is_safe(self, sample: Dict) -> bool:
        """
        Check if sample is safe for training.
        
        Args:
            sample: Data sample
            
        Returns:
            True if safe, False otherwise
        """
        if not self.safety_filter:
            return True
            
        text = sample.get('text', '').lower()
        
        # Check for unsafe keywords
        for keyword in self.unsafe_keywords:
            if keyword in text:
                return False
                
        # Additional safety checks could go here
        # (e.g., using a safety classifier)
        
        return True
    
    def is_high_quality(self, sample: Dict) -> bool:
        """
        Check if sample meets quality standards.
        
        Args:
            sample: Data sample
            
        Returns:
            True if high quality, False otherwise
        """
        text = sample.get('text', '')
        
        # Basic quality checks
        if len(text) < 10:  # Too short
            return False
            
        if len(text) > 10000:  # Too long
            return False
            
        # Check for repetition
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.3:  # Too repetitive
            return False
            
        # Could add more sophisticated quality metrics
        # (e.g., perplexity, coherence scores)
        
        return True


class HighStakesPreprocessor:
    """Enhanced preprocessing for high-stakes domains."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessing_config = config['data_quality']['high_stakes_preprocessing']
        self.imbalance_method = self.preprocessing_config['imbalance_handling']
        self.quality_checks = self.preprocessing_config['quality_checks']
        self.precision_focus = self.preprocessing_config['precision_focus']
        
    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for high-stakes training.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Preprocessed dataset
        """
        # Handle class imbalance
        if self.imbalance_method:
            dataset = self.handle_imbalance(dataset)
            
        # Quality checks
        if self.quality_checks:
            dataset = self.apply_quality_checks(dataset)
            
        # Precision-focused augmentation
        if self.precision_focus:
            dataset = self.augment_for_precision(dataset)
            
        return dataset
    
    def handle_imbalance(self, dataset: Dataset) -> Dataset:
        """
        Handle class imbalance in dataset.
        
        Args:
            dataset: Imbalanced dataset
            
        Returns:
            Balanced dataset
        """
        # Extract features and labels
        # This is simplified - actual implementation would depend on data format
        features = []
        labels = []
        
        for sample in dataset:
            # Convert text to features (e.g., embeddings)
            # Simplified: using length as feature
            features.append([len(sample.get('text', ''))])
            labels.append(sample.get('label', 0))
            
        features = np.array(features)
        labels = np.array(labels)
        
        if self.imbalance_method == 'smote':
            # Apply SMOTE oversampling
            smote = SMOTE(random_state=42)
            features_balanced, labels_balanced = smote.fit_resample(features, labels)
        elif self.imbalance_method == 'undersampling':
            # Apply random undersampling
            rus = RandomUnderSampler(random_state=42)
            features_balanced, labels_balanced = rus.fit_resample(features, labels)
        else:
            # Class weight adjustment (return original with weights)
            return dataset
            
        # Reconstruct dataset (simplified)
        balanced_samples = []
        for i in range(len(features_balanced)):
            balanced_samples.append({
                'text': f"Sample with feature {features_balanced[i][0]}",
                'label': int(labels_balanced[i])
            })
            
        return Dataset.from_list(balanced_samples)
    
    def apply_quality_checks(self, dataset: Dataset) -> Dataset:
        """Apply quality checks to dataset."""
        quality_samples = []
        
        for sample in dataset:
            # Check for required fields
            if 'text' not in sample or 'label' not in sample:
                continue
                
            # Check text quality
            text = sample['text']
            if not text or len(text.strip()) < 10:
                continue
                
            # Check label validity
            if sample['label'] not in [0, 1, 'relevant', 'irrelevant']:
                continue
                
            quality_samples.append(sample)
            
        logger.info(f"Quality checks: kept {len(quality_samples)}/{len(dataset)} samples")
        
        return Dataset.from_list(quality_samples)
    
    def augment_for_precision(self, dataset: Dataset) -> Dataset:
        """
        Augment dataset to improve precision.
        
        Args:
            dataset: Original dataset
            
        Returns:
            Augmented dataset with focus on reducing false positives
        """
        augmented_samples = list(dataset)
        
        # Add hard negative examples (to reduce false positives)
        for sample in dataset:
            if sample.get('label') == 0 or sample.get('label') == 'irrelevant':
                # Create variation that's clearly negative
                augmented_sample = sample.copy()
                augmented_sample['text'] = f"NOT RELEVANT: {sample['text']}"
                augmented_samples.append(augmented_sample)
                
        # Add boundary cases for better precision
        boundary_cases = [
            {"text": "This might be related but unclear", "label": 0},
            {"text": "Ambiguous content requiring abstention", "label": 0},
            {"text": "Edge case for classification boundary", "label": 0},
        ]
        augmented_samples.extend(boundary_cases)
        
        logger.info(f"Augmented dataset from {len(dataset)} to {len(augmented_samples)} samples")
        
        return Dataset.from_list(augmented_samples)