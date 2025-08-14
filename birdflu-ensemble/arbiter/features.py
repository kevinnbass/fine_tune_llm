"""Feature engineering for arbiter stacker."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import re
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class ArbiterFeatureEngineering:
    """Extract features from voter outputs for stacker."""
    
    def __init__(self, slice_definitions: Optional[Dict] = None):
        """
        Initialize feature engineering.
        
        Args:
            slice_definitions: Dictionary of slice definitions
        """
        self.slice_definitions = slice_definitions or {}
        self.feature_names = None
        
    def extract_features(
        self,
        voter_outputs: Dict[str, Dict],
        text: str,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract features from voter outputs and text.
        
        Args:
            voter_outputs: Dict mapping voter_id to output dict
            text: Original input text
            metadata: Optional metadata
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        feature_names = []
        
        # 1. Per-voter probability features
        for voter_id in sorted(voter_outputs.keys()):
            output = voter_outputs[voter_id]
            
            if output.get('abstain', False):
                # Abstained - use special values
                features.extend([0.0, 0.0, 0.0, 0.0])  # 4 class probs
                features.append(1.0)  # abstain flag
                features.append(0.0)  # max prob
                features.append(0.0)  # entropy
                
                feature_names.extend([
                    f'{voter_id}_prob_HIGH_RISK',
                    f'{voter_id}_prob_MEDIUM_RISK',
                    f'{voter_id}_prob_LOW_RISK',
                    f'{voter_id}_prob_NO_RISK',
                    f'{voter_id}_abstained',
                    f'{voter_id}_max_prob',
                    f'{voter_id}_entropy'
                ])
            else:
                probs = output.get('probs', {})
                
                # Class probabilities
                for label in ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']:
                    features.append(probs.get(label, 0.0))
                    feature_names.append(f'{voter_id}_prob_{label}')
                
                # Abstain flag
                features.append(0.0)
                feature_names.append(f'{voter_id}_abstained')
                
                # Max probability
                max_prob = max(probs.values()) if probs else 0.0
                features.append(max_prob)
                feature_names.append(f'{voter_id}_max_prob')
                
                # Entropy
                prob_values = list(probs.values())
                ent = entropy(prob_values) if prob_values else 0.0
                features.append(ent)
                feature_names.append(f'{voter_id}_entropy')
        
        # 2. Aggregate features across voters
        all_probs = {
            'HIGH_RISK': [],
            'MEDIUM_RISK': [],
            'LOW_RISK': [],
            'NO_RISK': []
        }
        
        max_probs = []
        abstain_count = 0
        
        for output in voter_outputs.values():
            if output.get('abstain', False):
                abstain_count += 1
            else:
                probs = output.get('probs', {})
                for label in all_probs:
                    all_probs[label].append(probs.get(label, 0.0))
                
                if probs:
                    max_probs.append(max(probs.values()))
        
        # Mean probability per class
        for label in all_probs:
            if all_probs[label]:
                features.append(np.mean(all_probs[label]))
            else:
                features.append(0.0)
            feature_names.append(f'mean_prob_{label}')
        
        # Std probability per class
        for label in all_probs:
            if len(all_probs[label]) > 1:
                features.append(np.std(all_probs[label]))
            else:
                features.append(0.0)
            feature_names.append(f'std_prob_{label}')
        
        # Agreement features
        if max_probs:
            # Overall max probability
            features.append(max(max_probs))
            feature_names.append('overall_max_prob')
            
            # Mean max probability
            features.append(np.mean(max_probs))
            feature_names.append('mean_max_prob')
            
            # Margin (difference between top 2 classes)
            mean_probs = [np.mean(all_probs[label]) if all_probs[label] else 0.0 
                         for label in all_probs]
            sorted_probs = sorted(mean_probs, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            features.append(margin)
            feature_names.append('margin')
        else:
            features.extend([0.0, 0.0, 0.0])
            feature_names.extend(['overall_max_prob', 'mean_max_prob', 'margin'])
        
        # Abstention rate
        abstain_rate = abstain_count / len(voter_outputs) if voter_outputs else 0.0
        features.append(abstain_rate)
        feature_names.append('abstain_rate')
        
        # 3. Text features
        text_features = self._extract_text_features(text)
        features.extend(text_features)
        feature_names.extend([
            'text_length',
            'word_count',
            'caps_ratio',
            'special_char_ratio',
            'has_negation',
            'has_medical_terms',
            'has_h5n1_mention'
        ])
        
        # 4. Slice features
        slice_features = self._extract_slice_features(text, metadata)
        features.extend(slice_features)
        for slice_name in self.slice_definitions:
            feature_names.append(f'slice_{slice_name}')
        
        # 5. Voter disagreement features
        disagreement_features = self._compute_disagreement_features(voter_outputs)
        features.extend(disagreement_features)
        feature_names.extend([
            'voter_entropy',
            'pairwise_disagreement',
            'has_split_decision'
        ])
        
        # Store feature names for reference
        if self.feature_names is None:
            self.feature_names = feature_names
        
        return np.array(features)
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract features from text."""
        features = []
        
        # Length features
        features.append(len(text))
        features.append(len(text.split()))
        
        # Character ratios
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features.append(caps_ratio)
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)
        features.append(special_ratio)
        
        # Content indicators
        negation_pattern = r'\b(not?|no|never|neither|none)\b'
        has_negation = 1.0 if re.search(negation_pattern, text, re.IGNORECASE) else 0.0
        features.append(has_negation)
        
        medical_pattern = r'\b(patient|symptom|treatment|diagnosis|clinical|virus|infection)\b'
        has_medical = 1.0 if re.search(medical_pattern, text, re.IGNORECASE) else 0.0
        features.append(has_medical)
        
        h5n1_pattern = r'\b[Hh]5[Nn]1\b'
        has_h5n1 = 1.0 if re.search(h5n1_pattern, text) else 0.0
        features.append(has_h5n1)
        
        return features
    
    def _extract_slice_features(self, text: str, metadata: Optional[Dict]) -> List[float]:
        """Extract slice membership features."""
        features = []
        
        for slice_name, slice_def in self.slice_definitions.items():
            is_member = self._check_slice_membership(text, metadata, slice_def)
            features.append(1.0 if is_member else 0.0)
        
        # If no slice definitions, return empty
        return features
    
    def _check_slice_membership(
        self,
        text: str,
        metadata: Optional[Dict],
        slice_def: Dict
    ) -> bool:
        """Check if text belongs to a slice."""
        condition = slice_def.get('condition', {})
        cond_type = condition.get('type')
        
        if cond_type == 'length':
            word_count = len(text.split())
            min_tokens = condition.get('min_tokens', 0)
            max_tokens = condition.get('max_tokens', float('inf'))
            return min_tokens <= word_count <= max_tokens
        
        elif cond_type == 'pattern':
            pattern = condition.get('regex')
            if pattern:
                flags = re.IGNORECASE if condition.get('case_insensitive') else 0
                matches = re.findall(pattern, text, flags)
                min_matches = condition.get('min_matches', 1)
                return len(matches) >= min_matches
        
        elif cond_type == 'metadata' and metadata:
            field = condition.get('field')
            values = condition.get('values', [])
            return metadata.get(field) in values
        
        elif cond_type == 'keyword_density':
            keywords = condition.get('keywords', [])
            min_density = condition.get('min_density', 0.0)
            
            word_count = len(text.split())
            if word_count == 0:
                return False
            
            keyword_count = sum(
                1 for word in text.lower().split()
                if any(kw in word for kw in keywords)
            )
            density = keyword_count / word_count
            return density >= min_density
        
        return False
    
    def _compute_disagreement_features(self, voter_outputs: Dict[str, Dict]) -> List[float]:
        """Compute voter disagreement features."""
        features = []
        
        # Get predictions from non-abstaining voters
        predictions = []
        for output in voter_outputs.values():
            if not output.get('abstain', False):
                probs = output.get('probs', {})
                if probs:
                    # Get argmax prediction
                    pred = max(probs, key=probs.get)
                    predictions.append(pred)
        
        if not predictions:
            return [0.0, 0.0, 0.0]
        
        # Voter entropy (disagreement)
        from collections import Counter
        pred_counts = Counter(predictions)
        pred_probs = [count/len(predictions) for count in pred_counts.values()]
        voter_entropy = entropy(pred_probs)
        features.append(voter_entropy)
        
        # Pairwise disagreement rate
        if len(predictions) > 1:
            disagreements = sum(
                1 for i in range(len(predictions))
                for j in range(i+1, len(predictions))
                if predictions[i] != predictions[j]
            )
            total_pairs = len(predictions) * (len(predictions) - 1) / 2
            disagreement_rate = disagreements / total_pairs
        else:
            disagreement_rate = 0.0
        features.append(disagreement_rate)
        
        # Has split decision (no majority)
        max_count = max(pred_counts.values())
        has_split = 1.0 if max_count <= len(predictions) / 2 else 0.0
        features.append(has_split)
        
        return features
    
    def create_feature_matrix(
        self,
        voter_outputs_list: List[Dict[str, Dict]],
        texts: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix for multiple samples.
        
        Args:
            voter_outputs_list: List of voter output dicts
            texts: List of input texts
            metadata_list: Optional list of metadata dicts
            
        Returns:
            DataFrame with features
        """
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        
        feature_rows = []
        for voter_outputs, text, metadata in zip(voter_outputs_list, texts, metadata_list):
            features = self.extract_features(voter_outputs, text, metadata)
            feature_rows.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(feature_rows, columns=self.feature_names)
        
        return df
    
    def get_feature_importance(
        self,
        model,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ or coef_
            feature_names: Optional custom feature names
            
        Returns:
            DataFrame with feature importance
        """
        if feature_names is None:
            feature_names = self.feature_names or []
        
        # Get importances based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear model
            importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df