"""Inference-time weak supervision label model voter."""

import numpy as np
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LabelModelVoter:
    """
    Probabilistic label model for weak supervision.
    Combines labeling function votes into probabilistic predictions.
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        abstain_value: int = -1,
        min_evidence: int = 2,
        abstain_threshold: float = 0.65
    ):
        """
        Initialize label model voter.
        
        Args:
            n_classes: Number of classes
            abstain_value: Value used for abstention in LF votes
            min_evidence: Minimum non-abstain votes required
            abstain_threshold: Max-prob threshold for abstention
        """
        self.n_classes = n_classes
        self.abstain_value = abstain_value
        self.min_evidence = min_evidence
        self.abstain_threshold = abstain_threshold
        
        # These will be learned from training data
        self.lf_accuracies = None
        self.class_balance = None
        self.conflict_resolution = None
        
    def fit(
        self,
        lf_votes: np.ndarray,
        labels: Optional[np.ndarray] = None,
        lf_names: Optional[List[str]] = None
    ):
        """
        Learn LF accuracies and class balance from data.
        
        Args:
            lf_votes: Matrix of LF votes (n_samples, n_lfs)
            labels: Optional gold labels for some samples
            lf_names: Optional names for LFs
        """
        n_samples, n_lfs = lf_votes.shape
        self.n_lfs = n_lfs
        self.lf_names = lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        # Estimate LF accuracies using agreement rates
        self.lf_accuracies = np.zeros(n_lfs)
        self.lf_coverage = np.zeros(n_lfs)
        
        for j in range(n_lfs):
            # Coverage: fraction of non-abstain votes
            coverage_mask = lf_votes[:, j] != self.abstain_value
            self.lf_coverage[j] = coverage_mask.mean()
            
            if labels is not None:
                # If we have gold labels, use them
                correct = (lf_votes[:, j] == labels) & coverage_mask
                self.lf_accuracies[j] = correct.sum() / max(coverage_mask.sum(), 1)
            else:
                # Otherwise, use agreement with majority vote as proxy
                majority_votes = self._get_majority_vote(lf_votes)
                agree_mask = (lf_votes[:, j] == majority_votes) & coverage_mask
                self.lf_accuracies[j] = agree_mask.sum() / max(coverage_mask.sum(), 1)
        
        # Smooth accuracies to avoid 0/1 values
        self.lf_accuracies = np.clip(self.lf_accuracies, 0.1, 0.9)
        
        # Estimate class balance
        if labels is not None:
            self.class_balance = np.bincount(labels, minlength=self.n_classes)
            self.class_balance = self.class_balance / self.class_balance.sum()
        else:
            # Use LF votes to estimate
            all_votes = lf_votes[lf_votes != self.abstain_value]
            if len(all_votes) > 0:
                counts = np.bincount(all_votes, minlength=self.n_classes)
                self.class_balance = counts / counts.sum()
            else:
                self.class_balance = np.ones(self.n_classes) / self.n_classes
        
        # Learn conflict resolution weights (which LFs to trust more in conflicts)
        self.conflict_resolution = self.lf_accuracies * self.lf_coverage
        self.conflict_resolution = self.conflict_resolution / self.conflict_resolution.sum()
        
        logger.info(f"Fitted label model with {n_lfs} LFs")
        logger.info(f"LF accuracies: {self.lf_accuracies}")
        logger.info(f"Class balance: {self.class_balance}")
    
    def predict_proba(self, lf_votes_row: np.ndarray) -> Dict[str, Any]:
        """
        Predict probabilities for a single sample.
        
        Args:
            lf_votes_row: LF votes for one sample (n_lfs,)
            
        Returns:
            Dictionary with probs and abstain flag
        """
        if self.lf_accuracies is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Count evidence
        non_abstain_mask = lf_votes_row != self.abstain_value
        n_evidence = non_abstain_mask.sum()
        
        # Check minimum evidence
        if n_evidence < self.min_evidence:
            return {
                'probs': {},
                'abstain': True,
                'reason': f"Insufficient evidence: {n_evidence} < {self.min_evidence}"
            }
        
        # Weighted probabilistic aggregation
        probs = np.zeros(self.n_classes)
        
        for j, vote in enumerate(lf_votes_row):
            if vote == self.abstain_value:
                continue
            
            # Weight by accuracy and coverage
            weight = self.lf_accuracies[j] * self.conflict_resolution[j]
            
            # Probabilistic vote
            for c in range(self.n_classes):
                if c == vote:
                    # LF votes for this class
                    probs[c] += weight * self.lf_accuracies[j]
                else:
                    # LF doesn't vote for this class
                    probs[c] += weight * (1 - self.lf_accuracies[j]) / (self.n_classes - 1)
        
        # Incorporate class balance as prior
        probs = probs * self.class_balance
        
        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = self.class_balance.copy()
        
        # Check abstention threshold
        max_prob = probs.max()
        if max_prob < self.abstain_threshold:
            return {
                'probs': {},
                'abstain': True,
                'reason': f"Low confidence: {max_prob:.3f} < {self.abstain_threshold}"
            }
        
        # Build result
        class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        probs_dict = {class_names[i]: float(probs[i]) for i in range(self.n_classes)}
        
        return {
            'probs': probs_dict,
            'abstain': False,
            'n_evidence': int(n_evidence),
            'max_prob': float(max_prob)
        }
    
    def _get_majority_vote(self, lf_votes: np.ndarray) -> np.ndarray:
        """Get majority vote for each sample."""
        n_samples = lf_votes.shape[0]
        majority = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = lf_votes[i, lf_votes[i] != self.abstain_value]
            if len(votes) > 0:
                counts = np.bincount(votes, minlength=self.n_classes)
                majority[i] = counts.argmax()
            else:
                majority[i] = self.abstain_value
        
        return majority
    
    def save(self, path: str):
        """Save model to disk."""
        model_dict = {
            'n_classes': self.n_classes,
            'abstain_value': self.abstain_value,
            'min_evidence': self.min_evidence,
            'abstain_threshold': self.abstain_threshold,
            'lf_accuracies': self.lf_accuracies,
            'class_balance': self.class_balance,
            'conflict_resolution': self.conflict_resolution,
            'lf_coverage': self.lf_coverage,
            'lf_names': self.lf_names,
            'n_lfs': self.n_lfs
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        logger.info(f"Saved label model to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.n_classes = model_dict['n_classes']
        self.abstain_value = model_dict['abstain_value']
        self.min_evidence = model_dict['min_evidence']
        self.abstain_threshold = model_dict['abstain_threshold']
        self.lf_accuracies = model_dict['lf_accuracies']
        self.class_balance = model_dict['class_balance']
        self.conflict_resolution = model_dict['conflict_resolution']
        self.lf_coverage = model_dict['lf_coverage']
        self.lf_names = model_dict['lf_names']
        self.n_lfs = model_dict['n_lfs']
        
        logger.info(f"Loaded label model from {path}")
    
    def get_lf_summary(self) -> str:
        """Get summary of LF performance."""
        if self.lf_accuracies is None:
            return "Model not fitted yet"
        
        lines = ["Labeling Function Summary:"]
        lines.append("-" * 50)
        
        for i, name in enumerate(self.lf_names):
            lines.append(
                f"{name:20s} | "
                f"Acc: {self.lf_accuracies[i]:.3f} | "
                f"Cov: {self.lf_coverage[i]:.3f} | "
                f"Weight: {self.conflict_resolution[i]:.3f}"
            )
        
        lines.append("-" * 50)
        lines.append(f"Class balance: {self.class_balance}")
        
        return "\n".join(lines)


class SnorkelLabelModel:
    """
    Wrapper for actual Snorkel label model if available.
    Falls back to our implementation if Snorkel not installed.
    """
    
    def __init__(self, use_snorkel: bool = True, **kwargs):
        """
        Initialize label model.
        
        Args:
            use_snorkel: Whether to try using Snorkel
            **kwargs: Arguments for LabelModelVoter
        """
        self.use_snorkel = use_snorkel
        self.model = None
        
        if use_snorkel:
            try:
                from snorkel.labeling.model import LabelModel
                self.model = LabelModel(verbose=True)
                self.use_snorkel = True
                logger.info("Using Snorkel LabelModel")
            except ImportError:
                logger.warning("Snorkel not available, using custom implementation")
                self.use_snorkel = False
        
        if not self.use_snorkel:
            self.model = LabelModelVoter(**kwargs)
    
    def fit(self, L_train: np.ndarray, **kwargs):
        """Fit the label model."""
        if self.use_snorkel:
            # Snorkel expects specific format
            self.model.fit(L_train, **kwargs)
        else:
            self.model.fit(L_train, **kwargs)
    
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.use_snorkel:
            return self.model.predict_proba(L)
        else:
            # Our model works row by row
            n_samples = L.shape[0]
            probs = []
            
            for i in range(n_samples):
                result = self.model.predict_proba(L[i])
                if result['abstain']:
                    # Return uniform probs for abstention
                    probs.append(np.ones(self.model.n_classes) / self.model.n_classes)
                else:
                    # Convert dict to array
                    prob_array = np.array([
                        result['probs'].get('HIGH_RISK', 0),
                        result['probs'].get('MEDIUM_RISK', 0),
                        result['probs'].get('LOW_RISK', 0),
                        result['probs'].get('NO_RISK', 0)
                    ])
                    probs.append(prob_array)
            
            return np.array(probs)